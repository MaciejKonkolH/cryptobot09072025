"""
Plik z funkcjami pomocniczymi dla modułu treningowego training3.
"""
import logging
import os
import sys
import re
from pathlib import Path

# Import config jako alias, aby uniknąć konfliktu nazw
from training3 import config as trainer_config

def extract_future_window(filename):
    """Wyciąga future_window z nazwy pliku."""
    match = re.search(r'fw(\d+)m', filename)
    if match:
        return f"{match.group(1)} minut"
    return "nieznany"

def find_project_root(marker_file=".project_root"):
    """
    Znajduje główny katalog projektu, szukając pliku-znacznika (.project_root).

    Funkcja przeszukuje drzewo katalogów w górę, zaczynając od lokalizacji 
    pliku, w którym jest wywoływana.

    Returns:
        Path: Obiekt Path wskazujący na główny katalog projektu.
    Raises:
        FileNotFoundError: Jeśli plik-znacznik nie zostanie znaleziony.
    """
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / marker_file).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Nie znaleziono pliku znacznika '{marker_file}'. "
                            f"Upewnij się, że istnieje on w głównym katalogu projektu.")


def setup_logging():
    """Konfiguruje system logowania dla całego modułu."""
    log_dir = trainer_config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    log_filepath = os.path.join(log_dir, trainer_config.LOG_FILENAME)
    
    # Konfiguracja loggera
    logging.basicConfig(
        level=trainer_config.LOG_LEVEL,
        format=trainer_config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # force=True jest ważne, jeśli funkcja może być wołana wielokrotnie
    )
    
    return logging.getLogger(__name__)


def balance_classes(X, y, class_weights=None):
    """
    Balansuje klasy poprzez undersampling klasy większościowej.
    
    Args:
        X: DataFrame z cechami
        y: Series z etykietami
        class_weights: słownik z wagami klas
    
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    if class_weights is None:
        return X, y
    
    from sklearn.utils import resample
    import pandas as pd
    import numpy as np
    
    # Znajdź klasę większościową
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    majority_count = class_counts.max()
    
    # Sprawdź czy wszystkie klasy mają próbki
    available_classes = set(class_counts.index)
    required_classes = set(class_weights.keys())
    missing_classes = required_classes - available_classes
    
    if missing_classes:
        # Jeśli brakuje klas, zwróć oryginalne dane
        return X, y
    
    # Oblicz docelową liczbę próbek dla każdej klasy
    target_counts = {}
    for class_label, weight in class_weights.items():
        current_count = class_counts.get(class_label, 0)
        if current_count == 0:
            # Jeśli klasa nie ma próbek, pomiń ją
            continue
            
        if class_label == majority_class:
            # Klasa większościowa - zmniejsz do średniej z mniejszościowych
            minority_classes = [cls for cls in class_weights.keys() if cls != majority_class and class_counts.get(cls, 0) > 0]
            if minority_classes:
                minority_avg = np.mean([class_counts.get(cls, 0) for cls in minority_classes])
                target_counts[class_label] = max(1, int(minority_avg * weight))
            else:
                target_counts[class_label] = max(1, int(current_count * 0.5))  # Zmniejsz o połowę
        else:
            # Klasy mniejszościowe - zwiększ proporcjonalnie do wagi
            target_counts[class_label] = max(1, int(current_count * weight))
    
    # Balansuj każdą klasę
    balanced_X = []
    balanced_y = []
    
    for class_label in target_counts.keys():
        class_mask = y == class_label
        X_class = X[class_mask]
        y_class = y[class_mask]
        
        target_count = target_counts[class_label]
        
        if len(X_class) > target_count:
            # Undersampling
            X_resampled = resample(X_class, n_samples=target_count, random_state=42, replace=False)
            y_resampled = resample(y_class, n_samples=target_count, random_state=42, replace=False)
        else:
            # Oversampling
            X_resampled = resample(X_class, n_samples=target_count, random_state=42, replace=True)
            y_resampled = resample(y_class, n_samples=target_count, random_state=42, replace=True)
        
        balanced_X.append(X_resampled)
        balanced_y.append(y_resampled)
    
    # Połącz wszystkie klasy
    X_balanced = pd.concat(balanced_X, ignore_index=True)
    y_balanced = pd.concat(balanced_y, ignore_index=True)
    
    # Przetasuj dane
    from sklearn.utils import shuffle
    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)
    
    return X_balanced, y_balanced 


def save_training_results_to_markdown(evaluation_results, model_params, data_info, cfg):
    """Zapisuje wyniki treningu do pliku markdown w określonym formacie."""
    from datetime import datetime
    import os
    import numpy as np
    
    # Utwórz nazwę pliku z timestampem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"results_{timestamp}.md"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename
    
    logger = logging.getLogger(__name__)
    logger.info(f"Zapisuję wyniki treningu do: {filepath}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Nagłówek
        f.write(f"# WYNIKI TRENINGU - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Parametry modelu
        f.write("## 📊 PARAMETRY MODELU\n")
        f.write(f"- NAZWA PLIKU: {cfg.MODEL_FILENAME}\n")
        f.write(f"- N_ESTIMATORS: {model_params.get('n_estimators', 'N/A')}\n")
        f.write(f"- LEARNING_RATE: {model_params.get('learning_rate', 'N/A')}\n")
        f.write(f"- MAX_DEPTH: {model_params.get('max_depth', 'N/A')}\n")
        f.write(f"- SUBSAMPLE: {model_params.get('subsample', 'N/A')}\n")
        f.write(f"- COLSAMPLE_BYTREE: {model_params.get('colsample_bytree', 'N/A')}\n")
        f.write(f"- EARLY_STOPPING_ROUNDS: {model_params.get('early_stopping_rounds', 'N/A')}\n")
        f.write(f"- CLASS_WEIGHTS: {model_params.get('class_weights', 'N/A')}\n")
        f.write(f"- ENABLE_CLASS_WEIGHTS_IN_TRAINING: {model_params.get('enable_class_weights_in_training', 'N/A')}\n")
        f.write(f"- ENABLE_WEIGHTED_LOSS: {model_params.get('enable_weighted_loss', 'N/A')}\n")
        f.write(f"- GAMMA: {model_params.get('gamma', 'N/A')}\n")
        f.write(f"- RANDOM_STATE: {model_params.get('random_state', 'N/A')}\n\n")
        
        # Informacje o danych
        f.write("## 📋 INFORMACJE O DANYCH\n")
        f.write(f"- Liczba cech: {data_info.get('n_features', 'N/A')}\n")
        f.write(f"- Liczba próbek treningowych: {data_info.get('n_train', 'N/A'):,}\n")
        f.write(f"- Liczba próbek walidacyjnych: {data_info.get('n_val', 'N/A'):,}\n")
        f.write(f"- Liczba próbek testowych: {data_info.get('n_test', 'N/A'):,}\n")
        
        # Dodaj informację o future window
        future_window = extract_future_window(cfg.INPUT_FILENAME)
        f.write(f"- Future Window: {future_window}\n")
        
        f.write(f"- Zakres czasowy treningu: {data_info.get('train_range', 'N/A')}\n")
        f.write(f"- Zakres czasowy testu: {data_info.get('test_range', 'N/A')}\n\n")
        
        # Wyniki dla każdego poziomu
        f.write("## 📈 WYNIKI DLA KAŻDEGO POZIOMU TP/SL\n\n")
        
        for level_idx, level_desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            f.write(f"### {level_desc}:\n")
            
            # Znajdź dane dla tego poziomu
            level_data = None
            for label_col, results in evaluation_results.items():
                if results.get('level_desc') == level_desc:
                    level_data = results
                    break
            
            if level_data:
                # Oblicz rzeczywisty rozkład etykiet z confusion matrix
                conf_matrix = level_data.get('confusion_matrix', [])
                if len(conf_matrix) >= 3:
                    total_samples = sum(sum(conf_matrix))
                    long_count = sum(conf_matrix[0])
                    short_count = sum(conf_matrix[1])
                    neutral_count = sum(conf_matrix[2])
                    
                    f.write("Stosunek wszystkich etykiet w zbiorze testowym\n")
                    f.write(f"LONG {long_count/total_samples*100:.0f}%\n")
                    f.write(f"SHORT {short_count/total_samples*100:.0f}%\n")
                    f.write(f"NEUTRAL {neutral_count/total_samples*100:.0f}%\n\n")
                
                # Standardowe metryki (bez progów)
                accuracy = level_data.get('accuracy', 0)
                class_report = level_data.get('classification_report', {})
                
                f.write("Standardowe metryki (bez progów):\n")
                
                if len(conf_matrix) >= 3:
                    f.write("Predicted\n")
                    f.write("Actual    LONG  SHORT  NEUTRAL\n")
                    f.write(f"LONG      {conf_matrix[0][0]:<6} {conf_matrix[0][1]:<6} {conf_matrix[0][2]:<6}\n")
                    f.write(f"SHORT     {conf_matrix[1][0]:<6} {conf_matrix[1][1]:<6} {conf_matrix[1][2]:<6}\n")
                    f.write(f"NEUTRAL   {conf_matrix[2][0]:<6} {conf_matrix[2][1]:<6} {conf_matrix[2][2]:<6}\n\n")
                
                f.write(f"Accuracy: {accuracy:.4f}\n")
                
                # Metryki per klasa
                for class_name in ['LONG', 'SHORT', 'NEUTRAL']:
                    if class_name in class_report:
                        precision = class_report[class_name].get('precision', 0)
                        recall = class_report[class_name].get('recall', 0)
                        f1 = class_report[class_name].get('f1-score', 0)
                        f.write(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n")
                
                # Analiza dochodowości dla standardowych metryk
                tp, sl = _get_tp_sl_values(level_desc)
                if tp and sl and conf_matrix is not None:
                    long_profit = _calculate_profit_from_confusion_matrix(conf_matrix, tp, sl, 'LONG')
                    short_profit = _calculate_profit_from_confusion_matrix(conf_matrix, tp, sl, 'SHORT')
                    combined_profit = _calculate_combined_profit_from_confusion_matrix(conf_matrix, tp, sl)
                    
                    # Oblicz precision dla wyświetlenia
                    long_precision = class_report.get('LONG', {}).get('precision', 0) if 'LONG' in class_report else 0
                    short_precision = class_report.get('SHORT', {}).get('precision', 0) if 'SHORT' in class_report else 0
                    
                    f.write(f"\n{level_desc} - LONG: {long_precision*100:.1f}% (dochód netto ~{long_profit:.1f}%)\n")
                    f.write(f"{level_desc} - SHORT: {short_precision*100:.1f}% (dochód netto ~{short_profit:.1f}%)\n")
                    f.write(f"{level_desc} - ŁĄCZNY DOCHÓD: {combined_profit:.1f}%\n")
                
                # Prawdziwe wyniki dla różnych progów pewności
                confidence_results = level_data.get('confidence_results', {})
                if confidence_results:
                    # Użyj progów z main.py (0.3, 0.4, 0.5, 0.6) i konwertuj na procenty
                    confidence_thresholds = [30.0, 40.0, 50.0, 60.0]  # Konwersja z 0.3, 0.4, 0.5, 0.6
                    
                    for threshold_percent in confidence_thresholds:
                        threshold = threshold_percent / 100.0  # Konwersja z powrotem na 0.5, 0.7, 0.8, 0.9
                        
                        if threshold in confidence_results and confidence_results[threshold] is not None:
                            conf_result = confidence_results[threshold]
                            
                            f.write(f"\nProgi pewności {threshold_percent:.1f}%:\n")
                            
                            # Confusion matrix dla tego progu
                            high_conf_conf_matrix = conf_result['confusion_matrix']
                            if len(high_conf_conf_matrix) >= 3:
                                f.write("Predicted\n")
                                f.write("Actual    LONG  SHORT  NEUTRAL\n")
                                f.write(f"LONG      {high_conf_conf_matrix[0][0]:<6} {high_conf_conf_matrix[0][1]:<6} {high_conf_conf_matrix[0][2]:<6}\n")
                                f.write(f"SHORT     {high_conf_conf_matrix[1][0]:<6} {high_conf_conf_matrix[1][1]:<6} {high_conf_conf_matrix[1][2]:<6}\n")
                                f.write(f"NEUTRAL   {high_conf_conf_matrix[2][0]:<6} {high_conf_conf_matrix[2][1]:<6} {high_conf_conf_matrix[2][2]:<6}\n\n")
                            
                            # Metryki dla tego progu
                            f.write(f"Próbki z wysoką pewnością: {conf_result['n_high_conf']:,}/{conf_result['n_total']:,} ({conf_result['percentage']:.1f}%)\n")
                            f.write(f"Accuracy: {conf_result['accuracy']:.4f}\n")
                            
                            # Metryki per klasa dla tego progu
                            high_conf_class_report = conf_result['classification_report']
                            for class_name in ['LONG', 'SHORT', 'NEUTRAL']:
                                if class_name in high_conf_class_report:
                                    precision = high_conf_class_report[class_name].get('precision', 0)
                                    recall = high_conf_class_report[class_name].get('recall', 0)
                                    f1 = high_conf_class_report[class_name].get('f1-score', 0)
                                    f.write(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n")
                            
                            # Analiza dochodowości dla tego progu
                            if tp and sl and high_conf_conf_matrix is not None:
                                long_profit = _calculate_profit_from_confusion_matrix(high_conf_conf_matrix, tp, sl, 'LONG')
                                short_profit = _calculate_profit_from_confusion_matrix(high_conf_conf_matrix, tp, sl, 'SHORT')
                                combined_profit = _calculate_combined_profit_from_confusion_matrix(high_conf_conf_matrix, tp, sl)
                                
                                # Oblicz precision dla wyświetlenia
                                long_precision = high_conf_class_report.get('LONG', {}).get('precision', 0) if 'LONG' in high_conf_class_report else 0
                                short_precision = high_conf_class_report.get('SHORT', {}).get('precision', 0) if 'SHORT' in high_conf_class_report else 0
                                
                                f.write(f"\n{level_desc} - LONG: {long_precision*100:.1f}% (dochód netto ~{long_profit:.1f}%)\n")
                                f.write(f"{level_desc} - SHORT: {short_precision*100:.1f}% (dochód netto ~{short_profit:.1f}%)\n")
                                f.write(f"{level_desc} - ŁĄCZNY DOCHÓD: {combined_profit:.1f}%\n")
                            
                            f.write("\n" + "-" * 68 + "\n")
                        else:
                            f.write(f"\nProgi pewności {threshold_percent:.1f}%:\n")
                            f.write("Brak próbek z tak wysoką pewnością\n")
                            f.write("\n" + "-" * 68 + "\n")
                else:
                    f.write("\nBrak danych o progach pewności\n")
                    f.write("\n" + "-" * 68 + "\n")
            
            f.write("\n" + "+" * 68 + "\n\n")
    
    logger.info(f"Wyniki treningu zapisane: {filepath}")
    return filepath


def _get_tp_sl_values(level_desc):
    """Wyciąga wartości TP i SL z opisu poziomu."""
    import re
    match = re.search(r'TP: ([\d.]+)%, SL: ([\d.]+)%', level_desc)
    if match:
        tp = float(match.group(1))
        sl = float(match.group(2))
        return tp, sl
    return None, None


def _calculate_profit_from_confusion_matrix(confusion_matrix, tp, sl, signal_type):
    """Oblicza dochód netto na podstawie confusion matrix z procentem składanym.
    
    Confusion matrix format:
    Predicted
    Actual    LONG  SHORT  NEUTRAL
    LONG      [0][0] [0][1] [0][2]
    SHORT     [1][0] [1][1] [1][2]
    NEUTRAL   [2][0] [2][1] [2][2]
    """
    if tp <= 0 or sl <= 0 or len(confusion_matrix) < 3:
        return 0.0
    
    if signal_type == 'LONG':
        # Dla LONG: kolumna 0 (przewidziane jako LONG)
        # confusion_matrix[0][0] = rzeczywiste LONG przewidziane jako LONG (zyskowne)
        # confusion_matrix[1][0] + confusion_matrix[2][0] = rzeczywiste SHORT/NEUTRAL przewidziane jako LONG (stratne)
        profitable = confusion_matrix[0][0]  # Prawidłowe LONG
        losing = confusion_matrix[1][0] + confusion_matrix[2][0]  # Błędne LONG
        
    elif signal_type == 'SHORT':
        # Dla SHORT: kolumna 1 (przewidziane jako SHORT)
        # confusion_matrix[1][1] = rzeczywiste SHORT przewidziane jako SHORT (zyskowne)
        # confusion_matrix[0][1] + confusion_matrix[2][1] = rzeczywiste LONG/NEUTRAL przewidziane jako SHORT (stratne)
        profitable = confusion_matrix[1][1]  # Prawidłowe SHORT
        losing = confusion_matrix[0][1] + confusion_matrix[2][1]  # Błędne SHORT
    else:
        return 0.0
    
    total_trades = profitable + losing
    if total_trades == 0:
        return 0.0
    
    # Procent składany: Kapitał_końcowy = Kapitał_początkowy × (1 + TP/100)^profitable × (1 - SL/100)^losing
    tp_multiplier = (1 + tp / 100) ** profitable
    sl_multiplier = (1 - sl / 100) ** losing
    
    final_capital = tp_multiplier * sl_multiplier
    
    # Dochód netto w procentach
    profit_percent = (final_capital - 1) * 100
    
    return profit_percent


def _calculate_combined_profit_from_confusion_matrix(confusion_matrix, tp, sl):
    """Oblicza łączny dochód netto z wszystkich transakcji LONG i SHORT razem."""
    if tp <= 0 or sl <= 0 or len(confusion_matrix) < 3:
        return 0.0
    
    # LONG transakcje (kolumna 0)
    profitable_long = confusion_matrix[0][0]  # Prawidłowe LONG
    losing_long = confusion_matrix[1][0] + confusion_matrix[2][0]  # Błędne LONG
    
    # SHORT transakcje (kolumna 1)
    profitable_short = confusion_matrix[1][1]  # Prawidłowe SHORT
    losing_short = confusion_matrix[0][1] + confusion_matrix[2][1]  # Błędne SHORT
    
    # Łączne liczby
    total_profitable = profitable_long + profitable_short
    total_losing = losing_long + losing_short
    total_trades = total_profitable + total_losing
    
    if total_trades == 0:
        return 0.0
    
    # Łączny procent składany
    tp_multiplier = (1 + tp / 100) ** total_profitable
    sl_multiplier = (1 - sl / 100) ** total_losing
    
    final_capital = tp_multiplier * sl_multiplier
    
    # Łączny dochód netto w procentach
    combined_profit_percent = (final_capital - 1) * 100
    
    return combined_profit_percent 