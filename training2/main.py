"""
Główny skrypt orkiestrujący proces treningu modelu.
"""
import sys
import os
from pathlib import Path

# --- Dynamiczne dodawanie ścieżki projektu ---
def _find_project_root(start_path: Path, marker_file: str = ".project_root") -> Path:
    path = start_path
    while path.parent != path:
        if (path / marker_file).is_file():
            return path
        path = path.parent
    raise FileNotFoundError(f"Nie znaleziono pliku znacznika '{marker_file}' w żadnym z nadrzędnych katalogów.")

try:
    project_root = _find_project_root(Path(__file__).resolve().parent)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except FileNotFoundError as e:
    print(f"BŁĄD KRYTYCZNY: {e}", file=sys.stderr)
    print("Upewnij się, że plik .project_root istnieje w głównym katalogu projektu.", file=sys.stderr)
    sys.exit(1)
# --- Koniec dynamicznego dodawania ścieżki ---

import json
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Importy z naszego modułu
from training2 import config as cfg
from training2.utils import setup_logging
from training2.reporter import MultiOutputTrainingReporter

logger = setup_logging()

class Trainer:
    """
    Główna klasa zarządzająca całym procesem treningowym z użyciem XGBoost.
    """
    def __init__(self):
        """Inicjalizuje i tworzy niezbędne katalogi."""
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        os.makedirs(cfg.REPORT_DIR, exist_ok=True)
        self.scaler = None
        self.model = None
        self.start_time = None
        self.end_time = None
        self.evaluation_results = None
        self.feature_names = None
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
    
    def _log_config_summary(self):
        """Loguje podsumowanie kluczowych parametrów treningu."""
        logger.info("=" * 60)
        logger.info("PODSUMOWANIE KONFIGURACJI TRENINGU (XGBoost)")
        logger.info("=" * 60)
        logger.info("Dane Wejściowe:")
        logger.info(f"  - Plik: {cfg.INPUT_FILENAME}")
        logger.info(f"  - Cechy: {cfg.FEATURES}")
        logger.info("-" * 60)
        logger.info("Parametry Podziału Danych:")
        logger.info(f"  - Podział Walidacyjny: {cfg.VALIDATION_SPLIT:.0%}")
        logger.info(f"  - Podział Testowy: {cfg.TEST_SPLIT:.0%}")
        logger.info("=" * 60)

    def run(self):
        """Uruchamia cały potok treningowy."""
        self.start_time = time.time()
        self._log_config_summary()

        logger.info(">>> KROK 1: Wczytywanie i Przygotowanie Danych <<<")
        df = self._load_data()
        if df is None:
            return

        logger.info(">>> KROK 2: Chronologiczny Podział na Zbiory <<<")
        self._split_data(df)

        # DIAGNOSTYKA: Sprawdź które cechy zawierają infinity
        logger.info("DIAGNOSTYKA INFINITY:")
        inf_mask = np.isinf(self.X_train)
        if inf_mask.any().any():  # Naprawka: .any() dla DataFrame
            inf_columns = self.X_train.columns[inf_mask.any(axis=0)]
            logger.error(f"Znaleziono {inf_mask.sum().sum()} wartości infinity")
            logger.error(f"Kolumny z inf: {list(inf_columns)}")
            
            # Sprawdź każdą kolumnę z inf
            for col in inf_columns:
                inf_count = inf_mask[col].sum()
                logger.error(f"  {col}: {inf_count} wartości inf")
                
                # Pokaż przykładowe wartości
                sample_values = self.X_train[col].head(10)
                logger.error(f"    Przykładowe wartości: {sample_values.values}")
        else:
            logger.info("Brak wartości infinity w danych")
        
        # Sprawdź też NaN
        nan_mask = np.isnan(self.X_train)
        if nan_mask.any().any():  # Naprawka: .any() dla DataFrame
            nan_columns = self.X_train.columns[nan_mask.any(axis=0)]
            logger.error(f"Znaleziono {nan_mask.sum().sum()} wartości NaN")
            logger.error(f"Kolumny z NaN: {list(nan_columns)}")
        
        logger.info(">>> KROK 3: Skalowanie Cech <<<")
        self._scale_features()

        logger.info(">>> KROK 4: Budowa, Trening i Ocena Modelu XGBoost <<<")
        self._build_train_and_evaluate()
        
        self._save_artifacts()

        self.end_time = time.time()
        logger.info("--- Proces treningowy (XGBoost) zakończony pomyślnie! ---")

        # Proste podsumowanie na koniec
        logger.info(f"Czas trwania: {(self.end_time - self.start_time):.2f} sekund.")
        logger.info(f"Model i raporty zapisane w: {cfg.MODEL_DIR}")
        logger.info(f"Ważność cech zapisana w: {os.path.join(cfg.REPORT_DIR, 'feature_importance.png')}")

    def _load_data(self) -> pd.DataFrame | None:
        """Wczytuje, filtruje i waliduje dane."""
        if not os.path.exists(cfg.INPUT_FILE_PATH):
            logger.error(f"Plik wejściowy nie istnieje: {cfg.INPUT_FILE_PATH}")
            return None
        
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
        
        # --- KROK DIAGNOSTYCZNY ---
        logger.info(f"Kolumny faktycznie wczytane w training2/main.py: {df.columns.tolist()}")
        # -------------------------

        # Ustaw indeks na timestamp jeśli istnieje
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        if cfg.ENABLE_DATE_FILTER:
            logger.info(f"Filtrowanie danych do zakresu: {cfg.START_DATE} - {cfg.END_DATE}")
            df = df[(df.index >= pd.to_datetime(cfg.START_DATE)) & (df.index <= pd.to_datetime(cfg.END_DATE))]
        
        logger.info(f"Liczba wierszy po wczytaniu: {len(df):,}")
        
        # Sprawdzamy, czy wszystkie potrzebne kolumny (cechy + etykiety) istnieją
        required_cols = set(cfg.FEATURES + cfg.LABEL_COLUMNS)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"W danych brakuje wymaganych kolumn: {missing_cols}. Zakończono.")
            return None
            
        # Loguj informacje o etykietach
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            unique_labels = df[label_col].value_counts().sort_index()
            logger.info(f"Rozkład etykiet {cfg.TP_SL_LEVELS_DESC[i]} ({label_col}): {unique_labels.to_dict()}")
            
        return df

    def _split_data(self, df: pd.DataFrame):
        """Dzieli dane stratyfikowane na X i y (multi-output) aby zapewnić reprezentację wszystkich klas."""
        from sklearn.model_selection import train_test_split
        
        self.feature_names = cfg.FEATURES
        X = df[self.feature_names]
        
        # y jest teraz macierzą z 3 kolumnami etykiet
        y_raw = df[cfg.LABEL_COLUMNS].values  # Shape: (n_samples, 3)
        
        # Preprocessing etykiet - XGBoost wymaga ciągłych indeksów klas
        from sklearn.preprocessing import LabelEncoder
        self.label_encoders = []
        y_encoded = np.zeros_like(y_raw)
        
        for i in range(y_raw.shape[1]):
            encoder = LabelEncoder()
            y_encoded[:, i] = encoder.fit_transform(y_raw[:, i])
            self.label_encoders.append(encoder)
            
            # Loguj mapowanie
            level_desc = cfg.TP_SL_LEVELS_DESC[i]
            unique_original = encoder.classes_
            unique_encoded = np.arange(len(unique_original))
            logger.info(f"Mapowanie etykiet dla {level_desc}:")
            for orig, enc in zip(unique_original, unique_encoded):
                class_name = cfg.CLASS_LABELS.get(orig, f'UNKNOWN_{orig}')
                logger.info(f"  {orig} ({class_name}) -> {enc}")
                
            # Sprawdź czy mamy ciągłe indeksy
            encoded_unique = np.unique(y_encoded[:, i])
            logger.info(f"Zakodowane klasy: {encoded_unique} (oczekiwane: {np.arange(len(encoded_unique))})")
            assert np.array_equal(encoded_unique, np.arange(len(encoded_unique))), f"Błąd kodowania dla poziomu {i}"

        # NAPRAWKA: Używam chronologicznego podziału zamiast stratyfikowanego
        logger.info("Używam chronologicznego podziału danych zamiast stratyfikowanego...")
        
        # Oblicz rozmiary zbiorów
        total_samples = len(X)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        # Chronologiczny podział
        self.X_train = X.iloc[:train_size]
        self.X_val = X.iloc[train_size:train_size+val_size]
        self.X_test = X.iloc[train_size+val_size:]
        
        self.y_train = y_encoded[:train_size]
        self.y_val = y_encoded[train_size:train_size+val_size]
        self.y_test = y_encoded[train_size+val_size:]

        logger.info(f"Podział danych (chronologiczny):")
        logger.info(f"  Trening: {len(self.X_train):,} próbek")
        logger.info(f"  Walidacja: {len(self.X_val):,} próbek")
        logger.info(f"  Test: {len(self.X_test):,} próbek")
        logger.info(f"  Wymiary y_train: {self.y_train.shape}")  # (n_samples, 3)
        
        # DIAGNOSTYKA: Sprawdź chronologię podziału
        if hasattr(self.X_train, 'index'):
            logger.info("DIAGNOSTYKA CHRONOLOGII:")
            logger.info(f"  Train: {self.X_train.index.min()} - {self.X_train.index.max()}")
            logger.info(f"  Val:   {self.X_val.index.min()} - {self.X_val.index.max()}")
            logger.info(f"  Test:  {self.X_test.index.min()} - {self.X_test.index.max()}")
            
            # Sprawdź czy nie ma overlap
            if self.X_train.index.max() >= self.X_val.index.min():
                logger.warning("UWAGA: Overlap między train a val!")
            if self.X_val.index.max() >= self.X_test.index.min():
                logger.warning("UWAGA: Overlap między val a test!")
            if self.X_train.index.max() >= self.X_test.index.min():
                logger.warning("UWAGA: Overlap między train a test!")

        # Sprawdź reprezentację klas w każdym zbiorze dla każdego poziomu
        for i, level_desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            logger.info(f"Rozkład klas w zbiorze treningowym dla {level_desc}:")
            unique, counts = np.unique(self.y_train[:, i], return_counts=True)
            for cls, count in zip(unique, counts):
                logger.info(f"  Klasa {cls}: {count} próbek")

    def _scale_features(self):
        """Dopasowuje skaler na danych treningowych i transformuje wszystkie zbiory."""
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        logger.info("Cechy zostały przeskalowane za pomocą RobustScaler.")

    def _build_train_and_evaluate(self):
        """Buduje, trenuje i ocenia 3 osobne modele XGBoost."""
        
        # Przygotuj parametry XGBoost
        xgb_params = {
            'n_estimators': cfg.XGB_N_ESTIMATORS,
            'learning_rate': cfg.XGB_LEARNING_RATE,
            'max_depth': cfg.XGB_MAX_DEPTH,
            'subsample': cfg.XGB_SUBSAMPLE,
            'colsample_bytree': cfg.XGB_COLSAMPLE_BYTREE,
            'gamma': cfg.XGB_GAMMA,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        logger.info("Rozpoczynanie treningu 3 osobnych modeli XGBoost...")
        logger.info(f"Parametry XGBoost: {xgb_params}")
        
        # Trenuj 3 osobne modele
        self.models = []
        
        for i, (label_col, level_desc) in enumerate(zip(cfg.LABEL_COLUMNS, cfg.TP_SL_LEVELS_DESC)):
            logger.info(f"\n--- Trening modelu {i+1}/3: {level_desc} ---")
            
            # Pobierz etykiety dla tego poziomu
            y_train_level = self.y_train[:, i]
            y_val_level = self.y_val[:, i]
            
            # Sprawdź unikalne klasy w tym poziomie
            unique_classes_train = np.unique(y_train_level)
            unique_classes_all = np.unique(self.y_train[:, i])  # To samo co wyżej, ale dla jasności
            logger.info(f"Klasy w zbiorze treningowym: {unique_classes_train}")
            
            # Sprawdź czy brakuje niektórych klas w zbiorze treningowym
            all_possible_classes = np.arange(len(self.label_encoders[i].classes_))
            missing_classes = set(all_possible_classes) - set(unique_classes_train)
            if missing_classes:
                logger.warning(f"UWAGA: Brakujące klasy w zbiorze treningowym: {missing_classes}")
                logger.warning(f"Może to wpłynąć na wydajność modelu dla tych klas.")
            
            # XGBoost poradzi sobie z brakującymi klasami, ale zarejestrujmy to
            logger.info(f"Oczekiwane klasy: {all_possible_classes}")
            logger.info(f"Dostępne klasy: {unique_classes_train}")
            
            # Utwórz i wytrenuj model z jawną liczbą klas
            num_classes = len(self.label_encoders[i].classes_)
            model_params = xgb_params.copy()
            model_params['objective'] = 'multi:softprob'
            model_params['num_class'] = num_classes
            
            model = xgb.XGBClassifier(**model_params)
            model.fit(
                self.X_train, y_train_level,
                eval_set=[(self.X_val, y_val_level)],
                verbose=False
            )
            
            self.models.append(model)
            logger.info(f"Model {i+1} wytrenowany pomyślnie.")
        
        logger.info("Wszystkie 3 modele wytrenowane.")
        
        # Ewaluacja na zbiorze testowym
        logger.info("Rozpoczynanie ewaluacji na zbiorze testowym...")
        
        # Przygotuj wyniki dla każdego poziomu TP/SL
        self.evaluation_results = {}
        
        for i, (label_col, level_desc) in enumerate(zip(cfg.LABEL_COLUMNS, cfg.TP_SL_LEVELS_DESC)):
            logger.info(f"\n--- Ewaluacja dla poziomu: {level_desc} ---")
            
            # === WALIDACJA ===
            logger.info(f"\n--- WYNIKI WALIDACYJNE dla {level_desc} ---")
            
            # Predykcja na zbiorze walidacyjnym
            y_val_pred_level = self.models[i].predict(self.X_val)
            y_val_true_level = self.y_val[:, i]
            
            # Dekoduj etykiety walidacyjne
            y_val_true_decoded = self.label_encoders[i].inverse_transform(y_val_true_level)
            y_val_pred_decoded = self.label_encoders[i].inverse_transform(y_val_pred_level)
            
            # Oblicz metryki walidacyjne
            val_accuracy = accuracy_score(y_val_true_decoded, y_val_pred_decoded)
            val_conf_matrix = confusion_matrix(y_val_true_decoded, y_val_pred_decoded)
            
            logger.info(f"VAL_ACCURACY {level_desc}: {val_accuracy:.4f}")
            
            # Wyświetl confusion matrix walidacyjną
            logger.info(f"CONFUSION MATRIX (WALIDACJA) dla {level_desc}:")
            logger.info("=" * 60)
            
            # Przygotuj etykiety dla confusion matrix
            unique_classes = self.label_encoders[i].classes_
            class_labels = [cfg.CLASS_LABELS.get(cls, f'CLASS_{cls}') for cls in unique_classes]
            
            # Wyświetl confusion matrix w czytelnym formacie
            logger.info("Predykcja ->")
            logger.info("Rzeczywistosc v")
            logger.info(" " * 15 + " | " + " | ".join(f"{label:>12}" for label in class_labels))
            logger.info("-" * 60)
            
            for j, true_label in enumerate(class_labels):
                row = val_conf_matrix[j]
                row_str = f"{true_label:>12} | " + " | ".join(f"{val:>12}" for val in row)
                logger.info(row_str)
            
            logger.info("=" * 60)
            
            # Analiza błędów walidacyjnych
            total_val_samples = val_conf_matrix.sum()
            correct_val_predictions = val_conf_matrix.diagonal().sum()
            val_overall_accuracy = correct_val_predictions / total_val_samples
            
            logger.info(f"VAL - Całkowita liczba próbek: {total_val_samples}")
            logger.info(f"VAL - Poprawne predykcje: {correct_val_predictions}")
            logger.info(f"VAL - Ogólna dokładność: {val_overall_accuracy:.4f}")
            
            # === TEST ===
            logger.info(f"\n--- WYNIKI TESTOWE dla {level_desc} ---")
            
            # Predykcja dla tego modelu na zbiorze testowym
            y_pred_level = self.models[i].predict(self.X_test)
            y_true_level = self.y_test[:, i]
            
            # Dekoduj etykiety z powrotem do oryginalnych wartości
            y_true_decoded = self.label_encoders[i].inverse_transform(y_true_level)
            y_pred_decoded = self.label_encoders[i].inverse_transform(y_pred_level)
            
            # Oblicz metryki na zdekodowanych etykietach
            test_accuracy = accuracy_score(y_true_decoded, y_pred_decoded)
            
            # Przygotuj nazwy klas dla raportu
            target_names = [cfg.CLASS_LABELS.get(cls, f'CLASS_{cls}') for cls in unique_classes]
            
            class_report = classification_report(
                y_true_decoded, y_pred_decoded, 
                target_names=target_names,
                output_dict=True, zero_division=0
            )
            test_conf_matrix = confusion_matrix(y_true_decoded, y_pred_decoded)
            
            # Zapisz wyniki
            self.evaluation_results[label_col] = {
                'level_desc': level_desc,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
                'classification_report': class_report,
                'val_confusion_matrix': val_conf_matrix,
                'test_confusion_matrix': test_conf_matrix,
                'y_val_true': y_val_true_decoded,
                'y_val_pred': y_val_pred_decoded,
                'y_test_true': y_true_decoded,
                'y_test_pred': y_pred_decoded,
                'unique_classes': unique_classes
            }
            
            logger.info(f"TEST_ACCURACY {level_desc}: {test_accuracy:.4f}")
            
            # Wyświetl confusion matrix testową
            logger.info(f"CONFUSION MATRIX (TEST) dla {level_desc}:")
            logger.info("=" * 60)
            
            # Wyświetl confusion matrix w czytelnym formacie
            logger.info("Predykcja ->")
            logger.info("Rzeczywistosc v")
            logger.info(" " * 15 + " | " + " | ".join(f"{label:>12}" for label in class_labels))
            logger.info("-" * 60)
            
            for j, true_label in enumerate(class_labels):
                row = test_conf_matrix[j]
                row_str = f"{true_label:>12} | " + " | ".join(f"{val:>12}" for val in row)
                logger.info(row_str)
            
            logger.info("=" * 60)
            
            # Analiza błędów testowych
            total_test_samples = test_conf_matrix.sum()
            correct_test_predictions = test_conf_matrix.diagonal().sum()
            test_overall_accuracy = correct_test_predictions / total_test_samples
            
            logger.info(f"TEST - Całkowita liczba próbek: {total_test_samples}")
            logger.info(f"TEST - Poprawne predykcje: {correct_test_predictions}")
            logger.info(f"TEST - Ogólna dokładność: {test_overall_accuracy:.4f}")
            
            # Porównanie walidacja vs test
            logger.info(f"\n--- PORÓWNANIE WALIDACJA vs TEST ---")
            logger.info(f"VAL_ACCURACY:  {val_accuracy:.4f}")
            logger.info(f"TEST_ACCURACY: {test_accuracy:.4f}")
            logger.info(f"RÓŻNICA:       {val_accuracy - test_accuracy:.4f}")
            
            if val_accuracy > test_accuracy:
                logger.info("UWAGA: Val > Test - możliwy overfitting lub data leakage!")
            elif test_accuracy > val_accuracy:
                logger.info("UWAGA: Test > Val - nietypowe, sprawdź podział danych!")
            else:
                logger.info("OK: Val ≈ Test - dobra generalizacja")
            
            logger.info("")
            
            # Loguj metryki dla klas PROFIT_SHORT i PROFIT_LONG (test)
            for class_idx in cfg.FOCUS_CLASSES:
                class_name = cfg.CLASS_LABELS[class_idx]
                # Sprawdź czy klasa istnieje w tym poziomie
                if class_idx in unique_classes:
                    class_key = target_names[list(unique_classes).index(class_idx)]
                    if class_key in class_report:
                        precision = class_report[class_key]['precision']
                        recall = class_report[class_key]['recall']
                        f1 = class_report[class_key]['f1-score']
                        logger.info(f"  {class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                else:
                    logger.info(f"  {class_name}: Brak w tym poziomie TP/SL")

        logger.info("Ewaluacja zakończona.")

    def _generate_feature_importance_report(self):
        """Generuje raport ważności cech dla 3 osobnych modeli."""
        logger.info("Generowanie raportu ważności cech...")
        
        # Pobierz ważności cech z każdego modelu
        feature_importances = {}
        
        for i, (label_col, level_desc) in enumerate(zip(cfg.LABEL_COLUMNS, cfg.TP_SL_LEVELS_DESC)):
            # Pobierz ważności z i-tego modelu
            model = self.models[i]
            importances = model.feature_importances_
            
            # Stwórz DataFrame z ważnościami
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            feature_importances[label_col] = {
                'level_desc': level_desc,
                'importances': importance_df
            }
            
            logger.info(f"Top 10 cech dla {level_desc}:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Zapisz wykresy ważności cech
        if cfg.SAVE_PLOTS:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle('Feature Importance - 3 Separate XGBoost Models', fontsize=16)
            
            for i, (label_col, data) in enumerate(feature_importances.items()):
                importance_df = data['importances']
                level_desc = data['level_desc']
                
                # Wykres dla top 15 cech
                top_features = importance_df.head(15)
                axes[i].barh(range(len(top_features)), top_features['importance'])
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features['feature'])
                axes[i].set_xlabel('Importance')
                axes[i].set_title(f'{level_desc}')
                axes[i].invert_yaxis()
            
            plt.tight_layout()
            plot_path = os.path.join(cfg.REPORT_DIR, 'feature_importance_3models.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Wykres ważności cech zapisany: {plot_path}")
        
        # Zapisz ważności do pliku CSV
        for label_col, data in feature_importances.items():
            csv_path = os.path.join(cfg.REPORT_DIR, f'feature_importance_{label_col}.csv')
            data['importances'].to_csv(csv_path, index=False)
            logger.info(f"Ważności cech zapisane: {csv_path}")


    def _save_artifacts(self):
        """Zapisuje 3 modele, scaler i wyniki."""
        # Zapisz 3 modele
        for i, (label_col, level_desc) in enumerate(zip(cfg.LABEL_COLUMNS, cfg.TP_SL_LEVELS_DESC)):
            model_filename = f"model_{label_col}.pkl"
            model_path = os.path.join(cfg.MODEL_DIR, model_filename)
            joblib.dump(self.models[i], model_path)
            logger.info(f"Model {i+1} ({level_desc}) zapisany: {model_path}")
        
        # Zapisz scaler
        scaler_path = os.path.join(cfg.MODEL_DIR, cfg.SCALER_FILENAME)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler zapisany: {scaler_path}")
        
        # Zapisz label encoders
        encoders_path = os.path.join(cfg.MODEL_DIR, "label_encoders.pkl")
        joblib.dump(self.label_encoders, encoders_path)
        logger.info(f"Label encoders zapisane: {encoders_path}")
        
        # Zapisz wyniki ewaluacji
        results_path = os.path.join(cfg.REPORT_DIR, 'evaluation_results.json')
        
        # Przygotuj wyniki do serializacji JSON
        json_results = {}
        for label_col, results in self.evaluation_results.items():
            json_results[label_col] = {
                'level_desc': results['level_desc'],
                'val_accuracy': float(results['val_accuracy']),
                'test_accuracy': float(results['test_accuracy']),
                'classification_report': results['classification_report'],
                'val_confusion_matrix': results['val_confusion_matrix'].tolist(),
                'test_confusion_matrix': results['test_confusion_matrix'].tolist()
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Wyniki ewaluacji zapisane: {results_path}")
        
        # Generuj raport ważności cech
        self._generate_feature_importance_report()

def main():
    """Główna funkcja uruchamiająca proces."""
    trainer = Trainer()
    trainer.run()

if __name__ == "__main__":
    main() 