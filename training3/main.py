"""
Główny skrypt orkiestrujący proces treningu modelu Multi-Output XGBoost.
"""
import sys
import os
import time
import json
import joblib
from pathlib import Path

# Dodaj ścieżkę projektu do sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importy z naszego modułu
from training3 import config as cfg
from training3.utils import setup_logging, save_training_results_to_markdown
from training5.report import save_markdown_report as save_markdown_report_t5
from training5.report import save_json_report as save_json_report_t5
from training3.data_loader import DataLoader
from training3.model_builder import MultiOutputXGBoost

logger = setup_logging()

class Trainer:
    """
    Główna klasa zarządzająca całym procesem treningowym Multi-Output XGBoost.
    """
    
    def __init__(self):
        """Inicjalizuje i tworzy niezbędne katalogi."""
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        os.makedirs(cfg.REPORT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        
        self.data_loader = DataLoader()
        self.model = MultiOutputXGBoost()
        self.start_time = None
        self.end_time = None
        self.evaluation_results = None
        
    def _log_config_summary(self):
        """Loguje podsumowanie kluczowych parametrów treningu."""
        logger.info("=" * 60)
        logger.info("PODSUMOWANIE KONFIGURACJI TRENINGU (Multi-Output XGBoost)")
        logger.info("=" * 60)
        logger.info("Dane Wejściowe:")
        logger.info(f"  - Plik: {cfg.INPUT_FILENAME}")
        logger.info(f"  - Oczekiwane cechy: {len(cfg.FEATURES)} cech")
        logger.info(f"  - Poziomy TP/SL: {len(cfg.LABEL_COLUMNS)}")
        logger.info(f"  - Nazwa pliku modelu: {cfg.MODEL_FILENAME}")
        logger.info("-" * 60)
        logger.info("Parametry Podziału Danych:")
        logger.info(f"  - Podział Walidacyjny: {cfg.VALIDATION_SPLIT:.0%}")
        logger.info(f"  - Podział Testowy: {cfg.TEST_SPLIT:.0%}")
        logger.info("-" * 60)
        logger.info("Parametry Modelu:")
        logger.info(f"  - Liczba drzew: {cfg.XGB_N_ESTIMATORS}")
        logger.info(f"  - Learning rate: {cfg.XGB_LEARNING_RATE}")
        logger.info(f"  - Max depth: {cfg.XGB_MAX_DEPTH}")
        logger.info(f"  - Balansowanie klas: {'WŁĄCZONE' if cfg.ENABLE_CLASS_BALANCING else 'WYŁĄCZONE'}")
        logger.info(f"  - Wagi klas w treningu: {'WŁĄCZONE' if hasattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING') and cfg.ENABLE_CLASS_WEIGHTS_IN_TRAINING else 'WYŁĄCZONE'}")
        if hasattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING') and cfg.ENABLE_CLASS_WEIGHTS_IN_TRAINING:
            logger.info(f"  - Wagi: LONG={cfg.CLASS_WEIGHTS[0]}, SHORT={cfg.CLASS_WEIGHTS[1]}, NEUTRAL={cfg.CLASS_WEIGHTS[2]}")
        logger.info(f"  - Weighted Loss: {'WŁĄCZONE' if cfg.ENABLE_WEIGHTED_LOSS else 'WYŁĄCZONE'}")
        logger.info("=" * 60)

    def run(self):
        """Uruchamia cały potok treningowy."""
        self.start_time = time.time()
        self._log_config_summary()

        try:
            logger.info(">>> KROK 1: Wczytywanie i Przygotowanie Danych <<<")
            df = self.data_loader.load_data()
            
            logger.info(">>> KROK 2: Przygotowanie Danych do Treningu <<<")
            self.data_loader.prepare_data(df)
            
            # Pobierz przygotowane dane
            (X_train, X_val, X_test, 
             y_train, y_val, y_test, 
             scaler) = self.data_loader.get_data()
            
            logger.info(">>> KROK 3: Trening Modelu Multi-Output XGBoost <<<")
            self.model.train_model(X_train, y_train, X_val, y_val)
            
            logger.info(">>> KROK 4: Ewaluacja Modelu <<<")
            self._evaluate_model(X_test, y_test)
            
            logger.info(">>> KROK 5: Zapisywanie Artifaktów <<<")
            self._save_artifacts(scaler)
            
            logger.info(">>> KROK 6: Generowanie Raportów <<<")
            self._generate_reports()
            
            self.end_time = time.time()
            logger.info("--- Proces treningowy zakończony pomyślnie! ---")
            logger.info(f"Czas trwania: {(self.end_time - self.start_time):.2f} sekund.")
            
        except Exception as e:
            logger.error(f"Błąd podczas treningu: {e}", exc_info=True)
            raise

    def _evaluate_model(self, X_test, y_test):
        """Ewaluuje model na zbiorze testowym z prawdopodobieństwami."""
        logger.info("Ewaluacja na zbiorze testowym...")
        
        # Predykcje testowe z prawdopodobieństwami
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)
        
        # Przygotuj wyniki dla każdego poziomu TP/SL
        self.evaluation_results = {}
        
        # Zapisz predykcje testowe do CSV (dla wybranego modelu z konfiguracji)
        self._save_test_predictions_to_csv(X_test, y_test, y_test_pred, y_test_proba, selected_model_index=cfg.CSV_PREDICTIONS_MODEL_INDEX)
        
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            level_desc = cfg.TP_SL_LEVELS_DESC[i]
            
            if cfg.EVAL_LOG_TO_CONSOLE:
                logger.info(f"\n--- Ewaluacja dla poziomu: {level_desc} ---")
            
            # Prawdopodobieństwa dla tego poziomu
            probas = y_test_proba[label_col]  # [n_samples, 3]
            
            # Analiza z różnymi progami pewności
            confidence_thresholds = [0.3, 0.4, 0.45, 0.5]
            
            for threshold in confidence_thresholds:
                if cfg.EVAL_LOG_TO_CONSOLE:
                    logger.info(f"\n  Progi pewności {threshold*100}%:")
                
                # Znajdź próbki z wysoką pewnością
                max_probas = np.max(probas, axis=1)
                high_conf_mask = max_probas >= threshold
                
                if np.sum(high_conf_mask) == 0:
                    if cfg.EVAL_LOG_TO_CONSOLE:
                        logger.info(f"    Brak próbek z pewnością >= {threshold*100}%")
                    continue
                
                # Predykcje tylko dla próbek z wysoką pewnością
                y_true_high_conf = y_test[label_col][high_conf_mask]
                y_pred_high_conf = y_test_pred[label_col][high_conf_mask]
                
                # Metryki dla próbek z wysoką pewnością
                accuracy = accuracy_score(y_true_high_conf, y_pred_high_conf)
                
                # Classification report
                class_report = classification_report(
                    y_true_high_conf, 
                    y_pred_high_conf,
                    target_names=['LONG', 'SHORT', 'NEUTRAL'],
                    output_dict=True,
                    zero_division=0
                )
                
                # Confusion matrix
                conf_matrix = confusion_matrix(
                    y_true_high_conf, 
                    y_pred_high_conf,
                    labels=[0, 1, 2]  # LONG, SHORT, NEUTRAL
                )
                
                if cfg.EVAL_LOG_TO_CONSOLE:
                    logger.info(f"    Próbki z wysoką pewnością: {np.sum(high_conf_mask):,}/{len(probas):,} ({np.sum(high_conf_mask)/len(probas)*100:.1f}%)")
                    logger.info(f"    Accuracy: {accuracy:.4f}")
                    logger.info(f"    LONG: P={class_report['LONG']['precision']:.3f}, R={class_report['LONG']['recall']:.3f}, F1={class_report['LONG']['f1-score']:.3f}")
                    logger.info(f"    SHORT: P={class_report['SHORT']['precision']:.3f}, R={class_report['SHORT']['recall']:.3f}, F1={class_report['SHORT']['f1-score']:.3f}")
                    logger.info(f"    NEUTRAL: P={class_report['NEUTRAL']['precision']:.3f}, R={class_report['NEUTRAL']['recall']:.3f}, F1={class_report['NEUTRAL']['f1-score']:.3f}")
            
            # Standardowe metryki (bez progów)
            accuracy = accuracy_score(y_test[label_col], y_test_pred[label_col])
            
            # Classification report
            class_report = classification_report(
                y_test[label_col], 
                y_test_pred[label_col],
                target_names=['LONG', 'SHORT', 'NEUTRAL'],
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(
                y_test[label_col], 
                y_test_pred[label_col],
                labels=[0, 1, 2]  # LONG, SHORT, NEUTRAL
            )
            
            # DODATKOWY ZAPIS DO CSV Z DANYCH EWALUACJI (dla porównania)
            if i == cfg.CSV_PREDICTIONS_MODEL_INDEX:  # Tylko dla wybranego modelu z konfiguracji
                self._save_evaluation_predictions_to_csv(X_test, y_test[label_col], y_test_pred[label_col], probas, level_desc)
            
            # Zapisz wyniki dla progów pewności
            confidence_results = {}
            for threshold in confidence_thresholds:
                # Znajdź próbki z wysoką pewnością
                max_probas = np.max(probas, axis=1)
                high_conf_mask = max_probas >= threshold
                
                if np.sum(high_conf_mask) == 0:
                    confidence_results[threshold] = None
                    continue
                
                # Predykcje tylko dla próbek z wysoką pewnością
                y_true_high_conf = y_test[label_col][high_conf_mask]
                y_pred_high_conf = y_test_pred[label_col][high_conf_mask]
                
                # Metryki dla próbek z wysoką pewnością
                high_conf_accuracy = accuracy_score(y_true_high_conf, y_pred_high_conf)
                
                # Classification report
                high_conf_class_report = classification_report(
                    y_true_high_conf, 
                    y_pred_high_conf,
                    target_names=['LONG', 'SHORT', 'NEUTRAL'],
                    output_dict=True,
                    zero_division=0
                )
                
                # Confusion matrix
                high_conf_conf_matrix = confusion_matrix(
                    y_true_high_conf, 
                    y_pred_high_conf,
                    labels=[0, 1, 2]  # LONG, SHORT, NEUTRAL
                )
                
                confidence_results[threshold] = {
                    'high_conf_mask': high_conf_mask,
                    'accuracy': high_conf_accuracy,
                    'classification_report': high_conf_class_report,
                    'confusion_matrix': high_conf_conf_matrix,
                    'n_high_conf': np.sum(high_conf_mask),
                    'n_total': len(probas),
                    'percentage': np.sum(high_conf_mask)/len(probas)*100
                }
            
            # Zapisz wyniki
            self.evaluation_results[label_col] = {
                'level_desc': level_desc,
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'y_true': y_test[label_col],
                'y_pred': y_test_pred[label_col],
                'probabilities': probas,
                'confidence_results': confidence_results,
                'level_index': i
            }
            
            if cfg.EVAL_LOG_TO_CONSOLE:
                logger.info(f"\n  Standardowe metryki (bez progów):")
                logger.info(f"    Accuracy: {accuracy:.4f}")
                logger.info(f"    LONG: P={class_report['LONG']['precision']:.3f}, R={class_report['LONG']['recall']:.3f}, F1={class_report['LONG']['f1-score']:.3f}")
                logger.info(f"    SHORT: P={class_report['SHORT']['precision']:.3f}, R={class_report['SHORT']['recall']:.3f}, F1={class_report['SHORT']['f1-score']:.3f}")
                logger.info(f"    NEUTRAL: P={class_report['NEUTRAL']['precision']:.3f}, R={class_report['NEUTRAL']['recall']:.3f}, F1={class_report['NEUTRAL']['f1-score']:.3f}")

    def _save_test_predictions_to_csv(self, X_test, y_test, y_test_pred, y_test_proba, selected_model_index=None):
        """Zapisuje predykcje testowe do pliku CSV w formacie identycznym jak strategia FreqTrade."""
        import pandas as pd
        from datetime import datetime
        
        # Użyj domyślnej wartości z konfiguracji jeśli nie podano
        if selected_model_index is None:
            selected_model_index = cfg.CSV_PREDICTIONS_MODEL_INDEX
        
        logger.info(f"Zapisywanie predykcji testowych do CSV dla modelu {selected_model_index} ({cfg.TP_SL_LEVELS_DESC[selected_model_index]})...")
        
        # Pobierz predykcje dla wybranego modelu
        selected_probas = y_test_proba[cfg.LABEL_COLUMNS[selected_model_index]]  # [n_samples, 3]
        selected_preds = y_test_pred[cfg.LABEL_COLUMNS[selected_model_index]]
        
        # DEBUG: Sprawdź dane
        logger.info(f"DEBUG: selected_probas shape: {selected_probas.shape}")
        logger.info(f"DEBUG: selected_preds shape: {selected_preds.shape}")
        logger.info(f"DEBUG: X_test shape: {X_test.shape}")
        
        # DEBUG: Sprawdź pierwsze predykcje
        logger.info(f"DEBUG: Pierwsze 5 predykcji: {selected_preds[:5]}")
        logger.info(f"DEBUG: Pierwsze 5 prawdopodobieństw:")
        for i in range(5):
            logger.info(f"  Próbka {i}: {selected_probas[i]}")
        
        # Mapowanie klas na sygnały
        class_to_signal = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
        
        # Przygotuj dane do CSV
        csv_data = []
        
        for i, (probs, pred) in enumerate(zip(selected_probas, selected_preds)):
            # Pobierz timestamp z indeksu X_test
            timestamp = X_test.index[i]
            
            # Określ sygnał i pewność
            signal = class_to_signal[pred]
            confidence = probs[pred]  # Prawdopodobieństwo wybranej klasy
            
            # Przygotuj wiersz
            row = {
                'timestamp': timestamp,
                'pair': 'BTC/USDT:USDT',  # Standardowa para
                'signal': signal,
                'confidence': confidence,
                'prob_SHORT': probs[1],  # XGBoost: 0=LONG, 1=SHORT, 2=NEUTRAL
                'prob_LONG': probs[0],
                'prob_NEUTRAL': probs[2]
            }
            csv_data.append(row)
        
        # Utwórz DataFrame i zapisz do CSV
        df_predictions = pd.DataFrame(csv_data)
        
        # Utwórz nazwę pliku z timestampem
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"training_predictions_BTCUSDTUSDT_{timestamp_str}.csv"
        csv_path = cfg.REPORT_DIR / csv_filename
        
        # Zapisz do CSV
        df_predictions.to_csv(csv_path, index=False)
        logger.info(f"Predykcje testowe zapisane: {csv_path}")
        logger.info(f"Liczba predykcji: {len(df_predictions)}")
        
        # Statystyki sygnałów
        signal_counts = df_predictions['signal'].value_counts()
        logger.info(f"Statystyki sygnałów:")
        for signal, count in signal_counts.items():
            logger.info(f"  {signal}: {count} ({count/len(df_predictions)*100:.1f}%)")
        
        # Dodatkowe informacje o zakresie czasowym
        logger.info(f"Zakres czasowy predykcji: {df_predictions['timestamp'].min()} - {df_predictions['timestamp'].max()}")

    def _save_evaluation_predictions_to_csv(self, X_test, y_true, y_pred, probas, level_desc):
        """Zapisuje predykcje ewaluacji do pliku CSV w formacie identycznym jak strategia FreqTrade."""
        import pandas as pd
        from datetime import datetime
        
        logger.info(f"Zapisywanie predykcji ewaluacji dla poziomu: {level_desc} do CSV...")
        
        # Mapowanie klas na sygnały
        class_to_signal = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
        
        # Przygotuj dane do CSV
        csv_data = []
        
        for i, (probs, pred) in enumerate(zip(probas, y_pred)):
            # Pobierz timestamp z indeksu X_test
            timestamp = X_test.index[i]
            
            # Określ sygnał i pewność
            signal = class_to_signal[pred]
            confidence = probs[pred]  # Prawdopodobieństwo wybranej klasy
            
            # Przygotuj wiersz
            row = {
                'timestamp': timestamp,
                'pair': 'BTC/USDT:USDT',  # Standardowa para
                'signal': signal,
                'confidence': confidence,
                'prob_SHORT': probs[1],  # XGBoost: 0=LONG, 1=SHORT, 2=NEUTRAL
                'prob_LONG': probs[0],
                'prob_NEUTRAL': probs[2]
            }
            csv_data.append(row)
        
        # Utwórz DataFrame i zapisz do CSV
        df_predictions = pd.DataFrame(csv_data)
        
        # Utwórz nazwę pliku z timestampem
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"evaluation_predictions_BTCUSDTUSDT_{timestamp_str}.csv"
        csv_path = cfg.REPORT_DIR / csv_filename
        
        # Zapisz do CSV
        df_predictions.to_csv(csv_path, index=False)
        logger.info(f"Predykcje ewaluacji zapisane: {csv_path}")
        logger.info(f"Liczba predykcji: {len(df_predictions)}")
        
        # Statystyki sygnałów
        signal_counts = df_predictions['signal'].value_counts()
        logger.info(f"Statystyki sygnałów:")
        for signal, count in signal_counts.items():
            logger.info(f"  {signal}: {count} ({count/len(df_predictions)*100:.1f}%)")
        
        # Dodatkowe informacje o zakresie czasowym
        logger.info(f"Zakres czasowy predykcji: {df_predictions['timestamp'].min()} - {df_predictions['timestamp'].max()}")

    def _save_artifacts(self, scaler):
        """Zapisuje model, scaler i wyniki."""
        # Zapisz modele (każdy osobno w JSON)
        model_path = cfg.MODEL_DIR / cfg.MODEL_FILENAME
        self.model.save_model(str(model_path))
        logger.info(f"Modele zapisane osobno w katalogu: {cfg.MODEL_DIR}")
        
        # Zapisz scaler
        scaler_path = cfg.MODEL_DIR / cfg.SCALER_FILENAME
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler zapisany: {scaler_path}")
        
        # Zapisz osobne metadata.json dla każdego modelu
        import re
        for i, level_desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            match = re.search(r'TP: ([\d.]+)%, SL: ([\d.]+)%', level_desc)
            if match:
                tp = match.group(1).replace('.', 'p')
                sl = match.group(2).replace('.', 'p')
                model_filename = f"model_tp{tp}_sl{sl}.json"
                metadata_filename = f"metadata_tp{tp}_sl{sl}.json"
            else:
                model_filename = f"model_level{i+1}.json"
                metadata_filename = f"metadata_level{i+1}.json"
            
            # Metadata dla konkretnego modelu
            model_metadata = {
                'n_features': len(cfg.FEATURES),
                'feature_names': cfg.FEATURES,
                'model_type': 'xgboost_individual',
                'scaler_type': 'robust_scaler',
                'training_date': time.strftime('%Y-%m-%d'),
                'version': '4.0',
                'model_index': i,
                'model_filename': model_filename,
                'model_description': level_desc,
                'tp_sl': match.groups() if match else None
            }
            
            metadata_path = cfg.MODEL_DIR / metadata_filename
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            logger.info(f"Metadata dla modelu {i+1} zapisana: {metadata_path}")
        
        # Zapisz również ogólny index wszystkich modeli
        models_info = []
        for i, level_desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            match = re.search(r'TP: ([\d.]+)%, SL: ([\d.]+)%', level_desc)
            if match:
                tp = match.group(1).replace('.', 'p')
                sl = match.group(2).replace('.', 'p')
                model_filename = f"model_tp{tp}_sl{sl}.json"
                metadata_filename = f"metadata_tp{tp}_sl{sl}.json"
            else:
                model_filename = f"model_level{i+1}.json"
                metadata_filename = f"metadata_level{i+1}.json"
            
            models_info.append({
                'index': i,
                'model_filename': model_filename,
                'metadata_filename': metadata_filename,
                'description': level_desc,
                'tp_sl': match.groups() if match else None
            })
        
        # Zapisz ogólny index
        index_path = cfg.MODEL_DIR / 'models_index.json'
        with open(index_path, 'w') as f:
            json.dump(models_info, f, indent=2)
        logger.info(f"Ogólny index modeli zapisany: {index_path}")
        
        # Zapisz wyniki ewaluacji
        results_path = cfg.REPORT_DIR / 'evaluation_results.json'
        
        # Przygotuj wyniki do serializacji JSON
        json_results = {}
        for label_col, results in self.evaluation_results.items():
            json_results[label_col] = {
                'level_desc': results['level_desc'],
                'accuracy': float(results['accuracy']),
                'classification_report': results['classification_report'],
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Wyniki ewaluacji zapisane: {results_path}")

        # Zapisz wyniki do markdown
        # Przygotuj parametry modelu
        model_params = {
            'n_estimators': cfg.XGB_N_ESTIMATORS,
            'learning_rate': cfg.XGB_LEARNING_RATE,
            'max_depth': cfg.XGB_MAX_DEPTH,
            'subsample': cfg.XGB_SUBSAMPLE,
            'colsample_bytree': cfg.XGB_COLSAMPLE_BYTREE,
            'early_stopping_rounds': cfg.XGB_EARLY_STOPPING_ROUNDS,
            'class_weights': cfg.CLASS_WEIGHTS,
            'enable_class_weights_in_training': hasattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING') and cfg.ENABLE_CLASS_WEIGHTS_IN_TRAINING,
            'enable_weighted_loss': cfg.ENABLE_WEIGHTED_LOSS,
            'gamma': cfg.XGB_GAMMA,
            'random_state': cfg.XGB_RANDOM_STATE
        }
        
        # Przygotuj informacje o danych
        data_info = {
            'n_features': len(cfg.FEATURES),
            'n_train': len(self.data_loader.X_train),
            'n_val': len(self.data_loader.X_val),
            'n_test': len(self.data_loader.X_test),
            'train_range': f"{self.data_loader.X_train.index.min()} - {self.data_loader.X_train.index.max()}",
            'test_range': f"{self.data_loader.X_test.index.min()} - {self.data_loader.X_test.index.max()}"
        }
        
        # Zapisz wyniki do markdown w formacie training5 (ujednolicone raportowanie)
        try:
            markdown_path = save_markdown_report_t5(self.evaluation_results, model_params, data_info, cfg, symbol='BTCUSDT')
            logger.info(f"Wyniki treningu zapisane do markdown (format t5): {markdown_path}")
        except Exception:
            # Fallback: oryginalny zapis
            markdown_path = save_training_results_to_markdown(self.evaluation_results, model_params, data_info, cfg)
            logger.info(f"Wyniki treningu zapisane do markdown (fallback): {markdown_path}")

        # JSON report (t5-json-1.2)
        try:
            json_path = save_json_report_t5(self.evaluation_results, model_params, data_info, cfg, symbol='BTCUSDT')
            logger.info(f"Wyniki treningu zapisane do JSON: {json_path}")
        except Exception as e:
            logger.warning(f"Nie udało się zapisać raportu JSON: {e}")

    def _generate_reports(self):
        """Generuje raporty i wykresy."""
        logger.info("Generowanie raportów...")
        
        # Feature importance
        feature_importance_list = self.model.get_feature_importance()
        if feature_importance_list is not None and len(feature_importance_list) > 0:
            if cfg.SAVE_PLOTS:
                # Użyj dynamicznych nazw cech
                feature_names = self.data_loader.feature_names if hasattr(self.data_loader, 'feature_names') else cfg.FEATURES
                # Konwertuj numpy array na DataFrame dla wykresu
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance_list
                }).sort_values('importance', ascending=False)
                self._plot_feature_importance(feature_importance_df)
        
        # Wykresy confusion matrix
        if cfg.SAVE_PLOTS:
            self._plot_confusion_matrices()
        
        # Raport porównawczy (bez zapisu do CSV)
        comparison_data = []
        for label_col, results in self.evaluation_results.items():
            level_desc = results['level_desc']
            accuracy = results['accuracy']
            class_report = results['classification_report']
            comparison_data.append({
                'Poziom': level_desc,
                'Accuracy': accuracy,
                'LONG_Precision': class_report['LONG']['precision'],
                'LONG_Recall': class_report['LONG']['recall'],
                'LONG_F1': class_report['LONG']['f1-score'],
                'SHORT_Precision': class_report['SHORT']['precision'],
                'SHORT_Recall': class_report['SHORT']['recall'],
                'SHORT_F1': class_report['SHORT']['f1-score'],
                'NEUTRAL_Precision': class_report['NEUTRAL']['precision'],
                'NEUTRAL_Recall': class_report['NEUTRAL']['recall'],
                'NEUTRAL_F1': class_report['NEUTRAL']['f1-score']
            })
        # Nie zapisuj comparison_df.to_csv(...)
        # Możesz dodać logowanie do konsoli jeśli chcesz

    def _plot_feature_importance(self, feature_importance):
        """Generuje wykres ważności cech."""
        plt.figure(figsize=(12, 8))
        
        # Top 20 cech
        top_features = feature_importance.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance - Multi-Output XGBoost')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plot_path = cfg.REPORT_DIR / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Wykres ważności cech zapisany: {plot_path}")

    def _plot_confusion_matrices(self):
        """Generuje wykresy confusion matrix dla wszystkich poziomów."""
        num_levels = len(self.evaluation_results)
        
        # Oblicz optymalny układ wykresów
        if num_levels <= 6:
            rows, cols = 2, 3
        elif num_levels <= 9:
            rows, cols = 3, 3
        elif num_levels <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        fig.suptitle('Confusion Matrices - Multi-Output XGBoost', fontsize=16)
        
        # Sprawdź czy axes jest 2D array
        if num_levels == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (label_col, results) in enumerate(self.evaluation_results.items()):
            row = i // cols
            col = i % cols
            
            conf_matrix = results['confusion_matrix']
            level_desc = results['level_desc']
            
            im = axes[row, col].imshow(conf_matrix, cmap='Blues', interpolation='nearest')
            axes[row, col].set_title(f'{level_desc}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('True')
            axes[row, col].set_xticks([0, 1, 2])
            axes[row, col].set_yticks([0, 1, 2])
            axes[row, col].set_xticklabels(['LONG', 'SHORT', 'NEUTRAL'])
            axes[row, col].set_yticklabels(['LONG', 'SHORT', 'NEUTRAL'])
            
            # Dodaj liczby do komórek
            for j in range(3):
                for k in range(3):
                    axes[row, col].text(k, j, str(conf_matrix[j, k]), 
                                      ha='center', va='center', fontsize=10)
        
        # Ukryj nieużywane subploty
        for i in range(num_levels, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plot_path = cfg.REPORT_DIR / 'confusion_matrices.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Wykresy confusion matrix zapisane: {plot_path}")

    def _generate_comparison_report(self):
        """Generuje raport porównawczy wszystkich poziomów."""
        logger.info("Generowanie raportu porównawczego...")
        
        comparison_data = []
        
        for label_col, results in self.evaluation_results.items():
            level_desc = results['level_desc']
            accuracy = results['accuracy']
            class_report = results['classification_report']
            
            comparison_data.append({
                'Poziom': level_desc,
                'Accuracy': accuracy,
                'LONG_Precision': class_report['LONG']['precision'],
                'LONG_Recall': class_report['LONG']['recall'],
                'LONG_F1': class_report['LONG']['f1-score'],
                'SHORT_Precision': class_report['SHORT']['precision'],
                'SHORT_Recall': class_report['SHORT']['recall'],
                'SHORT_F1': class_report['SHORT']['f1-score'],
                'NEUTRAL_Precision': class_report['NEUTRAL']['precision'],
                'NEUTRAL_Recall': class_report['NEUTRAL']['recall'],
                'NEUTRAL_F1': class_report['NEUTRAL']['f1-score']
            })
        
        # Zapisz do CSV
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = cfg.REPORT_DIR / 'level_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Raport porównawczy zapisany: {comparison_path}")
        
        # Znajdź najlepsze wyniki
        best_accuracy_idx = comparison_df['Accuracy'].idxmax()
        best_long_idx = comparison_df['LONG_F1'].idxmax()
        best_short_idx = comparison_df['SHORT_F1'].idxmax()
        
        logger.info("NAJLEPSZE WYNIKI:")
        logger.info(f"Ogólna dokładność: {comparison_df.iloc[best_accuracy_idx]['Poziom']} "
                   f"(Accuracy={comparison_df.iloc[best_accuracy_idx]['Accuracy']:.3f})")
        logger.info(f"LONG: {comparison_df.iloc[best_long_idx]['Poziom']} "
                   f"(F1={comparison_df.iloc[best_long_idx]['LONG_F1']:.3f})")
        logger.info(f"SHORT: {comparison_df.iloc[best_short_idx]['Poziom']} "
                   f"(F1={comparison_df.iloc[best_short_idx]['SHORT_F1']:.3f})")

def main():
    """Główna funkcja uruchamiająca proces."""
    trainer = Trainer()
    trainer.run()

if __name__ == "__main__":
    main() 