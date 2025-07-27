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
        logger.info(f"  - Cechy: {len(cfg.FEATURES)} cech (z rzeczywiście dostępnych)")
        logger.info(f"  - Poziomy TP/SL: {len(cfg.LABEL_COLUMNS)}")
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
        
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            level_desc = cfg.TP_SL_LEVELS_DESC[i]
            
            logger.info(f"\n--- Ewaluacja dla poziomu: {level_desc} ---")
            
            # Prawdopodobieństwa dla tego poziomu
            probas = y_test_proba[label_col]  # [n_samples, 3]
            
            # Analiza z różnymi progami pewności
            confidence_thresholds = [0.5, 0.7, 0.8, 0.9]
            
            for threshold in confidence_thresholds:
                logger.info(f"\n  Progi pewności {threshold*100}%:")
                
                # Znajdź próbki z wysoką pewnością
                max_probas = np.max(probas, axis=1)
                high_conf_mask = max_probas >= threshold
                
                if np.sum(high_conf_mask) == 0:
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
            
            # Zapisz wyniki
            self.evaluation_results[label_col] = {
                'level_desc': level_desc,
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'y_true': y_test[label_col],
                'y_pred': y_test_pred[label_col],
                'probabilities': probas
            }
            
            logger.info(f"\n  Standardowe metryki (bez progów):")
            logger.info(f"    Accuracy: {accuracy:.4f}")
            logger.info(f"    LONG: P={class_report['LONG']['precision']:.3f}, R={class_report['LONG']['recall']:.3f}, F1={class_report['LONG']['f1-score']:.3f}")
            logger.info(f"    SHORT: P={class_report['SHORT']['precision']:.3f}, R={class_report['SHORT']['recall']:.3f}, F1={class_report['SHORT']['f1-score']:.3f}")
            logger.info(f"    NEUTRAL: P={class_report['NEUTRAL']['precision']:.3f}, R={class_report['NEUTRAL']['recall']:.3f}, F1={class_report['NEUTRAL']['f1-score']:.3f}")

    def _save_artifacts(self, scaler):
        """Zapisuje model, scaler i wyniki."""
        # Zapisz model
        model_path = cfg.MODEL_DIR / cfg.MODEL_FILENAME
        self.model.save_model(str(model_path))
        logger.info(f"Model zapisany: {model_path}")
        
        # Zapisz scaler
        scaler_path = cfg.MODEL_DIR / cfg.SCALER_FILENAME
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler zapisany: {scaler_path}")
        
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
        
        # Zapisz wyniki do markdown
        markdown_path = save_training_results_to_markdown(self.evaluation_results, model_params, data_info, cfg)
        logger.info(f"Wyniki treningu zapisane do markdown: {markdown_path}")

    def _generate_reports(self):
        """Generuje raporty i wykresy."""
        logger.info("Generowanie raportów...")
        
        # Feature importance
        feature_importance_list = self.model.get_feature_importance()
        if feature_importance_list is not None and len(feature_importance_list) > 0:
            if cfg.SAVE_PLOTS:
                # Konwertuj numpy array na DataFrame dla wykresu
                feature_importance_df = pd.DataFrame({
                    'feature': cfg.FEATURES,
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices - Multi-Output XGBoost', fontsize=16)
        
        for i, (label_col, results) in enumerate(self.evaluation_results.items()):
            row = i // 3
            col = i % 3
            
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
                                      ha='center', va='center', fontsize=12)
        
        # Ukryj ostatni subplot jeśli nie ma wystarczająco danych
        if len(self.evaluation_results) < 6:
            axes[1, 2].set_visible(False)
        
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