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
from training import config as cfg
from training.utils import setup_logging
from training.reporter import TrainingReporter

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
        logger.info(f"Kolumny faktycznie wczytane w training/main.py: {df.columns.tolist()}")
        # -------------------------

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        if cfg.ENABLE_DATE_FILTER:
            logger.info(f"Filtrowanie danych do zakresu: {cfg.START_DATE} - {cfg.END_DATE}")
            df = df[(df.index >= pd.to_datetime(cfg.START_DATE)) & (df.index <= pd.to_datetime(cfg.END_DATE))]
        
        logger.info(f"Liczba wierszy po wczytaniu: {len(df):,}")
        
        # Sprawdzamy, czy wszystkie potrzebne kolumny (cechy + etykieta) istnieją
        required_cols = set(cfg.FEATURES + ['label'])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"W danych brakuje wymaganych kolumn: {missing_cols}. Zakończono.")
            return None
            
        return df

    def _split_data(self, df: pd.DataFrame):
        """Dzieli dane chronologicznie na X i y."""
        self.feature_names = cfg.FEATURES
        X = df[self.feature_names]
        y = df['label']

        val_split_idx = int(len(df) * (1 - cfg.VALIDATION_SPLIT - cfg.TEST_SPLIT))
        test_split_idx = int(len(df) * (1 - cfg.TEST_SPLIT))
        
        self.X_train = X.iloc[:val_split_idx]
        self.y_train = y.iloc[:val_split_idx]
        
        self.X_val = X.iloc[val_split_idx:test_split_idx]
        self.y_val = y.iloc[val_split_idx:test_split_idx]
        
        self.X_test = X.iloc[test_split_idx:]
        self.y_test = y.iloc[test_split_idx:]
        
        logger.info(f"Podział danych: Trening: {len(self.X_train):,}, Walidacja: {len(self.X_val):,}, Test: {len(self.X_test):,}")

    def _scale_features(self):
        """Dopasowuje skaler na danych treningowych i transformuje wszystkie zbiory."""
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        logger.info("Cechy zostały przeskalowane za pomocą RobustScaler.")

    def _build_train_and_evaluate(self):
        """Buduje, trenuje i ocenia model XGBoost w jednym kroku."""
        logger.info("Budowanie modelu XGBoost...")
        
        train_weights = None
        if cfg.ENABLE_CLASS_WEIGHTING:
            logger.info(f"Ważenie klas WŁĄCZONE. Stosowanie wag: {cfg.CLASS_WEIGHTS}")
            train_weights = self.y_train.map(cfg.CLASS_WEIGHTS).to_numpy()
        else:
            logger.info("Ważenie klas WYŁĄCZONE. Wszystkie próbki mają wagę 1.0.")

        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=6,
            eval_metric='mlogloss',
            n_estimators=cfg.XGB_N_ESTIMATORS,
            learning_rate=cfg.XGB_LEARNING_RATE,
            max_depth=cfg.XGB_MAX_DEPTH,
            subsample=cfg.XGB_SUBSAMPLE,
            colsample_bytree=cfg.XGB_COLSAMPLE_BYTREE,
            gamma=cfg.XGB_GAMMA,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=cfg.XGB_EARLY_STOPPING_ROUNDS  # Przeniesione do konstruktora
        )

        logger.info("Rozpoczynanie treningu modelu XGBoost...")
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=True,
            sample_weight=train_weights
        )
        logger.info("Trening zakończony.")

        # W XGBoost >= 2.0, najlepsza iteracja jest dostępna jako atrybut
        best_iteration = self.model.best_iteration
        logger.info(f"Najlepsza iteracja znaleziona w epoce: {best_iteration} (wynik walidacji: {self.model.best_score:.6f})")

        logger.info("Ocena modelu na zbiorze testowym przy użyciu najlepszej iteracji...")
        # Używamy `iteration_range`, aby dokonać predykcji na podstawie najlepszego modelu
        y_pred_proba = self.model.predict_proba(self.X_test, iteration_range=(0, best_iteration))
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Definiujemy nazwy naszych 6 klas dla czytelnego raportu
        target_names = [
            'PROFIT_SHORT (0)', 'TIMEOUT_HOLD (1)', 'PROFIT_LONG (2)',
            'LOSS_SHORT (3)', 'LOSS_LONG (4)', 'CHAOS_HOLD (5)'
        ]

        self.evaluation_results = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "classification_report": classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        logger.info("Raport Klasyfikacyjny (Zbiór Testowy):")
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        logger.info("Macierz Pomyłek (Zbiór Testowy):")
        # Musimy ponownie obliczyć macierz, bo evaluation_results jest oparte na ostatniej epoce
        cm = confusion_matrix(self.y_test, y_pred)
        self.evaluation_results['confusion_matrix'] = cm.tolist() # Aktualizujemy, żeby zapis był spójny
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df.to_string())

        logger.info("Generowanie raportu ważności cech...")
        self._generate_feature_importance_report()

    def _generate_feature_importance_report(self):
        """
        Generuje tekstowy raport ważności cech, zapisuje go do pliku
        i wyświetla w konsoli, zamiast tworzyć wykres.
        """
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Wyświetlanie w konsoli
        logger.info("Raport Ważności Cech (malejąco):")
        print(feature_importance_df.to_string())

        # Zapis do pliku
        report_path = os.path.join(cfg.REPORT_DIR, "feature_importance.txt")
        feature_importance_df.to_csv(report_path, sep='\t', index=False)
        logger.info(f"Raport ważności cech zapisany w {report_path}")


    def _save_artifacts(self):
        """Zapisuje model, skaler i wyniki oceny."""
        # Zapis modelu
        model_path = os.path.join(cfg.MODEL_DIR, "xgb_model.json")
        self.model.save_model(model_path)
        logger.info(f"Model zapisany w: {model_path}")

        # Zapis skalera
        scaler_path = os.path.join(cfg.MODEL_DIR, "scaler.gz")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Skaler zapisany w: {scaler_path}")

        # Zapis raportu
        report_path = os.path.join(cfg.REPORT_DIR, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        logger.info(f"Raport z oceny zapisany w: {report_path}")

def main():
    """Główna funkcja uruchamiająca proces."""
    trainer = Trainer()
    trainer.run()

if __name__ == "__main__":
    main() 