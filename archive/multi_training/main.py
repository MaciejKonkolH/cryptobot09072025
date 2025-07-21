"""
Główny skrypt orkiestrujący proces treningu modelu.
"""
import sys
import os
from pathlib import Path

# --- Dynamiczne dodawanie ścieżki projektu ---
# To pozwala na uruchamianie skryptu zarówno z katalogu głównego (python training/main.py)
# jak i bezpośrednio (cd training; python main.py)
def _find_project_root(start_path: Path, marker_file: str = ".project_root") -> Path:
    """Wspina się w górę drzewa katalogów w poszukiwaniu pliku znacznika."""
    path = start_path
    while path.parent != path:
        if (path / marker_file).is_file():
            return path
        path = path.parent
    raise FileNotFoundError(f"Nie znaleziono pliku znacznika '{marker_file}' w żadnym z nadrzędnych katalogów.")

try:
    # Znajdź katalog główny projektu, zaczynając od lokalizacji tego pliku
    project_root = _find_project_root(Path(__file__).resolve().parent)
    # Dodaj go do ścieżki systemowej, jeśli jeszcze go tam nie ma
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
from sklearn.preprocessing import RobustScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time
from pathlib import Path

# Importy z naszego modułu
from training import config as cfg
from training.utils import setup_logging
from multi_training.data_generator import TrainingGenerator, ValidationGenerator
from training.model_builder import build_model
from training.callbacks import BalancedUndersamplingCallback
from training.reporter import TrainingReporter

logger = setup_logging()

class Trainer:
    """
    Główna klasa zarządzająca całym procesem treningowym.
    """
    def __init__(self):
        """Inicjalizuje i tworzy niezbędne katalogi."""
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        os.makedirs(cfg.REPORT_DIR, exist_ok=True)
        self.scaler = None
        self.model = None
        self.history = None
        self.start_time = None
        self.end_time = None
        self.evaluation_results = None
        self.raw_probabilities = None
        self.predictions = None
        self.true_labels = None
        self.train_df, self.val_df, self.test_df = None, None, None

    def _log_config_summary(self):
        """Loguje podsumowanie kluczowych parametrów treningu."""
        logger.info("=" * 60)
        logger.info("🎯 PODSUMOWANIE KONFIGURACJI TRENINGU (MULTI-ASSET) 🎯")
        logger.info("=" * 60)
        logger.info("Dane Wejściowe:")
        logger.info(f"  - Pary Walutowe: {', '.join(cfg.PAIRS)}")
        logger.info(f"  - Take Profit: {cfg.TAKE_PROFIT_PCT}% / Stop Loss: {cfg.STOP_LOSS_PCT}%")
        logger.info(f"  - Okno Przyszłości (FW): {cfg.FUTURE_WINDOW} min")
        logger.info("-" * 60)
        logger.info("Parametry Modelu i Treningu:")
        logger.info(f"  - Długość Sekwencji: {cfg.SEQUENCE_LENGTH}")
        logger.info(f"  - Podział Walidacyjny: {cfg.VALIDATION_SPLIT:.0%}")
        logger.info(f"  - Podział Testowy: {cfg.TEST_SPLIT:.0%}")
        logger.info(f"  - Epoki: {cfg.EPOCHS}")
        logger.info(f"  - Rozmiar Batcha: {cfg.BATCH_SIZE}")
        logger.info("-" * 60)
        logger.info("Parametry Predykcji:")
        if cfg.ENABLE_CONFIDENCE_THRESHOLDING:
            thresholds = cfg.CONFIDENCE_THRESHOLDS
            logger.info(f"  - Progi Pewności: S={thresholds.get(0, 'N/A')}, H={thresholds.get(1, 'N/A')}, L={thresholds.get(2, 'N/A')}")
        else:
            logger.info("  - Progi Pewności: Wyłączone")
        logger.info("=" * 60)

    def run(self):
        """Uruchamia cały potok treningowy."""
        self.start_time = time.time()
        self._log_config_summary()

        logger.info(">>> KROK 1: Wczytywanie i Wstępne Przetworzenie Danych <<<")
        list_of_dfs = self._load_and_prepare_data()
        if not list_of_dfs:
            logger.error("Nie udało się wczytać żadnych danych. Zakończono.")
            return

        logger.info(">>> KROK 2: Chronologiczny Podział na Zbiory <<<")
        self.train_df, self.val_df, self.test_df = self._split_data(list_of_dfs)

        logger.info(">>> KROK 3: Skalowanie Cech <<<")
        train_scaled, val_scaled, test_scaled = self._scale_features(self.train_df, self.val_df, self.test_df)

        logger.info(">>> KROK 4: Przygotowanie Generatorów Danych <<<")
        train_gen, val_gen, test_gen = self._create_generators(train_scaled, val_scaled, test_scaled)

        logger.info(">>> KROK 6: Konfiguracja i Trening Modelu <<<")
        self._train_model(train_gen, val_gen)
        
        logger.info(">>> KROK 7: Ostateczny Egzamin i Analiza Predykcji <<<")
        self.evaluation_results = self._evaluate_model(test_gen)

        logger.info(">>> KROK 8: Zapisanie Wszystkich Artefaktów <<<")
        self._save_artifacts(self.evaluation_results)

        self.end_time = time.time()
        
        logger.info("--- Proces treningowy zakończony pomyślnie! ---")
        
        reporter = TrainingReporter(self)
        reporter.print_summary()


    def _load_and_prepare_data(self) -> list[pd.DataFrame]:
        """Wczytuje, filtruje i waliduje dane dla wszystkich par, zwracając listę ramek danych."""
        all_dfs = []
        for file_path in cfg.INPUT_FILE_PATHS:
            if not os.path.exists(file_path):
                logger.warning(f"Plik wejściowy nie istnieje i zostanie pominięty: {file_path}")
                continue

            logger.info(f"Wczytywanie danych z: {os.path.basename(file_path)}")
            df = pd.read_feather(file_path)

            # --- Walidacja i Konwersja Danych ---
        if 'date' not in df.columns:
                logger.warning(f"Brak kolumny 'date' w pliku {file_path}. Pomijanie.")
                continue
            if 'label' not in df.columns:
                logger.warning(f"Brak kolumny 'label' w pliku {file_path}. Pomijanie.")
                continue

        df['date'] = pd.to_datetime(df['date'])
        
        if cfg.ENABLE_DATE_FILTER:
            df = df[(df['date'] >= pd.to_datetime(cfg.START_DATE)) & (df['date'] <= pd.to_datetime(cfg.END_DATE))]

        df.set_index('date', inplace=True)
        
            if df.empty:
                logger.warning(f"Brak danych dla pliku {file_path} po filtracji. Pomijanie.")
                continue
            
        label_counts = df['label'].value_counts(normalize=True).mul(100).round(2)
            logger.info(f"  - Wczytano wierszy: {len(df):,}")
            logger.info(f"  - Rozkład etykiet: 0(S):{label_counts.get(0,0)}% 1(H):{label_counts.get(1,0)}% 2(L):{label_counts.get(2,0)}%")
        
            all_dfs.append(df)
        
        logger.info(f"Pomyślnie wczytano dane dla {len(all_dfs)} z {len(cfg.PAIRS)} par.")
        return all_dfs


    def _split_data(self, list_of_dfs: list[pd.DataFrame]) -> tuple[list, list, list]:
        """Dzieli każdą ramkę danych z listy chronologicznie."""
        train_list, val_list, test_list = [], [], []
        
        for i, df in enumerate(list_of_dfs):
            pair_name = cfg.PAIRS[i]
            logger.info(f"Dzielenie danych dla {pair_name}...")
            
            if len(df) < 3: # Potrzebujemy co najmniej 3 wierszy do podziału
                logger.warning(f"Zbyt mało danych dla {pair_name} ({len(df)} wierszy). Pomijanie.")
                continue

        val_split_idx = int(len(df) * (1 - cfg.VALIDATION_SPLIT - cfg.TEST_SPLIT))
        test_split_idx = int(len(df) * (1 - cfg.TEST_SPLIT))
        
        train_df = df.iloc[:val_split_idx]
        val_df = df.iloc[val_split_idx:test_split_idx]
        test_df = df.iloc[test_split_idx:]
        
            logger.info(f"  - Podział: Trening: {len(train_df):,}, Walidacja: {len(val_df):,}, Test: {len(test_df):,}")
            logger.info(f"  - Zakres dat (Trening): od {train_df.index.min().strftime('%Y-%m-%d %H:%M')} do {train_df.index.max().strftime('%Y-%m-%d %H:%M')}")
            
            train_list.append(train_df)
            val_list.append(val_df)
            test_list.append(test_df)

        return train_list, val_list, test_list

    def _scale_features(self, train_list, val_list, test_list) -> tuple[list, list, list]:
        """Uczy scaler na połączonych danych treningowych i transformuje wszystkie zbiory."""
        self.scaler = RobustScaler()
        
        # 1. Połącz wszystkie zbiory treningowe, aby nauczyć scaler
        combined_train_df = pd.concat(train_list, ignore_index=True)
        self.scaler.fit(combined_train_df[cfg.FEATURES])
        
        logger.info("Scaler nauczony na połączonych danych treningowych.")
        
        # 2. Funkcja pomocnicza do transformacji pojedynczej ramki danych
        def transform_df(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df_scaled = df.copy()
            df_scaled[cfg.FEATURES] = self.scaler.transform(df[cfg.FEATURES])
            return df_scaled

        # 3. Zastosuj transformację do każdej ramki z osobna
        train_scaled = [transform_df(df) for df in train_list]
        val_scaled = [transform_df(df) for df in val_list]
        test_scaled = [transform_df(df) for df in test_list]
        
        scaler_path = os.path.join(cfg.MODEL_DIR, cfg.SCALER_FILENAME)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler zapisany w: {scaler_path}")
        
        return train_scaled, val_scaled, test_scaled

    def _create_generators(self, train_list, val_list, test_list) -> tuple:
        """Tworzy instancje generatorów dla każdego zbioru."""
        
        train_gen = TrainingGenerator(
            list_of_dfs=train_list, 
            features=cfg.FEATURES, 
            sequence_length=cfg.SEQUENCE_LENGTH, 
            batch_size=cfg.BATCH_SIZE
        )
        
        val_gen = ValidationGenerator(
            list_of_dfs=val_list, 
            features=cfg.FEATURES, 
            sequence_length=cfg.SEQUENCE_LENGTH, 
            batch_size=cfg.BATCH_SIZE
        )
        
        test_gen = ValidationGenerator(
            list_of_dfs=test_list, 
            features=cfg.FEATURES, 
            sequence_length=cfg.SEQUENCE_LENGTH, 
            batch_size=cfg.BATCH_SIZE
        )
        
        logger.info("Generatory danych (treningowy, walidacyjny i testowy) zostały utworzone.")
        return train_gen, val_gen, test_gen

    def _train_model(self, train_gen, val_gen):
        """Buduje, kompiluje i trenuje model."""
        input_shape = (cfg.SEQUENCE_LENGTH, len(cfg.FEATURES))
        self.model = build_model(input_shape)
        self.model.summary(print_fn=logger.info)
        
        # Przygotowanie Callbacków
        model_path = os.path.join(cfg.MODEL_DIR, cfg.MODEL_FILENAME)
        callbacks = [
            ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
            EarlyStopping(monitor='val_loss', patience=cfg.EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=cfg.REDUCE_LR_FACTOR, patience=cfg.REDUCE_LR_PATIENCE, verbose=1)
        ]
        if cfg.ENABLE_CLASS_BALANCING:
            callbacks.append(BalancedUndersamplingCallback(train_gen))
        
        logger.info("Rozpoczynanie treningu modelu...")
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg.EPOCHS,
            callbacks=callbacks
        )
        logger.info("Trening zakończony.")
        # Model z najlepszymi wagami jest już wczytany dzięki `restore_best_weights=True`

    def _evaluate_model(self, test_gen) -> dict:
        """Ocenia model na zbiorze testowym i zapisuje predykcje."""
        logger.info("Ocenianie modelu na zbiorze testowym (wszystkie pary)...")
        # Generator walidacyjny (sekwencyjny) jest idealny do ewaluacji
        test_loss, test_accuracy = self.model.evaluate(test_gen, verbose=1)
        results = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
        logger.info(f"Ogólne wyniki na zbiorze testowym - Strata: {test_loss:.4f}, Dokładność: {test_accuracy:.4f}")

        logger.info("Generowanie predykcji dla zbioru testowego...")
        self.raw_probabilities = self.model.predict(test_gen, verbose=1)
        self.predictions = np.argmax(self.raw_probabilities, axis=1)
        
        # Przygotowanie danych do zapisu
        # Potrzebujemy prawdziwych etykiet i dat ze zbioru testowego.
        # Używamy `global_indices` z generatora, aby zrekonstruować kolejność.
        self.true_labels = np.array([test_gen.data_sources[src_idx]['labels'][seq_idx + cfg.SEQUENCE_LENGTH - 1] 
                                     for src_idx, seq_idx in test_gen.global_indices])
        
        all_dates = []
        for src_idx, seq_idx in test_gen.global_indices:
            # self.test_df to teraz lista, więc używamy src_idx
            original_df = self.test_df[src_idx]
            date = original_df.index[seq_idx + cfg.SEQUENCE_LENGTH - 1]
            all_dates.append(date)

        results_df = pd.DataFrame({
            'date': all_dates,
            'true_label': self.true_labels,
            'predicted_label': self.predictions,
            'prob_SHORT': self.raw_probabilities[:, 0],
            'prob_HOLD': self.raw_probabilities[:, 1],
            'prob_LONG': self.raw_probabilities[:, 2],
        })
        
        pred_path = os.path.join(cfg.REPORT_DIR, cfg.PREDICTIONS_FILENAME)
        results_df.to_csv(pred_path, index=False)
        logger.info(f"Predykcje zapisane w: {pred_path}")

        return results

    def _save_artifacts(self, evaluation_results: dict):
        """Zapisuje metadane i raport."""
        # --- Zapis metadanych ---
        # Zbieramy tylko zmienne z pliku config pisane wielkimi literami (nasze stałe)
        # i konwertujemy obiekty Path na stringi, aby były serializowalne w JSON.
        metadata = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(cfg).items()
            if k.isupper()
        }
        
        # Dodajemy wyniki ewaluacji do metadanych
        metadata.update(evaluation_results)

        # Dodajemy listę par do metadanych dla przejrzystości
        metadata['PAIRS'] = cfg.PAIRS

        meta_path = os.path.join(cfg.MODEL_DIR, cfg.METADATA_FILENAME)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=lambda o: '<not serializable>')
        logger.info(f"Metadane zapisane w: {meta_path}")

        # --- Zapis raportu tekstowego ---
        report_path = os.path.join(cfg.REPORT_DIR, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("--- Raport z Ewaluacji Modelu ---\n\n")
            for key, value in evaluation_results.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Raport z ewaluacji zapisany w: {report_path}")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run() 