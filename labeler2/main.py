"""
Nowy, uproszczony moduł do etykietowania danych (`labeler2`).
Dostosowany do nowego formatu danych z feature_calculator_snapshot.
Obsługuje wiele poziomów TP/SL jednocześnie.
"""
import logging
import os
import sys
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from pathlib import Path

# Opcjonalny import tqdm dla paska postępu
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Dodajemy ścieżkę do głównego katalogu, aby importy działały
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import labeler2.config as config
except ImportError:
    import config

def setup_logging():
    """Konfiguruje system logowania."""
    log_dir = config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, config.LOG_FILENAME)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class MultiLevelLabeler:
    """
    Klasa odpowiedzialna za obliczanie etykiet dla wielu poziomów TP/SL jednocześnie.
    Używa uproszczonego i poprawionego algorytmu 'competitive'.
    """

    def __init__(self):
        """Inicjalizuje klasę z parametrami z pliku konfiguracyjnego."""
        self.future_window = config.FUTURE_WINDOW_MINUTES
        self.tp_sl_levels = config.TP_SL_LEVELS
        self.label_map = config.LABEL_MAPPING
        
        logger.info("MultiLevelLabeler zainicjalizowany.")
        logger.info(f"Parametry: Okno przyszłości={self.future_window} min")
        logger.info(f"Poziomy TP/SL: {self.tp_sl_levels}")

    def calculate_labels_for_level(self, ohlc_data: np.ndarray, timestamps: np.ndarray, 
                                 tp_pct: float, sl_pct: float) -> np.ndarray:
        """
        Oblicza etykiety dla jednego poziomu TP/SL.
        
        Args:
            ohlc_data: Array z danymi [high, low, close]
            timestamps: Array z timestampami
            tp_pct: Procent Take Profit (np. 2.0 dla 2%)
            sl_pct: Procent Stop Loss (np. 1.0 dla 1%)
            
        Returns:
            np.ndarray: Array z etykietami
        """
        long_tp_pct = tp_pct / 100
        long_sl_pct = sl_pct / 100
        short_tp_pct = tp_pct / 100  # Symmetric
        short_sl_pct = sl_pct / 100  # Symmetric
        
        labels = np.full(len(ohlc_data), self.label_map['TIMEOUT_HOLD'], dtype=np.int8)
        total_rows = len(ohlc_data)
        
        for i in range(total_rows):
            if i + self.future_window >= len(ohlc_data):
                continue

            entry_price = ohlc_data[i, 2]  # close price
            
            long_tp_price = entry_price * (1 + long_tp_pct)
            long_sl_price = entry_price * (1 - long_sl_pct)
            short_tp_price = entry_price * (1 - short_tp_pct)
            short_sl_price = entry_price * (1 + short_sl_pct)

            zdarzenie_long = None
            zdarzenie_short = None

            future_window_slice = ohlc_data[i + 1 : i + 1 + self.future_window]

            for j, (future_high, future_low, _) in enumerate(future_window_slice):
                future_time = timestamps[i + 1 + j]

                if zdarzenie_long is None:
                    if future_high >= long_tp_price:
                        zdarzenie_long = ('TP', future_time)
                    elif future_low <= long_sl_price:
                        zdarzenie_long = ('SL', future_time)
                
                if zdarzenie_short is None:
                    if future_low <= short_tp_price:
                        zdarzenie_short = ('TP', future_time)
                    elif future_high >= short_sl_price:
                        zdarzenie_short = ('SL', future_time)
                
                if zdarzenie_long and zdarzenie_short:
                    break

            # --- 6-klasowa hierarchia decyzyjna ---

            # Scenariusz 1: Nic się nie stało -> TIMEOUT_HOLD
            if not zdarzenie_long and not zdarzenie_short:
                labels[i] = self.label_map['TIMEOUT_HOLD']
                continue

            # Scenariusz 2: Wystąpiło tylko jedno zdarzenie (czysty zysk lub czysta strata)
            if zdarzenie_long and not zdarzenie_short:
                if zdarzenie_long[0] == 'TP':
                    labels[i] = self.label_map['PROFIT_LONG']
                else: # SL
                    labels[i] = self.label_map['LOSS_LONG']
                continue

            if not zdarzenie_long and zdarzenie_short:
                if zdarzenie_short[0] == 'TP':
                    labels[i] = self.label_map['PROFIT_SHORT']
                else: # SL
                    labels[i] = self.label_map['LOSS_SHORT']
                continue
            
            # Scenariusz 3: Wystąpiły oba zdarzenia (złożone)
            long_type, long_time = zdarzenie_long
            short_type, short_time = zdarzenie_short

            # 3a: Zysk ma pierwszeństwo
            if long_type == 'TP' and short_type == 'TP': # Wyścig do TP
                labels[i] = self.label_map['PROFIT_LONG'] if long_time <= short_time else self.label_map['PROFIT_SHORT']
            elif long_type == 'TP': # short_type musi być SL
                labels[i] = self.label_map['PROFIT_LONG']
            elif short_type == 'TP': # long_type musi być SL
                labels[i] = self.label_map['PROFIT_SHORT']
            # 3b: Obie pozycje na stratach -> CHAOS_HOLD
            elif long_type == 'SL' and short_type == 'SL':
                labels[i] = self.label_map['CHAOS_HOLD']

        return labels

    def calculate_all_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Główna funkcja obliczająca etykiety dla wszystkich poziomów TP/SL.
        Przyjmuje DataFrame z cechami i dodaje kolumny z etykietami.
        """
        logger.info("Rozpoczynanie obliczania etykiet dla wszystkich poziomów...")
        
        # Sprawdź czy mamy wymagane kolumny
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Brakuje wymaganych kolumn OHLC: {missing_cols}")
        
        ohlc_data = df[['high', 'low', 'close']].to_numpy()
        timestamps = df.index.to_numpy()
        total_rows = len(ohlc_data)
        
        logger.info(f"Przetwarzanie {total_rows:,} wierszy dla {len(self.tp_sl_levels)} poziomów TP/SL")
        
        # Oblicz etykiety dla każdego poziomu
        for i, (tp_pct, sl_pct) in enumerate(self.tp_sl_levels):
            logger.info(f"  -> Poziom {i+1}/{len(self.tp_sl_levels)}: TP={tp_pct}%, SL={sl_pct}%")
            
            # Oblicz etykiety dla tego poziomu
            labels = self.calculate_labels_for_level(ohlc_data, timestamps, tp_pct, sl_pct)
            
            # Dodaj kolumnę z etykietami
            column_suffix = config.get_level_suffix(tp_pct, sl_pct)
            column_name = f"label_{column_suffix}"
            df[column_name] = labels
            
            # Statystyki dla tego poziomu
            unique_labels, counts = np.unique(labels, return_counts=True)
            stats = dict(zip(unique_labels, counts))
            logger.info(f"     Statystyki {column_name}: {stats}")
        
        logger.info("Zakończono obliczanie etykiet dla wszystkich poziomów.")
        return df

def main():
    """Główna funkcja uruchamiająca proces etykietowania."""
    logger.info("--- Rozpoczynanie procesu etykietowania (MultiLevel) ---")
    
    input_path = Path(config.INPUT_FILE_PATH)
    logger.info(f"Wczytywanie danych z cechami z: {input_path}")
    
    if not input_path.exists():
        logger.error(f"Plik wejściowy nie istnieje: {input_path}")
        sys.exit(1)
        
    try:
        # Wczytaj dane z cechami
        df_with_features = pd.read_feather(input_path)
        logger.info(f"Wczytano {len(df_with_features):,} wierszy danych z cechami.")

        # Ustaw indeks na timestamp jeśli potrzeba
        if 'timestamp' in df_with_features.columns and df_with_features.index.name != 'timestamp':
             df_with_features['timestamp'] = pd.to_datetime(df_with_features['timestamp'])
             df_with_features.set_index('timestamp', inplace=True)
        elif 'date' in df_with_features.columns and df_with_features.index.name != 'date':
             df_with_features['date'] = pd.to_datetime(df_with_features['date'])
             df_with_features.set_index('date', inplace=True)

        # Oblicz etykiety dla wszystkich poziomów
        labeler = MultiLevelLabeler()
        df_labeled = labeler.calculate_all_labels(df_with_features)
        
        # --- Tworzenie nazwy pliku wyjściowego ---
        base_name = config.INPUT_FILENAME.replace('_features.feather', '').replace('.feather', '')
        
        output_filename = config.OUTPUT_FILENAME_TEMPLATE.format(
            base_name=base_name,
            fw=config.FUTURE_WINDOW_MINUTES,
            levels_count=len(config.TP_SL_LEVELS)
        )
        output_path = config.OUTPUT_DIR / output_filename

        # --- Zapisywanie danych ---
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_labeled.reset_index().to_feather(output_path)
        logger.info(f"Zapisano dane z etykietami do: {output_path}")
        
        # Podsumowanie kolumn z etykietami
        label_columns = config.get_all_label_columns()
        logger.info(f"Utworzono {len(label_columns)} kolumn z etykietami: {label_columns}")

        if config.SAVE_CSV_COPY:
            csv_path = output_path.with_suffix('.csv')
            df_labeled.reset_index().to_csv(csv_path, index=False)
            logger.info(f"Zapisano kopię CSV do: {csv_path}")

        logger.info("--- Proces etykietowania zakończony pomyślnie ---")

    except Exception as e:
        logger.error(f"Główny proces etykietowania napotkał błąd: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
