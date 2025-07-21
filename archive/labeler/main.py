"""
Nowy, uproszczony moduł do etykietowania danych (`labeler`).
Jego jedynym zadaniem jest dodanie etykiet do pliku, który już zawiera cechy.
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
    import labeler.config as config
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

class SimpleLabeler:
    """
    Klasa odpowiedzialna za obliczanie etykiet na podstawie
    uproszczonego i poprawionego algorytmu 'competitive'.
    """

    def __init__(self):
        """Inicjalizuje klasę z parametrami z pliku konfiguracyjnego."""
        self.future_window = config.FUTURE_WINDOW_MINUTES
        self.long_tp_pct = config.LONG_TP_PCT / 100
        self.long_sl_pct = config.LONG_SL_PCT / 100
        self.short_tp_pct = config.SHORT_TP_PCT / 100
        self.short_sl_pct = config.SHORT_SL_PCT / 100
        self.label_map = config.LABEL_MAPPING
        logger.info("SimpleLabeler zainicjalizowany.")
        logger.info(f"Parametry: Okno przyszłości={self.future_window} min, "
                    f"Long TP={self.long_tp_pct:.2%}, Long SL={self.long_sl_pct:.2%}, "
                    f"Short TP={self.short_tp_pct:.2%}, Short SL={self.short_sl_pct:.2%}")

    def calculate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Główna funkcja obliczająca etykiety. Przyjmuje DataFrame, który
        musi zawierać kolumny OHLC, i dodaje do niego kolumnę 'label'.
        """
        logger.info("Rozpoczynanie obliczania etykiet...")
        
        ohlc_data = df[['high', 'low', 'close']].to_numpy()
        timestamps = df.index.to_numpy()
        labels = np.full(len(df), self.label_map['TIMEOUT_HOLD'], dtype=np.int8)
        total_rows = len(ohlc_data)
        
        iterable = range(total_rows)
        if TQDM_AVAILABLE:
            iterable = tqdm(iterable, desc="Obliczanie etykiet", total=total_rows, ncols=100)
        else:
            logger.info("Biblioteka 'tqdm' nie jest dostępna. Logowanie postępu co 5000 wierszy.")
            
        for i in iterable:
            if not TQDM_AVAILABLE and (i % 5000 == 0 or i == total_rows - 1):
                progress_pct = (i + 1) / total_rows * 100
                logger.info(f"Przetwarzanie wiersza: {i + 1}/{total_rows} ({progress_pct:.1f}%)")

            if i + self.future_window >= len(ohlc_data):
                continue

            entry_price = ohlc_data[i, 2]
            
            long_tp_price = entry_price * (1 + self.long_tp_pct)
            long_sl_price = entry_price * (1 - self.long_sl_pct)
            short_tp_price = entry_price * (1 - self.short_tp_pct)
            short_sl_price = entry_price * (1 + self.short_sl_pct)

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

            # --- Nowa, 6-klasowa hierarchia decyzyjna ---

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

        logger.info("Zakończono obliczanie etykiet.")
        df['label'] = labels
        logger.info("Dodano kolumnę 'label' do DataFrame.")
        return df

def main():
    """Główna funkcja uruchamiająca proces etykietowania."""
    logger.info("--- Rozpoczynanie procesu etykietowania ---")
    
    input_path = Path(config.INPUT_FILE_PATH)
    logger.info(f"Wczytywanie danych z cechami z: {input_path}")
    
    if not input_path.exists():
        logger.error(f"Plik wejściowy nie istnieje: {input_path}")
        sys.exit(1)
        
    try:
        df_with_features = pd.read_feather(input_path)
        logger.info(f"Wczytano {len(df_with_features):,} wierszy danych z cechami.")

        if 'date' in df_with_features.columns and df_with_features.index.name != 'date':
             df_with_features['date'] = pd.to_datetime(df_with_features['date'])
             df_with_features.set_index('date', inplace=True)

        labeler = SimpleLabeler()
        df_labeled = labeler.calculate_labels(df_with_features)
        
        # --- Tworzenie nazwy pliku wyjściowego ---
        base_name = config.INPUT_FILENAME.replace('_features.feather', '')
        
        output_filename = config.OUTPUT_FILENAME_TEMPLATE.format(
            base_name=base_name,
            tp=f"{int(config.LONG_TP_PCT*100):03d}",
            sl=f"{int(config.LONG_SL_PCT*100):03d}",
            fw=config.FUTURE_WINDOW_MINUTES
        )
        output_path = config.OUTPUT_DIR / output_filename

        # --- Zapisywanie danych ---
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_labeled.reset_index().to_feather(output_path)
        logger.info(f"Zapisano dane z etykietami do: {output_path}")

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
