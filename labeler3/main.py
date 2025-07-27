"""
Nowy moduł do etykietowania danych (`labeler3`).
Dostosowany do nowego formatu danych z feature_calculator_ohlc_snapshot (85 kolumn).
Obsługuje wiele poziomów TP/SL jednocześnie.
"""
import logging
import os
import sys
import time
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
    import labeler3.config as config
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
    Dostosowany do danych z feature_calculator_ohlc_snapshot (85 kolumn).
    """

    def __init__(self):
        """Inicjalizuje klasę z parametrami z pliku konfiguracyjnego."""
        self.future_window = config.FUTURE_WINDOW_MINUTES
        self.tp_sl_levels = config.TP_SL_LEVELS
        self.label_map = config.LABEL_MAPPING
        
        logger.info("MultiLevelLabeler (labeler3) zainicjalizowany.")
        logger.info(f"Parametry: Okno przyszłości={self.future_window} min")
        logger.info(f"Poziomy TP/SL: {self.tp_sl_levels}")
        logger.info(f"Mapowanie etykiet 3-klasowe: {self.label_map}")

    def calculate_labels_for_level(self, ohlc_data: np.ndarray, timestamps: np.ndarray, 
                                 tp_pct: float, sl_pct: float, level_info: str = "") -> np.ndarray:
        """
        Oblicza etykiety dla jednego poziomu TP/SL (3-klasowy system).
        
        Args:
            ohlc_data: Array z danymi [high, low, close]
            timestamps: Array z timestampami
            tp_pct: Procent Take Profit (np. 2.0 dla 2%)
            sl_pct: Procent Stop Loss (np. 1.0 dla 1%)
            level_info: Informacja o poziomie dla logów
            
        Returns:
            np.ndarray: Array z etykietami (0=LONG, 1=SHORT, 2=NEUTRAL)
        """
        long_tp_pct = tp_pct / 100
        long_sl_pct = sl_pct / 100
        short_tp_pct = tp_pct / 100  # Symmetric
        short_sl_pct = sl_pct / 100  # Symmetric
        
        labels = np.full(len(ohlc_data), self.label_map['NEUTRAL'], dtype=np.int8)
        total_rows = len(ohlc_data)
        
        # Ustawienie iteratora z paskiem postępu
        if TQDM_AVAILABLE:
            iterator = tqdm(range(total_rows), desc=f"{level_info}", unit="wiersz")
        else:
            iterator = range(total_rows)
            logger.info(f"{level_info} - Rozpoczynanie przetwarzania {total_rows:,} wierszy...")
        
        start_time = time.time()
        last_log_time = start_time
        
        for i in iterator:
            if i + self.future_window >= len(ohlc_data):
                continue

            entry_price = ohlc_data[i, 2]  # close price
            
            # Oblicz poziomy TP/SL
            long_tp_price = entry_price * (1 + long_tp_pct)
            long_sl_price = entry_price * (1 - long_sl_pct)
            short_tp_price = entry_price * (1 - short_tp_pct)
            short_sl_price = entry_price * (1 + short_sl_pct)

            # Inicjalizacja zmiennych dla pozycji
            long_result = None  # 'TP' lub 'SL' lub None
            short_result = None  # 'TP' lub 'SL' lub None

            # Sprawdź przyszłe 60 minut
            for j in range(self.future_window):
                if i + 1 + j >= len(ohlc_data):
                    break
                    
                future_high = ohlc_data[i + 1 + j, 0]  # high
                future_low = ohlc_data[i + 1 + j, 1]   # low

                # Sprawdź pozycję długą (jeśli jeszcze nie zamknięta)
                if long_result is None:
                    if future_high >= long_tp_price:
                        long_result = 'TP'
                    elif future_low <= long_sl_price:
                        long_result = 'SL'
                
                # Sprawdź pozycję krótką (jeśli jeszcze nie zamknięta)
                if short_result is None:
                    if future_low <= short_tp_price:
                        short_result = 'TP'
                    elif future_high >= short_sl_price:
                        short_result = 'SL'
                
                # Przerwij pętlę jeśli obie pozycje zamknięte
                if long_result is not None and short_result is not None:
                    break

            # Logika decyzyjna 3-klasowa
            if long_result == 'TP' and short_result != 'TP':
                # Tylko Long TP osiągnięty
                labels[i] = self.label_map['LONG']
            elif short_result == 'TP' and long_result != 'TP':
                # Tylko Short TP osiągnięty
                labels[i] = self.label_map['SHORT']
            elif long_result == 'TP' and short_result == 'TP':
                # Obie pozycje TP - sprawdź która pierwsza
                # (w tym uproszczonym algorytmie zakładamy, że Long TP ma pierwszeństwo)
                labels[i] = self.label_map['LONG']
            else:
                # Obie pozycje SL lub timeout
                labels[i] = self.label_map['NEUTRAL']
            
            # Logowanie postępu co 10% lub co 30 sekund
            current_time = time.time()
            if not TQDM_AVAILABLE and (current_time - last_log_time > 30 or i % max(1, total_rows // 10) == 0):
                progress_pct = (i / total_rows) * 100
                elapsed_time = current_time - start_time
                estimated_total = elapsed_time / (i + 1) * total_rows if i > 0 else 0
                remaining_time = estimated_total - elapsed_time
                
                logger.info(f"{level_info} - Postęp: {progress_pct:.1f}% ({i:,}/{total_rows:,}) - "
                           f"Czas: {elapsed_time:.0f}s, Pozostało: {remaining_time:.0f}s")
                last_log_time = current_time

        total_time = time.time() - start_time
        logger.info(f"{level_info} - Zakończono w {total_time:.1f} sekund ({total_time/60:.1f} minut)")
        
        return labels

    def calculate_all_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Główna funkcja obliczająca etykiety dla wszystkich poziomów TP/SL.
        Przyjmuje DataFrame z cechami (85 kolumn) i dodaje kolumny z etykietami.
        """
        logger.info("Rozpoczynanie obliczania etykiet 3-klasowych dla wszystkich poziomów...")
        
        # Sprawdź czy mamy wymagane kolumny OHLC
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Brakuje wymaganych kolumn OHLC: {missing_cols}")
        
        # Sprawdź czy to są dane z feature_calculator_ohlc_snapshot
        expected_cols = ['bb_width', 'rsi_14', 'macd_hist', 'buy_sell_ratio_s1']
        found_feature_cols = [col for col in expected_cols if col in df.columns]
        if len(found_feature_cols) >= 2:
            logger.info(f"Wykryto dane z feature_calculator_ohlc_snapshot (znalezione cechy: {found_feature_cols})")
        else:
            logger.warning("Nie wykryto typowych cech z feature_calculator_ohlc_snapshot")
        
        ohlc_data = df[['high', 'low', 'close']].to_numpy()
        timestamps = df.index.to_numpy()
        total_rows = len(ohlc_data)
        
        logger.info(f"Przetwarzanie {total_rows:,} wierszy dla {len(self.tp_sl_levels)} poziomów TP/SL")
        logger.info(f"Zakres czasowy: {df.index.min()} do {df.index.max()}")
        
        overall_start_time = time.time()
        
        # Oblicz etykiety dla każdego poziomu
        for i, (tp_pct, sl_pct) in enumerate(self.tp_sl_levels):
            level_start_time = time.time()
            level_info = f"Poziom {i+1}/{len(self.tp_sl_levels)}: TP={tp_pct}%, SL={sl_pct}%"
            
            logger.info(f"  -> {level_info}")
            
            # Oblicz etykiety dla tego poziomu
            labels = self.calculate_labels_for_level(ohlc_data, timestamps, tp_pct, sl_pct, level_info)
            
            # Dodaj kolumnę z etykietami
            column_suffix = config.get_level_suffix(tp_pct, sl_pct)
            column_name = f"label_{column_suffix}"
            df[column_name] = labels
            
            # Statystyki dla tego poziomu
            unique_labels, counts = np.unique(labels, return_counts=True)
            stats = dict(zip(unique_labels, counts))
            
            # Konwertuj liczby na nazwy etykiet dla czytelności
            readable_stats = {}
            total_samples = len(labels)
            
            for label_num, count in stats.items():
                label_name = config.REVERSE_LABEL_MAPPING.get(label_num, f"UNKNOWN_{label_num}")
                readable_stats[label_name] = count
            
            # Oblicz procenty
            percentages = {}
            for label_name, count in readable_stats.items():
                percentages[label_name] = (count / total_samples) * 100
            
            level_time = time.time() - level_start_time
            logger.info(f"     Statystyki {column_name}: {readable_stats}")
            logger.info(f"     Rozkład procentowy: LONG {percentages.get('LONG', 0):.1f}%, SHORT {percentages.get('SHORT', 0):.1f}%, NEUTRAL {percentages.get('NEUTRAL', 0):.1f}%")
            logger.info(f"     Czas wykonania poziomu: {level_time:.1f}s ({level_time/60:.1f} min)")
            
            # Szacowany czas pozostały
            if i < len(self.tp_sl_levels) - 1:
                avg_time_per_level = (time.time() - overall_start_time) / (i + 1)
                remaining_levels = len(self.tp_sl_levels) - i - 1
                estimated_remaining = avg_time_per_level * remaining_levels
                logger.info(f"     Szacowany czas pozostały: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} min)")
        
        total_time = time.time() - overall_start_time
        logger.info(f"Zakończono obliczanie etykiet dla wszystkich poziomów w {total_time:.1f}s ({total_time/60:.1f} min)")
        return df

    def analyze_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Analizuje rozkład etykiet i zwraca statystyki."""
        logger.info("Analizowanie rozkładu etykiet...")
        
        label_columns = config.get_all_label_columns()
        analysis = {}
        
        for col in label_columns:
            if col in df.columns:
                labels = df[col]
                total = len(labels)
                
                # Rozkład etykiet
                label_counts = labels.value_counts().to_dict()
                
                # Konwertuj na nazwy
                readable_counts = {}
                for label_num, count in label_counts.items():
                    label_name = config.REVERSE_LABEL_MAPPING.get(label_num, f"UNKNOWN_{label_num}")
                    readable_counts[label_name] = count
                
                # Procenty
                percentages = {}
                for label_name, count in readable_counts.items():
                    percentages[label_name] = (count / total) * 100
                
                analysis[col] = {
                    'total_samples': total,
                    'label_counts': readable_counts,
                    'percentages': percentages
                }
        
        return analysis

def main():
    """Główna funkcja uruchamiająca proces etykietowania."""
    logger.info("--- Rozpoczynanie procesu etykietowania (labeler3) ---")
    logger.info(f"Moduł: {config.MODULE_INFO['name']} v{config.MODULE_INFO['version']}")
    logger.info(f"Opis: {config.MODULE_INFO['description']}")
    
    input_path = Path(config.INPUT_FILE_PATH)
    logger.info(f"Wczytywanie danych z cechami z: {input_path}")
    
    if not input_path.exists():
        logger.error(f"Plik wejściowy nie istnieje: {input_path}")
        logger.error(f"Sprawdź czy plik {config.INPUT_FILENAME} istnieje w {config.INPUT_DIR}")
        sys.exit(1)
        
    try:
        # Wczytaj dane z cechami
        df_with_features = pd.read_feather(input_path)
        logger.info(f"Wczytano {len(df_with_features):,} wierszy danych z cechami.")
        logger.info(f"Liczba kolumn: {len(df_with_features.columns)}")

        # Ustaw indeks na timestamp jeśli potrzeba
        if 'timestamp' in df_with_features.columns and df_with_features.index.name != 'timestamp':
             df_with_features['timestamp'] = pd.to_datetime(df_with_features['timestamp'])
             df_with_features.set_index('timestamp', inplace=True)
             logger.info("Ustawiono timestamp jako indeks")

        # Sprawdź czy mamy dane OHLC
        ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
        found_ohlc = [col for col in ohlc_cols if col in df_with_features.columns]
        logger.info(f"Znalezione kolumny OHLC: {found_ohlc}")

        # Oblicz etykiety dla wszystkich poziomów
        labeler = MultiLevelLabeler()
        df_labeled = labeler.calculate_all_labels(df_with_features)
        
        # Analiza rozkładu etykiet
        analysis = labeler.analyze_label_distribution(df_labeled)
        logger.info("Rozkład etykiet:")
        for col, stats in analysis.items():
            logger.info(f"  {col}: {stats['percentages']}")
            logger.info(f"    Procentowy rozkład: LONG {stats['percentages'].get('LONG', 0):.1f}%, SHORT {stats['percentages'].get('SHORT', 0):.1f}%, NEUTRAL {stats['percentages'].get('NEUTRAL', 0):.1f}%")
        
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

        # Finalne statystyki
        logger.info(f"Finalny rozmiar danych: {len(df_labeled):,} wierszy, {len(df_labeled.columns)} kolumn")
        logger.info("--- Proces etykietowania zakończony pomyślnie ---")

    except Exception as e:
        logger.error(f"Główny proces etykietowania napotkał błąd: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 