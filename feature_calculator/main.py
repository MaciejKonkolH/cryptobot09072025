"""
Moduł do obliczania cech (features) na podstawie danych z etykietami.
"""
import logging
import os
import sys
import argparse
from typing import List, Optional

import pandas as pd
import numpy as np
import re
import bamboo_ta as bta

# Dodajemy ścieżkę do głównego katalogu, aby importy działały poprawnie
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import feature_calculator.config as config
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

class FeatureCalculator:
    """
    Główna klasa odpowiedzialna za transformację danych OHLCV z etykietami
    do finalnego zbioru cech gotowego do treningu modelu.
    """
    def __init__(self, ma_periods: List[int]):
        """Inicjalizuje klasę z parametrami."""
        self.ma_periods = ma_periods
        logger.info(f"FeatureCalculator zainicjalizowany z okresami MA: {self.ma_periods}")

    def _calculate_manual_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Oblicza wskaźnik ADX ręcznie przy użyciu podstawowych operacji pandas.
        Jest to niezależna implementacja, aby uniknąć błędów w bibliotekach zewnętrznych.
        """
        df_adx = df.copy()
        
        # Krok 1: Obliczenie True Range (TR) oraz +DM i -DM
        df_adx['tr'] = bta.true_range(df_adx)
        
        dm_result = bta.directional_movement(df_adx, length=period)
        df_adx['plus_dm'] = dm_result['dmp']
        df_adx['minus_dm'] = dm_result['dmn']

        # Krok 2: Wygładzenie za pomocą EMA (Wilder's Smoothing)
        alpha = 1 / period
        df_adx['plus_di'] = 100 * (df_adx['plus_dm'].ewm(alpha=alpha, adjust=False).mean() / df_adx['tr'].ewm(alpha=alpha, adjust=False).mean())
        df_adx['minus_di'] = 100 * (df_adx['minus_dm'].ewm(alpha=alpha, adjust=False).mean() / df_adx['tr'].ewm(alpha=alpha, adjust=False).mean())

        # Krok 3: Obliczenie ADX
        df_adx['dx'] = 100 * (abs(df_adx['plus_di'] - df_adx['minus_di']) / (df_adx['plus_di'] + df_adx['minus_di']))
        adx = df_adx['dx'].ewm(alpha=alpha, adjust=False).mean()
        
        return adx

    def _calculate_manual_chop(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Oblicza wskaźnik Choppiness Index (CHOP) ręcznie.
        Formuła: 100 * log10(sum(ATR, n) / (max(high, n) - min(low, n))) / log10(n)
        """
        df_chop = df.copy()
        
        # Używamy True Range, który jest już dostępny w bamboo_ta
        df_chop['tr'] = bta.true_range(df_chop)
        
        # Suma True Range w okresie 'n'
        sum_tr = df_chop['tr'].rolling(window=period).sum()
        
        # Najwyższy high i najniższy low w okresie 'n'
        highest_high = df_chop['high'].rolling(window=period).max()
        lowest_low = df_chop['low'].rolling(window=period).min()
        
        # Główna formuła wskaźnika
        chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
        
        return chop

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Wczytuje i przygotowuje dane wejściowe z modułu 'labeler'."""
        logger.info(f"Wczytywanie danych z: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Plik wejściowy nie istnieje: {file_path}")
            return None
        
        try:
            df = pd.read_feather(file_path)
            logger.info(f"Wczytano {len(df):,} wierszy danych.")

            required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(df.columns):
                logger.error(f"Brak wymaganych kolumn. Wymagane: {required_cols}. Znaleziono: {df.columns}")
                return None

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info("Dane wczytane i przygotowane pomyślnie.")
            return df
        except Exception as e:
            logger.error(f"Wystąpił błąd podczas wczytywania danych: {e}", exc_info=True)
            return None

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Główna funkcja, która wykonuje wszystkie kroki obliczeniowe:
        1. Oblicza wskaźniki techniczne za pomocą bamboo_ta.
        2. Oblicza cechy niestandardowe (np. stosunek ceny do MA).
        3. Czyści dane z wartości NaN.
        4. Selekcjonuje finalne kolumny, zachowując OHLCV.
        """
        logger.info("Rozpoczynanie obliczania nowego zestawu cech...")
        
        # Upewnijmy się, że DataFrame ma standardowe nazwy kolumn, których oczekuje bamboo_ta
        df.rename(columns={
            config.COL_OPEN: 'open',
            config.COL_HIGH: 'high',
            config.COL_LOW: 'low',
            config.COL_CLOSE: 'close',
            config.COL_VOLUME: 'volume'
        }, inplace=True)

        # --- Krok 1: Obliczanie wskaźników z biblioteki bamboo_ta ---
        logger.info("Obliczanie wskaźników z biblioteki bamboo_ta...")
        
        # Grupa B: Zmienność (Volatility)
        logger.info("  -> Obliczanie Wstęg Bollingera i cech kanałowych (okres 20)...")
        bbands = bta.bollinger_bands(df, 'close', period=20, std_dev=2.0)
        
        # Cecha 1: Szerokość kanału Bollingera (znormalizowana)
        df['bb_width'] = (bbands['bb_upper'] - bbands['bb_lower']) / bbands['bb_middle']

        # Cecha 2: Pozycja ceny w kanale Bollingera
        # Wartość od -1 (na dolnej bandzie) do +1 (na górnej bandzie)
        df['bb_position'] = (df['close'] - bbands['bb_lower']) / (bbands['bb_upper'] - bbands['bb_lower'])
        # Przeskalowanie do zakresu [-1, 1] - odejmujemy 0.5 i mnożymy przez 2
        df['bb_position'] = (df['bb_position'] - 0.5) * 2

        # Grupa C: Pęd/Siła Ruchu (Momentum)
        logger.info("  -> Obliczanie RSI (okres 14)...")
        df['rsi_14'] = bta.relative_strength_index(df, column='close', period=14)['rsi']
        
        logger.info("  -> Obliczanie MACD (12, 26, 9)...")
        macd = bta.macd(df, 'close', short_window=12, long_window=26, signal_window=9)
        # Zmiana nazwy bb_width dla spójności
        df.rename(columns={'BBB_20_2.0': 'bb_width'}, inplace=True)

        # Poprawione obliczanie histogramu MACD.
        # Używamy surowej wartości histogramu, bez błędnej normalizacji przez cenę.
        df['macd_hist'] = macd['macd_histogram']

        # Grupa D: Siła i Kierunku Trendu (Trend)
        logger.info("  -> Obliczanie ADX (okres 14) - implementacja własna...")
        df['adx_14'] = self._calculate_manual_adx(df, period=14)

        logger.info("  -> Obliczanie Choppiness Index (okres 14) - implementacja własna...")
        df['choppiness_index'] = self._calculate_manual_chop(df, period=14)

        logger.info("Wskaźniki z bamboo_ta i własne obliczone.")

        # --- Krok 2: Obliczanie cech niestandardowych (średnie i stosunki) ---
        logger.info("Rozpoczynanie obliczania średnich kroczących...")
        ma_periods_all = [60, 240, 1440, 43200]
        for period in ma_periods_all:
            logger.info(f"  -> Obliczanie średniej kroczącej ceny (okres {period})...")
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        
        volume_ma_periods = [1440, 43200]
        for period in volume_ma_periods:
            logger.info(f"  -> Obliczanie średniej kroczącej wolumenu (okres {period})...")
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()

        logger.info("Średnie kroczące obliczone.")
        
        logger.info("Obliczanie cech opartych na stosunkach...")
        df['price_to_ma_60'] = df['close'] / df['ma_60']
        df['price_to_ma_240'] = df['close'] / df['ma_240']
        df['ma_60_to_ma_240'] = df['ma_60'] / df['ma_240']
        
        # Grupa E: Cechy Wolumenu i Kontekstu Długoterminowego
        df['volume_change_norm'] = df['volume'].pct_change().fillna(0)
        
        df['price_to_ma_1440'] = df['close'] / df['ma_1440']
        df['price_to_ma_43200'] = df['close'] / df['ma_43200']
        df['volume_to_ma_1440'] = df['volume'] / df['volume_ma_1440']
        df['volume_to_ma_43200'] = df['volume'] / df['volume_ma_43200']
        
        logger.info("Obliczono cechy niestandardowe (stosunki do MA, zmiany wolumenu).")

        # --- Krok 3: Cechy dedykowane do wykrywania pułapek i chaosu ---
        logger.info("Obliczanie cech dedykowanych do wykrywania pułapek i chaosu...")

        # Cecha do rozróżniania Nudy od Chaosu (Whipsaw vs. Stagnation)
        logger.info("  -> Obliczanie 'whipsaw_range_15m'...")
        highest_high_15m = df['high'].rolling(window=15).max()
        lowest_low_15m = df['low'].rolling(window=15).min()
        df['whipsaw_range_15m'] = (highest_high_15m - lowest_low_15m) / df['close']

        # Cechy do wykrywania Pułapek (analiza cieni świec)
        logger.info("  -> Obliczanie 'upper_wick_ratio_5m' i 'lower_wick_ratio_5m'...")
        candle_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']

        # Używamy np.where, aby uniknąć dzielenia przez zero, gdy świeca nie ma zakresu
        df['upper_wick_ratio'] = np.where(candle_range > 0, upper_wick / candle_range, 0)
        df['lower_wick_ratio'] = np.where(candle_range > 0, lower_wick / candle_range, 0)

        # Używamy średniej z 5 minut, aby wygładzić sygnał
        df['upper_wick_ratio_5m'] = df['upper_wick_ratio'].rolling(window=5).mean()
        df['lower_wick_ratio_5m'] = df['lower_wick_ratio'].rolling(window=5).mean()

        # Usuwamy kolumny pomocnicze
        df.drop(columns=['upper_wick_ratio', 'lower_wick_ratio'], inplace=True)


        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # --- KROK DIAGNOSTYCZNY ---
        logger.info("--- DEBUG: Sprawdzanie wartości NaN PRZED dropna() ---")
        nan_counts = df.isnull().sum()
        logger.info("Liczba NaN w każdej kolumnie (tylko kolumny z NaN):")
        print(nan_counts[nan_counts > 0].to_string())
        # -------------------------

        # --- Krok 3: Czyszczenie danych ---
        initial_rows = len(df)
        df.dropna(inplace=True)
        final_rows = len(df)
        logger.info(f"Usunięto {initial_rows - final_rows:,} wierszy z brakującymi danymi (NaN).")
        logger.info(f"Pozostało {final_rows:,} kompletnych wierszy.")

        # --- Krok 4: Selekcja finalnych kolumn ---
        # Usuwamy ręcznie zdefiniowaną, sztywną listę cech.
        # Zamiast tego, dynamicznie wybieramy wszystkie kolumny, które nie są OHLCV,
        # co automatycznie uwzględni wszystkie nowo dodane wskaźniki.
        raw_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Wszystkie kolumny w ramce danych
        all_columns = df.columns.tolist()
        
        # Cechy to wszystkie kolumny, które NIE SĄ w `raw_columns`
        feature_columns = [col for col in all_columns if col not in raw_columns]
        
        # Łączymy wszystko w finalną ramkę danych.
        # Kolumna 'label', jeśli istnieje, zostanie automatycznie zachowana,
        # ponieważ nie ma jej w 'raw_columns'.
        final_columns = raw_columns + feature_columns
        
        # Upewniamy się, że nie ma duplikatów i zachowujemy oryginalną kolejność,
        # jednocześnie upewniając się, że wszystkie kolumny z df są uwzględnione.
        final_df = df[[col for col in final_columns if col in df.columns]]
        
        logger.info(f"Wybrano finalne kolumny: {len(raw_columns)} surowych i {len(feature_columns)} cech.")
        
        # --- KROK DIAGNOSTYCZNY ---
        logger.info(f"Finalne kolumny zapisywane do pliku: {final_df.columns.tolist()}")
        # -------------------------

        return final_df

    def save_data(self, df: pd.DataFrame, file_path: str, to_csv: bool = False):
        """Zapisuje DataFrame do formatu feather i opcjonalnie CSV."""
        # Upewniamy się, że katalog wyjściowy istnieje
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Zapis do feather
        feather_path = f"{os.path.splitext(file_path)[0]}.feather"
        df.reset_index().to_feather(feather_path)
        logger.info(f"Dane zapisane pomyślnie do: {feather_path}")

        # Opcjonalny zapis do CSV
        if to_csv:
            csv_path = f"{os.path.splitext(file_path)[0]}.csv"
            df.to_csv(csv_path)
            logger.info(f"Dane zapisane pomyślnie do: {csv_path}")

def main():
    """Główna pętla programu."""
    setup_logging()
    logger.info("--- Rozpoczynanie procesu obliczania cech ---")
    
    parser = argparse.ArgumentParser(description="Kalkulator Cech dla Danych Finansowych")
    parser.add_argument(
        '--input', 
        type=str, 
        default=config.INPUT_FILE_PATH,
        help=f"Ścieżka do pliku wejściowego .feather. Domyślnie: {config.INPUT_FILE_PATH}"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=os.path.join(config.OUTPUT_DIR, os.path.basename(config.INPUT_FILE_PATH).replace('.feather', '_features.feather')),
        help="Ścieżka do pliku wyjściowego. Domyślnie: [nazwa_wejściowa]_features.feather"
    )
    parser.add_argument(
        '--to-csv',
        action='store_true',
        help="Jeśli podano, zapisuje również kopię wynikową w formacie .csv"
    )
    args = parser.parse_args()

    calculator = FeatureCalculator(ma_periods=config.MA_WINDOWS)
    
    df = calculator.load_data(args.input)
    if df is not None:
        final_df = calculator.calculate_features(df)
        
        if config.COL_TARGET in df.columns:
            logger.info(f"Znaleziono kolumnę '{config.COL_TARGET}'. Dołączanie do zbioru wyjściowego.")
            # Używamy reindex, aby dopasować etykiety do wierszy, które przetrwały dropna()
            final_df = final_df.join(df[[config.COL_TARGET]], how='inner')
            logger.info(f"Rozkład etykiet w finalnym zbiorze: \n{final_df[config.COL_TARGET].value_counts(normalize=True)}")

        # --- KROK DIAGNOSTYCZNY ---
        logger.info(f"Kolumny, które zostaną zapisane przez feature_calculator: {final_df.columns.tolist()}")
        # -------------------------

        calculator.save_data(final_df, args.output, args.to_csv)
        logger.info("--- Proces obliczania cech zakończony pomyślnie. ---")
        
        feather_path = f"{os.path.splitext(args.output)[0]}.feather"
        logger.info(f"Wynikowy plik (feather): {feather_path}")
        if args.to_csv:
            csv_path = f"{os.path.splitext(args.output)[0]}.csv"
            logger.info(f"Wynikowy plik (csv):    {csv_path}")

if __name__ == "__main__":
    main() 