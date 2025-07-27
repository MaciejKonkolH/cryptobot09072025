"""
Moduł do obliczania cech (features) na podstawie danych OHLC + Order Book.
Obsługuje format feather z danymi order book i świeczkami OHLC.
"""
import logging
import os
import sys
import argparse
from typing import Optional

import pandas as pd
import numpy as np
import bamboo_ta as bta
from scipy import stats

# Dodajemy ścieżkę do głównego katalogu, aby importy działały poprawnie
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import feature_calculator_snapshot.config as config
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

class OrderBookFeatureCalculator:
    """
    Główna klasa odpowiedzialna za transformację danych OHLCV + Order Book
    do finalnego zbioru cech gotowego do treningu modelu.
    """
    def __init__(self):
        """Inicjalizuje klasę z parametrami."""
        self.ma_periods = config.MA_WINDOWS
        self.history_window = config.ORDERBOOK_HISTORY_WINDOW
        self.short_window = config.ORDERBOOK_SHORT_WINDOW
        self.momentum_window = config.ORDERBOOK_MOMENTUM_WINDOW
        
        logger.info(f"OrderBookFeatureCalculator zainicjalizowany")
        logger.info(f"MA okresy: {self.ma_periods}")
        logger.info(f"Order book history window: {self.history_window}")

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Wczytuje i przygotowuje dane wejściowe z pliku feather."""
        logger.info(f"Wczytywanie danych z: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Plik wejściowy nie istnieje: {file_path}")
            return None
        
        try:
            # Wczytaj bezpośrednio z pliku feather
            df = pd.read_feather(file_path)
            
            logger.info(f"Wczytano {len(df)} wierszy danych z feather.")
            
            # Ustaw timestamp jako indeks
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            logger.info("Dane wczytane i przygotowane pomyślnie.")
            return df
        except Exception as e:
            logger.error(f"Wystąpił błąd podczas wczytywania danych: {e}", exc_info=True)
            return None

    def calculate_traditional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza tradycyjne wskaźniki techniczne (bez order book)."""
        logger.info("Obliczanie tradycyjnych wskaźników technicznych...")
        
        # Grupa B: Zmienność (Volatility)
        logger.info("  -> Obliczanie Wstęg Bollingera...")
        # Grupa A: Wstęgi Bollingera
        bbands = bta.bollinger_bands(df, 'close', period=20, std_dev=2.0)
        df['bb_width'] = np.where(bbands['bb_middle'] != 0, (bbands['bb_upper'] - bbands['bb_lower']) / bbands['bb_middle'], 0)
        df['bb_position'] = np.where((bbands['bb_upper'] - bbands['bb_lower']) != 0, (df['close'] - bbands['bb_lower']) / (bbands['bb_upper'] - bbands['bb_lower']), 0)
        df['bb_position'] = (df['bb_position'] - 0.5) * 2

        # Grupa C: Pęd/Siła Ruchu (Momentum)
        logger.info("  -> Obliczanie RSI...")
        df['rsi_14'] = bta.relative_strength_index(df, column='close', period=14)['rsi']
        
        logger.info("  -> Obliczanie MACD...")
        macd = bta.macd(df, 'close', short_window=12, long_window=26, signal_window=9)
        df['macd_hist'] = macd['macd_histogram']

        # Grupa D: Siła i Kierunku Trendu (Trend)
        logger.info("  -> Obliczanie ADX...")
        df['adx_14'] = self._calculate_manual_adx(df, period=14)

        logger.info("  -> Obliczanie Choppiness Index...")
        df['choppiness_index'] = self._calculate_manual_chop(df, period=14)

        # Średnie kroczące
        logger.info("Obliczanie średnich kroczących...")
        for period in self.ma_periods:
            logger.info(f"  -> MA {period}...")
            # NAPRAWKA: Dodaję shift(1) aby używać tylko przeszłości
            df[f'ma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean().shift(1)
        
        # Cechy stosunkowe
        logger.info("Obliczanie cech stosunkowych...")
        df['price_to_ma_60'] = np.where(df['ma_60'] != 0, df['close'] / df['ma_60'], 1)
        df['price_to_ma_240'] = np.where(df['ma_240'] != 0, df['close'] / df['ma_240'], 1)
        df['ma_60_to_ma_240'] = np.where(df['ma_240'] != 0, df['ma_60'] / df['ma_240'], 1)
        df['price_to_ma_1440'] = np.where(df['ma_1440'] != 0, df['close'] / df['ma_1440'], 1)
        
        # Cechy wolumenu
        df['volume_change_norm'] = df['volume'].pct_change().fillna(0)
        
        # Cechy dedykowane
        logger.info("Obliczanie cech dedykowanych...")
        # 12. Zakres whipsaw 15-minutowy
        highest_high_15m = df['high'].rolling(window=15, min_periods=1).max()
        lowest_low_15m = df['low'].rolling(window=15, min_periods=1).min()
        df['whipsaw_range_15m'] = np.where(df['close'] != 0, (highest_high_15m - lowest_low_15m) / df['close'], 0)

        candle_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']

        df['upper_wick_ratio'] = np.where(candle_range > 0, upper_wick / candle_range, 0)
        df['lower_wick_ratio'] = np.where(candle_range > 0, lower_wick / candle_range, 0)
        df['upper_wick_ratio_5m'] = df['upper_wick_ratio'].rolling(window=5).mean()
        df['lower_wick_ratio_5m'] = df['lower_wick_ratio'].rolling(window=5).mean()

        df.drop(columns=['upper_wick_ratio', 'lower_wick_ratio'], inplace=True)
        
        logger.info("Tradycyjne wskaźniki obliczone.")
        return df

    def calculate_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza wszystkie 30 cech order book."""
        logger.info("Rozpoczynanie obliczania cech order book...")
        
        # Sprawdź czy mamy dane order book
        complete_data = df['data_quality'] == 'complete'
        logger.info(f"Wiersze z kompletnymi danymi order book: {complete_data.sum()}/{len(df)}")
        
        if complete_data.sum() == 0:
            logger.warning("Brak kompletnych danych order book - pomijam cechy order book")
            return df
        
        # Grupa A: Cechy Podstawowe (z 2 snapshotów)
        logger.info("  -> Grupa A: Cechy podstawowe...")
        df = self._calculate_basic_features(df)
        
        # Grupa B: Cechy Głębokości Poziomów
        logger.info("  -> Grupa B: Cechy głębokości poziomów...")
        df = self._calculate_level_features(df)
        
        # Grupa C: Cechy Dynamiczne
        logger.info("  -> Grupa C: Cechy dynamiczne...")
        df = self._calculate_dynamic_features(df)
        
        # Grupa D: Cechy Historyczne
        logger.info("  -> Grupa D: Cechy historyczne...")
        df = self._calculate_historical_features(df)
        
        # Grupa E: Cechy Korelacyjne
        logger.info("  -> Grupa E: Cechy korelacyjne...")
        df = self._calculate_correlation_features(df)
        
        # Grupa F: Cechy Momentum
        logger.info("  -> Grupa F: Cechy momentum...")
        df = self._calculate_momentum_features(df)
        
        # Grupa G: Cechy Koncentracji
        logger.info("  -> Grupa G: Cechy koncentracji...")
        df = self._calculate_concentration_features(df)
        
        logger.info("Wszystkie cechy order book obliczone.")
        return df

    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa A: Cechy Podstawowe (z 2 snapshotów)."""
        
        # Pomocnicze funkcje
        def calc_bid_sum(prefix):
            return sum(df[f'{prefix}depth_{level}'] for level in config.BID_LEVELS)
        
        def calc_ask_sum(prefix):
            return sum(df[f'{prefix}depth_{level}'] for level in config.ASK_LEVELS)
        
        # 1-2. Stosunek presji kupna do sprzedaży
        bid_sum_s1 = calc_bid_sum('snapshot1_')
        ask_sum_s1 = calc_ask_sum('snapshot1_')
        bid_sum_s2 = calc_bid_sum('snapshot2_')
        ask_sum_s2 = calc_ask_sum('snapshot2_')
        
        # 3-4. Stosunek kupna/sprzedaży
        df['buy_sell_ratio_s1'] = np.where(ask_sum_s1 != 0, bid_sum_s1 / ask_sum_s1, 1)
        df['buy_sell_ratio_s2'] = np.where(ask_sum_s2 != 0, bid_sum_s2 / ask_sum_s2, 1)
        
        # 5-6. Imbalans kupna/sprzedaży
        df['imbalance_s1'] = np.where((bid_sum_s1 + ask_sum_s1) != 0, (bid_sum_s1 - ask_sum_s1) / (bid_sum_s1 + ask_sum_s1), 0)
        df['imbalance_s2'] = np.where((bid_sum_s2 + ask_sum_s2) != 0, (bid_sum_s2 - ask_sum_s2) / (bid_sum_s2 + ask_sum_s2), 0)
        
        # 7. Zmiana presji między snapshotami
        df['pressure_change'] = np.where(df['buy_sell_ratio_s1'] != 0, (df['buy_sell_ratio_s2'] - df['buy_sell_ratio_s1']) / df['buy_sell_ratio_s1'], 0)
        
        return df

    def _calculate_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa B: Cechy Głębokości Poziomów."""
        
        # 6-8. Głębokość na poziomach TP/SL
        df['tp_1pct_depth_s1'] = df['snapshot1_depth_1']
        df['tp_2pct_depth_s1'] = df['snapshot1_depth_2']
        df['sl_1pct_depth_s1'] = df['snapshot1_depth_-1']
        df['tp_sl_ratio_1pct'] = np.where(df['snapshot1_depth_-1'] != 0, df['snapshot1_depth_1'] / df['snapshot1_depth_-1'], 1)
        df['tp_sl_ratio_2pct'] = np.where(df['snapshot1_depth_-2'] != 0, df['snapshot1_depth_2'] / df['snapshot1_depth_-2'], 1)
        
        return df

    def _calculate_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa C: Cechy Dynamiczne (zmiany między snapshotami)."""
        
        # Pomocnicze funkcje
        def calc_total_depth(prefix):
            return sum(df[f'{prefix}depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        def calc_total_notional(prefix):
            return sum(df[f'{prefix}notional_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        total_depth_s1 = calc_total_depth('snapshot1_')
        total_depth_s2 = calc_total_depth('snapshot2_')
        total_notional_s1 = calc_total_notional('snapshot1_')
        total_notional_s2 = calc_total_notional('snapshot2_')
        
        # 11. Zmiana głębokości totalnej
        df['total_depth_change'] = np.where(total_depth_s1 != 0, (total_depth_s2 - total_depth_s1) / total_depth_s1, 0)
        
        # 12. Zmiana wartości nominalnej
        df['notional_change'] = np.where(total_notional_s1 != 0, (total_notional_s2 - total_notional_s1) / total_notional_s1, 0)
        
        # 13-14. Zmiany głębokości na poziomach +1/-1
        df['depth_1_change'] = np.where(df['snapshot1_depth_1'] != 0, (df['snapshot2_depth_1'] - df['snapshot1_depth_1']) / df['snapshot1_depth_1'], 0)
        df['depth_neg1_change'] = np.where(df['snapshot1_depth_-1'] != 0, (df['snapshot2_depth_-1'] - df['snapshot1_depth_-1']) / df['snapshot1_depth_-1'], 0)
        
        return df

    def _calculate_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa D: Cechy Historyczne (ostatnie 30 snapshotów)."""
        
        # Pomocnicze funkcje dla total depth
        def calc_total_depth_series():
            return sum(df[f'snapshot1_depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        total_depth = calc_total_depth_series()
        
        # 15. Trend głębokości totalnej
        df['depth_trend'] = total_depth.rolling(window=self.history_window, min_periods=1).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == self.history_window else np.nan
        ).shift(1)
        
        # 16. Volatilność order book
        df['ob_volatility'] = total_depth.pct_change().rolling(window=self.history_window, min_periods=1).std().shift(1)
        
        # 17. Średnia głębokość historyczna
        df['avg_depth_30'] = total_depth.rolling(window=self.history_window, min_periods=1).mean().shift(1)
        
        # 18. Anomalia głębokości
        rolling_std = total_depth.rolling(window=self.history_window, min_periods=1).std().shift(1)
        df['avg_depth_30_shifted'] = total_depth.rolling(window=self.history_window, min_periods=1).mean().shift(1)
        # NAPRAWKA: Dodaję lepsze zabezpieczenie przed bardzo małymi wartościami
        df['depth_anomaly'] = np.where(
            (rolling_std > 1e-10) & (rolling_std != 0),  # Sprawdź czy nie jest zbyt małe
            (total_depth - df['avg_depth_30_shifted']) / rolling_std,
            0
        )
        df.drop(columns=['avg_depth_30_shifted'], inplace=True)
        
        # 19-20. Trendy presji kupna/sprzedaży
        buy_pressure = sum(df[f'snapshot1_depth_{level}'] for level in config.BID_LEVELS)
        sell_pressure = sum(df[f'snapshot1_depth_{level}'] for level in config.ASK_LEVELS)
        
        df['buy_pressure_trend'] = buy_pressure.rolling(window=self.history_window, min_periods=1).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == self.history_window else np.nan
        ).shift(1)
        df['sell_pressure_trend'] = sell_pressure.rolling(window=self.history_window, min_periods=1).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == self.history_window else np.nan
        ).shift(1)
        
        return df

    def _calculate_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa E: Cechy Korelacyjne."""
        
        total_depth = sum(df[f'snapshot1_depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        # 21. Korelacja głębokości z ceną
        depth_price_corr = total_depth.rolling(window=self.history_window, min_periods=1).corr(df['close']).shift(1)
        # NAPRAWKA: Zastąp inf/NaN wartościami 0
        df['depth_price_corr'] = depth_price_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 22. Korelacja presji z wolumenem
        pressure_volume_corr = df['buy_sell_ratio_s1'].rolling(window=self.history_window, min_periods=1).corr(df['volume']).shift(1)
        # NAPRAWKA: Zastąp inf/NaN wartościami 0
        df['pressure_volume_corr'] = pressure_volume_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        return df

    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa F: Cechy Momentum."""
        
        total_depth = sum(df[f'snapshot1_depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        depth_change = total_depth.pct_change()
        
        # 23. Przyspieszenie zmian głębokości
        df['depth_acceleration'] = depth_change.diff()
        
        # 24. Momentum order book
        df['ob_momentum'] = depth_change.rolling(window=self.momentum_window, min_periods=1).mean().shift(1)
        
        # 25. Breakout signal
        rolling_max = total_depth.rolling(window=self.history_window, min_periods=1).max().shift(1)
        rolling_std = total_depth.rolling(window=self.history_window, min_periods=1).std().shift(1)
        # NAPRAWKA: Dodaję lepsze zabezpieczenie przed bardzo małymi wartościami
        df['breakout_signal'] = np.where(
            (rolling_std > 1e-10) & (rolling_std != 0),  # Sprawdź czy nie jest zbyt małe
            (total_depth - rolling_max) / rolling_std,
            0
        )
        
        return df

    def _calculate_concentration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Grupa G: Cechy Koncentracji."""
        
        # Pomocnicze funkcje
        def calc_all_depths():
            return [df[f'snapshot1_depth_{level}'] for level in config.ORDERBOOK_LEVELS]
        
        all_depths = calc_all_depths()
        total_depth = sum(all_depths)
        max_depth = pd.concat(all_depths, axis=1).max(axis=1)
        
        # 26. Koncentracja głębokości
        df['depth_concentration'] = np.where(total_depth != 0, max_depth / total_depth, 0)
        
        # 27. Asymetria poziomów
        near_bid = sum(df[f'snapshot1_depth_{level}'] for level in [-1, -2, -3])
        near_ask = sum(df[f'snapshot1_depth_{level}'] for level in [1, 2, 3])
        df['level_asymmetry'] = np.where(total_depth != 0, (near_ask - near_bid) / total_depth, 0)
        
        # 28. Dominacja poziomów bliskich
        df['near_level_dominance'] = np.where(total_depth != 0, (df['snapshot1_depth_1'] + df['snapshot1_depth_-1']) / total_depth, 0)
        
        # 29. Dominacja poziomów dalekich
        df['far_level_dominance'] = np.where(total_depth != 0, (df['snapshot1_depth_5'] + df['snapshot1_depth_-5']) / total_depth, 0)
        
        # 30. Stabilność order book
        depth_changes = total_depth.pct_change().rolling(window=self.short_window, min_periods=1).std().shift(1)
        df['ob_stability'] = np.where(depth_changes != 0, 1 / depth_changes, 0)
        
        return df

    def _calculate_manual_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Oblicza wskaźnik ADX ręcznie."""
        df_adx = df.copy()
        
        df_adx['tr'] = bta.true_range(df_adx)
        dm_result = bta.directional_movement(df_adx, length=period)
        df_adx['plus_dm'] = dm_result['dmp']
        df_adx['minus_dm'] = dm_result['dmn']

        alpha = 1 / period
        df_adx['plus_di'] = 100 * np.where(
            df_adx['tr'].ewm(alpha=alpha, adjust=False).mean() != 0,
            df_adx['plus_dm'].ewm(alpha=alpha, adjust=False).mean() / df_adx['tr'].ewm(alpha=alpha, adjust=False).mean(),
            0
        )
        df_adx['minus_di'] = 100 * np.where(
            df_adx['tr'].ewm(alpha=alpha, adjust=False).mean() != 0,
            df_adx['minus_dm'].ewm(alpha=alpha, adjust=False).mean() / df_adx['tr'].ewm(alpha=alpha, adjust=False).mean(),
            0
        )

        df_adx['dx'] = 100 * np.where(
            (df_adx['plus_di'] + df_adx['minus_di']) != 0,
            abs(df_adx['plus_di'] - df_adx['minus_di']) / (df_adx['plus_di'] + df_adx['minus_di']),
            0
        )
        adx = df_adx['dx'].ewm(alpha=alpha, adjust=False).mean()
        
        return adx

    def _calculate_manual_chop(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Oblicza wskaźnik Choppiness Index (CHOP) ręcznie."""
        df_chop = df.copy()
        
        df_chop['tr'] = bta.true_range(df_chop)
        sum_tr = df_chop['tr'].rolling(window=period).sum()
        highest_high = df_chop['high'].rolling(window=period).max()
        lowest_low = df_chop['low'].rolling(window=period).min()
        
        chop = 100 * np.where(
            (highest_high - lowest_low) != 0,
            np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period),
            0
        )
        
        return chop

    def calculate_features(self, df: pd.DataFrame, user_start_dt=None, user_end_dt=None) -> pd.DataFrame:
        """
        Oblicza wszystkie cechy i filtruje do zakresu użytkownika na końcu.
        
        Args:
            df: DataFrame z danymi OHLCV + Order Book
            user_start_dt: Data początkowa zakresu użytkownika (opcjonalna)
            user_end_dt: Data końcowa zakresu użytkownika (opcjonalna)
        """
        logger.info("Rozpoczynanie obliczania wszystkich cech...")
        
        # Oblicz tradycyjne cechy na pełnym zbiorze danych
        df_with_traditional = self.calculate_traditional_features(df.copy())
        
        # Oblicz cechy order book na pełnym zbiorze danych
        df_with_orderbook = self.calculate_orderbook_features(df_with_traditional)
        
        # Usuń wiersze z NaN (po obliczeniu wszystkich cech)
        initial_rows = len(df_with_orderbook)
        df_clean = df_with_orderbook.dropna()
        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Usunięto {removed_rows:,} wierszy z brakującymi danymi (NaN).")
        logger.info(f"Pozostało {len(df_clean):,} kompletnych wierszy.")
        
        # NOWE: Filtruj do zakresu użytkownika (jeśli podano)
        if user_start_dt is not None and user_end_dt is not None:
            logger.info(f"🔄 Filtruję dane do zakresu użytkownika: {user_start_dt} - {user_end_dt}")
            
            # Dodaj 10 minut historii na początku
            filter_start_dt = user_start_dt - pd.Timedelta(minutes=10)
            filter_end_dt = user_end_dt + pd.Timedelta(days=1)  # Cały dzień końcowy
            
            # Filtruj dane
            df_filtered = df_clean[
                (df_clean.index >= filter_start_dt) & 
                (df_clean.index < filter_end_dt)
            ].copy()
            
            logger.info(f"STATYSTYKI: Przed filtrowaniem: {len(df_clean):,} wierszy")
            logger.info(f"STATYSTYKI: Po filtrowaniu: {len(df_filtered):,} wierszy")
            logger.info(f"CZAS: Zakres wynikowy: {df_filtered.index.min()} do {df_filtered.index.max()}")
            
            df_final = df_filtered
        else:
            logger.info("UWAGA: Brak zakresu użytkownika - zwracam wszystkie dane")
            df_final = df_clean
        
        # Wybierz finalne kolumny
        raw_columns = ['open', 'high', 'low', 'close', 'volume', 'data_quality']
        feature_columns = [col for col in df_final.columns if col not in raw_columns]
        
        final_columns = raw_columns + feature_columns
        df_result = df_final[final_columns]
        
        logger.info(f"Wybrano finalne kolumny: {len(raw_columns)} surowych i {len(feature_columns)} cech.")
        logger.info(f"Finalne kolumny: {final_columns}")
        
        return df_result

    def save_data(self, df: pd.DataFrame, file_path: str, to_csv: bool = False):
        """Zapisuje DataFrame do formatu feather i opcjonalnie CSV."""
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        feather_path = f"{os.path.splitext(file_path)[0]}.feather"
        df.reset_index().to_feather(feather_path)
        logger.info(f"Dane zapisane pomyślnie do: {feather_path}")

        if to_csv:
            csv_path = f"{os.path.splitext(file_path)[0]}.csv"
            df.to_csv(csv_path)
            logger.info(f"Dane zapisane pomyślnie do: {csv_path}")

def main():
    """Główna pętla programu."""
    setup_logging()
    logger.info("--- Rozpoczynanie procesu obliczania cech Order Book ---")
    
    parser = argparse.ArgumentParser(description="Kalkulator Cech Order Book dla Danych Finansowych")
    parser.add_argument(
        '--input', 
        type=str, 
        default=str(config.INPUT_FILE_PATH),
        help=f"Ścieżka do pliku wejściowego JSON. Domyślnie: {config.INPUT_FILE_PATH}"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=os.path.join(config.OUTPUT_DIR, config.DEFAULT_OUTPUT_FILENAME),
        help="Ścieżka do pliku wyjściowego. Domyślnie: orderbook_ohlc_features.feather"
    )
    parser.add_argument(
        '--user-start',
        type=str,
        help="Data początkowa zakresu użytkownika (YYYY-MM-DD). Jeśli nie podano, nie filtruje."
    )
    parser.add_argument(
        '--user-end',
        type=str,
        help="Data końcowa zakresu użytkownika (YYYY-MM-DD). Jeśli nie podano, nie filtruje."
    )
    parser.add_argument(
        '--to-csv',
        action='store_true',
        help="Jeśli podano, zapisuje również kopię wynikową w formacie .csv"
    )
    args = parser.parse_args()

    # Parsuj daty użytkownika jeśli podano
    user_start_dt = None
    user_end_dt = None
    if args.user_start and args.user_end:
        try:
            user_start_dt = pd.to_datetime(args.user_start)
            user_end_dt = pd.to_datetime(args.user_end)
            logger.info(f"DATA: Zakres użytkownika: {user_start_dt} do {user_end_dt}")
        except Exception as e:
            logger.error(f"Błąd parsowania dat użytkownika: {e}")
            sys.exit(1)

    calculator = OrderBookFeatureCalculator()
    
    df = calculator.load_data(args.input)
    if df is not None:
        final_df = calculator.calculate_features(df, user_start_dt, user_end_dt)
        calculator.save_data(final_df, args.output, args.to_csv)
        logger.info("--- Proces obliczania cech Order Book zakończony pomyślnie. ---")
        
        feather_path = f"{os.path.splitext(args.output)[0]}.feather"
        logger.info(f"Wynikowy plik (feather): {feather_path}")
        if args.to_csv:
            csv_path = f"{os.path.splitext(args.output)[0]}.csv"
            logger.info(f"Wynikowy plik (csv):    {csv_path}")

if __name__ == "__main__":
    main() 