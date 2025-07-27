"""
Moduł do obliczania cech (features) na podstawie danych OHLC + Order Book.
Obsługuje format feather z danymi order book i świeczkami OHLC.
"""
import logging
import os
import sys
import argparse
from typing import Optional
import time

import pandas as pd
import numpy as np
import bamboo_ta as bta
from scipy import stats

# Dodajemy ścieżkę do głównego katalogu, aby importy działały poprawnie
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import feature_calculator_ohlc_snapshot.config as config
except ImportError:
    import config

def setup_logging():
    """Konfiguruje system logowania z logami w jednej linii."""
    log_dir = config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # Handler dla pliku
    file_handler = logging.FileHandler(os.path.join(log_dir, config.LOG_FILENAME), encoding='utf-8')
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    
    # Handler dla konsoli (jedna linia) - bez emoji dla kompatybilności
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Konfiguracja loggera
    logger = logging.getLogger(__name__)
    logger.setLevel(config.LOG_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class OHLCOrderBookFeatureCalculator:
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
        self.warmup_period = config.WARMUP_PERIOD_MINUTES
        
        logger.info(f"OHLCOrderBookFeatureCalculator zainicjalizowany")
        logger.info(f"MA okresy: {self.ma_periods}")
        logger.info(f"Order book history window: {self.history_window}")
        logger.info(f"Warmup period: {self.warmup_period} minut ({self.warmup_period/1440:.1f} dni)")

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Wczytuje i przygotowuje dane wejściowe z pliku feather."""
        logger.info(f"Wczytywanie danych z: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Plik wejściowy nie istnieje: {file_path}")
            return None
        
        try:
            # Wczytaj bezpośrednio z pliku feather
            df = pd.read_feather(file_path)
            
            logger.info(f"Wczytano {len(df):,} wierszy danych z feather")
            
            # Ustaw timestamp jako indeks
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            logger.info(f"Zakres czasowy: {df.index.min()} do {df.index.max()}")
            logger.info("Dane wczytane i przygotowane pomyślnie")
            return df
        except Exception as e:
            logger.error(f"Wystąpił błąd podczas wczytywania danych: {e}", exc_info=True)
            return None

    def calculate_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy OHLC (15 cech)."""
        logger.info("Obliczanie cech OHLC...")
        
        # 1. Wstęgi Bollingera (3 cechy)
        logger.info("  -> Wstęgi Bollingera...")
        bbands = bta.bollinger_bands(df, 'close', period=config.BOLLINGER_PERIOD, std_dev=config.BOLLINGER_STD_DEV)
        df['bb_width'] = np.where(bbands['bb_middle'] != 0, (bbands['bb_upper'] - bbands['bb_lower']) / bbands['bb_middle'], 0)
        df['bb_position'] = np.where((bbands['bb_upper'] - bbands['bb_lower']) != 0, (df['close'] - bbands['bb_lower']) / (bbands['bb_upper'] - bbands['bb_lower']), 0)
        df['bb_position'] = (df['bb_position'] - 0.5) * 2

        # 2. RSI (1 cecha)
        logger.info("  -> RSI...")
        df['rsi_14'] = bta.relative_strength_index(df, column='close', period=config.RSI_PERIOD)['rsi']
        
        # 3. MACD (1 cecha)
        logger.info("  -> MACD...")
        macd = bta.macd(df, 'close', short_window=config.MACD_SHORT, long_window=config.MACD_LONG, signal_window=config.MACD_SIGNAL)
        df['macd_hist'] = macd['macd_histogram']

        # 4. ADX (1 cecha)
        logger.info("  -> ADX...")
        df['adx_14'] = self._calculate_manual_adx(df, period=config.ADX_PERIOD)

        # 5. Średnie kroczące (3 cechy: 1h, 4h, 1d)
        logger.info("  -> Średnie kroczące...")
        for period in self.ma_periods[:-1]:  # Bez 30-dniowej na razie
            df[f'ma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean().shift(1)
        
        # 6. Cechy cenowe (4 cechy)
        logger.info("  -> Cechy cenowe...")
        df['price_to_ma_60'] = np.where(df['ma_60'] != 0, df['close'] / df['ma_60'], 1)
        df['price_to_ma_240'] = np.where(df['ma_240'] != 0, df['close'] / df['ma_240'], 1)
        df['ma_60_to_ma_240'] = np.where(df['ma_240'] != 0, df['ma_60'] / df['ma_240'], 1)
        df['price_to_ma_1440'] = np.where(df['ma_1440'] != 0, df['close'] / df['ma_1440'], 1)
        
        # 7. Cechy wolumenu (1 cecha)
        logger.info("  -> Cechy wolumenu...")
        # Bezpieczne obliczanie zmiany wolumenu - użyj logarytmu żeby uniknąć infinity
        volume_ratio = df['volume'] / (df['volume'].shift(1) + 1e-8)
        df['volume_change_norm'] = np.log(volume_ratio).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 8. Cechy świec (2 cechy)
        logger.info("  -> Cechy świec...")
        candle_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        upper_wick_ratio = np.where(candle_range > 0, upper_wick / candle_range, 0)
        lower_wick_ratio = np.where(candle_range > 0, lower_wick / candle_range, 0)
        
        df['upper_wick_ratio_5m'] = pd.Series(upper_wick_ratio, index=df.index).rolling(window=5).mean()
        df['lower_wick_ratio_5m'] = pd.Series(lower_wick_ratio, index=df.index).rolling(window=5).mean()
        
        logger.info("Cechy OHLC obliczone (15 cech)")
        return df

    def calculate_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy orderbook (12 cech)."""
        logger.info("Obliczanie cech orderbook...")
        
        # Sprawdź czy mamy dane orderbook
        orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
        if len(orderbook_columns) == 0:
            logger.warning("Brak danych orderbook - pomijam cechy orderbook")
            return df
        
        # 1. Cechy podstawowe (5 cech)
        logger.info("  -> Cechy podstawowe...")
        df = self._calculate_basic_orderbook_features(df)
        
        # 2. Cechy głębokości (4 cechy)
        logger.info("  -> Cechy głębokości...")
        df = self._calculate_depth_features(df)
        
        # 3. Cechy dynamiczne (3 cechy)
        logger.info("  -> Cechy dynamiczne...")
        df = self._calculate_dynamic_features(df)
        
        logger.info("Cechy orderbook obliczone (12 cech)")
        return df

    def _calculate_basic_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza podstawowe cechy orderbook."""
        # Pomocnicze funkcje
        def calc_bid_sum(prefix):
            return sum(df[f'{prefix}depth_{level}'] for level in config.BID_LEVELS)
        
        def calc_ask_sum(prefix):
            return sum(df[f'{prefix}depth_{level}'] for level in config.ASK_LEVELS)
        
        # Stosunek presji kupna do sprzedaży
        bid_sum_s1 = calc_bid_sum('snapshot1_')
        ask_sum_s1 = calc_ask_sum('snapshot1_')
        bid_sum_s2 = calc_bid_sum('snapshot2_')
        ask_sum_s2 = calc_ask_sum('snapshot2_')
        
        # Buy/sell ratios
        df['buy_sell_ratio_s1'] = np.where(ask_sum_s1 != 0, bid_sum_s1 / ask_sum_s1, 1)
        df['buy_sell_ratio_s2'] = np.where(ask_sum_s2 != 0, bid_sum_s2 / ask_sum_s2, 1)
        
        # Imbalances
        df['imbalance_s1'] = np.where((bid_sum_s1 + ask_sum_s1) != 0, (bid_sum_s1 - ask_sum_s1) / (bid_sum_s1 + ask_sum_s1), 0)
        df['imbalance_s2'] = np.where((bid_sum_s2 + ask_sum_s2) != 0, (bid_sum_s2 - ask_sum_s2) / (bid_sum_s2 + ask_sum_s2), 0)
        
        # Pressure change
        df['pressure_change'] = np.where(df['buy_sell_ratio_s1'] != 0, (df['buy_sell_ratio_s2'] - df['buy_sell_ratio_s1']) / df['buy_sell_ratio_s1'], 0)
        
        return df

    def _calculate_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy głębokości."""
        # TP/SL depths
        df['tp_1pct_depth_s1'] = df['snapshot1_depth_1']
        df['tp_2pct_depth_s1'] = df['snapshot1_depth_2']
        df['sl_1pct_depth_s1'] = df['snapshot1_depth_-1']
        
        # TP/SL ratios
        df['tp_sl_ratio_1pct'] = np.where(df['snapshot1_depth_-1'] != 0, df['snapshot1_depth_1'] / df['snapshot1_depth_-1'], 1)
        df['tp_sl_ratio_2pct'] = np.where(df['snapshot1_depth_-2'] != 0, df['snapshot1_depth_2'] / df['snapshot1_depth_-2'], 1)
        
        return df

    def _calculate_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy dynamiczne."""
        # Pomocnicze funkcje
        def calc_total_depth(prefix):
            return sum(df[f'{prefix}depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        def calc_total_notional(prefix):
            return sum(df[f'{prefix}notional_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        total_depth_s1 = calc_total_depth('snapshot1_')
        total_depth_s2 = calc_total_depth('snapshot2_')
        total_notional_s1 = calc_total_notional('snapshot1_')
        total_notional_s2 = calc_total_notional('snapshot2_')
        
        # Zmiany
        df['total_depth_change'] = np.where(total_depth_s1 != 0, (total_depth_s2 - total_depth_s1) / total_depth_s1, 0)
        df['total_notional_change'] = np.where(total_notional_s1 != 0, (total_notional_s2 - total_notional_s1) / total_notional_s1, 0)
        
        # Spread (różnica między najlepszym bid i ask)
        df['spread'] = df['snapshot1_depth_1'] - df['snapshot1_depth_-1']
        
        return df

    def calculate_hybrid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy hybrydowe (5 cech)."""
        logger.info("Obliczanie cech hybrydowych...")
        
        # 1. Korelacje (2 cechy)
        logger.info("  -> Korelacje...")
        total_depth = sum(df[f'snapshot1_depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        # Korelacja głębokości z ceną
        depth_price_corr = total_depth.rolling(window=self.history_window, min_periods=1).corr(df['close']).shift(1)
        df['depth_price_corr'] = depth_price_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Korelacja presji z wolumenem
        pressure_volume_corr = df['buy_sell_ratio_s1'].rolling(window=self.history_window, min_periods=1).corr(df['volume']).shift(1)
        df['pressure_volume_corr'] = pressure_volume_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 2. Cechy czasowe (2 cechy)
        logger.info("  -> Cechy czasowe...")
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 3. Cecha momentum (1 cecha)
        logger.info("  -> Momentum...")
        df['price_momentum'] = df['close'].pct_change(periods=5).fillna(0)
        
        logger.info("Cechy hybrydowe obliczone (5 cech)")
        return df

    def calculate_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza nowe 18 względnych cech."""
        logger.info("Obliczanie względnych cech...")
        
        # 1. CECHY TRENDU CENY (5 cech)
        logger.info("  -> Cechy trendu ceny...")
        
        # Cechy 1-3: Trendy cenowe
        df['price_trend_30m'] = df['close'].pct_change(periods=config.PRICE_TREND_PERIODS[0]).fillna(0).replace([np.inf, -np.inf], 0)
        df['price_trend_2h'] = df['close'].pct_change(periods=config.PRICE_TREND_PERIODS[1]).fillna(0).replace([np.inf, -np.inf], 0)
        df['price_trend_6h'] = df['close'].pct_change(periods=config.PRICE_TREND_PERIODS[2]).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 4: Siła trendu
        price_strength = np.where(
            np.abs(df['price_trend_30m']) > 0.001,
            np.abs(df['price_trend_2h']) / (np.abs(df['price_trend_30m']) + 0.001),
            0
        )
        df['price_strength'] = pd.Series(price_strength, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 5: Spójność kierunku (numeryczna)
        def sign(x):
            return np.where(x > 0, 1, np.where(x < 0, -1, 0))
        
        df['price_consistency_score'] = (
            sign(df['price_trend_30m']) + 
            sign(df['price_trend_2h']) + 
            sign(df['price_trend_6h'])
        ) / 3
        
        # 2. CECHY POZYCJI CENY (4 cechy)
        logger.info("  -> Cechy pozycji ceny...")
        
        # Cechy 6-7: Relacje do średnich
        price_vs_ma_60 = np.where(df['ma_60'] != 0, (df['close'] - df['ma_60']) / df['ma_60'], 0)
        df['price_vs_ma_60'] = pd.Series(price_vs_ma_60, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        price_vs_ma_240 = np.where(df['ma_240'] != 0, (df['close'] - df['ma_240']) / df['ma_240'], 0)
        df['price_vs_ma_240'] = pd.Series(price_vs_ma_240, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 8: Trend średnich
        ma_trend = np.where(df['ma_240'] != 0, (df['ma_60'] - df['ma_240']) / df['ma_240'], 0)
        df['ma_trend'] = pd.Series(ma_trend, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 9: Zmienność cenowa (rolling std)
        df['price_volatility_rolling'] = df['close'].pct_change().rolling(window=config.ROLLING_WINDOWS[0], min_periods=1).std().fillna(0).replace([np.inf, -np.inf], 0)
        
        # 3. CECHY WOLUMENU (5 cech)
        logger.info("  -> Cechy wolumenu...")
        
        # Cecha 10: Trend wolumenu
        df['volume_trend_1h'] = df['volume'].pct_change(periods=config.VOLUME_TREND_PERIODS[0]).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 11: Intensywność wolumenu
        volume_ma_60 = df['volume'].rolling(window=config.ROLLING_WINDOWS[1], min_periods=1).mean()
        volume_intensity = np.where(volume_ma_60 != 0, df['volume'] / volume_ma_60, 1)
        df['volume_intensity'] = pd.Series(volume_intensity, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Cecha 12: Zmienność wolumenu (rolling std)
        df['volume_volatility_rolling'] = df['volume'].pct_change().rolling(window=config.ROLLING_WINDOWS[0], min_periods=1).std().fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 13: Korelacja wolumenu z ceną
        volume_price_corr = df['volume'].rolling(window=config.ROLLING_WINDOWS[1], min_periods=1).corr(df['close']).shift(1)
        df['volume_price_correlation'] = volume_price_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Cecha 14: Momentum wolumenu
        volume_momentum = df['volume_trend_1h'] - df['volume_trend_1h'].shift(config.MOMENTUM_PERIODS[0]).fillna(0)
        df['volume_momentum'] = volume_momentum.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 4. CECHY ORDERBOOK (4 cechy)
        logger.info("  -> Cechy orderbook...")
        
        # Cecha 15: Tightness spreadu
        spread_ma_60 = df['spread'].rolling(window=config.ROLLING_WINDOWS[1], min_periods=1).mean()
        spread_tightness = np.where(spread_ma_60 != 0, df['spread'] / spread_ma_60, 1)
        df['spread_tightness'] = pd.Series(spread_tightness, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Cecha 16: Asymetria głębokości snapshot1
        bid_depth_s1 = sum(df[f'snapshot1_depth_{level}'] for level in config.BID_LEVELS)
        ask_depth_s1 = sum(df[f'snapshot1_depth_{level}'] for level in config.ASK_LEVELS)
        depth_ratio_s1 = np.where(ask_depth_s1 != 0, bid_depth_s1 / ask_depth_s1, 1)
        df['depth_ratio_s1'] = pd.Series(depth_ratio_s1, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Cecha 17: Asymetria głębokości snapshot2
        bid_depth_s2 = sum(df[f'snapshot2_depth_{level}'] for level in config.BID_LEVELS)
        ask_depth_s2 = sum(df[f'snapshot2_depth_{level}'] for level in config.ASK_LEVELS)
        depth_ratio_s2 = np.where(ask_depth_s2 != 0, bid_depth_s2 / ask_depth_s2, 1)
        df['depth_ratio_s2'] = pd.Series(depth_ratio_s2, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Cecha 18: Momentum asymetrii głębokości
        depth_ratio_s1_1h_ago = df['depth_ratio_s1'].shift(config.MOMENTUM_PERIODS[0]).fillna(1)
        depth_momentum = np.where(
            depth_ratio_s1_1h_ago != 0,
            (df['depth_ratio_s1'] - depth_ratio_s1_1h_ago) / depth_ratio_s1_1h_ago,
            0
        )
        df['depth_momentum'] = pd.Series(depth_momentum, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        logger.info("Względne cechy obliczone (18 cech)")
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
        
        return df_adx['dx'].ewm(alpha=alpha, adjust=False).mean()

    def trim_warmup_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obcina okres rozgrzewania z danych."""
        if len(df) <= self.warmup_period:
            logger.warning(f"Dane są krótsze niż okres rozgrzewania ({self.warmup_period} minut)")
            return df
        
        original_length = len(df)
        df_trimmed = df.iloc[self.warmup_period:].copy()
        
        logger.info(f"Obcięto okres rozgrzewania: {original_length:,} -> {len(df_trimmed):,} wierszy")
        logger.info(f"Utrata: {self.warmup_period:,} wierszy ({self.warmup_period/1440:.1f} dni)")
        logger.info(f"Nowy zakres: {df_trimmed.index.min()} do {df_trimmed.index.max()}")
        
        return df_trimmed

    def calculate_features(self, df: pd.DataFrame, user_start_dt=None, user_end_dt=None) -> pd.DataFrame:
        """Główna funkcja obliczająca wszystkie cechy."""
        start_time = time.time()
        logger.info("Rozpoczynanie obliczania wszystkich cech...")
        
        # Filtrowanie zakresu czasowego jeśli podano
        if user_start_dt or user_end_dt:
            if user_start_dt:
                df = df[df.index >= user_start_dt]
            if user_end_dt:
                df = df[df.index <= user_end_dt]
            logger.info(f"Przefiltrowano dane: {len(df):,} wierszy")
        
        # Obliczanie cech OHLC
        df = self.calculate_ohlc_features(df)
        
        # Obliczanie cech orderbook
        df = self.calculate_orderbook_features(df)
        
        # Obliczanie cech hybrydowych
        df = self.calculate_hybrid_features(df)
        
        # Obliczanie cech względnych
        df = self.calculate_relative_features(df)
        
        # Obcinanie okresu rozgrzewania
        df = self.trim_warmup_period(df)
        
        # Usuwanie kolumn pomocniczych
        columns_to_drop = ['upper_wick_ratio_5m', 'lower_wick_ratio_5m', 'hour_of_day', 'day_of_week']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Lista nowych względnych cech do zachowania
        relative_features = config.RELATIVE_FEATURES
        
        logger.info(f"Nowe względne cechy: {len(relative_features)} cech")
        for feature in relative_features:
            if feature in df.columns:
                logger.info(f"  ✅ {feature}")
            else:
                logger.warning(f"  ❌ {feature} - brakuje!")
        
        # Podsumowanie
        feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        logger.info(f"Obliczono {len(feature_columns)} cech (32 stare + 18 nowe = 50 total)")
        logger.info(f"Finalny rozmiar: {len(df):,} wierszy, {len(df.columns)} kolumn")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Czas obliczeń: {elapsed_time:.2f} sekund")
        
        return df

    def save_data(self, df: pd.DataFrame, file_path: str):
        """Zapisuje dane do pliku feather."""
        logger.info(f"Zapisuję dane do: {file_path}")
        
        # Utwórz katalog jeśli nie istnieje
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Resetuj indeks przed zapisem
        df_to_save = df.reset_index()
        
        # Zapisz do feather
        df_to_save.to_feather(file_path)
        
        # Sprawdź rozmiar pliku
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        logger.info(f"Zapisano: {file_path} ({file_size:.2f} MB)")

def main():
    """Główna funkcja."""
    parser = argparse.ArgumentParser(description='Oblicza cechy OHLC + Orderbook')
    parser.add_argument('--input', default=str(config.INPUT_FILE_PATH), 
                       help='Ścieżka do pliku wejściowego')
    parser.add_argument('--output', default=str(config.OUTPUT_DIR / config.DEFAULT_OUTPUT_FILENAME),
                       help='Ścieżka do pliku wyjściowego')
    parser.add_argument('--start-date', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Data końcowa (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    logger.info("ROZPOCZYNAM OBLICZANIE CECH OHLC + ORDERBOOK")
    logger.info("=" * 60)
    
    # Inicjalizacja kalkulatora
    calculator = OHLCOrderBookFeatureCalculator()
    
    # Wczytanie danych
    df = calculator.load_data(args.input)
    if df is None:
        return
    
    # Konwersja dat jeśli podano
    start_dt = pd.to_datetime(args.start_date) if args.start_date else None
    end_dt = pd.to_datetime(args.end_date) if args.end_date else None
    
    # Obliczanie cech
    df_features = calculator.calculate_features(df, start_dt, end_dt)
    
    # Zapisanie wyników
    calculator.save_data(df_features, args.output)
    
    logger.info("OBLICZANIE CECH ZAKONCZONE POMYSLNIE!")
    logger.info(f"Plik wynikowy: {args.output}")
    logger.info(f"Wierszy: {len(df_features):,}")
    logger.info(f"Kolumn: {len(df_features.columns)}")

if __name__ == "__main__":
    main() 