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
from pathlib import Path

import pandas as pd
import numpy as np
import bamboo_ta as bta
from scipy import stats

# Dodajemy ścieżkę do głównego katalogu, aby importy działały poprawnie
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Dodaj ścieżkę do modułu konfiguracyjnego
sys.path.append('../download2/orderbook')
try:
    import config as orderbook_config
except ImportError:
    # Fallback - zdefiniuj pary lokalnie
    orderbook_config = None

# Import konfiguracji
try:
    import feature_calculator_ohlc_snapshot.config as config
except ImportError:
    try:
        import config
    except ImportError:
        print("Błąd: Nie można zaimportować konfiguracji!")
        sys.exit(1)

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
        volume_shifted = df['volume'].shift(1)
        
        # Bezpieczne obliczanie ratio
        volume_ratio = np.where(
            (volume_shifted > 0) & (df['volume'] > 0),
            df['volume'] / volume_shifted,
            np.where(
                (volume_shifted == 0) & (df['volume'] > 0),
                10.0,  # Jeśli poprzedni volume był 0, a obecny > 0, ustaw ratio na 10
                np.where(
                    (volume_shifted > 0) & (df['volume'] == 0),
                    0.1,  # Jeśli obecny volume jest 0, a poprzedni > 0, ustaw ratio na 0.1
                    1.0   # W innych przypadkach (0/0) ustaw ratio na 1
                )
            )
        )
        
        # Bezpieczne logowanie
        volume_change = np.log(volume_ratio)
        df['volume_change_norm'] = pd.Series(volume_change, index=df.index).replace([np.inf, -np.inf], 0).fillna(0)
        
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

    def calculate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza 5 cech market regime."""
        logger.info("Obliczanie cech market regime...")
        
        # 1. Market Trend Strength (0-100)
        logger.info("  -> Market trend strength...")
        df['market_trend_strength'] = self._calculate_market_trend_strength(df)
        
        # 2. Market Trend Direction (-1 do 1)
        logger.info("  -> Market trend direction...")
        df['market_trend_direction'] = self._calculate_market_trend_direction(df)
        
        # 3. Market Choppiness (0-100)
        logger.info("  -> Market choppiness...")
        df['market_choppiness'] = self._calculate_market_choppiness(df)
        
        # 4. Bollinger Band Width (0-1)
        logger.info("  -> Bollinger band width...")
        df['bollinger_band_width'] = self._calculate_bollinger_band_width(df)
        
        # 5. Market Regime Classification (0/1/2)
        logger.info("  -> Market regime classification...")
        df['market_regime'] = self._calculate_market_regime(df)
        
        logger.info("Cechy market regime obliczone (5 cech)")
        return df

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza 6 cech volatility clustering."""
        logger.info("Obliczanie cech volatility clustering...")
        
        # 1. Volatility Regime (0/1/2)
        logger.info("  -> Volatility regime...")
        df['volatility_regime'] = self._calculate_volatility_regime(df)
        
        # 2. Volatility Percentile (0-100)
        logger.info("  -> Volatility percentile...")
        df['volatility_percentile'] = self._calculate_volatility_percentile(df)
        
        # 3. Volatility Persistence (0-1)
        logger.info("  -> Volatility persistence...")
        df['volatility_persistence'] = self._calculate_volatility_persistence(df)
        
        # 4. Volatility Momentum (-1 do 1)
        logger.info("  -> Volatility momentum...")
        df['volatility_momentum'] = self._calculate_volatility_momentum(df)
        
        # 5. Volatility of Volatility (0-1)
        logger.info("  -> Volatility of volatility...")
        df['volatility_of_volatility'] = self._calculate_volatility_of_volatility(df)
        
        # 6. Volatility Term Structure (-1 do 1)
        logger.info("  -> Volatility term structure...")
        df['volatility_term_structure'] = self._calculate_volatility_term_structure(df)
        
        logger.info("Cechy volatility clustering obliczone (6 cech)")
        return df

    def calculate_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza 8 cech order book imbalance."""
        logger.info("Obliczanie cech order book imbalance...")
        
        # Sprawdź czy mamy dane orderbook
        orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
        if len(orderbook_columns) == 0:
            logger.warning("Brak danych orderbook - pomijam cechy imbalance")
            return df
        
        # 1. Volume Imbalance (-1 do 1)
        logger.info("  -> Volume imbalance...")
        df['volume_imbalance'] = self._calculate_volume_imbalance(df)
        
        # 2. Weighted Volume Imbalance (-1 do 1)
        logger.info("  -> Weighted volume imbalance...")
        df['weighted_volume_imbalance'] = self._calculate_weighted_volume_imbalance(df)
        
        # 3. Volume Imbalance Trend (-1 do 1)
        logger.info("  -> Volume imbalance trend...")
        df['volume_imbalance_trend'] = self._calculate_volume_imbalance_trend(df)
        
        # 4. Price Pressure (-1 do 1)
        logger.info("  -> Price pressure...")
        df['price_pressure'] = self._calculate_price_pressure(df)
        
        # 5. Weighted Price Pressure (-1 do 1)
        logger.info("  -> Weighted price pressure...")
        df['weighted_price_pressure'] = self._calculate_weighted_price_pressure(df)
        
        # 6. Price Pressure Momentum (-1 do 1)
        logger.info("  -> Price pressure momentum...")
        df['price_pressure_momentum'] = self._calculate_price_pressure_momentum(df)
        
        # 7. Order Flow Imbalance (-1 do 1)
        logger.info("  -> Order flow imbalance...")
        df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        
        # 8. Order Flow Trend (-1 do 1)
        logger.info("  -> Order flow trend...")
        df['order_flow_trend'] = self._calculate_order_flow_trend(df)
        
        logger.info("Cechy order book imbalance obliczone (8 cech)")
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

    # --- METODY POMOCNICZE DLA MARKET REGIME ---
    
    def _calculate_market_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza siłę trendu (0-100) na podstawie ADX."""
        adx = self._calculate_manual_adx(df, period=config.ADX_PERIOD)
        # Normalizacja ADX do zakresu 0-100
        trend_strength = np.clip(adx, 0, 100)
        return trend_strength.fillna(0)

    def _calculate_market_trend_direction(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza kierunek trendu (-1 do 1)."""
        # Użyj średnich kroczących do określenia kierunku
        ma_short = df['close'].rolling(window=config.MARKET_REGIME_PERIODS[0]).mean()
        ma_long = df['close'].rolling(window=config.MARKET_REGIME_PERIODS[1]).mean()
        
        # Kierunek trendu
        trend_direction = np.where(
            ma_long != 0,
            (ma_short - ma_long) / ma_long,
            0
        )
        
        # Normalizacja do zakresu [-1, 1]
        trend_direction = np.clip(trend_direction, -1, 1)
        return pd.Series(trend_direction, index=df.index).fillna(0)

    def _calculate_market_choppiness(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza wskaźnik chaotyczności (0-100)."""
        period = config.CHOPPINESS_PERIOD
        
        # True Range
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Suma True Range
        tr_sum = tr.rolling(window=period).sum()
        
        # Długość ścieżki (suma zmian)
        path_length = (df['high'] - df['low']).rolling(window=period).sum()
        
        # Choppiness Index
        choppiness = np.where(
            tr_sum > 0,
            100 * np.log10(path_length / tr_sum) / np.log10(period),
            0
        )
        
        # Clipping do zakresu 0-100
        choppiness = np.clip(choppiness, 0, 100)
        return pd.Series(choppiness, index=df.index).fillna(0)

    def _calculate_bollinger_band_width(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza szerokość pasm Bollingera (0-1)."""
        period = config.BOLLINGER_WIDTH_PERIOD
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        
        # Szerokość pasm
        bb_width = np.where(
            bb_middle != 0,
            (bb_upper - bb_lower) / bb_middle,
            0
        )
        
        # Normalizacja do zakresu 0-1
        bb_width = np.clip(bb_width, 0, 1)
        return pd.Series(bb_width, index=df.index).fillna(0)

    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Klasyfikuje reżim rynkowy (0=sideways, 1=trend, 2=volatile)."""
        # Użyj ADX do określenia siły trendu
        adx = self._calculate_manual_adx(df, period=config.ADX_PERIOD)
        
        # Użyj choppiness do określenia chaotyczności
        choppiness = self._calculate_market_choppiness(df)
        
        # Klasyfikacja
        regime = np.where(
            adx > 25,  # Silny trend
            1,  # Trend
            np.where(
                choppiness > 60,  # Wysoka chaotyczność
                2,  # Volatile
                0   # Sideways
            )
        )
        
        return pd.Series(regime, index=df.index).fillna(0)

    # --- METODY POMOCNICZE DLA VOLATILITY CLUSTERING ---
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Klasyfikuje reżim zmienności (0=low, 1=normal, 2=high)."""
        # Oblicz volatility na różnych okresach
        volatility_20 = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[0]).std()
        volatility_60 = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
        
        # Percentyl volatility w długim oknie
        vol_percentile = volatility_60.rolling(window=config.VOLATILITY_PERCENTILE_WINDOW).rank(pct=True) * 100
        
        # Klasyfikacja
        regime = np.where(
            vol_percentile > 80,  # Wysoka zmienność
            2,  # High volatility
            np.where(
                vol_percentile < 20,  # Niska zmienność
                0,  # Low volatility
                1   # Normal volatility
            )
        )
        
        return pd.Series(regime, index=df.index).fillna(1)

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza percentyl zmienności (0-100)."""
        # Volatility na długim okresie
        volatility = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[2]).std()
        
        # Percentyl w rolling window
        percentile = volatility.rolling(window=config.VOLATILITY_PERCENTILE_WINDOW).rank(pct=True) * 100
        
        # Clipping do zakresu 0-100
        percentile = np.clip(percentile, 0, 100)
        return pd.Series(percentile, index=df.index).fillna(50)

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza trwałość zmienności (0-1)."""
        # Volatility
        volatility = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
        
        # Uproszczona autokorelacja - użyj korelacji z poprzednim okresem
        volatility_lag = volatility.shift(1)
        
        # Korelacja między obecną a poprzednią volatility w rolling window
        persistence = volatility.rolling(window=config.VOLATILITY_WINDOWS[1], min_periods=2).corr(volatility_lag)
        
        # Clipping do zakresu 0-1
        persistence = np.clip(persistence, 0, 1)
        return pd.Series(persistence, index=df.index).fillna(0)

    def _calculate_volatility_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza momentum zmienności (-1 do 1)."""
        # Volatility na różnych okresach
        vol_short = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[0]).std()
        vol_long = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[2]).std()
        
        # Momentum (różnica między krótko- i długoterminową zmiennością)
        momentum = np.where(
            vol_long > config.VOLATILITY_MIN_THRESHOLD,
            (vol_short - vol_long) / vol_long,
            0
        )
        
        # Normalizacja do zakresu [-1, 1]
        momentum = np.clip(momentum, -1, 1)
        return pd.Series(momentum, index=df.index).fillna(0)

    def _calculate_volatility_of_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza zmienność zmienności (0-1)."""
        # Volatility
        volatility = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
        
        # Volatility of volatility
        vol_of_vol = np.where(
            volatility > config.VOLATILITY_MIN_THRESHOLD,
            volatility.rolling(window=config.VOLATILITY_WINDOWS[0]).std() / volatility,
            0
        )
        
        # Clipping do zakresu 0-1
        vol_of_vol = np.clip(vol_of_vol, 0, 1)
        return pd.Series(vol_of_vol, index=df.index).fillna(0)

    def _calculate_volatility_term_structure(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza strukturę terminową zmienności (-1 do 1)."""
        # Volatility na różnych okresach
        vol_short = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[0]).std()
        vol_medium = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
        vol_long = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[2]).std()
        
        # Term structure slope
        term_structure = np.where(
            vol_medium > config.VOLATILITY_MIN_THRESHOLD,
            (vol_short - vol_long) / vol_medium,
            0
        )
        
        # Normalizacja do zakresu [-1, 1]
        term_structure = np.clip(term_structure, -1, 1)
        return pd.Series(term_structure, index=df.index).fillna(0)

    # --- METODY POMOCNICZE DLA ORDER BOOK IMBALANCE ---
    
    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza volume imbalance (-1 do 1)."""
        # Sprawdź czy mamy dane orderbook
        if 'snapshot1_bid_volume' not in df.columns or 'snapshot1_ask_volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        bid_volume = df['snapshot1_bid_volume']
        ask_volume = df['snapshot1_ask_volume']
        
        # Volume imbalance
        total_volume = bid_volume + ask_volume
        imbalance = np.where(
            total_volume > 0,
            (bid_volume - ask_volume) / total_volume,
            0
        )
        
        # Clipping do zakresu [-1, 1]
        imbalance = np.clip(imbalance, -1, 1)
        return pd.Series(imbalance, index=df.index).fillna(0)

    def _calculate_weighted_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza weighted volume imbalance (-1 do 1)."""
        # Sprawdź czy mamy dane orderbook
        if 'snapshot1_bid_volume' not in df.columns or 'snapshot1_ask_volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        bid_volume = df['snapshot1_bid_volume']
        ask_volume = df['snapshot1_ask_volume']
        
        # Weighted imbalance (większa waga dla bliższych poziomów)
        weighted_imbalance = np.where(
            (bid_volume + ask_volume) > 0,
            (bid_volume - ask_volume) / (bid_volume + ask_volume),
            0
        )
        
        # Clipping do zakresu [-1, 1]
        weighted_imbalance = np.clip(weighted_imbalance, -1, 1)
        return pd.Series(weighted_imbalance, index=df.index).fillna(0)

    def _calculate_volume_imbalance_trend(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza trend volume imbalance (-1 do 1)."""
        # Oblicz volume imbalance
        imbalance = self._calculate_volume_imbalance(df)
        
        # Trend (różnica między obecnym a średnim)
        trend = imbalance - imbalance.rolling(window=config.PRESSURE_WINDOW).mean()
        
        # Normalizacja do zakresu [-1, 1]
        trend = np.clip(trend, -1, 1)
        return pd.Series(trend, index=df.index).fillna(0)

    def _calculate_price_pressure(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza price pressure (-1 do 1)."""
        # Sprawdź czy mamy spread
        if 'snapshot1_spread' not in df.columns:
            return pd.Series(0, index=df.index)
        
        spread = df['snapshot1_spread']
        volume_imbalance = self._calculate_volume_imbalance(df)
        
        # Price pressure (imbalance * inverse spread)
        pressure = np.where(
            spread > config.MIN_SPREAD_THRESHOLD,
            volume_imbalance / spread,
            0
        )
        
        # Normalizacja do zakresu [-1, 1]
        pressure = np.clip(pressure, -1, 1)
        return pd.Series(pressure, index=df.index).fillna(0)

    def _calculate_weighted_price_pressure(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza weighted price pressure (-1 do 1)."""
        # Sprawdź czy mamy spread
        if 'snapshot1_spread' not in df.columns:
            return pd.Series(0, index=df.index)
        
        spread = df['snapshot1_spread']
        weighted_imbalance = self._calculate_weighted_volume_imbalance(df)
        
        # Weighted price pressure
        pressure = np.where(
            spread > config.MIN_SPREAD_THRESHOLD,
            weighted_imbalance / spread,
            0
        )
        
        # Normalizacja do zakresu [-1, 1]
        pressure = np.clip(pressure, -1, 1)
        return pd.Series(pressure, index=df.index).fillna(0)

    def _calculate_price_pressure_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza momentum price pressure (-1 do 1)."""
        # Oblicz price pressure
        pressure = self._calculate_price_pressure(df)
        
        # Momentum (różnica między obecnym a poprzednim)
        momentum = pressure.diff()
        
        # Normalizacja do zakresu [-1, 1]
        momentum = np.clip(momentum, -1, 1)
        return pd.Series(momentum, index=df.index).fillna(0)

    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza order flow imbalance (-1 do 1)."""
        # Sprawdź czy mamy dane orderbook
        if 'snapshot1_bid_volume' not in df.columns or 'snapshot1_ask_volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        bid_volume = df['snapshot1_bid_volume']
        ask_volume = df['snapshot1_ask_volume']
        
        # Order flow imbalance (zmiana volume)
        bid_change = bid_volume.diff()
        ask_change = ask_volume.diff()
        
        total_change = bid_change + ask_change
        flow_imbalance = np.where(
            total_change != 0,
            (bid_change - ask_change) / total_change,
            0
        )
        
        # Clipping do zakresu [-1, 1]
        flow_imbalance = np.clip(flow_imbalance, -1, 1)
        return pd.Series(flow_imbalance, index=df.index).fillna(0)

    def _calculate_order_flow_trend(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza trend order flow (-1 do 1)."""
        # Oblicz order flow imbalance
        flow_imbalance = self._calculate_order_flow_imbalance(df)
        
        # Trend (rolling mean)
        trend = flow_imbalance.rolling(window=config.PRESSURE_WINDOW).mean()
        
        # Normalizacja do zakresu [-1, 1]
        trend = np.clip(trend, -1, 1)
        return pd.Series(trend, index=df.index).fillna(0)

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
        
        # Obliczanie nowych zaawansowanych cech
        df = self.calculate_market_regime_features(df)
        df = self.calculate_volatility_features(df)
        df = self.calculate_imbalance_features(df)
        
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
        logger.info(f"Obliczono {len(feature_columns)} cech (32 stare + 18 względne + 19 zaawansowane = 69 total)")
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

def process_all_pairs(input_dir: str, output_dir: str, start_date: str = None, end_date: str = None):
    """Przetwarza wszystkie pary z konfiguracji orderbook."""
    
    # Pobierz listę par z konfiguracji
    if orderbook_config:
        pairs = orderbook_config.PAIRS
        logger.info(f"Pobrano {len(pairs)} par z konfiguracji orderbook")
    else:
        # Fallback - standardowe pary
        pairs = [
            "ETHUSDT", "BCHUSDT", "XRPUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT", 
            "LINKUSDT", "XLMUSDT", "ADAUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT", 
            "XTZUSDT", "ATOMUSDT", "BNBUSDT", "ONTUSDT", "IOTAUSDT", "BATUSDT", 
            "VETUSDT", "NEOUSDT"
        ]
        logger.info(f"Używam standardowej listy {len(pairs)} par")
    
    # Inicjalizacja kalkulatora
    calculator = OHLCOrderBookFeatureCalculator()
    
    # Konwersja dat jeśli podano
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    
    # Statystyki
    total_start_time = time.time()
    successful_pairs = 0
    failed_pairs = []
    
    logger.info("=" * 80)
    logger.info(f"ROZPOCZYNAM PRZETWARZANIE WSZYSTKICH {len(pairs)} PAR")
    logger.info("=" * 80)
    
    for i, pair in enumerate(pairs, 1):
        logger.info(f"\n[{i}/{len(pairs)}] Przetwarzanie pary: {pair}")
        logger.info("-" * 50)
        
        # Ścieżki plików
        input_file = os.path.join(input_dir, f"merged_{pair}.feather")
        output_file = os.path.join(output_dir, f"features_{pair}.feather")
        
        # Sprawdź czy plik wejściowy istnieje
        if not os.path.exists(input_file):
            logger.error(f"❌ Plik wejściowy nie istnieje: {input_file}")
            failed_pairs.append((pair, "Plik wejściowy nie istnieje"))
            continue
        
        try:
            # Wczytanie danych
            df = calculator.load_data(input_file)
            if df is None:
                logger.error(f"❌ Nie można wczytać danych dla {pair}")
                failed_pairs.append((pair, "Błąd wczytywania danych"))
                continue
            
            # Obliczanie cech
            start_time = time.time()
            df_features = calculator.calculate_features(df, start_dt, end_dt)
            elapsed_time = time.time() - start_time
            
            # Zapisanie wyników
            calculator.save_data(df_features, output_file)
            
            logger.info(f"✅ {pair}: {len(df_features):,} wierszy, {len(df_features.columns)} kolumn, {elapsed_time:.2f}s")
            successful_pairs += 1
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania {pair}: {str(e)}")
            failed_pairs.append((pair, str(e)))
            continue
    
    # Podsumowanie
    total_time = time.time() - total_start_time
    logger.info("\n" + "=" * 80)
    logger.info("PODSUMOWANIE PRZETWARZANIA")
    logger.info("=" * 80)
    logger.info(f"✅ Pomyślnie przetworzono: {successful_pairs}/{len(pairs)} par")
    logger.info(f"❌ Nieudane: {len(failed_pairs)} par")
    logger.info(f"⏱️  Całkowity czas: {total_time:.2f} sekund ({total_time/60:.2f} minut)")
    logger.info(f"📊 Średni czas na parę: {total_time/len(pairs):.2f} sekund")
    
    if failed_pairs:
        logger.info("\n❌ Lista nieudanych par:")
        for pair, error in failed_pairs:
            logger.info(f"  - {pair}: {error}")
    
    logger.info(f"\n📁 Pliki wynikowe w: {output_dir}")
    logger.info("=" * 80)

def main():
    """Główna funkcja."""
    parser = argparse.ArgumentParser(description='Oblicza cechy OHLC + Orderbook')
    parser.add_argument('--input', default=str(config.INPUT_FILE_PATH), 
                       help='Ścieżka do pliku wejściowego')
    parser.add_argument('--output', default=str(config.OUTPUT_DIR / config.DEFAULT_OUTPUT_FILENAME),
                       help='Ścieżka do pliku wyjściowego')
    parser.add_argument('--start-date', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Data końcowa (YYYY-MM-DD)')
    parser.add_argument('--all-pairs', action='store_true',
                       help='Przetwarzaj wszystkie pary z konfiguracji orderbook')
    parser.add_argument('--input-dir', default='../download2/merge/merged_data',
                       help='Katalog z plikami merged (używany z --all-pairs)')
    parser.add_argument('--output-dir', default='output',
                       help='Katalog wyjściowy (używany z --all-pairs)')
    
    args = parser.parse_args()
    
    # Jeśli używamy --all-pairs, przetwarzaj wszystkie pary
    if args.all_pairs:
        # Utwórz katalog wyjściowy jeśli nie istnieje
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Przetwarzaj wszystkie pary
        process_all_pairs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date
        )
        return
    
    # Standardowe przetwarzanie pojedynczej pary
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