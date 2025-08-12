#!/usr/bin/env python3
"""
Moduł obliczający cechy dla danych OHLC + Order Book.
Obsługuje dane w formacie Feather z modułu download2.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import warnings
import time
from tqdm import tqdm

# Import bibliotek technicznych
import bamboo_ta as bta
from scipy import stats
from scipy.signal import savgol_filter

# Import konfiguracji
import config

# Ignoruj ostrzeżenia
warnings.filterwarnings('ignore')

def setup_logging():
    """Konfiguruje system logowania."""
    # Utwórz katalogi jeśli nie istnieją
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Konfiguracja logowania
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_DIR / config.LOG_FILENAME, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class OHLCOrderBookFeatureCalculator:
    """Kalkulator cech dla danych OHLC + Order Book."""
    
    def __init__(self):
        """Inicjalizacja kalkulatora cech."""
        self.logger = logging.getLogger(__name__)
        
        # Okresy dla średnich kroczących (w minutach)
        self.ma_periods = config.MA_WINDOWS
        self.history_window = config.ORDERBOOK_HISTORY_WINDOW
        self.short_window = config.ORDERBOOK_SHORT_WINDOW
        self.momentum_window = config.ORDERBOOK_MOMENTUM_WINDOW
        self.warmup_period = config.WARMUP_PERIOD_MINUTES
        
        # Lista cech do treningu - tylko względne i zaawansowane
        self.feature_groups = {
            'training_features_basic': 37,     # Cechy z training3 (zalecane)
            'additional_relative_features': 34, # Dodatkowe cechy względne (do eksperymentów)
            'total_training_features': 71      # Łącznie cech do treningu
        }
        
        self.logger.info("Kalkulator cech zainicjalizowany")
        self.logger.info(f"Oczekiwane cechy: {sum(self.feature_groups.values())}")
        self.logger.info(f"Order book history window: {self.history_window}")

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Wczytuje dane z pliku Feather."""
        try:
            self.logger.info(f"Wczytywanie danych z: {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"Plik nie istnieje: {file_path}")
                return None
            
            # Wczytaj dane
            df = pd.read_feather(file_path)
            
            # Ustaw timestamp jako indeks
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            # Podstawowe informacje
            self.logger.info(f"Wczytano {len(df):,} wierszy, {len(df.columns)} kolumn")
            self.logger.info(f"Zakres czasowy: {df.index.min()} - {df.index.max()}")
            
            # Sprawdź wymagane kolumny
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Brak wymaganych kolumn: {missing_columns}")
                return None
            
            # Sprawdź kolumny orderbook
            orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
            self.logger.info(f"Znaleziono {len(orderbook_columns)} kolumn orderbook")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wczytywania danych: {e}")
            return None

    def calculate_training_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza tylko względne cechy OHLC do treningu (12 cech)."""
        self.logger.info("Obliczanie względnych cech OHLC do treningu...")
        
        # 1. Wstęgi Bollingera (2 cechy względne)
        self.logger.info("  -> Wstęgi Bollingera (względne)...")
        bbands = bta.bollinger_bands(df, 'close', period=config.BOLLINGER_PERIOD, std_dev=config.BOLLINGER_STD_DEV)
        df['bb_width'] = np.where(bbands['bb_middle'] != 0, (bbands['bb_upper'] - bbands['bb_lower']) / bbands['bb_middle'], 0)
        df['bb_position'] = np.where((bbands['bb_upper'] - bbands['bb_lower']) != 0, (df['close'] - bbands['bb_lower']) / (bbands['bb_upper'] - bbands['bb_lower']), 0)
        df['bb_position'] = (df['bb_position'] - 0.5) * 2

        # 2. RSI (1 cecha względna)
        self.logger.info("  -> RSI...")
        df['rsi_14'] = bta.relative_strength_index(df, column='close', period=config.RSI_PERIOD)['rsi']
        
        # 3. MACD (1 cecha względna)
        self.logger.info("  -> MACD...")
        macd = bta.macd(df, 'close', short_window=config.MACD_SHORT, long_window=config.MACD_LONG, signal_window=config.MACD_SIGNAL)
        df['macd_hist'] = macd['macd_histogram']

        # 4. ADX (1 cecha względna)
        self.logger.info("  -> ADX...")
        df['adx_14'] = self._calculate_manual_adx(df, period=config.ADX_PERIOD)

        # 5. Średnie kroczące (3 cechy bezwzględne - potrzebne do obliczenia względnych)
        self.logger.info("  -> Średnie kroczące (do obliczenia względnych)...")
        for period in self.ma_periods[:-1]:  # Bez 30-dniowej na razie
            df[f'ma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean().shift(1)
        
        # 6. Cechy cenowe względne (4 cechy)
        self.logger.info("  -> Cechy cenowe względne...")
        df['price_to_ma_60'] = np.where(df['ma_60'] != 0, df['close'] / df['ma_60'], 1)
        df['price_to_ma_240'] = np.where(df['ma_240'] != 0, df['close'] / df['ma_240'], 1)
        df['ma_60_to_ma_240'] = np.where(df['ma_240'] != 0, df['ma_60'] / df['ma_240'], 1)
        df['price_to_ma_1440'] = np.where(df['ma_1440'] != 0, df['close'] / df['ma_1440'], 1)
        
        # 7. Cechy wolumenu względne (1 cecha)
        self.logger.info("  -> Cechy wolumenu względne...")
        # Bezpieczne obliczanie zmiany wolumenu - użyj logarytmu żeby uniknąć infinity
        volume_ratio = df['volume'] / (df['volume'].shift(1) + 1e-8)
        df['volume_change_norm'] = np.log(volume_ratio).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 8. Cechy świec względne (2 cechy)
        self.logger.info("  -> Cechy świec względne...")
        candle_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        upper_wick_ratio = np.where(candle_range > 0, upper_wick / candle_range, 0)
        lower_wick_ratio = np.where(candle_range > 0, lower_wick / candle_range, 0)
        
        df['upper_wick_ratio_5m'] = pd.Series(upper_wick_ratio, index=df.index).rolling(window=5).mean()
        df['lower_wick_ratio_5m'] = pd.Series(lower_wick_ratio, index=df.index).rolling(window=5).mean()
        
        self.logger.info("Względne cechy OHLC obliczone (12 cech)")
        return df

    def calculate_training_bamboo_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza tylko względne cechy z biblioteki bamboo_ta do treningu (6 cech)."""
        self.logger.info("Obliczanie względnych cech bamboo_ta do treningu...")
        
        # 1. Stochastic (2 cechy względne)
        self.logger.info("  -> Stochastic (względne)...")
        stoch = bta.stochastics_oscillator(df, 'high', 'low', 'close', window=14, smooth_window=3)
        df['stoch_k'] = stoch['stoch']
        df['stoch_d'] = stoch['stoch_signal']
        
        # 2. CCI (1 cecha względna)
        self.logger.info("  -> CCI...")
        df['cci'] = bta.commodity_channel_index(df, length=20)['cci']
        
        # 3. Williams %R (1 cecha względna)
        self.logger.info("  -> Williams %R...")
        df['williams_r'] = bta.williams_r(df, 'high', 'low', 'close', lbp=14)['williams_r']
        
        # 4. MFI (1 cecha względna)
        self.logger.info("  -> MFI...")
        df['mfi'] = bta.money_flow_index(df, window=14)['mfi']
        
        # 5. True Range (1 cecha względna)
        self.logger.info("  -> True Range...")
        df['trange'] = bta.true_range(df)['true_range']
        
        # POMIJAMY: OBV, VWAP, Bollinger Bands - są bezwzględne
        
        self.logger.info("Względne cechy bamboo_ta obliczone (6 cech)")
        return df

    def calculate_training_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza tylko względne cechy orderbook do treningu (6 cech)."""
        self.logger.info("Obliczanie względnych cech orderbook do treningu...")
        
        # Sprawdź czy mamy dane orderbook
        orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
        if len(orderbook_columns) == 0:
            self.logger.warning("Brak danych orderbook - pomijam cechy orderbook")
            return df
        
        # 1. Cechy podstawowe względne (4 cechy)
        self.logger.info("  -> Cechy podstawowe względne...")
        df = self._calculate_basic_orderbook_features(df)
        
        # 2. Cechy spreadu względne (1 cecha)
        self.logger.info("  -> Cechy spreadu względne...")
        df = self._calculate_spread_features(df)
        
        # 3. Cechy agregowane względne (1 cecha)
        self.logger.info("  -> Cechy agregowane względne...")
        df = self._calculate_aggregated_orderbook_features(df)
        
        # POMIJAMY: głębokości, dynamiczne, wolumeny, ceny - są bezwzględne
        
        self.logger.info("Względne cechy orderbook obliczone (6 cech)")
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



    def _calculate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy spreadu."""
        # Spread (różnica między najlepszym bid i ask)
        df['spread'] = df['snapshot1_depth_1'] - df['snapshot1_depth_-1']
        df['spread_pct'] = np.where(df['snapshot1_depth_-1'] != 0, df['spread'] / df['snapshot1_depth_-1'], 0)
        
        return df



    def _calculate_aggregated_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza cechy agregowane orderbook (względne)."""
        # Oblicz price imbalance (względne)
        bid_price_s1 = df['snapshot1_notional_-1']
        ask_price_s1 = df['snapshot1_notional_1']
        mid_price = (ask_price_s1 + bid_price_s1) / 2
        df['price_imbalance'] = np.where(mid_price != 0, (ask_price_s1 - bid_price_s1) / mid_price, 0)
        
        return df

    def calculate_training_hybrid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza tylko względne cechy hybrydowe do treningu (10 cech)."""
        self.logger.info("Obliczanie względnych cech hybrydowych do treningu...")
        
        # 1. Korelacje (2 cechy względne)
        self.logger.info("  -> Korelacje...")
        total_depth = sum(df[f'snapshot1_depth_{level}'] for level in config.ORDERBOOK_LEVELS)
        
        # Korelacja głębokości z ceną
        depth_price_corr = total_depth.rolling(window=self.history_window, min_periods=1).corr(df['close']).shift(1)
        df['depth_price_corr'] = depth_price_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Korelacja presji z wolumenem
        pressure_volume_corr = df['buy_sell_ratio_s1'].rolling(window=self.history_window, min_periods=1).corr(df['volume']).shift(1)
        df['pressure_volume_corr'] = pressure_volume_corr.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 2. Cechy czasowe (2 cechy względne)
        self.logger.info("  -> Cechy czasowe...")
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 3. Cecha momentum (1 cecha względna)
        self.logger.info("  -> Momentum...")
        df['price_momentum'] = df['close'].pct_change(periods=5).fillna(0)
        
        # 4. Cechy efektywności (3 cechy względne)
        self.logger.info("  -> Cechy efektywności...")
        df['market_efficiency_ratio'] = self._calculate_market_efficiency_ratio(df)
        df['price_efficiency_ratio'] = self._calculate_price_efficiency_ratio(df)
        df['volume_efficiency_ratio'] = self._calculate_volume_efficiency_ratio(df)
        
        # 5. Dodatkowe cechy hybrydowe (2 cechy względne)
        self.logger.info("  -> Dodatkowe cechy hybrydowe...")
        # Market microstructure score - będzie obliczone później
        # Liquidity score
        liquidity_score = np.where(df['spread_pct'] != 0, 1 / df['spread_pct'], 0)
        df['liquidity_score'] = pd.Series(liquidity_score, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        self.logger.info("Względne cechy hybrydowe obliczone (10 cech)")
        return df

    def calculate_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza względne cechy (16 cech)."""
        self.logger.info("Obliczanie względnych cech...")
        
        # 1. CECHY TRENDU CENY (5 cech)
        self.logger.info("  -> Cechy trendu ceny...")
        
        # Cechy 1-3: Trendy cenowe
        price_trend_30m = df['close'].pct_change(periods=config.PRICE_TREND_PERIODS[0])
        df['price_trend_30m'] = price_trend_30m.fillna(0).replace([np.inf, -np.inf], 0)
        
        price_trend_2h = df['close'].pct_change(periods=config.PRICE_TREND_PERIODS[1])
        df['price_trend_2h'] = price_trend_2h.fillna(0).replace([np.inf, -np.inf], 0)
        
        price_trend_6h = df['close'].pct_change(periods=config.PRICE_TREND_PERIODS[2])
        df['price_trend_6h'] = price_trend_6h.fillna(0).replace([np.inf, -np.inf], 0)
        
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
        self.logger.info("  -> Cechy pozycji ceny...")
        
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
        self.logger.info("  -> Cechy wolumenu...")
        
        # Cecha 10: Trend wolumenu
        volume_trend = df['volume'].pct_change(periods=config.VOLUME_TREND_PERIODS[0])
        df['volume_trend_1h'] = volume_trend.fillna(0).replace([np.inf, -np.inf], 0)
        
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
        self.logger.info("  -> Cechy orderbook...")
        
        # Cecha 15: Tightness spreadu
        spread_ma_60 = df['spread'].rolling(window=config.ROLLING_WINDOWS[1], min_periods=1).mean()
        spread_tightness = np.where(spread_ma_60 != 0, df['spread'] / spread_ma_60, 1)
        df['spread_tightness'] = pd.Series(spread_tightness, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Cecha 16: Ratio głębokości
        # Snapshot1
        bid_volume_s1 = sum(df[f'snapshot1_depth_{level}'] for level in config.BID_LEVELS)
        ask_volume_s1 = sum(df[f'snapshot1_depth_{level}'] for level in config.ASK_LEVELS)
        depth_ratio_s1 = np.where(ask_volume_s1 != 0, bid_volume_s1 / ask_volume_s1, 1)
        df['depth_ratio_s1'] = pd.Series(depth_ratio_s1, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Snapshot2
        bid_volume_s2 = sum(df[f'snapshot2_depth_{level}'] for level in config.BID_LEVELS)
        ask_volume_s2 = sum(df[f'snapshot2_depth_{level}'] for level in config.ASK_LEVELS)
        depth_ratio_s2 = np.where(ask_volume_s2 != 0, bid_volume_s2 / ask_volume_s2, 1)
        df['depth_ratio_s2'] = pd.Series(depth_ratio_s2, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
        
        # Total volume dla momentum
        total_volume = bid_volume_s1 + ask_volume_s1
        
        # Cecha 17: Momentum asymetrii głębokości
        depth_ratio_s1_1h_ago = df['depth_ratio_s1'].shift(config.MOMENTUM_PERIODS[0]).fillna(1)
        depth_momentum = np.where(
            depth_ratio_s1_1h_ago != 0,
            (df['depth_ratio_s1'] - depth_ratio_s1_1h_ago) / depth_ratio_s1_1h_ago,
            0
        )
        df['depth_momentum'] = pd.Series(depth_momentum, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)
        
        self.logger.info("Względne cechy obliczone (18 cech)")
        return df

    def _calculate_market_efficiency_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza ratio efektywności rynku."""
        returns = df['close'].pct_change().fillna(0)
        variance_ratio = returns.rolling(window=20).var() / returns.rolling(window=5).var()
        return variance_ratio.fillna(1)

    def _calculate_price_efficiency_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza ratio efektywności ceny."""
        price_changes = df['close'].diff().fillna(0)
        efficiency = price_changes.rolling(window=20).std() / price_changes.rolling(window=5).std()
        return efficiency.fillna(1)

    def _calculate_volume_efficiency_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza ratio efektywności wolumenu."""
        volume_changes = df['volume'].diff().fillna(0)
        efficiency = volume_changes.rolling(window=20).std() / volume_changes.rolling(window=5).std()
        return efficiency.fillna(1)

    def _calculate_price_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Oblicza spójność trendu ceny."""
        returns = df['close'].pct_change().fillna(0)
        consistency = returns.rolling(window=20).apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x))
        return consistency.fillna(0.5)

    def calculate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza 5 cech market regime."""
        self.logger.info("Obliczanie cech market regime...")
        
        # 1. Market Trend Strength (0-100)
        self.logger.info("  -> Market trend strength...")
        df['market_trend_strength'] = self._calculate_market_trend_strength(df)
        
        # 2. Market Trend Direction (-1 do 1)
        self.logger.info("  -> Market trend direction...")
        df['market_trend_direction'] = self._calculate_market_trend_direction(df)
        
        # 3. Market Choppiness (0-100)
        self.logger.info("  -> Market choppiness...")
        df['market_choppiness'] = self._calculate_market_choppiness(df)
        
        # 4. Bollinger Band Width (0-1)
        self.logger.info("  -> Bollinger band width...")
        df['bollinger_band_width'] = self._calculate_bollinger_band_width(df)
        
        # 5. Market Regime Classification (0/1/2)
        self.logger.info("  -> Market regime classification...")
        df['market_regime'] = self._calculate_market_regime(df)
        
        self.logger.info("Cechy market regime obliczone (5 cech)")
        return df

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza 6 cech volatility clustering."""
        self.logger.info("Obliczanie cech volatility clustering...")
        
        # 1. Volatility Regime (0/1/2)
        self.logger.info("  -> Volatility regime...")
        df['volatility_regime'] = self._calculate_volatility_regime(df)
        
        # 2. Volatility Percentile (0-100)
        self.logger.info("  -> Volatility percentile...")
        df['volatility_percentile'] = self._calculate_volatility_percentile(df)
        
        # 3. Volatility Persistence (0-1)
        self.logger.info("  -> Volatility persistence...")
        df['volatility_persistence'] = self._calculate_volatility_persistence(df)
        
        # 4. Volatility Momentum (-1 do 1)
        self.logger.info("  -> Volatility momentum...")
        df['volatility_momentum'] = self._calculate_volatility_momentum(df)
        
        # 5. Volatility of Volatility (0-1)
        self.logger.info("  -> Volatility of volatility...")
        df['volatility_of_volatility'] = self._calculate_volatility_of_volatility(df)
        
        # 6. Volatility Term Structure (-1 do 1)
        self.logger.info("  -> Volatility term structure...")
        df['volatility_term_structure'] = self._calculate_volatility_term_structure(df)
        
        self.logger.info("Cechy volatility clustering obliczone (6 cech)")
        return df

    def calculate_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oblicza 8 cech order book imbalance."""
        self.logger.info("Obliczanie cech order book imbalance...")
        
        # Sprawdź czy mamy dane orderbook
        orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
        if len(orderbook_columns) == 0:
            self.logger.warning("Brak danych orderbook - pomijam cechy imbalance")
            return df
        
        # 1. Volume Imbalance (-1 do 1)
        self.logger.info("  -> Volume imbalance...")
        df['volume_imbalance'] = self._calculate_volume_imbalance(df)
        
        # 2. Weighted Volume Imbalance (-1 do 1)
        self.logger.info("  -> Weighted volume imbalance...")
        df['weighted_volume_imbalance'] = self._calculate_weighted_volume_imbalance(df)
        
        # 3. Volume Imbalance Trend (-1 do 1)
        self.logger.info("  -> Volume imbalance trend...")
        df['volume_imbalance_trend'] = self._calculate_volume_imbalance_trend(df)
        
        # 4. Price Pressure (-1 do 1)
        self.logger.info("  -> Price pressure...")
        df['price_pressure'] = self._calculate_price_pressure(df)
        
        # 5. Weighted Price Pressure (-1 do 1)
        self.logger.info("  -> Weighted price pressure...")
        df['weighted_price_pressure'] = self._calculate_weighted_price_pressure(df)
        
        # 6. Price Pressure Momentum (-1 do 1)
        self.logger.info("  -> Price pressure momentum...")
        df['price_pressure_momentum'] = self._calculate_price_pressure_momentum(df)
        
        # 7. Order Flow Imbalance (-1 do 1)
        self.logger.info("  -> Order flow imbalance...")
        df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        
        # 8. Order Flow Trend (-1 do 1)
        self.logger.info("  -> Order flow trend...")
        df['order_flow_trend'] = self._calculate_order_flow_trend(df)
        
        self.logger.info("Cechy order book imbalance obliczone (8 cech)")
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
        
        # Autokorelacja volatility (lag=1)
        persistence = volatility.rolling(window=config.VOLATILITY_WINDOWS[1]).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
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
        if len(df) <= config.WARMUP_PERIOD_MINUTES:
            self.logger.warning(f"Dane są krótsze niż okres rozgrzewania ({config.WARMUP_PERIOD_MINUTES} minut)")
            return df
        
        original_length = len(df)
        df_trimmed = df.iloc[config.WARMUP_PERIOD_MINUTES:].copy()
        
        self.logger.info(f"Obcięto okres rozgrzewania: {original_length:,} -> {len(df_trimmed):,} wierszy")
        self.logger.info(f"Utrata: {config.WARMUP_PERIOD_MINUTES:,} wierszy ({config.WARMUP_PERIOD_MINUTES/1440:.1f} dni)")
        self.logger.info(f"Nowy zakres: {df_trimmed.index.min()} do {df_trimmed.index.max()}")
        
        return df_trimmed

    def calculate_features(self, df: pd.DataFrame, user_start_dt=None, user_end_dt=None) -> pd.DataFrame:
        """Główna funkcja obliczająca tylko cechy do treningu."""
        import time
        start_time = time.time()
        self.logger.info("Rozpoczynanie obliczania cech do treningu...")
        
        # Konwersja dat jeśli podano
        if user_start_dt:
            if isinstance(user_start_dt, str):
                user_start_dt = pd.to_datetime(user_start_dt)
        if user_end_dt:
            if isinstance(user_end_dt, str):
                user_end_dt = pd.to_datetime(user_end_dt)
        
        # Filtrowanie zakresu czasowego jeśli podano
        if user_start_dt or user_end_dt:
            if user_start_dt:
                df = df[df.index >= user_start_dt]
            if user_end_dt:
                df = df[df.index <= user_end_dt]
            self.logger.info(f"Przefiltrowano dane: {len(df):,} wierszy")
        
        # Obliczanie tylko cech do treningu
        self.logger.info("=== KROK 1: Cechy OHLC (względne) ===")
        df = self.calculate_training_ohlc_features(df)
        
        self.logger.info("=== KROK 2: Cechy bamboo_ta (względne) ===")
        df = self.calculate_training_bamboo_ta_features(df)
        
        self.logger.info("=== KROK 3: Cechy orderbook (względne) ===")
        df = self.calculate_training_orderbook_features(df)
        
        self.logger.info("=== KROK 4: Cechy hybrydowe (względne) ===")
        df = self.calculate_training_hybrid_features(df)
        
        self.logger.info("=== KROK 5: Cechy względne (trendy) ===")
        df = self.calculate_relative_features(df)
        
        self.logger.info("=== KROK 6: Cechy market regime ===")
        df = self.calculate_market_regime_features(df)
        
        self.logger.info("=== KROK 7: Cechy volatility clustering ===")
        df = self.calculate_volatility_features(df)
        
        self.logger.info("=== KROK 8: Cechy order book imbalance ===")
        df = self.calculate_imbalance_features(df)
        
        # Obcięcie okresu rozgrzewania
        self.logger.info("=== KROK 9: Obcięcie okresu rozgrzewania ===")
        df = self.trim_warmup_period(df)
        
        # Obliczenie market microstructure score (po wszystkich cechach imbalance)
        self.logger.info("=== KROK 10: Market microstructure score ===")
        df['market_microstructure_score'] = df['volume_imbalance'] * df['price_imbalance']
        
        # Sprawdzenie cech do treningu
        training_features = config.TRAINING_FEATURES_EXTENDED
        
        self.logger.info(f"Cechy do treningu: {len(training_features)} cech")
        missing_features = []
        for feature in training_features:
            if feature in df.columns:
                self.logger.info(f"  ✅ {feature}")
            else:
                self.logger.warning(f"  ❌ {feature} - brakuje!")
                missing_features.append(feature)
        
        if missing_features:
            self.logger.warning(f"Brakuje {len(missing_features)} cech: {missing_features}")
        
        # Podsumowanie
        feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        self.logger.info(f"Obliczono {len(feature_columns)} cech do treningu")
        self.logger.info(f"Finalny rozmiar: {len(df):,} wierszy, {len(df.columns)} kolumn")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Czas obliczeń: {elapsed_time:.2f} sekund")
        
        return df

    def save_data(self, df: pd.DataFrame, file_path: str):
        """Zapisuje dane do pliku feather."""
        self.logger.info(f"Zapisuję dane do: {file_path}")
        
        # Utwórz katalog jeśli nie istnieje
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Resetuj indeks przed zapisem
        df_to_save = df.reset_index()
        
        # Zapisz dane
        df_to_save.to_feather(file_path)
        
        # Sprawdź rozmiar pliku
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        self.logger.info(f"Zapisano: {file_path} ({file_size:.2f} MB)")
        self.logger.info(f"Zapisano {len(df.columns)} kolumn")

def process_all_pairs(input_dir: str, output_dir: str, start_date: str = None, end_date: str = None):
    """Przetwarza wszystkie pary z konfiguracji orderbook."""
    
    # Pobierz listę par z konfiguracji
    pairs = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
        "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
        "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT",
        "FILUSDT", "TRXUSDT", "XLMUSDT", "VETUSDT", "NEOUSDT"
    ]
    
    # Inicjalizacja kalkulatora
    calculator = OHLCOrderBookFeatureCalculator()
    
    # Statystyki
    total_start_time = time.time()
    successful_pairs = 0
    failed_pairs = []
    
    calculator.logger.info("=" * 80)
    calculator.logger.info(f"ROZPOCZYNAM PRZETWARZANIE WSZYSTKICH {len(pairs)} PAR")
    calculator.logger.info("=" * 80)
    
    for i, pair in enumerate(pairs, 1):
        calculator.logger.info(f"\n[{i}/{len(pairs)}] Przetwarzanie pary: {pair}")
        calculator.logger.info("-" * 50)
        
        # Ścieżki plików
        input_file = os.path.join(input_dir, f"merged_{pair}.feather")
        output_file = os.path.join(output_dir, f"features_{pair}.feather")
        
        # Sprawdź czy plik wejściowy istnieje
        if not os.path.exists(input_file):
            calculator.logger.error(f"❌ Plik wejściowy nie istnieje: {input_file}")
            failed_pairs.append((pair, "Plik wejściowy nie istnieje"))
            continue
        
        try:
            # Wczytaj dane
            df = calculator.load_data(input_file)
            if df is None:
                calculator.logger.error(f"❌ Nie można wczytać danych dla {pair}")
                failed_pairs.append((pair, "Błąd wczytywania danych"))
                continue
            
            # Oblicz cechy
            start_time = time.time()
            df_features = calculator.calculate_features(df, start_date, end_date)
            elapsed_time = time.time() - start_time
            
            # Zapisz wyniki
            calculator.save_data(df_features, output_file)
            
            calculator.logger.info(f"✅ {pair}: {len(df_features):,} wierszy, {len(df_features.columns)} kolumn, {elapsed_time:.2f}s")
            successful_pairs += 1
            
        except Exception as e:
            calculator.logger.error(f"❌ Błąd podczas przetwarzania {pair}: {str(e)}")
            failed_pairs.append((pair, str(e)))
            continue
    
    # Podsumowanie
    total_time = time.time() - total_start_time
    calculator.logger.info("\n" + "=" * 80)
    calculator.logger.info("PODSUMOWANIE PRZETWARZANIA")
    calculator.logger.info("=" * 80)
    calculator.logger.info(f"✅ Pomyślnie przetworzono: {successful_pairs}/{len(pairs)} par")
    calculator.logger.info(f"❌ Nieudane: {len(failed_pairs)} par")
    calculator.logger.info(f"⏱️  Całkowity czas: {total_time:.2f} sekund ({total_time/60:.2f} minut)")
    calculator.logger.info(f"📊 Średni czas na parę: {total_time/len(pairs):.2f} sekund")
    
    if failed_pairs:
        calculator.logger.info("\n❌ Lista nieudanych par:")
        for pair, error in failed_pairs:
            calculator.logger.info(f"  - {pair}: {error}")
    
    calculator.logger.info(f"\n📁 Pliki wynikowe w: {output_dir}")
    calculator.logger.info("=" * 80)

def main():
    """Główna funkcja programu."""
    import time
    
    # Konfiguracja logowania
    logger = setup_logging()
    
    # Parsowanie argumentów
    parser = argparse.ArgumentParser(description='Obliczanie cech OHLC + Order Book')
    parser.add_argument('--input', type=str, help='Ścieżka do pliku wejściowego')
    parser.add_argument('--output', type=str, help='Ścieżka do pliku wyjściowego')
    parser.add_argument('--all-pairs', action='store_true', help='Przetwarzaj wszystkie pary')
    parser.add_argument('--input-dir', type=str, default='../download2/merge/merged_data', help='Katalog z plikami wejściowymi')
    parser.add_argument('--output-dir', type=str, default='output', help='Katalog z plikami wyjściowymi')
    parser.add_argument('--start-date', type=str, help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Data końcowa (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Sprawdź argumenty
    if not args.all_pairs and (not args.input or not args.output):
        logger.error("Musisz podać --input i --output lub --all-pairs")
        parser.print_help()
        return
    
    # Przetwarzanie wszystkich par
    if args.all_pairs:
        process_all_pairs(args.input_dir, args.output_dir, args.start_date, args.end_date)
        return
    
    # Standardowe przetwarzanie pojedynczej pary
    logger.info("ROZPOCZYNAM OBLICZANIE CECH OHLC + ORDERBOOK")
    logger.info("=" * 60)
    
    # Inicjalizacja kalkulatora
    calculator = OHLCOrderBookFeatureCalculator()
    
    # Wczytanie danych
    df = calculator.load_data(args.input)
    if df is None:
        logger.error("Nie można wczytać danych!")
        return
    
    # Obliczenie cech
    start_time = time.time()
    df_features = calculator.calculate_features(df, args.start_date, args.end_date)
    elapsed_time = time.time() - start_time
    
    # Zapisanie wyników
    calculator.save_data(df_features, args.output)
    
    logger.info("OBLICZANIE CECH ZAKONCZONE POMYSLNIE!")
    logger.info(f"Plik wynikowy: {args.output}")
    logger.info(f"Wierszy: {len(df_features):,}")
    logger.info(f"Kolumn: {len(df_features.columns)}")
    logger.info(f"Czas: {elapsed_time:.2f} sekund")

if __name__ == "__main__":
    main() 