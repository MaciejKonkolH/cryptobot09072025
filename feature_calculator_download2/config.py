"""
Konfiguracja dla modułu obliczającego cechy (`feature_calculator_download2`).
Obsługuje dane OHLC + Order Book w formacie Feather.
"""
import os
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "feature_calculator_download2"

# --- Konfiguracja Wejścia ---
INPUT_DIR = PROJECT_ROOT / "download2" / "merge" / "merged_data"
INPUT_FILENAME = "merged_{symbol}.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'feature_calculator_download2.log'

# --- Parametry Obliczeń Cech ---
# Okna dla średnich kroczących (w minutach)
MA_WINDOWS = [60, 240, 1440, 43200]  # 1h, 4h, 1d, 30d

# Okna dla cech historycznych order book (w liczbie snapshotów)
ORDERBOOK_HISTORY_WINDOW = 10  # Ostatnie 10 snapshotów dla trendów
ORDERBOOK_SHORT_WINDOW = 5     # Ostatnie 5 snapshotów dla stabilności
ORDERBOOK_MOMENTUM_WINDOW = 3  # Ostatnie 3 snapshoty dla momentum

# Okres rozgrzewania (w minutach) - 30 dni dla MA_43200
WARMUP_PERIOD_MINUTES = 30 * 24 * 60  # 30 dni

# Nazwy kolumn OHLCV
COL_TIMESTAMP = 'timestamp'
COL_OPEN = 'open'
COL_CLOSE = 'close'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_VOLUME = 'volume'

# Nazwy poziomów order book
ORDERBOOK_LEVELS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
BID_LEVELS = [-5, -4, -3, -2, -1]  # Poziomy kupna
ASK_LEVELS = [1, 2, 3, 4, 5]       # Poziomy sprzedaży

# Parametry dla cech TP/SL
TP_LEVELS = [1, 2]  # Poziomy Take Profit (1%, 2%)
SL_LEVELS = [-1, -2]  # Poziomy Stop Loss (-1%, -2%)

# Nazwy kolumn order book
SNAPSHOT1_PREFIX = 'snapshot1_'
SNAPSHOT2_PREFIX = 'snapshot2_'
DEPTH_SUFFIX = 'depth_'
NOTIONAL_SUFFIX = 'notional_'
TIMESTAMP_SUFFIX = 'timestamp'

# Domyślna nazwa pliku wyjściowego
DEFAULT_OUTPUT_FILENAME = 'features_{symbol}.feather'

# Parametry dla wskaźników technicznych
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2.0
RSI_PERIOD = 14
MACD_SHORT = 12
MACD_LONG = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
CHOPPINESS_PERIOD = 14

# Parametry dla cech czasowych
HOURS_OF_DAY = list(range(24))
DAYS_OF_WEEK = list(range(7))  # 0=Monday, 6=Sunday

# --- PARAMETRY DLA NOWYCH WZGLĘDNYCH CECH ---

# Okresy dla trendów cenowych (w minutach)
PRICE_TREND_PERIODS = [30, 120, 360]  # 30m, 2h, 6h

# Okresy dla trendów wolumenu (w minutach)
VOLUME_TREND_PERIODS = [60]  # 1h

# Okna dla obliczeń rolling (w minutach)
ROLLING_WINDOWS = [30, 60]  # 30m dla volatility, 60m dla średnich

# Okresy dla momentum (w minutach)
MOMENTUM_PERIODS = [60]  # 1h dla momentum

# Lista cech do treningu - PODSTAWOWA (37 cech z training3)
TRAINING_FEATURES_BASIC = [
    # Cechy trendu ceny (5 cech)
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
    
    # Cechy pozycji ceny (4 cechy)
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    
    # Cechy wolumenu (5 cech)
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
    
    # Cechy orderbook (4 cechy)
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    
    # Market regime (5 cech)
    'market_trend_strength', 'market_trend_direction', 'market_choppiness',
    'bollinger_band_width', 'market_regime',
    
    # Volatility clustering (6 cech)
    'volatility_regime', 'volatility_percentile', 'volatility_persistence',
    'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    
    # Order book imbalance (8 cech)
    'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
    'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
    'order_flow_imbalance', 'order_flow_trend'
]

# Lista dodatkowych cech względnych - DO EKSPERYMENTÓW
ADDITIONAL_RELATIVE_FEATURES = [
    # Dodatkowe cechy OHLC (12)
    'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'adx_14',
    'price_to_ma_60', 'price_to_ma_240', 'ma_60_to_ma_240', 'price_to_ma_1440',
    'volume_change_norm', 'upper_wick_ratio_5m', 'lower_wick_ratio_5m',
    
    # Dodatkowe cechy bamboo_ta (6)
    'stoch_k', 'stoch_d', 'cci', 'williams_r', 'mfi', 'trange',
    
    # Dodatkowe cechy orderbook (6)
    'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'imbalance_s1', 'imbalance_s2',
    'spread_pct', 'price_imbalance',
    
    # Cechy hybrydowe (10)
    'market_microstructure_score', 'liquidity_score', 'depth_price_corr',
    'pressure_volume_corr', 'hour_of_day', 'day_of_week', 'price_momentum',
    'market_efficiency_ratio', 'price_efficiency_ratio', 'volume_efficiency_ratio'
]

# Lista wszystkich cech do treningu (podstawowe + dodatkowe)
TRAINING_FEATURES_EXTENDED = TRAINING_FEATURES_BASIC + ADDITIONAL_RELATIVE_FEATURES

# Zachowaj kompatybilność wsteczną
RELATIVE_FEATURES = TRAINING_FEATURES_BASIC

# --- PARAMETRY DLA NOWYCH ZAAWANSOWANYCH CECH ---

# Market Regime parameters
MARKET_REGIME_PERIODS = [20, 50]  # Okresy dla trend detection
CHOPPINESS_PERIOD = 14
BOLLINGER_WIDTH_PERIOD = 20

# Volatility Clustering parameters  
VOLATILITY_WINDOWS = [20, 60, 240]  # Okresy dla volatility
VOLATILITY_PERCENTILE_WINDOW = 60   # Okres dla percentyla (zmniejszony z 240 na 60)
VOLATILITY_MIN_THRESHOLD = 0.001    # Minimalny próg dla volatility

# Order Book Imbalance parameters
IMBALANCE_LEVELS = [1, 2, 3]  # Poziomy do analizy
PRESSURE_WINDOW = 10  # Okres dla pressure trend
MIN_SPREAD_THRESHOLD = 0.0001  # Minimalny spread dla price pressure

# Lista nowych zaawansowanych cech
NEW_ADVANCED_FEATURES = [
    # Market Regime (5 cech)
    'market_trend_strength', 'market_trend_direction', 'market_choppiness',
    'bollinger_band_width', 'market_regime',
    
    # Volatility Clustering (6 cech)
    'volatility_regime', 'volatility_percentile', 'volatility_persistence',
    'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    
    # Order Book Imbalance (8 cech)
    'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
    'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
    'order_flow_imbalance', 'order_flow_trend'
] 