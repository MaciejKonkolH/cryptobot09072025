"""
Konfiguracja dla modułu obliczającego cechy (`feature_calculator_ohlc_snapshot`).
Obsługuje dane OHLC + Order Book w formacie Feather.
"""
import os
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "feature_calculator_ohlc_snapshot"

# --- Konfiguracja Wejścia ---
INPUT_DIR = PROJECT_ROOT / "merge"
INPUT_FILENAME = "merged_ohlc_orderbook.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'feature_calculator_ohlc_snapshot.log'

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
DEFAULT_OUTPUT_FILENAME = 'ohlc_orderbook_features.feather'

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

# Lista nowych względnych cech
RELATIVE_FEATURES = [
    # Cechy trendu ceny (5 cech)
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
    
    # Cechy pozycji ceny (4 cechy)
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    
    # Cechy wolumenu (5 cech)
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
    
    # Cechy orderbook (4 cechy)
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum'
]

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