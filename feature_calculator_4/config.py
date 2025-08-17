import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "feature_calculator_4"

# Input (merged wide parquet from download3)
MERGE_DIR = PROJECT_ROOT / "download3" / "merge" / "merged_data"
INPUT_TEMPLATE = "merged_{symbol}.parquet"

# Output
OUTPUT_DIR = MODULE_DIR / "output"
METADATA_DIR = MODULE_DIR / "metadata"
LOG_DIR = MODULE_DIR / "logs"

# Symbols
DEFAULT_SYMBOL = "BTCUSDT"

# Channel windows (minutes)
# Base set aligned to current pipeline
CHANNEL_WINDOWS = [240, 180, 120]
# Optional extended windows prepared for experiments (keep disabled by default)
ENABLE_EXTENDED_CHANNELS = True
EXTENDED_CHANNEL_WINDOWS = [360, 480]

# OB EMA lengths for persistence
IMB_EMA_LENS = [5, 10]

# EMA slope lookback for slope_ema features
EMA_SLOPE_K = 10

# Progress / verbosity
SHOW_PROGRESS = True

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# Training3 core (37) features support
# =====================

# Master switch: include legacy training3 37-feature core in output
ENABLE_T3_FEATURES = True

# Periods/parameters aligned with feature_calculator_ohlc_snapshot
PRICE_TREND_PERIODS = [30, 120, 360]  # 30m, 2h, 6h
ROLLING_WINDOWS = [30, 60]            # for volatility and moving averages
VOLUME_TREND_PERIODS = [60]           # 1h
MOMENTUM_PERIODS = [60]               # 1h

# Market regime / TA params
ADX_PERIOD = 14
MARKET_REGIME_PERIODS = [20, 50]
CHOPPINESS_PERIOD = 14
BOLLINGER_WIDTH_PERIOD = 20

# Volatility clustering params
VOLATILITY_WINDOWS = [20, 60, 240]
VOLATILITY_PERCENTILE_WINDOW = 60
VOLATILITY_MIN_THRESHOLD = 0.001

# Orderbook layout assumptions
BID_LEVELS = [-5, -4, -3, -2, -1]
ASK_LEVELS = [1, 2, 3, 4, 5]
PRESSURE_WINDOW = 10
MIN_SPREAD_THRESHOLD = 0.0001

# Winsorization (optional; applied in main after feature computation, before fillna)
WINSORIZE_ENABLED = False
# Quantiles expressed as fractions (e.g., 0.005 = 0.5%, 0.995 = 99.5%)
WINSORIZE_FEATURES = {
    # Prepare for outlier control; can be toggled on when testing ETH/XRP
    "OBV_slope_over_ATR": (0.005, 0.995),
}

