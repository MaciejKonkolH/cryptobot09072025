import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "feature_calculator3"

# Input (merged wide parquet from download3)
MERGE_DIR = PROJECT_ROOT / "download3" / "merge" / "merged_data"
INPUT_TEMPLATE = "merged_{symbol}.parquet"

# Output
OUTPUT_DIR = MODULE_DIR / "output"
METADATA_DIR = MODULE_DIR / "metadata"
LOG_DIR = MODULE_DIR / "logs"

# Symbols
DEFAULT_SYMBOL = "BTCUSDT"

# Hybrid sequence configuration (Option A)
USE_HYBRID = True

# Lags for key short-term signals
SHORT_LAGS = [1, 2, 5]
KEY_LAG_FEATURES = [
    "imb_s1", "imb_delta", "delta_depth_bid_rel", "delta_depth_ask_rel", "rv_5m", "ret_1m",
]

# Binning buckets and base features for time aggregation
BIN_BUCKETS = {
    "bin13": (1, 3),
    "bin410": (4, 10),
    "bin1130": (11, 30),
    "bin3160": (31, 60),
}

BIN_FEATURES = [
    "imb_s1", "imb_s2", "imb_delta", "near_pressure_ratio_s1",
    "wadl_bid_share_s1", "wadl_ask_share_s1",
    "delta_depth_bid_rel", "delta_depth_ask_rel",
    "rv_5m", "ret_1m", "price_vs_ma_240", "price_vs_ma_1440",
    # extended long-trend features
    "price_vs_ma_60", "price_vs_ma_360",
]

# Raw columns to carry over from merged parquet into output (for labeling)
RAW_COLUMNS = [
    "open", "high", "low", "close", "volume",
]

# Moving averages for price ratios: SMA windows
MA_WINDOWS = {
    60: "price_vs_ma_60",
    240: "price_vs_ma_240",
    360: "price_vs_ma_360",
    720: "price_vs_ma_720",
    1440: "price_vs_ma_1440",
    2880: "price_vs_ma_2880",
    4320: "price_vs_ma_4320",
    10080: "price_vs_ma_10080",
}

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

