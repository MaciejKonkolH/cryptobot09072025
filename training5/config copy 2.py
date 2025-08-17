import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "training5"

# Input from labeler5
INPUT_DIR = PROJECT_ROOT / "labeler5" / "output"
INPUT_TEMPLATE = "labeled_{symbol}.feather"

# Output
OUTPUT_DIR = MODULE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

DEFAULT_SYMBOL = "BTCUSDT"

# Features: use all non-label, non-OHLC columns except timestamp as X
# Will be detected dynamically in data_loader

# Label columns: build from TP/SL (match labeler5)
TP_SL_LEVELS = [
    (0.6, 0.2), (0.6, 0.3), (0.8, 0.2), (0.8, 0.3), (0.8, 0.4),
    (1.0, 0.3), (1.0, 0.4), (1.0, 0.5), (1.2, 0.4), (1.2, 0.5), (1.2, 0.6),
    (1.4, 0.4), (1.4, 0.5), (1.4, 0.6), (1.4, 0.7),
]

def label_col(tp, sl):
    return f"label_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

LABEL_COLUMNS = [label_col(tp, sl) for tp, sl in TP_SL_LEVELS]

# Splits
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# XGBoost params
XGB_N_ESTIMATORS = 700
XGB_LEARNING_RATE = 0.04
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.75
XGB_COLSAMPLE_BYTREE = 0.65
XGB_GAMMA = 0.25
XGB_RANDOM_STATE = 42
XGB_EARLY_STOPPING_ROUNDS = 30
XGB_REG_ALPHA = 0.05
XGB_REG_LAMBDA = 1.5
XGB_MIN_CHILD_WEIGHT = 3

# Progress logging
TRAIN_VERBOSE_EVAL = 50  # print eval metric every N iterations

# Predictions CSV: which model index (0-14)
CSV_PREDICTIONS_MODEL_INDEX = 3

def get_report_dir(symbol: str) -> Path:
    d = REPORTS_DIR / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# Class weighting (to improve detection of rare LONG/SHORT classes)
# Enable mapping-based class weights during training (applied to train/val DMatrix)
ENABLE_CLASS_WEIGHTS_IN_TRAINING = True
# Mapping for 3-class labels: 0=LONG, 1=SHORT, 2=NEUTRAL
# Adjust as needed; defaults put more weight on LONG/SHORT vs NEUTRAL
CLASS_WEIGHTS = {
    0: 2.0,  # LONG
    1: 2.0,  # SHORT
    2: 1.0,  # NEUTRAL
}


# Feature selection options
# Modes:
# - 'all'         -> use all numeric features (excluding OHLC/labels)
# - 't3_37'       -> use training3 37-feature whitelist when available
# - 'custom'      -> use intersection of CUSTOM_FEATURE_LIST with available features (warn on missing)
# - 'custom_strict' -> use EXACTLY CUSTOM_FEATURE_LIST and FAIL if any listed feature is missing
FEATURE_SELECTION_MODE = 'custom_strict'

# Backward-compat flag
USE_TRAINING3_FEATURE_WHITELIST = False

# Optional custom feature list (used if FEATURE_SELECTION_MODE in {'custom','custom_strict'})
CUSTOM_FEATURE_LIST = [
    # training3 core (37)
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
    'price_strength', 'price_consistency_score',
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    'market_trend_strength', 'market_trend_direction', 'bollinger_band_width', 'market_regime',
    'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
    'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend',

    # channel features (12)
    'pos_in_channel_240', 'pos_in_channel_180', 'pos_in_channel_120',
    'width_over_ATR_240', 'width_over_ATR_180', 'width_over_ATR_120',
    'slope_over_ATR_window_240', 'slope_over_ATR_window_180', 'slope_over_ATR_window_120',
    'channel_fit_score_240', 'channel_fit_score_180', 'channel_fit_score_120',

    # additions (targeted): channel×OB interactions (240), BB position, EMA position/slope, OBV slope
    'pos_in_channel_240_x_imbalance_1pct',
    'slope_over_ATR_window_240_x_imbalance_1pct',
    'width_over_ATR_240_x_imbalance_1pct',
    # extend interactions for 180 window as suggested
    'pos_in_channel_180_x_imbalance_1pct',
    'slope_over_ATR_window_180_x_imbalance_1pct',
    'width_over_ATR_180_x_imbalance_1pct',
    'bb_pos_20',
    'close_vs_ema_120',
    'slope_ema_60_over_ATR',
    'OBV_slope_over_ATR',
]

# Proposed extended feature set to experiment with (training3 whitelist + TA + microstructure + channels)
# To use: set FEATURE_SELECTION_MODE = 'custom' above.
CUSTOM_FEATURE_LIST_PROPOSED = [
    # training3 whitelist (37)
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    'market_trend_strength', 'market_trend_direction', 'market_choppiness', 'bollinger_band_width', 'market_regime',
    'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility',
    'volatility_term_structure', 'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
    'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend',

    # Candlestick structure & short-term returns
    'body_ratio', 'wick_up_ratio', 'wick_down_ratio',
    'r_1', 'r_5', 'r_15', 'slope_return_120',

    # Volatility/regime (alt windows)
    'vol_regime_120', 'vol_of_vol_120', 'r2_trend_120',

    # TA indicators
    'RSI_14', 'RSI_30', 'StochK_14_3', 'StochD_14_3', 'MACD_hist_over_ATR', 'ADX_14', 'di_diff_14', 'CCI_20_over_ATR',
    'bb_pos_20', 'bb_width_over_ATR_20', 'donch_pos_60', 'donch_width_over_ATR_60',
    'close_vs_ema_60', 'close_vs_ema_120', 'slope_ema_60_over_ATR', 'MFI_14', 'OBV_slope_over_ATR',

    # Orderbook microstructure
    'imbalance_1pct_notional', 'log_ratio_ask_2_over_1', 'log_ratio_bid_2_over_1', 'ask_near_ratio', 'bid_near_ratio',
    'concentration_near_mkt', 'ask_com', 'bid_com', 'com_diff', 'pressure_12', 'pressure_12_norm', 'side_skew',
    'persistence_imbalance_1pct_ema5', 'persistence_imbalance_1pct_ema10', 'dA1', 'dB1', 'dImb1', 'ema_dImb1_5', 'ema_dImb1_10',

    # Channel-based structure (multi-window)
    'pos_in_channel_240', 'width_over_ATR_240', 'slope_over_ATR_window_240', 'channel_fit_score_240',
    'pos_in_channel_180', 'width_over_ATR_180', 'slope_over_ATR_window_180', 'channel_fit_score_180',
    'pos_in_channel_120', 'width_over_ATR_120', 'slope_over_ATR_window_120', 'channel_fit_score_120',

    # Cross features: channel position/slope/width x imbalance
    'pos_in_channel_240_x_imbalance_1pct', 'slope_over_ATR_window_240_x_imbalance_1pct', 'width_over_ATR_240_x_imbalance_1pct',
    'pos_in_channel_180_x_imbalance_1pct', 'slope_over_ATR_window_180_x_imbalance_1pct', 'width_over_ATR_180_x_imbalance_1pct',
    'pos_in_channel_120_x_imbalance_1pct', 'slope_over_ATR_window_120_x_imbalance_1pct', 'width_over_ATR_120_x_imbalance_1pct',
]

# Optional: align training window to training3 input range
# When enabled, training5 will cut its dataframe to the min/max timestamp
# found in the reference file produced for training3.
ALIGN_TO_TRAINING3_RANGE = False
TRAINING3_REF_FILE = PROJECT_ROOT / "labeler3" / "output" / "ohlc_orderbook_labeled_3class_fw120m_15levels.feather"

# Optional explicit date filter overrides (set to ISO strings or None)
DATE_FILTER_START = None  # e.g. "2022-01-01"
DATE_FILTER_END = None    # e.g. "2023-01-01"

