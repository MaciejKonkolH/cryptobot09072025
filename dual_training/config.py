import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "dual_training"

# Input from dual_labeler
INPUT_DIR = PROJECT_ROOT / "dual_labeler" / "output"
INPUT_TEMPLATE = "labeled_{symbol}.feather"

# Output
OUTPUT_DIR = MODULE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

DEFAULT_SYMBOL = "BTCUSDT"

# Symmetric TP levels (TP=SL). We'll derive TP_SL_LEVELS for compatibility
TP_LEVELS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]
TP_SL_LEVELS = [(tp, tp) for tp in TP_LEVELS]


def label_col(tp, sl):
    return f"label_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"


LABEL_COLUMNS = [label_col(tp, sl) for tp, sl in TP_SL_LEVELS]

# Splits
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# XGBoost params (binary)
XGB_N_ESTIMATORS = 400
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.7
XGB_GAMMA = 0.1
XGB_RANDOM_STATE = 42
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_REG_ALPHA = 0.0
XGB_REG_LAMBDA = 0.0
XGB_MIN_CHILD_WEIGHT = 1

# Progress logging
TRAIN_VERBOSE_EVAL = 50

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

# Feature selection (same interface as training5)
FEATURE_SELECTION_MODE = 'custom_strict'
USE_TRAINING3_FEATURE_WHITELIST = False
CUSTOM_FEATURE_LIST = [
    'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
    'price_vs_ma_240', 'close_vs_ema_120', 'close_vs_ema_60', 'slope_ema_60_over_ATR',
    'slope_return_120', 'r2_trend_120',
    'price_volatility_rolling', 'vol_regime_120', 'vol_of_vol_120',
    'volatility_momentum', 'volatility_percentile', 'volatility_term_structure', 'volatility_persistence', 'volatility_of_volatility',
    'bb_pos_20', 'bb_width_over_ATR_20',
    'donch_pos_60', 'donch_width_over_ATR_60',
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    'side_skew', 'pressure_12', 'pressure_12_norm',
    'volume_imbalance', 'price_pressure', 'weighted_price_pressure',
    'persistence_imbalance_1pct_ema5', 'persistence_imbalance_1pct_ema10',
    'market_trend_strength', 'market_trend_direction', 'market_regime',
    'pos_in_channel_120', 'pos_in_channel_180', 'pos_in_channel_240', 'pos_in_channel_360', 'pos_in_channel_480',
    'width_over_ATR_120', 'width_over_ATR_180', 'width_over_ATR_240', 'width_over_ATR_360', 'width_over_ATR_480',
    'slope_over_ATR_window_120', 'slope_over_ATR_window_180', 'slope_over_ATR_window_240', 'slope_over_ATR_window_360', 'slope_over_ATR_window_480',
    'channel_fit_score_120', 'channel_fit_score_180', 'channel_fit_score_240', 'channel_fit_score_360', 'channel_fit_score_480',
    'slope_over_ATR_window_240_x_imbalance_1pct',
    'slope_over_ATR_window_360_x_imbalance_1pct',
    'slope_over_ATR_window_480_x_imbalance_1pct',
    'width_over_ATR_240_x_imbalance_1pct',
    'width_over_ATR_360_x_imbalance_1pct',
    'width_over_ATR_480_x_imbalance_1pct',
]

