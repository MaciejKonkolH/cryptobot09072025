import logging
from pathlib import Path
from training5 import config as t5cfg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "multi_training"

# Input from labeler5
INPUT_DIR = t5cfg.INPUT_DIR
INPUT_TEMPLATE = t5cfg.INPUT_TEMPLATE

# Output
OUTPUT_DIR = MODULE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

# Which symbols to train jointly
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]

# TP/SL levels and label columns reused from training5 for full compatibility
TP_SL_LEVELS = t5cfg.TP_SL_LEVELS


def label_col(tp, sl):
    return f"label_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"


LABEL_COLUMNS = [label_col(tp, sl) for tp, sl in TP_SL_LEVELS]

# Splits
VALIDATION_SPLIT = t5cfg.VALIDATION_SPLIT
TEST_SPLIT = t5cfg.TEST_SPLIT

# XGBoost params (reuse from training5)
XGB_N_ESTIMATORS = t5cfg.XGB_N_ESTIMATORS
XGB_LEARNING_RATE = t5cfg.XGB_LEARNING_RATE
XGB_MAX_DEPTH = t5cfg.XGB_MAX_DEPTH
XGB_SUBSAMPLE = t5cfg.XGB_SUBSAMPLE
XGB_COLSAMPLE_BYTREE = t5cfg.XGB_COLSAMPLE_BYTREE
XGB_GAMMA = t5cfg.XGB_GAMMA
XGB_RANDOM_STATE = t5cfg.XGB_RANDOM_STATE
XGB_EARLY_STOPPING_ROUNDS = t5cfg.XGB_EARLY_STOPPING_ROUNDS
XGB_REG_ALPHA = t5cfg.XGB_REG_ALPHA
XGB_REG_LAMBDA = t5cfg.XGB_REG_LAMBDA
XGB_MIN_CHILD_WEIGHT = t5cfg.XGB_MIN_CHILD_WEIGHT

# Progress logging
TRAIN_VERBOSE_EVAL = t5cfg.TRAIN_VERBOSE_EVAL

# Class weighting (to improve detection of rare LONG/SHORT classes)
ENABLE_CLASS_WEIGHTS_IN_TRAINING = t5cfg.ENABLE_CLASS_WEIGHTS_IN_TRAINING
CLASS_WEIGHTS = dict(t5cfg.CLASS_WEIGHTS)

# Feature selection options — mirror training5
FEATURE_SELECTION_MODE = ('custom' if getattr(t5cfg, 'FEATURE_SELECTION_MODE', 'all') == 'custom_strict' else getattr(t5cfg, 'FEATURE_SELECTION_MODE', 'all'))
USE_TRAINING3_FEATURE_WHITELIST = t5cfg.USE_TRAINING3_FEATURE_WHITELIST
CUSTOM_FEATURE_LIST = list(t5cfg.CUSTOM_FEATURE_LIST)

# Optional explicit date filter overrides (set to ISO strings or None)
# Limit to last 24 months ending at 2025-08-15
DATE_FILTER_START = "2023-08-15"
DATE_FILTER_END = "2025-08-15"

# Symbol encoding controls
ADD_SYMBOL_ONEHOTS = True  # adds columns sym_{symbol} to X (they bypass whitelists)


def get_report_dir(symbol: str = "MULTI") -> Path:
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

