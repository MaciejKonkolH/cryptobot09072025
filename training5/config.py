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

