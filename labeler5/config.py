from pathlib import Path
import logging


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "labeler5"

# Input from feature_calculator3
INPUT_DIR = PROJECT_ROOT / "feature_calculator3" / "output"
INPUT_TEMPLATE = "features_{symbol}.feather"

# Output
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = MODULE_DIR / "logs"
METADATA_DIR = MODULE_DIR / "metadata"

# Symbols
DEFAULT_SYMBOL = "BTCUSDT"

# Labeling params
FUTURE_WINDOW = 120  # minutes

# TP/SL levels (match training4)
TP_SL_LEVELS = [
    (0.6, 0.2), (0.6, 0.3), (0.8, 0.2), (0.8, 0.3), (0.8, 0.4),
    (1.0, 0.3), (1.0, 0.4), (1.0, 0.5), (1.2, 0.4), (1.2, 0.5), (1.2, 0.6),
    (1.4, 0.4), (1.4, 0.5), (1.4, 0.6), (1.4, 0.7),
]

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

