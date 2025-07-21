"""
Konfiguracja dla modułu obliczającego cechy (`feature_calculator_snapshot`).
Obsługuje dane OHLC + Order Book w formacie JSON.
"""
import os
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "feature_calculator_snapshot"

# --- Konfiguracja Wejścia ---
# Ścieżka do pliku JSON z danymi OHLC + Order Book
INPUT_DIR = PROJECT_ROOT / "download"  # Zmieniono ze "skrypty" na "download"
INPUT_FILENAME = "orderbook_ohlc_merged.json"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs" # Zapisuj logi w podkatalogu output

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'feature_calculator_snapshot.log'

# --- Parametry Obliczeń Cech ---
# Okna dla średnich kroczących (w minutach) - teraz z pełnymi danymi historycznymi
MA_WINDOWS = [60, 240, 1440, 43200]  # 1h, 4h, 1d, 30d - przywrócono 30d

# Okna dla cech historycznych order book (w liczbie snapshotów) - dostosowane do małego zbioru
ORDERBOOK_HISTORY_WINDOW = 10  # Ostatnie 10 snapshotów dla trendów (zamiast 30)
ORDERBOOK_SHORT_WINDOW = 5     # Ostatnie 5 snapshotów dla stabilności (zamiast 10)
ORDERBOOK_MOMENTUM_WINDOW = 3  # Ostatnie 3 snapshoty dla momentum (zamiast 5)

# Nazwy kolumn OHLCV
COL_TIMESTAMP = 'timestamp'
COL_OPEN = 'open'
COL_CLOSE = 'close'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_VOLUME = 'volume'
COL_DATA_QUALITY = 'data_quality'

# Nazwy poziomów order book
ORDERBOOK_LEVELS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
BID_LEVELS = [-5, -4, -3, -2, -1]  # Poziomy kupna
ASK_LEVELS = [1, 2, 3, 4, 5]       # Poziomy sprzedaży

# Parametry dla cech TP/SL
TP_LEVELS = [1, 2]  # Poziomy Take Profit (1%, 2%)
SL_LEVELS = [-1, -2]  # Poziomy Stop Loss (-1%, -2%)

# Nazwy kolumn order book w JSON
SNAPSHOT1_PREFIX = 'snapshot1_'
SNAPSHOT2_PREFIX = 'snapshot2_'
DEPTH_SUFFIX = 'depth_'
NOTIONAL_SUFFIX = 'notional_'
TIMESTAMP_SUFFIX = 'timestamp'

# Domyślna nazwa pliku wyjściowego
DEFAULT_OUTPUT_FILENAME = 'orderbook_ohlc_features.feather' 