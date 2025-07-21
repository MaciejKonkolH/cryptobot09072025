"""
Konfiguracja dla modułu obliczającego cechy (`feature_calculator`).
"""
import os
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "feature_calculator"

# --- Konfiguracja Wejścia ---
# Ścieżka do katalogu z danymi z modułu `labeler`
INPUT_DIR = PROJECT_ROOT / "validation_and_labeling" / "raw_validated"
# Nazwa konkretnego pliku, który ma zostać przetworzony
INPUT_FILENAME = "BTCUSDT-1m-futures.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs" # Zapisuj logi w podkatalogu output

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'feature_calculator.log'

# --- Parametry Obliczeń Cech ---
# Okna dla średnich kroczących
MA_WINDOWS = [1440, 43200]

# Domyślna nazwa pliku wejściowego (może być nadpisana przez argumenty CLI)
# Oczekujemy, że plik wejściowy będzie miał format <para>_<okres>.feather
# Przykład: 'BTCUSDT_1m.feather'
# Tutaj możemy zostawić pustą wartość lub przykładową
DEFAULT_INPUT_FILENAME = ''

# Nazwy kolumn
COL_OPEN = 'open'
COL_CLOSE = 'close'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_VOLUME = 'volume'
COL_TARGET = 'label'

# Usunięto COL_LABEL, ponieważ ten moduł już nie obsługuje etykiet 