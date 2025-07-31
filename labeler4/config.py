"""
Konfiguracja dla modułu etykietowania (`labeler4`).
Dostosowany do nowego pipeline z feature_calculator_download2 (113 kolumn).
System 3-klasowy: LONG, SHORT, NEUTRAL
Obsługuje etykietowanie pojedynczej pary i wszystkich par na raz.
"""
import os
import sys
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "labeler4"

# Lista par z konfiguracji orderbook
PAIRS = [
    "ETHUSDT", "BCHUSDT", "XRPUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT", 
    "LINKUSDT", "XLMUSDT", "ADAUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT", 
    "XTZUSDT", "ATOMUSDT", "BNBUSDT", "ONTUSDT", "IOTAUSDT", "BATUSDT", 
    "VETUSDT", "NEOUSDT"
]

# --- Konfiguracja Wejścia ---
# Używamy plików z cechami z feature_calculator_download2
INPUT_DIR = PROJECT_ROOT / "feature_calculator_download2" / "output"
INPUT_FILENAME_TEMPLATE = "features_{symbol}.feather"

# --- Konfiguracja Wyjścia ---
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'labeler4.log'

# --- Parametry Etykietowania ---
# Okno przyszłości w minutach (jak długo czekamy na TP/SL)
FUTURE_WINDOW_MINUTES = 120  # 2 godziny

# Poziomy TP/SL w procentach [(TP%, SL%), ...]
TP_SL_LEVELS = [
    (0.6, 0.2),
    (0.6, 0.3),
    (0.8, 0.2),
    (0.8, 0.3),
    (0.8, 0.4),
    (1.0, 0.3),
    (1.0, 0.4),
    (1.0, 0.5),
    (1.2, 0.4),
    (1.2, 0.5),
    (1.2, 0.6),  
    (1.4, 0.4),   
    (1.4, 0.5),   
    (1.4, 0.6),   
    (1.4, 0.7),   
]

# Mapowanie etykiet 3-klasowe na liczby całkowite
LABEL_MAPPING = {
    'LONG': 0,      # Rynkowy trend wzrostowy
    'SHORT': 1,     # Rynkowy trend spadkowy
    'NEUTRAL': 2    # Brak wyraźnego trendu
}

# Odwrotne mapowanie (liczba -> nazwa)
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# --- Konfiguracja Wyjścia ---
OUTPUT_FILENAME_TEMPLATE = "labeled_{symbol}.feather"
SAVE_CSV_COPY = False  # Czy zapisać kopię CSV

def get_level_suffix(tp_pct: float, sl_pct: float) -> str:
    """Generuje suffix dla kolumny etykiety na podstawie poziomów TP/SL."""
    return f"tp{tp_pct:.1f}_sl{sl_pct:.1f}".replace('.0', '').replace('.', 'p')

def get_all_label_columns() -> list:
    """Zwraca listę wszystkich kolumn z etykietami."""
    label_columns = []
    for tp_pct, sl_pct in TP_SL_LEVELS:
        suffix = get_level_suffix(tp_pct, sl_pct)
        label_columns.append(f"label_{suffix}")
    return label_columns

def get_input_file_path(symbol: str) -> Path:
    """Zwraca ścieżkę do pliku wejściowego dla danej pary."""
    return INPUT_DIR / INPUT_FILENAME_TEMPLATE.format(symbol=symbol)

def get_output_file_path(symbol: str) -> Path:
    """Zwraca ścieżkę do pliku wyjściowego dla danej pary."""
    return OUTPUT_DIR / OUTPUT_FILENAME_TEMPLATE.format(symbol=symbol)

# --- Informacje o module ---
MODULE_INFO = {
    'name': 'labeler4',
    'version': '1.0.0',
    'description': 'Moduł etykietowania 3-klasowy dla nowego pipeline (feature_calculator_download2)',
    'input_format': 'feather (113 kolumn)',
    'output_format': 'feather + CSV',
    'labeling_strategy': '3-class directional (LONG/SHORT/NEUTRAL)',
    'future_window': f'{FUTURE_WINDOW_MINUTES} minut',
    'tp_sl_levels': len(TP_SL_LEVELS),
    'label_classes': 3,
    'supported_pairs': len(PAIRS),
} 