"""
Konfiguracja dla modułu etykietowania (`labeler3`).
Dostosowany do nowego formatu danych z feature_calculator_ohlc_snapshot (85 kolumn).
System 3-klasowy: LONG, SHORT, NEUTRAL
"""
import os
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "labeler3"

# --- Konfiguracja Wejścia ---
# Używamy nowego pliku z cechami z feature_calculator_ohlc_snapshot
INPUT_DIR = PROJECT_ROOT / "feature_calculator_ohlc_snapshot" / "output"
INPUT_FILENAME = "ohlc_orderbook_features.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'labeler3.log'

# --- Parametry Etykietowania ---
# Okno przyszłości w minutach (jak długo czekamy na TP/SL)
FUTURE_WINDOW_MINUTES = 60  # 1 godzina

# Poziomy TP/SL w procentach [(TP%, SL%), ...]
TP_SL_LEVELS = [
    (0.8, 0.2),   # TP 0.8%, SL 0.4%
    (0.6, 0.3),   # TP 0.8%, SL 0.4%
    (0.8, 0.4),   # TP 0.8%, SL 0.4%
    (1.0, 0.5),   # TP 1.0%, SL 0.5%
    (1.2, 0.6),   # TP 1.2%, SL 0.6%
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
OUTPUT_FILENAME_TEMPLATE = "{base_name}_labeled_3class_fw{fw}m_{levels_count}levels.feather"
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

# --- Informacje o module ---
MODULE_INFO = {
    'name': 'labeler3',
    'version': '2.0.0',
    'description': 'Moduł etykietowania 3-klasowy dla danych z feature_calculator_ohlc_snapshot',
    'input_format': 'feather (85 kolumn)',
    'output_format': 'feather + CSV',
    'labeling_strategy': '3-class directional (LONG/SHORT/NEUTRAL)',
    'future_window': f'{FUTURE_WINDOW_MINUTES} minut',
    'tp_sl_levels': len(TP_SL_LEVELS),
    'label_classes': 3,
} 