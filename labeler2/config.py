"""
Konfiguracja dla nowego modułu etykietowania (`labeler2`)
Dostosowany do nowego formatu danych z feature_calculator_snapshot
"""
import os
from pathlib import Path
import logging

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "labeler2"
INPUT_DIR = PROJECT_ROOT / "feature_calculator_snapshot" / "output" # Zmieniono na nowy moduł
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Wejścia ---
# Teraz plik wejściowy to plik z cechami z feature_calculator_snapshot
INPUT_FILENAME = "orderbook_ohlc_features.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
# Szablon dla nazwy pliku wyjściowego z wieloma poziomami TP/SL
OUTPUT_FILENAME_TEMPLATE = "{base_name}_labels_FW-{fw:03d}_levels-{levels_count}.feather"

# --- Parametry labelingu ---
FUTURE_WINDOW_MINUTES = 120 # Horyzont czasowy (w minutach) do sprawdzania TP/SL

# NOWE: Lista poziomów TP/SL jako pary [TP%, SL%]
# Każda para będzie generować osobne kolumny etykiet
# DOSTOSOWANE do rzeczywistej zmienności danych (~0.4% dziennie)
TP_SL_LEVELS = [
    [0.8, 0.4],    # TP: 0.2% (~135 pkt), SL: 0.1% (~67 pkt)
    [0.6, 0.3], # TP: 0.15% (~100 pkt), SL: 0.075% (~50 pkt)
    [0.4, 0.2]    # TP: 0.1% (~67 pkt), SL: 0.05% (~33 pkt)
]

# Stare parametry - zachowane dla kompatybilności wstecznej (używany pierwszy poziom)
LONG_TP_PCT = TP_SL_LEVELS[0][0]
LONG_SL_PCT = TP_SL_LEVELS[0][1]
SHORT_TP_PCT = TP_SL_LEVELS[0][0]  # Symmetric
SHORT_SL_PCT = TP_SL_LEVELS[0][1]  # Symmetric

# --- Mapowanie Etykiet ---
# 'SHORT' (0), 'HOLD' (1), 'LONG' (2), itd.
LABEL_MAPPING = {
    'PROFIT_SHORT': 0,
    'TIMEOUT_HOLD': 1,
    'PROFIT_LONG': 2,
    'LOSS_SHORT': 3,
    'LOSS_LONG': 4,
    'CHAOS_HOLD': 5,
}

# --- Ustawienia Logiki ---
# Czy wiersze, dla których nie można obliczyć etykiety (np. ostatnie N świec),
# powinny być wypełnione wartością HOLD?
# False = zostaną jako puste (NaN) - zalecane do diagnostyki.
# True = zostaną wypełnione jako HOLD (1).
FILL_UNCOMPUTED_WITH_HOLD = True

# --- Opcje Zapisu ---
INCLUDE_OHLCV_IN_OUTPUT = True
SAVE_CSV_COPY = False # Zmieniono z SAVE_TO_CSV dla spójności

# --- Logowanie ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'labeler2.log'

# --- Funkcje pomocnicze ---
def get_level_suffix(tp_pct: float, sl_pct: float) -> str:
    """Generuje sufiks dla nazwy kolumny na podstawie poziomów TP/SL."""
    tp_str = str(tp_pct).replace('.', '')
    sl_str = str(sl_pct).replace('.', '')
    return f"tp{tp_str}_sl{sl_str}"

def get_all_label_columns() -> list:
    """Zwraca listę wszystkich nazw kolumn z etykietami dla wszystkich poziomów."""
    columns = []
    for tp_pct, sl_pct in TP_SL_LEVELS:
        suffix = get_level_suffix(tp_pct, sl_pct)
        columns.append(f"label_{suffix}")
    return columns
