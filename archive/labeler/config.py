"""
Konfiguracja dla nowego modułu etykietowania (`labeler`)
"""
import os
from pathlib import Path
import logging

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "labeler"
INPUT_DIR = PROJECT_ROOT / "feature_calculator" / "output" # <-- Zmienione
OUTPUT_DIR = MODULE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Wejścia ---
# Teraz plik wejściowy to plik z cechami z poprzedniego kroku
INPUT_FILENAME = "BTCUSDT-1m-futures_features.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# --- Konfiguracja Wyjścia ---
# Szablon dla nazwy pliku wyjściowego
OUTPUT_FILENAME_TEMPLATE = "{base_name}_features_and_labels_FW-{fw:03d}_SL-{sl}_TP-{tp}.feather"

# --- Parametry labelingu ---
FUTURE_WINDOW_MINUTES = 120 # Horyzont czasowy (w minutach) do sprawdzania TP/SL
LONG_TP_PCT = 0.8
LONG_SL_PCT = 0.4
# Poniższe nie są jeszcze używane w logice, ale są dla spójności
SHORT_TP_PCT = 0.8 
SHORT_SL_PCT = 0.4

# --- Nazewnictwo Plików ---
# Ta funkcja nie jest już potrzebna, nazwa generowana jest w main.py
# def get_output_filename(pair_name="BTCUSDT"):
#     """Generuje nazwę pliku wyjściowego na podstawie parametrów."""
#     def format_pct(value):
#         return str(value).replace('.', '')
    
#     tp_str = format_pct(LONG_TP_PCT)
#     sl_str = format_pct(LONG_SL_PCT)
#     fw_str = FUTURE_WINDOW_MINUTES
    
#     return f"{pair_name}_TP{tp_str}_SL{sl_str}_FW{fw_str}_labeled"

# --- Mapowanie Etykiet ---
# 'SHORT' (0), 'HOLD' (1), 'LONG' (2)
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
LOG_FILENAME = 'labeler.log'
