"""
Konfiguracja dla modułu walidacji i etykietowania danych.
"""
from pathlib import Path

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "validation_and_labeling"
INPUT_DIR = PROJECT_ROOT / "feature_calculator" / "output" # <-- Zmienione na wyjście z feature_calculator
OUTPUT_DIR = MODULE_DIR / "output"
RAW_VALIDATED_DIR = MODULE_DIR / "raw_validated"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Konfiguracja Wejścia/Wyjścia ---
# Teraz plik wejściowy to plik z cechami
INPUT_FILENAME = "BTCUSDT-1m-futures_features.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME
# Plik wyjściowy będzie miał dodane parametry etykietowania
OUTPUT_FILENAME_TEMPLATE = "{base_name}_TP{tp}_SL{sl}_FW{fw}_labeled.feather"

# --- Parametry labelingu ---
TAKE_PROFIT_PCT = 0.012   # 1.2%
STOP_LOSS_PCT = 0.004     # 0.4%
FUTURE_WINDOW = 240       # 240 minut (4 godziny)

# --- Konfiguracja Logowania ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'labeling_process.log'

# 🛠️ PARAMETRY ZAAWANSOWANE (ZAZWYCZAJ BEZ ZMIAN)
# ==============================================================================

# ===== FORMATOWANIE ETYKIET DLA ZGODNOŚCI Z TRENINGIEM =====
# Tryb zgodności: Włącza specjalne formatowanie etykiet dla różnych frameworków
TRAINING_COMPATIBILITY_MODE = True

# Dostępne formaty:
# - "onehot":           [[1,0,0], [0,1,0], ...] (dla Keras/TF categorical_crossentropy)
# - "sparse_categorical": [0, 1, 2, ...] (dla Keras/TF sparse_categorical_crossentropy)
# - "int8":             [0, 1, 2, ...] (jako int8, kompaktowe)
LABEL_OUTPUT_FORMAT = "sparse_categorical"
LABEL_DTYPE = "int32" # Zgodnie z wymaganiami TensorFlow dla sparse labels

# Dołączanie metadanych do raportu - przydatne do automatyzacji potoków
INCLUDE_TRAINING_METADATA = True

def get_training_metadata():
    """Zwraca słownik z metadanymi o formacie etykiet."""
    return {
        "label_output_format": LABEL_OUTPUT_FORMAT,
        "label_dtype": str(LABEL_DTYPE),
        "label_source": "CompetitiveLabeler",
    } 