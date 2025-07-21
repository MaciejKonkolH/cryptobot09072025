"""
Konfiguracja dla nowego, czystego modułu treningowego (`trainer`).
"""
import os
import logging
from training.utils import find_project_root

# ==============================================================================
# 🎯 GŁÓWNE PARAMETRY KONFIGURACYJNE (DO ZMIANY PRZEZ UŻYTKOWNIKA)
# ==============================================================================

# --- Parametry Danych Wejściowych ---
# Lista par, na których model będzie trenowany.
PAIRS = ["BTCUSDT", "ETHUSDT"] 

# Parametry, na podstawie których zostaną znalezione odpowiednie pliki cech.
TAKE_PROFIT_PCT = 2.0
STOP_LOSS_PCT = 1.0
FUTURE_WINDOW = 60
TIMEFRAME = "1m"

# --- Nazewnictwo Plików Wejściowych ---
def get_input_filenames() -> list[str]:
    """Dynamicznie generuje listę nazw plików wejściowych dla wszystkich par."""
    def format_pct(value):
        return str(value).replace('.', '')

    tp_str = format_pct(TAKE_PROFIT_PCT)
    sl_str = format_pct(STOP_LOSS_PCT)
    
    filenames = []
    for pair in PAIRS:
        # Przykład: BTCUSDT_TP20_SL10_FW60_features.feather
        filename = f"{pair}_TP{tp_str}_SL{sl_str}_FW{FUTURE_WINDOW}_features.feather"
        filenames.append(filename)
    return filenames

# --- Ścieżki ---
# Automatyczne wykrywanie głównego katalogu projektu
# Zakładamy, że ten skrypt jest w podkatalogu 'multi_training'
PROJECT_ROOT = find_project_root()
MODULE_DIR = os.path.join(PROJECT_ROOT, 'multi_training')

# Ścieżka do danych wejściowych (z feature_calculator)
INPUT_DIR = os.path.join(PROJECT_ROOT, 'feature_calculator', 'output')
# Generujemy listę pełnych ścieżek do plików wejściowych
INPUT_FILENAMES = get_input_filenames()
INPUT_FILE_PATHS = [os.path.join(INPUT_DIR, fname) for fname in INPUT_FILENAMES]


# Ścieżki wyjściowe
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
SCALER_FILENAME = "scaler.pkl"
MODEL_FILENAME = "model.h5"
METADATA_FILENAME = "metadata.json"
PREDICTIONS_FILENAME = "test_predictions.csv"

# --- Parametry Danych ---
SEQUENCE_LENGTH = 120
FEATURES = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200', 
    'volume_to_ma1440', 'volume_to_ma43200'
]

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = False
START_DATE = "2022-01-01"  # Format: YYYY-MM-DD
END_DATE = "2023-01-01"    # Format: YYYY-MM-DD

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15  # 15% na walidację
TEST_SPLIT = 0.15        # 15% na test
# Reszta (70%) pójdzie na trening

# --- Parametry Skalowania ---
SCALER_TYPE = 'robust'  # 'robust', 'standard', 'minmax'

# --- Parametry Treningu ---
EPOCHS = 50
BATCH_SIZE = 2048

# --- Parametry Architektury Modelu LSTM ---
LSTM_UNITS = [128, 64, 32]
DENSE_UNITS = [64, 32]
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0001

# --- Parametry Callbacków ---
# EarlyStopping: zatrzymaj, jeśli metryka 'val_loss' nie poprawi się przez X epok
EARLY_STOPPING_PATIENCE = 10
# ReduceLROnPlateau: zmniejsz LR, jeśli 'val_loss' nie poprawi się przez X epok
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5  # Współczynnik zmniejszenia LR (new_lr = lr * factor)

# === Parametry Progu Pewności (Confidence Thresholding) ===
# Aktywuje filtrowanie predykcji na podstawie ich "pewności".
# Predykcje z prawdopodobieństwem niższym niż próg dla danej klasy zostaną odrzucone.
ENABLE_CONFIDENCE_THRESHOLDING = True
CONFIDENCE_THRESHOLDS = {
    0: 0.47,  # Próg dla klasy SHORT
    1: 0.30,  # Próg dla klasy HOLD
    2: 0.47   # Próg dla klasy LONG
}

# --- Parametry Balansowania Klas ---
ENABLE_CLASS_BALANCING = True

# --- Parametry Logowania ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'multi_trainer.log' 