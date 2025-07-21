"""
Konfiguracja dla modułu treningowego (`trainer`) z użyciem XGBoost.
"""
import os
import logging
from training.utils import find_project_root

# ==============================================================================
# 🎯 GŁÓWNE PARAMETRY KONFIGURACYJNE (DO ZMIANY PRZEZ UŻYTKOWNIKA)
# ==============================================================================

# --- Ścieżki ---
PROJECT_ROOT = find_project_root()
MODULE_DIR = os.path.join(PROJECT_ROOT, 'training')

# Dane wejściowe pochodzą teraz z modułu `labeler`
INPUT_DIR = os.path.join(PROJECT_ROOT, 'labeler', 'output')
INPUT_FILENAME = "BTCUSDT-1m-futures_features_and_labels_FW-120_SL-040_TP-080.feather" 
INPUT_FILE_PATH = os.path.join(INPUT_DIR, INPUT_FILENAME)

# Ścieżki wyjściowe
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
SCALER_FILENAME = "scaler.pkl"

# --- Lista Cech do Treningu ---
# Wszystkie cechy obliczone przez feature_calculator.py
FEATURES = [
    'bb_width', 
    # 'bb_position', # Usunięta z powodu bardzo niskiej ważności
    # 'rsi_14', # Usunięta z powodu bardzo niskiej ważności
    'macd_hist', 
    'adx_14', 
    'choppiness_index',
    'price_to_ma_60',
    'price_to_ma_240',
    'ma_60_to_ma_240',
    # 'volume_change_norm', # Usunięta z powodu bardzo niskiej ważności
    'price_to_ma_1440',
    'price_to_ma_43200',
    'volume_to_ma_1440',
    'volume_to_ma_43200',
    # --- Nowe cechy dedykowane (v2) ---
    'whipsaw_range_15m',   # Rozpiętość ceny w krótkim oknie (chaos vs nuda)
    'upper_wick_ratio_5m', # Stosunek górnego cienia (pułapki na byki)
    'lower_wick_ratio_5m', # Stosunek dolnego cienia (pułapki na niedźwiedzie)
]

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = False
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Parametry Modelu XGBoost ---
XGB_N_ESTIMATORS = 500          # Maksymalna liczba drzew
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia
XGB_MAX_DEPTH = 5               # Maksymalna głębokość drzewa
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa
XGB_COLSAMPLE_BYTREE = 0.8      # Procent cech użytych do budowy każdego drzewa
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału
XGB_EARLY_STOPPING_ROUNDS = 15  # Zatrzymaj trening, jeśli metryka na zbiorze walidacyjnym nie poprawi się przez 15 rund

# --- Strategiczne Wagi Klas ---
# Ustaw na True, aby aktywować poniższe wagi. Ustaw na False, aby wszystkie
# klasy były traktowane jednakowo (bez ważenia).
ENABLE_CLASS_WEIGHTING = True

# Ustawiamy własne wagi, aby nadać priorytet najważniejszym lekcjom dla modelu.
# Największą wagę dajemy błędom na klasach stratnych (LOSS_*), aby model
# nauczył się ich unikać za wszelką cenę.
CLASS_WEIGHTS = {
    0: 5.0,   # PROFIT_SHORT
    1: 1.0,   # TIMEOUT_HOLD
    2: 5.0,   # PROFIT_LONG
    3: 15.0,  # LOSS_SHORT (kluczowa lekcja!)
    4: 15.0,  # LOSS_LONG (kluczowa lekcja!)
    5: 2.0    # CHAOS_HOLD
}


# --- Parametry Logowania ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'trainer_xgboost.log' 

# Zapisz także wykresy
SAVE_PLOTS = True


# -----------------------------------------------------------------------------
# USTAWIENIA FUNKCJI STRATY (LOSS FUNCTION)
# -----------------------------------------------------------------------------
# Wybierz funkcję straty: 'categorical_crossentropy' lub 'focal_loss'
# 'focal_loss' jest zalecana przy problemach z niezbalansowanymi klasami.
LOSS_FUNCTION = 'categorical_crossentropy'  # Domyślnie standardowa funkcja

# Parametry dla Focal Loss (ignorowane, jeśli LOSS_FUNCTION != 'focal_loss')
# ---
# Gamma (γ): Kontroluje tempo, w jakim łatwe przykłady są "wyciszane".
# Wyższa gamma bardziej skupia model na trudnych, błędnie klasyfikowanych przykładach.
# Wartość 2.0 jest standardowym i dobrym punktem wyjścia.
FOCAL_LOSS_GAMMA = 2.0

# Alpha (α): Kontroluje wagę każdej z klas. Działa podobnie do `class_weights`.
# Ustaw wyższe wartości dla klas, które model ma problemy poprawnie przewidzieć.
# Kolejność: [SHORT, HOLD, LONG]
FOCAL_LOSS_ALPHA = [0.45, 0.1, 0.45]


# -----------------------------------------------------------------------------
# USTAWIENIA BALANSOWANIA DANYCH
# ----------------------------------------------------------------------------- 