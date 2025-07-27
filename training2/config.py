"""
Konfiguracja dla modułu treningowego (`training2`) z użyciem XGBoost Multi-Output.
Dostosowany do nowego formatu danych z 3 poziomami TP/SL.
"""
import os
import logging
from training2.utils import find_project_root

# ==============================================================================
# 🎯 GŁÓWNE PARAMETRY KONFIGURACYJNE (DO ZMIANY PRZEZ UŻYTKOWNIKA)
# ==============================================================================

# --- Ścieżki ---
PROJECT_ROOT = find_project_root()
MODULE_DIR = os.path.join(PROJECT_ROOT, 'training2')

# Dane wejściowe pochodzą teraz z modułu `labeler2`
INPUT_DIR = os.path.join(PROJECT_ROOT, 'labeler2', 'output')
INPUT_FILENAME = "orderbook_ohlc_labels_FW-120_levels-3.feather" 
INPUT_FILE_PATH = os.path.join(INPUT_DIR, INPUT_FILENAME)

# Ścieżki wyjściowe
OUTPUT_DIR = os.path.join(MODULE_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
SCALER_FILENAME = "scaler.pkl"
MODEL_FILENAME = "model.pkl"  # Jeden model dla wszystkich poziomów

# --- Konfiguracja Multi-Output ---
# Nazwy kolumn z etykietami dla różnych poziomów TP/SL
LABEL_COLUMNS = [
    'label_tp08_sl04',     # TP: 0.8%, SL: 0.4%
    'label_tp06_sl03',     # TP: 0.6%, SL: 0.3%
    'label_tp04_sl02'      # TP: 0.4%, SL: 0.2%
]

# Opis poziomów dla raportowania
TP_SL_LEVELS_DESC = [
    "TP: 0.8%, SL: 0.4%",
    "TP: 0.6%, SL: 0.3%", 
    "TP: 0.4%, SL: 0.2%"
]

# --- Lista Cech do Treningu ---
# Wszystkie cechy obliczone przez feature_calculator_snapshot
FEATURES = [
    # Tradycyjne wskaźniki techniczne
    'bb_width', 
    'bb_position',
    'rsi_14',
    'macd_hist', 
    'adx_14', 
    'choppiness_index',
    
    # Średnie kroczące i relacje cenowe
    'ma_60', 'ma_240', 'ma_1440',
    'price_to_ma_60',
    'price_to_ma_240', 
    'price_to_ma_1440',
    'ma_60_to_ma_240',
    
    # Cechy volume i dedykowane
    'volume_change_norm',
    'whipsaw_range_15m',
    'upper_wick_ratio_5m',
    'lower_wick_ratio_5m',
    
    # Cechy Order Book - podstawowe
    'buy_sell_ratio_s1', 'buy_sell_ratio_s2',
    'imbalance_s1', 'imbalance_s2',
    'pressure_change',
    
    # Cechy Order Book - głębokość poziomów
    'tp_1pct_depth_s1', 'tp_2pct_depth_s1', 'sl_1pct_depth_s1',
    'tp_sl_ratio_1pct', 'tp_sl_ratio_2pct',
    
    # Cechy Order Book - dynamiczne
    'total_depth_change', 'notional_change',
    'depth_1_change', 'depth_neg1_change',
    
    # Cechy Order Book - historyczne
    'depth_trend', 'ob_volatility', 'avg_depth_30',
    'depth_anomaly', 'buy_pressure_trend', 'sell_pressure_trend',
    
    # Cechy Order Book - korelacyjne
    'depth_price_corr', 'pressure_volume_corr',
    
    # Cechy Order Book - momentum
    'depth_acceleration', 'ob_momentum', 'breakout_signal',
    
    # Cechy Order Book - koncentracja
    'depth_concentration', 'level_asymmetry',
    'near_level_dominance', 'far_level_dominance', 'ob_stability'
]

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = False
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Parametry Modelu XGBoost Multi-Output ---
XGB_N_ESTIMATORS = 500          # Maksymalna liczba drzew
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia
XGB_MAX_DEPTH = 5               # Maksymalna głębokość drzewa
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa
XGB_COLSAMPLE_BYTREE = 0.8      # Procent cech użytych do budowy każdego drzewa
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału
XGB_EARLY_STOPPING_ROUNDS = 15  # Zatrzymaj trening, jeśli metryka na zbiorze walidacyjnym nie poprawi się przez 15 rund

# --- Strategiczne Wagi Klas dla każdego poziomu TP/SL ---
# Każdy poziom ma te same klasy, ale możemy je ważyć różnie
ENABLE_CLASS_WEIGHTING = True

# Wagi klas: 0=PROFIT_SHORT, 1=TIMEOUT_HOLD, 2=PROFIT_LONG, 3=LOSS_SHORT, 4=LOSS_LONG, 5=CHAOS_HOLD
CLASS_WEIGHTS = {
    0: 5.0,   # PROFIT_SHORT - ważne sygnały
    1: 1.0,   # TIMEOUT_HOLD - neutralne
    2: 5.0,   # PROFIT_LONG - ważne sygnały  
    3: 15.0,  # LOSS_SHORT - krytyczne do nauczenia
    4: 15.0,  # LOSS_LONG - krytyczne do nauczenia
    5: 2.0    # CHAOS_HOLD - umiarkowanie ważne
}

# --- Mapowanie klas dla raportowania ---
CLASS_LABELS = {
    0: 'PROFIT_SHORT',
    1: 'TIMEOUT_HOLD', 
    2: 'PROFIT_LONG',
    3: 'LOSS_SHORT',
    4: 'LOSS_LONG',
    5: 'CHAOS_HOLD'
}

# --- Parametry Logowania ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'trainer_multioutput.log' 

# Zapisz także wykresy
SAVE_PLOTS = True

# --- Metryki ewaluacji ---
# Skupiamy się na skuteczności SHORT i LONG jak żądał użytkownik
PRIMARY_METRICS = ['precision', 'recall', 'f1-score']
FOCUS_CLASSES = [0, 2]  # PROFIT_SHORT, PROFIT_LONG 