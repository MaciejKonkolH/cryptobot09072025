"""
Konfiguracja dla modułu treningowego (`training3`) z użyciem XGBoost Multi-Output.
Dostosowany do nowego formatu danych z labeler3 (3-klasowe etykiety).
"""
import os
import logging
from pathlib import Path

# ==============================================================================
# 🎯 GŁÓWNE PARAMETRY KONFIGURACYJNE (DO ZMIANY PRZEZ UŻYTKOWNIKA)
# ==============================================================================

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "training3"

# Dane wejściowe pochodzą z modułu `labeler3` (po przetworzeniu przez feature_calculator_ohlc_snapshot z nowymi cechami)
INPUT_DIR = PROJECT_ROOT / "labeler3" / "output"
INPUT_FILENAME = "ohlc_orderbook_labeled_3class_fw60m_5levels.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# Ścieżki wyjściowe
OUTPUT_DIR = MODULE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
SCALER_FILENAME = "scaler.pkl"
MODEL_FILENAME = "model_multioutput.pkl"

# --- Konfiguracja Multi-Output ---
# Nazwy kolumn z etykietami 3-klasowymi z labeler3
LABEL_COLUMNS = [
    'label_tp0p8_sl0p2',     # TP: 0.8%, SL: 0.2%
    'label_tp0p6_sl0p3',     # TP: 0.6%, SL: 0.3%
    'label_tp0p8_sl0p4',     # TP: 0.8%, SL: 0.4%
    'label_tp1_sl0p5',       # TP: 1.0%, SL: 0.5%
    'label_tp1p2_sl0p6'      # TP: 1.2%, SL: 0.6%
]

# Opis poziomów dla raportowania
TP_SL_LEVELS_DESC = [
    "TP: 0.8%, SL: 0.2%",
    "TP: 0.6%, SL: 0.3%", 
    "TP: 0.8%, SL: 0.4%",
    "TP: 1.0%, SL: 0.5%",
    "TP: 1.2%, SL: 0.6%"
]

# --- Lista Cech do Treningu ---
# TYLKO nowe względne cechy (18 sztuk) z idealne_cechy_treningowe.md
# Zastąpiły stare 73 cechy absolutne - nowe podejście oparte na trendach i względnych wartościach
FEATURES = [
    # 1. CECHY TRENDU CENY (5 cech)
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
    'price_strength', 'price_consistency_score',
    
    # 2. CECHY POZYCJI CENY (4 cechy)
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    
    # 3. CECHY VOLUME (5 cech)
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling',
    'volume_price_correlation', 'volume_momentum',
    
    # 4. CECHY ORDERBOOK (4 cechy)
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum'
]

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = False
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Parametry Modelu XGBoost Multi-Output ---
XGB_N_ESTIMATORS = 300          # Maksymalna liczba drzew (zmniejszone z 500 dla 18 cech)
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia
XGB_MAX_DEPTH = 5               # Maksymalna głębokość drzewa
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa
XGB_COLSAMPLE_BYTREE = 0.8      # Procent cech użytych do budowy każdego drzewa
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału
XGB_RANDOM_STATE = 42           # Ziarno losowości dla powtarzalności wyników
XGB_N_JOBS = -1                 # Liczba procesów (-1 = wszystkie dostępne)
XGB_EARLY_STOPPING_ROUNDS = 15  # Zatrzymaj trening, jeśli metryka na zbiorze walidacyjnym nie poprawi się przez 15 rund

# --- Balansowanie Klas ---
ENABLE_CLASS_BALANCING = False   # Wyłączone - bez balansowania

# Wagi klas dla 3-klasowego systemu: 0=LONG, 1=SHORT, 2=NEUTRAL
# WYŁĄCZONE - wszystkie klasy mają równe wagi
CLASS_WEIGHTS = {
    0: 1.0,   # LONG - równe wagi
    1: 1.0,   # SHORT - równe wagi
    2: 1.0    # NEUTRAL - równe wagi
}

# --- Weighted Loss Configuration ---
ENABLE_WEIGHTED_LOSS = False     # WYŁĄCZONE - bez wag
WEIGHTED_LOSS_MULTIPLIER = 1.0   # Neutralny mnożnik
NEUTRAL_WEIGHT_REDUCTION = 1.0   # Brak redukcji wagi

# --- Mapowanie klas dla raportowania ---
CLASS_LABELS = {
    0: 'LONG',
    1: 'SHORT', 
    2: 'NEUTRAL'
}

# --- Parametry Logowania ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILENAME = 'trainer_3class.log' 

# Zapisz także wykresy
SAVE_PLOTS = True

# --- Metryki ewaluacji ---
# Wszystkie klasy mają równe znaczenie
PRIMARY_METRICS = ['precision', 'recall', 'f1-score']
FOCUS_CLASSES = [0, 1, 2]  # LONG, SHORT, NEUTRAL 