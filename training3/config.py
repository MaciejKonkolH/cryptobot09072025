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
INPUT_FILENAME = "ohlc_orderbook_labeled_3class_fw120m_15levels.feather"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# Ścieżki wyjściowe
OUTPUT_DIR = MODULE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
SCALER_FILENAME = "scaler.pkl"

# --- Konfiguracja Multi-Output ---
# Definicja poziomów TP/SL (jedyna tablica źródłowa)
# Zaktualizowane na podstawie rzeczywistych kolumn w pliku
TP_SL_LEVELS = [
    {"tp": 0.6, "sl": 0.2},  # Poziom 1 - label_tp0p6_sl0p2
    {"tp": 0.6, "sl": 0.3},  # Poziom 2 - label_tp0p6_sl0p3
    {"tp": 0.8, "sl": 0.2},  # Poziom 3 - label_tp0p8_sl0p2
    {"tp": 0.8, "sl": 0.3},  # Poziom 4 - label_tp0p8_sl0p3
    {"tp": 0.8, "sl": 0.4},  # Poziom 5 - label_tp0p8_sl0p4
    {"tp": 1.0, "sl": 0.3},  # Poziom 6 - label_tp1_sl0p3
    {"tp": 1.0, "sl": 0.4},  # Poziom 7 - label_tp1_sl0p4
    {"tp": 1.0, "sl": 0.5},  # Poziom 8 - label_tp1_sl0p5
    {"tp": 1.2, "sl": 0.4},  # Poziom 9 - label_tp1p2_sl0p4
    {"tp": 1.2, "sl": 0.5},  # Poziom 10 - label_tp1p2_sl0p5
    {"tp": 1.2, "sl": 0.6},  # Poziom 11 - label_tp1p2_sl0p6
    {"tp": 1.4, "sl": 0.4},  # Poziom 12 - label_tp1p4_sl0p4
    {"tp": 1.4, "sl": 0.5},  # Poziom 13 - label_tp1p4_sl0p5
    {"tp": 1.4, "sl": 0.6},  # Poziom 14 - label_tp1p4_sl0p6
    {"tp": 1.4, "sl": 0.7}   # Poziom 15 - label_tp1p4_sl0p7
]


# --- Konfiguracja predykcji CSV ---
# Indeks modelu używany do generowania pliku CSV z predykcjami testowymi
# Dostępne indeksy: 0-4 (odpowiadają TP_SL_LEVELS)
CSV_PREDICTIONS_MODEL_INDEX = 3  

def get_model_filename():
    """Generuje nazwę pliku modelu z informacją o poziomach TP/SL."""
    # Generowanie skróconych opisów poziomów na podstawie TP_SL_LEVELS
    level_abbreviations = [
        f"tp{level['tp']:.1f}".replace('.', '') + f"_sl{level['sl']:.1f}".replace('.', '')
        for level in TP_SL_LEVELS
    ]
    
    # Połącz wszystkie poziomy w nazwę
    levels_str = "_".join(level_abbreviations)
    return f"models_{levels_str}"

MODEL_FILENAME = get_model_filename()

# Generowanie nazw kolumn z etykietami na podstawie TP_SL_LEVELS
def format_tp_sl_name(tp, sl):
    """Formatuje nazwę kolumny TP/SL zgodnie z rzeczywistym formatem w danych."""
    # Dla TP=1.0 używamy 'tp1', dla TP=2.0 używamy 'tp2'
    if tp == 1.0:
        tp_str = "tp1"
    elif tp == 2.0:
        tp_str = "tp2"
    else:
        tp_str = f"tp{tp:.1f}".replace('.', 'p')
    
    # Dla SL=0.5 używamy 'sl0p5'
    sl_str = f"sl{sl:.1f}".replace('.', 'p')
    
    return f"label_{tp_str}_{sl_str}"

LABEL_COLUMNS = [
    format_tp_sl_name(level["tp"], level["sl"])
    for level in TP_SL_LEVELS
]

# Generowanie opisów poziomów na podstawie TP_SL_LEVELS
TP_SL_LEVELS_DESC = [
    f'TP: {level["tp"]:.1f}%, SL: {level["sl"]:.1f}%'
    for level in TP_SL_LEVELS
]

def get_report_dir(symbol: str):
    d = REPORT_DIR / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d

# --- Lista Cech do Treningu ---
# Wszystkie cechy z feature_calculator_ohlc_snapshot: 18 względnych + 19 zaawansowanych = 37 cech
# Nowe podejście oparte na trendach, względnych wartościach i zaawansowanych wskaźnikach
FEATURES = [
    # 1. CECHY TRENDU CENY (5 cech) - WZGLĘDNE
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
    'price_strength', 'price_consistency_score',
    
    # 2. CECHY POZYCJI CENY (4 cechy) - WZGLĘDNE
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    
    # 3. CECHY VOLUME (5 cech) - WZGLĘDNE
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling',
    'volume_price_correlation', 'volume_momentum',
    
    # 4. CECHY ORDERBOOK (4 cechy) - WZGLĘDNE
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    
    # 5. MARKET REGIME (5 cech) - ZAAWANSOWANE
    'market_trend_strength', 'market_trend_direction', 'market_choppiness',
    'bollinger_band_width', 'market_regime',
    
    # 6. VOLATILITY CLUSTERING (6 cech) - ZAAWANSOWANE
    'volatility_regime', 'volatility_percentile', 'volatility_persistence',
    'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    
    # 7. ORDER BOOK IMBALANCE (8 cech) - ZAAWANSOWANE
    'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
    'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
    'order_flow_imbalance', 'order_flow_trend'
]

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = False
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Parametry Modelu XGBoost Multi-Output ---
XGB_N_ESTIMATORS = 400          # Maksymalna liczba drzew (zwiększone dla 37 cech)
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia
XGB_MAX_DEPTH = 6               # Maksymalna głębokość drzewa (zwiększone dla większej liczby cech)
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa
XGB_COLSAMPLE_BYTREE = 0.7      # Procent cech użytych do budowy każdego drzewa (zmniejszone dla 37 cech)
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału
XGB_RANDOM_STATE = 42           # Ziarno losowości dla powtarzalności wyników
XGB_N_JOBS = -1                 # Liczba procesów (-1 = wszystkie dostępne)
XGB_EARLY_STOPPING_ROUNDS = 20  # Zatrzymaj trening, jeśli metryka na zbiorze walidacyjnym nie poprawi się przez 20 rund

# --- Balansowanie Klas ---
ENABLE_CLASS_BALANCING = False   # Wyłączone - bez balansowania

# Wagi klas dla 3-klasowego systemu: 0=LONG, 1=SHORT, 2=NEUTRAL
# WŁĄCZONE - wagi wpływają na trening modelu
CLASS_WEIGHTS = {
    0: 2.0,   # LONG - wyższa waga (3x)
    1: 2.0,   # SHORT - wyższa waga (3x)
    2: 1.0    # NEUTRAL - standardowa waga (1x)
}

# Włącz używanie wag w treningu
ENABLE_CLASS_WEIGHTS_IN_TRAINING = True

def detect_available_features(df_columns):
    """
    Automatycznie wykrywa dostępne cechy w danych.
    Zwraca tylko te cechy, które rzeczywiście istnieją w danych.
    """
    available_features = []
    missing_features = []
    
    for feature in FEATURES:
        if feature in df_columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    if missing_features:
        logger = logging.getLogger(__name__)
        logger.warning(f"Brakuje {len(missing_features)} cech w danych: {missing_features}")
    
    return available_features

def get_feature_groups():
    """
    Zwraca grupy cech dla lepszej organizacji i raportowania.
    """
    return {
        'relative_features': [
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
            'price_strength', 'price_consistency_score',
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling',
            'volume_price_correlation', 'volume_momentum',
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum'
        ],
        'market_regime_features': [
            'market_trend_strength', 'market_trend_direction', 'market_choppiness',
            'bollinger_band_width', 'market_regime'
        ],
        'volatility_features': [
            'volatility_regime', 'volatility_percentile', 'volatility_persistence',
            'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure'
        ],
        'imbalance_features': [
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
            'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
            'order_flow_imbalance', 'order_flow_trend'
        ]
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

# Sterowanie logowaniem ewaluacji do konsoli (INFO). Domyślnie wyłączone –
# pełna ewaluacja trafia do raportu markdown.
EVAL_LOG_TO_CONSOLE = False
# Wyłączanie logów metryk walidacyjnych (po treningu); raporty pozostają zapisywane do plików
VALIDATION_LOG_TO_CONSOLE = False

