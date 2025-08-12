"""
Konfiguracja dla modułu treningowego (`training4`) z użyciem XGBoost Multi-Output.
Dostosowany do nowego pipeline'u z feature_calculator_download2 i labeler4.
Obsługuje trening dla pojedynczej pary i wszystkich par na raz.
"""
import os
import logging
from pathlib import Path

# ==============================================================================
# 🎯 GŁÓWNE PARAMETRY KONFIGURACYJNE (DO ZMIANY PRZEZ UŻYTKOWNIKA)
# ==============================================================================

# --- Ścieżki ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = PROJECT_ROOT / "training4"

# Dane wejściowe pochodzą z modułu `labeler4` (po przetworzeniu przez feature_calculator_download2)
INPUT_DIR = PROJECT_ROOT / "labeler4" / "output"
INPUT_FILENAME_TEMPLATE = "labeled_{symbol}.feather"  # Template dla różnych par

# Ścieżki wyjściowe
OUTPUT_DIR = MODULE_DIR / "output"
MODELS_BASE_DIR = OUTPUT_DIR / "models"
REPORTS_BASE_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

# Lista par z konfiguracji orderbook (skopiowana z labeler4)
PAIRS = [
    "ETHUSDT", "BCHUSDT", "XRPUSDT", "LTCUSDT", "TRXUSDT", "ETCUSDT",
    "LINKUSDT", "XLMUSDT", "ADAUSDT", "XMRUSDT", "DASHUSDT", "ZECUSDT",
    "XTZUSDT", "ATOMUSDT", "BNBUSDT", "ONTUSDT", "IOTAUSDT", "BATUSDT",
    "VETUSDT", "NEOUSDT"
]

# --- Konfiguracja Multi-Output ---
# Definicja poziomów TP/SL (jedyna tablica źródłowa)
# Zaktualizowane na podstawie rzeczywistych kolumn w pliku labeler4
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
# Dostępne indeksy: 0-14 (odpowiadają TP_SL_LEVELS)
CSV_PREDICTIONS_MODEL_INDEX = 3  

def get_model_filename(symbol=None):
    """Generuje nazwę pliku modelu z informacją o poziomach TP/SL."""
    # Generowanie skróconych opisów poziomów na podstawie TP_SL_LEVELS
    level_abbreviations = [
        f"tp{level['tp']:.1f}".replace('.', '') + f"_sl{level['sl']:.1f}".replace('.', '')
        for level in TP_SL_LEVELS
    ]
    
    # Połącz wszystkie poziomy w nazwę
    levels_str = "_".join(level_abbreviations)
    
    if symbol:
        return f"models_{symbol}_{levels_str}"
    else:
        return f"models_{levels_str}"

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

# --- Konfiguracja Cech do Treningu ---
# Opcja wyboru między pełnymi 71 cechami a podstawowymi 37 cechami z training3
USE_BASIC_FEATURES_ONLY = True   # True = tylko 37 cech z training3 (jak w training3), False = wszystkie 71 cech

# Podstawowe cechy z training3 (37 cech)
BASIC_FEATURES = [
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

# Wszystkie cechy z feature_calculator_download2 (71 cech)
EXTENDED_FEATURES = [
    # Cechy trendu ceny (5 cech) - PODSTAWOWE
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
    
    # Cechy pozycji ceny (4 cechy) - PODSTAWOWE
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    
    # Cechy wolumenu (5 cech) - PODSTAWOWE
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
    
    # Cechy orderbook (4 cechy) - PODSTAWOWE
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    
    # Market regime (5 cech) - PODSTAWOWE
    'market_trend_strength', 'market_trend_direction', 'market_choppiness',
    'bollinger_band_width', 'market_regime',
    
    # Volatility clustering (6 cech) - PODSTAWOWE
    'volatility_regime', 'volatility_percentile', 'volatility_persistence',
    'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    
    # Order book imbalance (8 cech) - PODSTAWOWE
    'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
    'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
    'order_flow_imbalance', 'order_flow_trend',
    
    # Dodatkowe cechy OHLC (12) - DODATKOWE
    'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'adx_14',
    'price_to_ma_60', 'price_to_ma_240', 'ma_60_to_ma_240', 'price_to_ma_1440',
    'volume_change_norm', 'upper_wick_ratio_5m', 'lower_wick_ratio_5m',
    
    # Dodatkowe cechy bamboo_ta (6) - DODATKOWE
    'stoch_k', 'stoch_d', 'cci', 'williams_r', 'mfi', 'trange',
    
    # Dodatkowe cechy orderbook (6) - DODATKOWE
    'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'imbalance_s1', 'imbalance_s2',
    'spread_pct', 'price_imbalance',
    
    # Cechy hybrydowe (10) - DODATKOWE
    'market_microstructure_score', 'liquidity_score', 'depth_price_corr',
    'pressure_volume_corr', 'hour_of_day', 'day_of_week', 'price_momentum',
    'market_efficiency_ratio', 'price_efficiency_ratio', 'volume_efficiency_ratio'
]

# Automatyczny wybór cech na podstawie ustawienia
FEATURES = BASIC_FEATURES if USE_BASIC_FEATURES_ONLY else EXTENDED_FEATURES

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = True
START_DATE = "2023-01-31 00:00:00"  # Dokładnie jak w training3
END_DATE = "2025-06-30 23:59:59"    # Dokładnie jak w training3 (koniec testu)

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Parametry Modelu XGBoost Multi-Output ---
XGB_N_ESTIMATORS = 400          # Maksymalna liczba drzew
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia (jak w training3)
XGB_MAX_DEPTH = 6               # Maksymalna głębokość drzewa (jak w training3)
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa (jak w training3)
XGB_COLSAMPLE_BYTREE = 0.7      # Procent cech użytych do budowy każdego drzewa (jak w training3)
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału (jak w training3)
XGB_RANDOM_STATE = 42           # Ziarno losowości dla powtarzalności wyników
XGB_N_JOBS = -1                 # Liczba procesów (-1 = wszystkie dostępne)
XGB_EARLY_STOPPING_ROUNDS = 20  # Zatrzymaj trening, jeśli metryka na zbiorze walidacyjnym nie poprawi się przez 20 rund

# --- Parametry Regularyzacji (przeciwko overfitting) ---
# WYŁĄCZONE - jak w training3 (brak regularyzacji L1/L2)
XGB_REG_ALPHA = 0.0             # L1 regularization (Lasso) - wyłączone
XGB_REG_LAMBDA = 0.0            # L2 regularization (Ridge) - wyłączone
XGB_MIN_CHILD_WEIGHT = 1        # Minimalna suma wag w liściu - zmniejszone (jak w training3)

# --- Balansowanie Klas ---
ENABLE_CLASS_BALANCING = False   # Wyłączone - bez balansowania

# Wagi klas dla 3-klasowego systemu: 0=LONG, 1=SHORT, 2=NEUTRAL
# WŁĄCZONE - wagi wpływają na trening modelu
CLASS_WEIGHTS = {
    0: 2.0,   # LONG - wyższa waga (2x)
    1: 2.0,   # SHORT - wyższa waga (2x)
    2: 1.0    # NEUTRAL - standardowa waga (1x)
}

# Włącz używanie wag w treningu
ENABLE_CLASS_WEIGHTS_IN_TRAINING = True

def get_input_file_path(symbol):
    """Generuje ścieżkę do pliku wejściowego dla danej pary."""
    return INPUT_DIR / INPUT_FILENAME_TEMPLATE.format(symbol=symbol)

def get_model_dir(symbol):
    """Generuje ścieżkę do katalogu modeli dla danej pary."""
    return MODELS_BASE_DIR / symbol

def get_report_dir(symbol):
    """Generuje ścieżkę do katalogu raportów dla danej pary."""
    return REPORTS_BASE_DIR / symbol

def get_scaler_path(symbol):
    """Generuje ścieżkę do pliku scalera dla danej pary."""
    return get_model_dir(symbol) / "scaler.pkl"

def get_metadata_path(symbol):
    """Generuje ścieżkę do pliku metadata dla danej pary."""
    return get_model_dir(symbol) / "metadata.json"

def get_models_index_path(symbol):
    """Generuje ścieżkę do pliku index modeli dla danej pary."""
    return get_model_dir(symbol) / "models_index.json"

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
    Dostosowane do wybranej konfiguracji cech.
    """
    if USE_BASIC_FEATURES_ONLY:
        # Tryb podstawowy (37 cech z training3)
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
    else:
        # Tryb rozszerzony (71 cech)
        return {
            'basic_features': [
                # Cechy trendu ceny (5 cech)
                'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
                # Cechy pozycji ceny (4 cechy)
                'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
                # Cechy wolumenu (5 cech)
                'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
                # Cechy orderbook (4 cechy)
                'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
                # Market regime (5 cech)
                'market_trend_strength', 'market_trend_direction', 'market_choppiness',
                'bollinger_band_width', 'market_regime',
                # Volatility clustering (6 cech)
                'volatility_regime', 'volatility_percentile', 'volatility_persistence',
                'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
                # Order book imbalance (8 cech)
                'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
                'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
                'order_flow_imbalance', 'order_flow_trend'
            ],
            'additional_features': [
                # Dodatkowe cechy OHLC (12)
                'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'adx_14',
                'price_to_ma_60', 'price_to_ma_240', 'ma_60_to_ma_240', 'price_to_ma_1440',
                'volume_change_norm', 'upper_wick_ratio_5m', 'lower_wick_ratio_5m',
                # Dodatkowe cechy bamboo_ta (6)
                'stoch_k', 'stoch_d', 'cci', 'williams_r', 'mfi', 'trange',
                # Dodatkowe cechy orderbook (6)
                'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'imbalance_s1', 'imbalance_s2',
                'spread_pct', 'price_imbalance',
                # Cechy hybrydowe (10)
                'market_microstructure_score', 'liquidity_score', 'depth_price_corr',
                'pressure_volume_corr', 'hour_of_day', 'day_of_week', 'price_momentum',
                'market_efficiency_ratio', 'price_efficiency_ratio', 'volume_efficiency_ratio'
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
LOG_FILENAME = 'trainer_4class.log' 

# Zapisz także wykresy
SAVE_PLOTS = True

# --- Metryki ewaluacji ---
# Wszystkie klasy mają równe znaczenie
PRIMARY_METRICS = ['precision', 'recall', 'f1-score']
FOCUS_CLASSES = [0, 1, 2]  # LONG, SHORT, NEUTRAL

# --- Informacje o module ---
MODULE_INFO = {
    'name': 'training4',
    'version': '1.1.0',
    'description': 'Moduł treningowy Multi-Output XGBoost dla nowego pipeline\'u (feature_calculator_download2 + labeler4) - ZOPTYMALIZOWANY Z PARAMETRAMI TRAINING3',
    'author': 'AI Assistant',
    'features': 'Multi-pair training, individual scalers, batch processing, training3 parameters'
}

