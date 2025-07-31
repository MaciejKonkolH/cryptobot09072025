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

# --- Lista Cech do Treningu ---
# Wszystkie cechy z feature_calculator_download2 (113 kolumn - 15 etykiet = 98 cech)
# Automatycznie wykrywane z danych, ale tutaj lista oczekiwanych cech
FEATURES = [
    # OHLC cechy (5)
    'open', 'high', 'low', 'close', 'volume',
    
    # Bamboo TA cechy (około 50+)
    'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'macd_signal', 'macd',
    'stoch_k', 'stoch_d', 'adx', 'cci', 'williams_r', 'mfi', 'obv',
    'ema_12', 'ema_26', 'sma_20', 'sma_50', 'sma_200',
    'atr', 'natr', 'trange', 'hlc3', 'typical_price',
    'vwap', 'vwap_upper', 'vwap_lower',
    'supertrend', 'supertrend_direction', 'supertrend_signal',
    'ichimoku_a', 'ichimoku_b', 'ichimoku_base', 'ichimoku_conversion',
    'kst', 'kst_sig', 'kst_diff',
    'tsi', 'tsi_signal', 'tsi_diff',
    'uo', 'uo_bull', 'uo_bear',
    'ao', 'ao_signal',
    'mom', 'mom_signal',
    'roc', 'roc_signal',
    'stoch_rsi_k', 'stoch_rsi_d',
    'wma', 'hma', 'dema', 'tema',
    'kama', 't3', 'trix', 'trix_signal',
    'aroon_up', 'aroon_down', 'aroon_ind',
    'psar', 'psar_up', 'psar_down',
    'bbands_upper', 'bbands_middle', 'bbands_lower',
    'kc_upper', 'kc_middle', 'kc_lower',
    'dc_upper', 'dc_middle', 'dc_lower',
    
    # Orderbook cechy (około 40+)
    'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'buy_sell_ratio_s3',
    'buy_sell_ratio_s4', 'buy_sell_ratio_s5',
    'depth_imbalance_s1', 'depth_imbalance_s2', 'depth_imbalance_s3',
    'depth_imbalance_s4', 'depth_imbalance_s5',
    'spread', 'spread_pct', 'mid_price',
    'bid_volume_s1', 'bid_volume_s2', 'bid_volume_s3', 'bid_volume_s4', 'bid_volume_s5',
    'ask_volume_s1', 'ask_volume_s2', 'ask_volume_s3', 'ask_volume_s4', 'ask_volume_s5',
    'bid_price_s1', 'bid_price_s2', 'bid_price_s3', 'bid_price_s4', 'bid_price_s5',
    'ask_price_s1', 'ask_price_s2', 'ask_price_s3', 'ask_price_s4', 'ask_price_s5',
    'total_bid_volume', 'total_ask_volume', 'total_volume',
    'volume_imbalance', 'price_imbalance',
    'order_flow_imbalance', 'order_flow_trend',
    'market_microstructure_score', 'liquidity_score',
    
    # Hybrid cechy (około 10+)
    'price_volume_trend', 'volume_price_trend',
    'orderbook_price_correlation', 'orderbook_volume_correlation',
    'market_efficiency_ratio', 'price_efficiency_ratio',
    'volume_efficiency_ratio', 'orderbook_efficiency_ratio',
    
    # Relative cechy (około 15+)
    'price_change_1m', 'price_change_5m', 'price_change_15m', 'price_change_30m',
    'volume_change_1m', 'volume_change_5m', 'volume_change_15m', 'volume_change_30m',
    'spread_change_1m', 'spread_change_5m', 'spread_change_15m', 'spread_change_30m',
    'depth_change_1m', 'depth_change_5m', 'depth_change_15m', 'depth_change_30m',
    
    # Advanced cechy (około 20+)
    'volatility_1m', 'volatility_5m', 'volatility_15m', 'volatility_30m',
    'volatility_regime', 'volatility_percentile', 'volatility_persistence',
    'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    'market_regime', 'market_trend_strength', 'market_trend_direction',
    'market_choppiness', 'market_momentum', 'market_efficiency',
    'orderbook_regime', 'orderbook_trend_strength', 'orderbook_trend_direction',
    'orderbook_choppiness', 'orderbook_momentum', 'orderbook_efficiency',
    'volume_regime', 'volume_trend_strength', 'volume_trend_direction',
    'volume_choppiness', 'volume_momentum', 'volume_efficiency'
]

# Opcjonalne filtrowanie po dacie
ENABLE_DATE_FILTER = False
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"

# --- Parametry Podziału Danych ---
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Parametry Modelu XGBoost Multi-Output ---
XGB_N_ESTIMATORS = 400          # Maksymalna liczba drzew
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia
XGB_MAX_DEPTH = 6               # Maksymalna głębokość drzewa
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa
XGB_COLSAMPLE_BYTREE = 0.7      # Procent cech użytych do budowy każdego drzewa
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału
XGB_RANDOM_STATE = 42           # Ziarno losowości dla powtarzalności wyników
XGB_N_JOBS = -1                 # Liczba procesów (-1 = wszystkie dostępne)
XGB_EARLY_STOPPING_ROUNDS = 20  # Zatrzymaj trening, jeśli metryka na zbiorze walidacyjnym nie poprawi się przez 20 rund

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
    """
    return {
        'ohlc_features': ['open', 'high', 'low', 'close', 'volume'],
        'bamboo_ta_features': [
            'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'macd_signal', 'macd',
            'stoch_k', 'stoch_d', 'adx', 'cci', 'williams_r', 'mfi', 'obv',
            'ema_12', 'ema_26', 'sma_20', 'sma_50', 'sma_200',
            'atr', 'natr', 'trange', 'hlc3', 'typical_price',
            'vwap', 'vwap_upper', 'vwap_lower',
            'supertrend', 'supertrend_direction', 'supertrend_signal',
            'ichimoku_a', 'ichimoku_b', 'ichimoku_base', 'ichimoku_conversion',
            'kst', 'kst_sig', 'kst_diff',
            'tsi', 'tsi_signal', 'tsi_diff',
            'uo', 'uo_bull', 'uo_bear',
            'ao', 'ao_signal',
            'mom', 'mom_signal',
            'roc', 'roc_signal',
            'stoch_rsi_k', 'stoch_rsi_d',
            'wma', 'hma', 'dema', 'tema',
            'kama', 't3', 'trix', 'trix_signal',
            'aroon_up', 'aroon_down', 'aroon_ind',
            'psar', 'psar_up', 'psar_down',
            'bbands_upper', 'bbands_middle', 'bbands_lower',
            'kc_upper', 'kc_middle', 'kc_lower',
            'dc_upper', 'dc_middle', 'dc_lower'
        ],
        'orderbook_features': [
            'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'buy_sell_ratio_s3',
            'buy_sell_ratio_s4', 'buy_sell_ratio_s5',
            'depth_imbalance_s1', 'depth_imbalance_s2', 'depth_imbalance_s3',
            'depth_imbalance_s4', 'depth_imbalance_s5',
            'spread', 'spread_pct', 'mid_price',
            'bid_volume_s1', 'bid_volume_s2', 'bid_volume_s3', 'bid_volume_s4', 'bid_volume_s5',
            'ask_volume_s1', 'ask_volume_s2', 'ask_volume_s3', 'ask_volume_s4', 'ask_volume_s5',
            'bid_price_s1', 'bid_price_s2', 'bid_price_s3', 'bid_price_s4', 'bid_price_s5',
            'ask_price_s1', 'ask_price_s2', 'ask_price_s3', 'ask_price_s4', 'ask_price_s5',
            'total_bid_volume', 'total_ask_volume', 'total_volume',
            'volume_imbalance', 'price_imbalance',
            'order_flow_imbalance', 'order_flow_trend',
            'market_microstructure_score', 'liquidity_score'
        ],
        'hybrid_features': [
            'price_volume_trend', 'volume_price_trend',
            'orderbook_price_correlation', 'orderbook_volume_correlation',
            'market_efficiency_ratio', 'price_efficiency_ratio',
            'volume_efficiency_ratio', 'orderbook_efficiency_ratio'
        ],
        'relative_features': [
            'price_change_1m', 'price_change_5m', 'price_change_15m', 'price_change_30m',
            'volume_change_1m', 'volume_change_5m', 'volume_change_15m', 'volume_change_30m',
            'spread_change_1m', 'spread_change_5m', 'spread_change_15m', 'spread_change_30m',
            'depth_change_1m', 'depth_change_5m', 'depth_change_15m', 'depth_change_30m'
        ],
        'advanced_features': [
            'volatility_1m', 'volatility_5m', 'volatility_15m', 'volatility_30m',
            'volatility_regime', 'volatility_percentile', 'volatility_persistence',
            'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            'market_regime', 'market_trend_strength', 'market_trend_direction',
            'market_choppiness', 'market_momentum', 'market_efficiency',
            'orderbook_regime', 'orderbook_trend_strength', 'orderbook_trend_direction',
            'orderbook_choppiness', 'orderbook_momentum', 'orderbook_efficiency',
            'volume_regime', 'volume_trend_strength', 'volume_trend_direction',
            'volume_choppiness', 'volume_momentum', 'volume_efficiency'
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
    'version': '1.0.0',
    'description': 'Moduł treningowy Multi-Output XGBoost dla nowego pipeline\'u (feature_calculator_download2 + labeler4)',
    'author': 'AI Assistant',
    'features': 'Multi-pair training, individual scalers, batch processing'
}

