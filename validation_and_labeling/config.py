"""
Konfiguracja modułu validation_and_labeling
Wszystkie parametry systemu w jednym miejscu
"""
import os
from pathlib import Path

# ===== ŚCIEŻKI (USTALONE) =====
# Katalogi względem głównego katalogu validation_and_labeling
BASE_DIR = Path(__file__).parent
INPUT_DATA_PATH = BASE_DIR / "input"
OUTPUT_DATA_PATH = BASE_DIR / "output"
REPORTS_PATH = OUTPUT_DATA_PATH / "reports"

# 🆕 RAW VALIDATED DATA EXPORT - NOWA FUNKCJONALNOŚĆ
RAW_VALIDATED_OUTPUT_PATH = BASE_DIR / "raw_validated"
SAVE_RAW_VALIDATED_DATA = True  # Włącz/wyłącz zapis raw validated data

# ===== PARAMETRY COMPETITIVE LABELING (USTALONE) =====
LONG_TP_PCT = 1.0      # Take Profit dla pozycji LONG (%)
LONG_SL_PCT = 0.5      # Stop Loss dla pozycji LONG (%)
SHORT_TP_PCT = 1.0     # Take Profit dla pozycji SHORT (%)
SHORT_SL_PCT = 0.5     # Stop Loss dla pozycji SHORT (%)
FUTURE_WINDOW = 120    # Okno prognozy (minuty)

# ===== PARAMETRY ŚREDNICH KROCZĄCYCH (USTALONE) =====
MA_SHORT_WINDOW = 1440    # Krótka MA (1 dzień w minutach)
MA_LONG_WINDOW = 43200    # Długa MA (1 miesiąc w minutach)

# ===== PARAMETRY WALIDACJI JAKOŚCI FEATURES (USTALONE) =====
MAX_CHANGE_THRESHOLD = 50.0     # Maksymalna zmiana % (ostrzeżenie)
MAX_VOLUME_CHANGE = 1000.0      # Maksymalna zmiana volume % (ostrzeżenie)
MAX_MA_RATIO = 3.0              # Maksymalny stosunek price/MA (ostrzeżenie)

# ===== TIMEFRAME (USTALONE) =====
TIMEFRAME = "1m"

# ===== STRATEGIA BŁĘDÓW (USTALONE) =====
ERROR_STRATEGY = "all_or_nothing"  # Kasuj cały plik po błędzie
OVERWRITE_FILES = True             # Nadpisuj istniejące pliki bez pytania

# ===== KONFIGURACJA LOGOWANIA =====
LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ===== PROGRESS REPORTING =====
PROGRESS_REPORT_EVERY_N_ROWS = 100000  # Co ile wierszy raportować postęp (zwiększone dla wydajności)
PERFORMANCE_METRICS_ENABLED = True     # Czy zbierać metryki wydajności

# ===== WALIDACJA PLIKÓW WEJŚCIOWYCH =====
SUPPORTED_INPUT_FORMATS = ['.feather', '.csv']  # Obsługiwane formaty (.feather priorytet)
REQUIRED_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

# ===== PARAMETRY ALGORYTMU BRIDGE (WYPEŁNIANIE LUK) =====
BRIDGE_NOISE_PCT = 0.01           # Procent szumu dla interpolacji (±0.01%)
BRIDGE_VOLUME_RANDOM_FACTOR = (0.8, 1.2)  # Zakres losowy dla volume

# ===== WYMAGANIA MINIMUM DANYCH =====
MIN_ROWS_REQUIRED = MA_LONG_WINDOW  # Minimum 43,200 wierszy dla pełnej długiej MA

# ===== 🎯 TRAINING COMPATIBILITY CONFIGURATION ===== 
# Nowe parametry dla generowania training-ready output
TRAINING_COMPATIBILITY_MODE = True    # Włącz training-ready output
LABEL_OUTPUT_FORMAT = "sparse_categorical"  # "int8", "onehot", "sparse_categorical"
LABEL_DTYPE = "uint8"                 # "int8", "float32", "int32", "uint8"
INCLUDE_TRAINING_METADATA = True      # Dodaj metadata dla training module
STANDARDIZE_TIMESTAMP_FORMAT = True   # Zunifikuj format timestamp

# Lista dostępnych formatów etykiet
SUPPORTED_LABEL_FORMATS = {
    "int8": "Compact format: [0, 1, 2] as int8",
    "onehot": "One-hot encoding: [[1,0,0], [0,1,0], [0,0,1]] as float32", 
    "sparse_categorical": "Sparse format: [0, 1, 2] as uint8"
}

# ===== NAZEWNICTWO PLIKÓW =====
# Zaktualizowane wzorce nazw plików
if TRAINING_COMPATIBILITY_MODE:
    if LABEL_OUTPUT_FORMAT == "sparse_categorical":
        OUTPUT_FILE_PATTERN = "{pair}_TF-{timeframe}__FW-{future_window}__SL-{sl_formatted}__TP-{tp_formatted}__single_label.feather"
    else:
        OUTPUT_FILE_PATTERN = "{pair}_TF-{timeframe}__FW-{future_window}__SL-{sl_formatted}__TP-{tp_formatted}__training_ready.feather"
else:
    OUTPUT_FILE_PATTERN = "{pair}_TF-{timeframe}__FW-{future_window}__SL-{sl_formatted}__TP-{tp_formatted}__features_labels.feather"

REPORT_FILE_PATTERN = "{pair}_TF-{timeframe}__report.json"

def format_percentage_for_filename(value):
    """
    Formatuje wartość procentową do nazwy pliku
    1.0 -> '010'
    0.5 -> '005'
    2.5 -> '025'
    """
    return f"{int(value * 100):03d}"

def get_output_filename(pair, timeframe="1m"):
    """Generuje nazwę pliku wyjściowego zgodnie z konwencją"""
    sl_formatted = format_percentage_for_filename(LONG_SL_PCT)
    tp_formatted = format_percentage_for_filename(LONG_TP_PCT)
    
    return OUTPUT_FILE_PATTERN.format(
        pair=pair,
        timeframe=timeframe,
        future_window=FUTURE_WINDOW,
        sl_formatted=sl_formatted,
        tp_formatted=tp_formatted
    )

def get_report_filename(pair, timeframe="1m"):
    """Generuje nazwę pliku raportu"""
    return REPORT_FILE_PATTERN.format(
        pair=pair,
        timeframe=timeframe
    )

def validate_training_config():
    """
    🔍 VALIDATE TRAINING COMPATIBILITY CONFIG
    Sprawdza poprawność konfiguracji training compatibility
    """
    errors = []
    
    # Sprawdź format etykiet
    if LABEL_OUTPUT_FORMAT not in SUPPORTED_LABEL_FORMATS:
        errors.append(f"Invalid LABEL_OUTPUT_FORMAT: {LABEL_OUTPUT_FORMAT}. Supported: {list(SUPPORTED_LABEL_FORMATS.keys())}")
    
    # Sprawdź typ danych dla etykiet
    supported_dtypes = ["int8", "float32", "int32", "uint8"]
    if LABEL_DTYPE not in supported_dtypes:
        errors.append(f"Invalid LABEL_DTYPE: {LABEL_DTYPE}. Supported: {supported_dtypes}")
    
    # Sprawdź kompatybilność format + dtype
    if LABEL_OUTPUT_FORMAT == "onehot" and LABEL_DTYPE not in ["float32"]:
        errors.append(f"LABEL_OUTPUT_FORMAT 'onehot' requires LABEL_DTYPE 'float32', got '{LABEL_DTYPE}'")
    
    if LABEL_OUTPUT_FORMAT == "sparse_categorical" and LABEL_DTYPE not in ["uint8", "int8", "int32"]:
        errors.append(f"LABEL_OUTPUT_FORMAT 'sparse_categorical' requires integer dtype, got '{LABEL_DTYPE}'")
    
    if errors:
        raise ValueError("Training compatibility configuration errors:\n" + "\n".join(f"  - {error}" for error in errors))
    
    return True

def get_training_metadata():
    """
    📋 GET TRAINING METADATA
    Zwraca metadata dla compatibility z training module
    """
    return {
        "label_format": LABEL_OUTPUT_FORMAT,
        "label_dtype": LABEL_DTYPE,
        "features_count": 8,  # Zawsze 8 features
        "features_list": [
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200', 
            'volume_to_ma1440', 'volume_to_ma43200'
        ],
        "label_mapping": {
            0: "SHORT", 
            1: "HOLD", 
            2: "LONG"
        },
        "minimum_window_size": MA_LONG_WINDOW,  # Minimum history needed for features
        "timestamp_format": "pandas_datetime64_utc_naive",
        "timestamp_column": "timestamp",
        "includes_timestamp": True,
        "compatible_training_params": {
            "LONG_TP_PCT": LONG_TP_PCT / 100,  # Convert to decimal
            "LONG_SL_PCT": LONG_SL_PCT / 100,
            "SHORT_TP_PCT": SHORT_TP_PCT / 100,
            "SHORT_SL_PCT": SHORT_SL_PCT / 100,
            "FUTURE_WINDOW": FUTURE_WINDOW,
            "MA_SHORT_WINDOW": MA_SHORT_WINDOW,
            "MA_LONG_WINDOW": MA_LONG_WINDOW
        },
        "training_compatibility_version": "v1.1"
    }

# ===== AUTOMATYCZNE TWORZENIE KATALOGÓW =====
def ensure_directories_exist():
    """Tworzy wszystkie potrzebne katalogi jeśli nie istnieją"""
    directories = [INPUT_DATA_PATH, OUTPUT_DATA_PATH, REPORTS_PATH]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def ensure_raw_validated_directory():
    """Tworzy katalog raw_validated jeśli nie istnieje"""
    if SAVE_RAW_VALIDATED_DATA:
        RAW_VALIDATED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Automatyczne tworzenie katalogów przy imporcie
ensure_directories_exist()
ensure_raw_validated_directory()

# Automatyczna walidacja konfiguracji training compatibility
if TRAINING_COMPATIBILITY_MODE:
    validate_training_config()

# ========================
# WALIDACJA DANYCH - GŁÓWNE
# ========================

# ==================================================
# ANALIZA I OBCINANIE EKSTREMALNYCH ZMIAN OHLCV
# ==================================================

OHLCV_VALIDATION_CONFIG = {
    # Włączenie całego modułu analizy i obcinania
    'enabled': True,
    
    # Włączenie samego mechanizmu obcinania (clipping)
    'enable_clipping': True,

    # Progi do analizy statystycznej (w procentach)
    'statistics_thresholds': {
        'open_vs_prev_close': [0.5, 1.0, 2.0, 3.0, 5.0],
        'high_vs_open':       [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        'low_vs_open':        [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        'close_vs_open':      [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        'volume_vs_prev_volume': [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    },

    # Progi (w procentach) dla samego obcinania wartości
    'clipping_thresholds': {
        'open_vs_prev_close': 3.0,
        'high_vs_open': 2.0,
        'low_vs_open': 2.0,  # Zawsze dodatnie, logika obsługuje kierunek
        'close_vs_open': 2.0,
        'volume_vs_prev_volume': 5000.0
    }
}

# ================================
# ALGORYTM INTERPOLACJI - PARAMETRY
# ================================

# Włączenie/wyłączenie interpolacji
INTERPOLATION_ENABLED = True

# Maksymalna liczba iteracji naprawy
MAX_INTERPOLATION_ITERATIONS = 3

# Procent szumu dodawanego dla realizmu (±%)
NOISE_PERCENTAGE = 2.0

# Minimalny dopuszczalny volume (praktycznie > 0)
MIN_VALID_VOLUME = 0.0001

# Maksymalny rozsądny volume jako wielokrotność średniej
MAX_REASONABLE_VOLUME_MULTIPLIER = 10.0

# Maksymalna rozsądna zmiana ceny między świecami (%)
MAX_REASONABLE_PRICE_CHANGE = 50.0

# FALLBACK STRATEGIES
INTERPOLATION_FALLBACK_ON_FAILURE = True  # Użyj starych metod jeśli fail
INTERPOLATION_MAX_PROCESSING_TIME = 300   # 5 minut timeout
INTERPOLATION_MAX_CORRUPTED_PERCENTAGE = 50  # Skip jeśli >50% zepsute

# ============================
# OBLICZANIE FEATURES - GŁÓWNE
# ============================

# Lista 8 features, które mają być obliczone i zwrócone
FEATURE_COLUMNS = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200',
    'volume_to_ma1440', 'volume_to_ma43200'
]

# Parametry średnich kroczących
MA_SHORT_WINDOW = 1440   # 1 dzień
MA_LONG_WINDOW = 43200   # 30 dni

# =========================
# WALIDACJA JAKOŚCI FEATURES
# =========================

# Włączenie/wyłączenie walidacji jakości features
ENABLE_FEATURE_QUALITY_VALIDATION = True

# Progi dla wykrywania ekstremalnych wartości (jako wielokrotność IQR)
EXTREME_VALUE_IQR_MULTIPLIER = 3.0

# Próg procentowy dla anomalii NaN/Inf, powyżej którego rzucany jest błąd
ANOMALY_THRESHOLD_PERCENT = 5.0

# ===========================
# OBLICZANIE LABELS - GŁÓWNE
# ===========================

# Horyzont predykcji (w minutach)
LABELING_HORIZON = 60

# Próg procentowy dla zysku/straty
PROFIT_THRESHOLD = 0.5   # 0.5%
LOSS_THRESHOLD = -0.5    # -0.5%

# Kolumny używane do tworzenia etykiet
LABELING_COLUMNS = ['high', 'low']

# =================================
# KOMPATYBILNOŚĆ Z MODUŁEM TRENINGU
# =================================

# Włącz tryb kompatybilności z modułem treningowym
TRAINING_COMPATIBILITY_MODE = True

# Standaryzuj format timestamp (usuń timezone, konwertuj do datetime64[ns])
STANDARDIZE_TIMESTAMP_FORMAT = True 