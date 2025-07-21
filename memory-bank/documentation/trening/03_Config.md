# 📖 KONFIGURACJA MODUŁU TRENUJĄCEGO V3 (config.py)

## 🎯 PRZEGLĄD KONFIGURACJI

**config.py** (276 linii) to serce modułu trenującego V3 - **single source of truth** dla wszystkich parametrów systemu. Eliminuje chaos hierarchii klas V2 na rzecz prostej, flat konfiguracji.

### ✨ **Kluczowe Zalety V3 Config**
- ✅ **Single Section** - wszystkie parametry w jednej sekcji
- ✅ **Zero Dependencies** - brak importów z innych modułów
- ✅ **Explicit Values** - jasne, konkretne wartości
- ✅ **Comprehensive Validation** - zaawansowana walidacja parametrów
- ✅ **Helper Functions** - funkcje pomocnicze dla filename generation
- ✅ **CLI Override Support** - możliwość nadpisania przez argumenty CLI

## 📋 **PARAMETRY KONFIGURACJI**

### 1. **🎯 CRYPTO PAIR SELECTION**

```python
# CRYPTO PAIR
PAIR = "BTCUSDT"  # Explicit pair selection
```

**Opis:** Podstawowy parametr definiujący parę kryptowalutową do trenowania.

**Zastosowanie:**
- Generowanie nazw plików wejściowych i wyjściowych
- Organizacja katalogów output
- Walidacja zgodności z filename

**Przykłady:**
```python
PAIR = "BTCUSDT"    # Bitcoin/USDT
PAIR = "ETHUSDT"    # Ethereum/USDT  
PAIR = "ADAUSDT"    # Cardano/USDT
```

**Walidacja:**
```python
if not PAIR or len(PAIR) < 6:
    errors.append("PAIR must be valid crypto pair (e.g. BTCUSDT)")
```

### 2. **📁 DATA PATHS (Docker Container Paths)**

```python
# DATA PATHS (Docker container paths)
TRAINING_DATA_PATH = "/freqtrade/user_data/trening2/inputs/"  # Explicit path to training-ready files
OUTPUT_BASE_PATH = "/freqtrade/user_data/trening2/outputs/"  # Base output directory
```

**TRAINING_DATA_PATH:**
- **Cel:** Ścieżka do training-ready files
- **Format:** Absolutna ścieżka Docker container
- **Wymagania:** Musi zaczynać się od `/`
- **Zawartość:** Pliki `.feather` z modułu walidacji

**OUTPUT_BASE_PATH:**
- **Cel:** Bazowa ścieżka dla wszystkich artifacts
- **Auto-generation:** `{OUTPUT_BASE_PATH}models/{PAIR}/`
- **Zawartość:** Modele, scalers, metadata, logs

**Walidacja:**
```python
if not TRAINING_DATA_PATH.startswith('/'):
    errors.append("TRAINING_DATA_PATH must be absolute Docker path")
```

### 3. **🏋️ TRAINING PARAMETERS**

```python
# TRAINING PARAMETERS
EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
```

#### **EPOCHS**
- **Typ:** Integer
- **Zakres:** 1-1000 (praktycznie 50-200)
- **Opis:** Liczba epochs trenowania
- **Default:** 100
- **Recommendations:**
  - Quick test: 10-20
  - Full training: 100-200
  - Heavy datasets: 50-100

#### **BATCH_SIZE**
- **Typ:** Integer
- **Zakres:** 16-512 (zalecane: 128, 256, 512)
- **Opis:** Rozmiar batcha dla trenowania
- **Memory Impact:** Większy batch = więcej pamięci
- **Recommendations:**
  - GPU (8GB): 256
  - GPU (16GB+): 512
  - CPU: 128

#### **VALIDATION_SPLIT**
- **Typ:** Float
- **Zakres:** 0.1-0.3
- **Opis:** Procent danych na walidację
- **Default:** 0.2 (20% validation, 80% training)

### 4. **📋 CALLBACKS CONFIGURATION**

```python
# CALLBACKS CONFIGURATION
EARLY_STOPPING_PATIENCE = 10         # Number of epochs with no improvement after which training will be stopped
REDUCE_LR_PATIENCE = 5               # Number of epochs with no improvement after which learning rate will be reduced
REDUCE_LR_FACTOR = 0.5               # Factor by which the learning rate will be reduced
MIN_LEARNING_RATE = 1e-7             # Lower bound on the learning rate
```

#### **EARLY_STOPPING_PATIENCE**
- **Opis:** Zatrzymaj trening po N epochs bez poprawy
- **Monitor:** `val_accuracy` (domyślnie)
- **Zalecenia:** 
  - Fast iteration: 5-7
  - Production: 10-15
  - Conservative: 20+

#### **REDUCE_LR_PATIENCE**
- **Opis:** Zmniejsz learning rate po N epochs bez poprawy
- **Monitor:** `val_loss`
- **Zalecenia:** Połowa Early Stopping Patience

#### **REDUCE_LR_FACTOR**
- **Opis:** Mnożnik zmniejszania learning rate
- **Zakres:** 0.1-0.8
- **Default:** 0.5 (zmniejszenie o połowę)

#### **MIN_LEARNING_RATE**
- **Opis:** Minimalna wartość learning rate
- **Default:** 1e-7
- **Cel:** Prevent learning rate od zbyt małych wartości

### 5. **⚖️ CLASS BALANCING CONFIGURATION**

```python
# CLASS BALANCING CONFIGURATION
ENABLE_CLASS_BALANCING = True                    # Enable/disable class balancing
CLASS_BALANCING_METHOD = "systematic_undersampling"  # "systematic_undersampling", "class_weights", "none"

# SYSTEMATIC UNDERSAMPLING SETTINGS
UNDERSAMPLING_PRESERVE_RATIO = True              # Keep proportions between minority classes
UNDERSAMPLING_MIN_SAMPLES = 50000                # Minimum samples per class (safety limit)
UNDERSAMPLING_SEED = 42                          # Random seed for reproducibility

# CLASS WEIGHTS SETTINGS (if using class_weights method)
CLASS_WEIGHT_METHOD = "balanced"                 # "balanced", "manual", "none"
MANUAL_CLASS_WEIGHTS = {                         # Used only when CLASS_WEIGHT_METHOD = "manual"
    0: 4.0,  # SHORT weight (minority class)
    1: 1.0,  # HOLD weight (majority class)  
    2: 4.0   # LONG weight (minority class)
}
```

#### **Available Methods:**

**1. Systematic Undersampling (Recommended)**
```python
CLASS_BALANCING_METHOD = "systematic_undersampling"
```
- **Algorytm:** Reduce majority classes systematically
- **Zalety:** Zachowuje temporal diversity, fast training
- **Parametry:** `UNDERSAMPLING_PRESERVE_RATIO`, `UNDERSAMPLING_MIN_SAMPLES`

**2. Class Weights**
```python
CLASS_BALANCING_METHOD = "class_weights"
```
- **Algorytm:** Apply weights to loss function
- **Zalety:** Uses all data, no data loss
- **Parametry:** `CLASS_WEIGHT_METHOD`, `MANUAL_CLASS_WEIGHTS`

**3. No Balancing**
```python
CLASS_BALANCING_METHOD = "none"
```
- **Algorytm:** Use raw class distribution
- **Use case:** Balanced datasets, testing

### 6. **📅 DATE RANGE CONTROL**

```python
# DATE RANGE CONTROL (Optional - if None, uses full dataset)
START_DATE = "2021-01-01"  # Format: YYYY-MM-DD or None for full range
END_DATE = "2025-05-30"    # Format: YYYY-MM-DD or None for full range
ENABLE_DATE_FILTER = True  # Set to False to disable date filtering
```

#### **Date Filtering Logic:**
1. **Gdy ENABLE_DATE_FILTER = True:**
   - Filtruje dane po datetime index lub timestamp column
   - Wymagane: START_DATE i END_DATE
   - Walidacja: START_DATE < END_DATE

2. **Gdy ENABLE_DATE_FILTER = False:**
   - Używa pełnego datasetu
   - Ignoruje START_DATE i END_DATE

#### **Date Format:**
```python
# Valid formats
START_DATE = "2021-01-01"    # YYYY-MM-DD
END_DATE = "2025-05-30"      # YYYY-MM-DD

# Disable filtering
ENABLE_DATE_FILTER = False
START_DATE = None
END_DATE = None
```

### 7. **📏 FEATURE SCALING CONFIGURATION**

```python
# FEATURE SCALING CONFIGURATION
ENABLE_FEATURE_SCALING = True     # Enable/disable feature scaling
SCALER_TYPE = "robust"            # "standard", "robust", "minmax"
SCALER_FIT_ONLY_TRAIN = True      # Fit scaler only on train data (prevent data leakage)
VALIDATE_SCALING_STATS = True     # Print scaling statistics for validation
```

#### **Available Scalers:**

**1. RobustScaler (Recommended)**
```python
SCALER_TYPE = "robust"
```
- **Algorytm:** Uses median and IQR
- **Zalety:** Robust to outliers
- **Use case:** Crypto data (high volatility)

**2. StandardScaler**
```python
SCALER_TYPE = "standard"
```
- **Algorytm:** Zero mean, unit variance
- **Zalety:** Classical normalization
- **Use case:** Well-distributed data

**3. MinMaxScaler**
```python
SCALER_TYPE = "minmax"
```
- **Algorytm:** Scale to [0, 1] range
- **Zalety:** Bounded output
- **Use case:** Neural networks with specific input ranges

#### **Zero Data Leakage Prevention:**
```python
SCALER_FIT_ONLY_TRAIN = True  # CRITICAL for production!
```
- **True:** Fit scaler ONLY on training data, transform train & validation
- **False:** ⚠️ DATA LEAKAGE - fit on full dataset

### 8. **🧠 MODEL PARAMETERS**

```python
# MODEL PARAMETERS
SEQUENCE_LENGTH = 120  # Time window for LSTM
LSTM_UNITS = [128, 64, 32]  # LSTM layer sizes
DENSE_UNITS = [32, 16]      # Dense layer sizes
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
```

#### **SEQUENCE_LENGTH**
- **Opis:** Liczba timesteps dla LSTM (time window)
- **Default:** 120 (2 godziny dla 1m data)
- **Zakres:** 60-240 (1-4 godziny)
- **Impact:** Więcej timesteps = więcej kontekstu, ale większa złożoność

#### **LSTM_UNITS**
- **Opis:** Lista rozmiarów warstw LSTM
- **Default:** [128, 64, 32] (progressive reduction)
- **Format:** List[int]
- **Przykłady:**
  ```python
  LSTM_UNITS = [64, 32]        # Smaller, faster
  LSTM_UNITS = [256, 128, 64]  # Larger, more complex
  ```

#### **DENSE_UNITS**
- **Opis:** Lista rozmiarów warstw Dense
- **Default:** [32, 16]
- **Format:** List[int]
- **Cel:** Feature extraction po LSTM

#### **DROPOUT_RATE**
- **Opis:** Dropout rate dla regularization
- **Zakres:** 0.0-0.5
- **Default:** 0.3
- **Impact:** Higher = more regularization, less overfitting

#### **LEARNING_RATE**
- **Opis:** Initial learning rate dla Adam optimizer
- **Default:** 0.001
- **Zakres:** 1e-5 to 1e-2
- **Note:** Will be reduced by ReduceLROnPlateau callback

### 9. **💰 TRADING PARAMETERS**

```python
# TRADING PARAMETERS (will be validated against filename)
LONG_TP_PCT = 1.0    # Take profit %
LONG_SL_PCT = 0.5    # Stop loss %
SHORT_TP_PCT = 1.0   # Take profit %
SHORT_SL_PCT = 0.5   # Stop loss %
FUTURE_WINDOW = 120  # Future bars for labeling
```

#### **Critical Validation:**
Te parametry **MUSZĄ** być zgodne z filename:
```python
# Expected filename pattern:
# BTCUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather
#                ^^^     ^^^     ^^^
#            FW=120  SL=0.5%  TP=1.0%

# Config must match:
FUTURE_WINDOW = 120    # FW-120
LONG_SL_PCT = 0.5      # SL-050 (0.5% * 100)
LONG_TP_PCT = 1.0      # TP-100 (1.0% * 100)
```

#### **Parameter Meanings:**
- **LONG_TP_PCT:** Take profit threshold dla LONG positions (%)
- **LONG_SL_PCT:** Stop loss threshold dla LONG positions (%)
- **SHORT_TP_PCT:** Take profit threshold dla SHORT positions (%)
- **SHORT_SL_PCT:** Stop loss threshold dla SHORT positions (%)
- **FUTURE_WINDOW:** Liczba bars w przyszłość dla label calculation

### 10. **💾 OUTPUT SETTINGS**

```python
# OUTPUT SETTINGS
OVERWRITE_EXISTING = True  # Overwrite model if exists
SAVE_METADATA = True       # Save training metadata JSON
VERBOSE_LOGGING = True     # Detailed progress logs
```

#### **OVERWRITE_EXISTING**
- **True:** Nadpisz istniejący model
- **False:** Error jeśli model już istnieje
- **Use case:** Development vs Production

#### **SAVE_METADATA**
- **True:** Zapisz training metadata do JSON
- **False:** Skip metadata saving
- **Content:** Training parameters, model info, performance metrics

#### **VERBOSE_LOGGING**
- **True:** Detailed progress logs
- **False:** Minimal logging
- **Impact:** Debug information, training progress

## 🔧 **HELPER FUNCTIONS**

### 1. **Filename Generation Functions**

#### **get_expected_filename()**
```python
def get_expected_filename():
    """Generate expected training-ready filename based on config"""
    return f"{PAIR}_TF-1m__FW-{FUTURE_WINDOW:03d}__SL-{int(LONG_SL_PCT*100):03d}__TP-{int(LONG_TP_PCT*100):03d}__training_ready.feather"

# Example output:
# "BTCUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather"
```

#### **get_model_output_dir()**
```python
def get_model_output_dir():
    """Get output directory for this crypto pair"""
    return f"{OUTPUT_BASE_PATH}models/{PAIR}/"

# Example output:
# "/freqtrade/user_data/trening2/outputs/models/BTCUSDT/"
```

#### **get_model_filename()**
```python
def get_model_filename():
    """Generate model filename with parameters"""
    return f"model_{PAIR}_FW{FUTURE_WINDOW:03d}_SL{int(LONG_SL_PCT*100):03d}_TP{int(LONG_TP_PCT*100):03d}.keras"

# Example output:
# "model_BTCUSDT_FW120_SL050_TP100.keras"
```

#### **get_scaler_filename() & get_metadata_filename()**
```python
def get_scaler_filename():
    return f"scaler_{PAIR}_FW{FUTURE_WINDOW:03d}_SL{int(LONG_SL_PCT*100):03d}_TP{int(LONG_TP_PCT*100):03d}.pkl"

def get_metadata_filename():
    return f"metadata_{PAIR}_FW{FUTURE_WINDOW:03d}_SL{int(LONG_SL_PCT*100):03d}_TP{int(LONG_TP_PCT*100):03d}.json"

# Example outputs:
# "scaler_BTCUSDT_FW120_SL050_TP100.pkl"
# "metadata_BTCUSDT_FW120_SL050_TP100.json"
```

### 2. **Configuration Validation**

#### **validate_config() Function**
```python
def validate_config():
    """Validate configuration parameters - returns list of errors"""
    errors = []
    
    # Path validation
    if not TRAINING_DATA_PATH.startswith('/'):
        errors.append("TRAINING_DATA_PATH must be absolute Docker path")
    
    # Pair validation  
    if not PAIR or len(PAIR) < 6:
        errors.append("PAIR must be valid crypto pair (e.g. BTCUSDT)")
    
    # Training params validation
    if EPOCHS <= 0:
        errors.append("EPOCHS must be positive")
    
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    # Callbacks validation
    if EARLY_STOPPING_PATIENCE <= 0:
        errors.append("EARLY_STOPPING_PATIENCE must be positive")
    
    if not (0 < REDUCE_LR_FACTOR < 1):
        errors.append(f"REDUCE_LR_FACTOR must be between 0 and 1, got {REDUCE_LR_FACTOR}")
    
    # Date range validation
    if ENABLE_DATE_FILTER and START_DATE and END_DATE:
        try:
            start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
            end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
            if start_dt >= end_dt:
                errors.append(f"START_DATE must be before END_DATE")
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")
    
    # Feature scaling validation
    if ENABLE_FEATURE_SCALING:
        valid_scalers = ["standard", "robust", "minmax"]
        if SCALER_TYPE not in valid_scalers:
            errors.append(f"SCALER_TYPE must be one of {valid_scalers}")
    
    # Model architecture validation
    if not LSTM_UNITS or len(LSTM_UNITS) == 0:
        errors.append("LSTM_UNITS cannot be empty")
    
    if not (0 <= DROPOUT_RATE <= 1):
        errors.append(f"DROPOUT_RATE must be 0-1, got {DROPOUT_RATE}")
    
    return errors
```

### 3. **Configuration Summary**

#### **print_config_summary() Function**
```python
def print_config_summary():
    """Print comprehensive configuration summary"""
    print("🎯 STANDALONE TRAINING MODULE V3 - CONFIGURATION")
    print("=" * 60)
    print(f"📍 Crypto Pair: {PAIR}")
    print(f"📁 Data Path: {TRAINING_DATA_PATH}")
    print(f"📁 Output Path: {get_model_output_dir()}")
    print(f"📄 Expected File: {get_expected_filename()}")
    print("")
    print(f"🏋️ Training:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Validation Split: {VALIDATION_SPLIT}")
    print(f"   Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print("")
    print(f"📏 Feature Scaling:")
    if ENABLE_FEATURE_SCALING:
        print(f"   Enabled: YES")
        print(f"   Scaler Type: {SCALER_TYPE}")
        print(f"   Fit Only Train: {SCALER_FIT_ONLY_TRAIN}")
    else:
        print(f"   Enabled: NO")
    print("")
    print(f"⚖️ Class Balancing:")
    if ENABLE_CLASS_BALANCING:
        print(f"   Enabled: YES")
        print(f"   Method: {CLASS_BALANCING_METHOD}")
        if CLASS_BALANCING_METHOD == "systematic_undersampling":
            print(f"   Min Samples: {UNDERSAMPLING_MIN_SAMPLES}")
            print(f"   Preserve Ratio: {UNDERSAMPLING_PRESERVE_RATIO}")
    else:
        print(f"   Enabled: NO")
```

## 🎯 **CLI OVERRIDE SUPPORT**

### **Supported CLI Parameters**
```python
# trainer.py main() function supports:
python trainer.py --pair ETHUSDT          # Override PAIR
python trainer.py --epochs 50             # Override EPOCHS  
python trainer.py --batch-size 128        # Override BATCH_SIZE
python trainer.py --config-test           # Test configuration only
```

### **CLI Implementation**
```python
# In trainer.py main() function:
if args.pair:
    config.PAIR = args.pair
if args.epochs:
    config.EPOCHS = args.epochs
if args.batch_size:
    config.BATCH_SIZE = args.batch_size
```

## 📊 **EXAMPLE CONFIGURATIONS**

### **Configuration 1: Quick Test**
```python
# Quick iteration configuration
PAIR = "BTCUSDT"
EPOCHS = 20                    # Fast training
BATCH_SIZE = 128               # Memory efficient
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 5   # Quick stopping
ENABLE_FEATURE_SCALING = True
SCALER_TYPE = "robust"
ENABLE_CLASS_BALANCING = True
CLASS_BALANCING_METHOD = "systematic_undersampling"
```

### **Configuration 2: Production Training**
```python
# Full production configuration
PAIR = "BTCUSDT"
EPOCHS = 100                   # Full training
BATCH_SIZE = 256               # Balanced performance
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10  # Patient training
ENABLE_FEATURE_SCALING = True
SCALER_TYPE = "robust"
ENABLE_CLASS_BALANCING = True
CLASS_BALANCING_METHOD = "systematic_undersampling"
UNDERSAMPLING_MIN_SAMPLES = 50000
```

### **Configuration 3: GPU Optimized**
```python
# High-performance GPU configuration
PAIR = "BTCUSDT"
EPOCHS = 150
BATCH_SIZE = 512               # Large batches for GPU
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15
LSTM_UNITS = [256, 128, 64]    # Deeper network
DENSE_UNITS = [64, 32]
DROPOUT_RATE = 0.4             # More regularization
```

---

**🎯 KLUCZOWE ZALETY CONFIG V3:**
- ✅ **Single Source of Truth** - wszystkie parametry w jednym miejscu
- ✅ **Zero Dependencies** - brak importów z validation module
- ✅ **Comprehensive Validation** - walidacja wszystkich parametrów
- ✅ **Clear Structure** - logical grouping of parameters
- ✅ **Helper Functions** - filename generation, validation, summary
- ✅ **CLI Support** - możliwość override przez command line

**📈 NEXT:** [04_Data_loader.md](./04_Data_loader.md) - Szczegółowa dokumentacja data loading 