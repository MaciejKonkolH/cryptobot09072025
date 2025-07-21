# 📖 STRUKTURA PROJEKTU MODUŁU TRENUJĄCEGO V3

## 📁 RZECZYWISTA STRUKTURA PLIKÓW

### **Główny Katalog Trenowania**
```
ft_bot_docker_compose/user_data/trening2/
├── 📄 config.py                    (276 linii) - Standalone configuration
├── 📄 data_loader.py               (641 linii) - Explicit path data loading  
├── 📄 sequence_generator.py        (570 linii) - Memory-efficient generators
├── 📄 model_builder.py             (503 linie) - LSTM architecture builder
├── 📄 trainer.py                   (724 linie) - Main training pipeline
├── 📄 utils.py                     (422 linie) - Helper functions & monitoring
├── 📁 __pycache__/                 - Python cache files
├── 📁 inputs/                      - Training-ready data files
├── 📁 outputs/                     - Models, scalers, metadata
└── 📁 temp/                        - Temporary processing files
```

### **Katalogi Data Flow**
```
📁 inputs/                          # Training-ready files from validation module
├── BTCUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather
├── ETHUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather
└── ... (other pairs)

📁 outputs/                         # Generated artifacts
├── 📁 models/
│   ├── 📁 BTCUSDT/
│   │   ├── model_BTCUSDT_FW120_SL050_TP100.keras      # TensorFlow model
│   │   ├── scaler_BTCUSDT_FW120_SL050_TP100.pkl       # Feature scaler
│   │   ├── metadata_BTCUSDT_FW120_SL050_TP100.json    # Training metadata
│   │   └── 📁 logs/                                    # TensorBoard logs
│   └── 📁 ETHUSDT/
│       └── ... (same structure for other pairs)
└── 📁 reports/                     # Training reports & confusion matrices

📁 temp/                            # Temporary files (auto-cleanup)
├── temp_sequences_*.npy
├── temp_labels_*.npy  
└── memory_monitor.log
```

## 🔗 **DEPENDENCIES DIAGRAM**

### **File Dependency Graph**
```
config.py                           # ← CORE CONFIG (no dependencies)
    ↑
    ├── data_loader.py              # ← Uses config for paths, parameters
    │   ↑
    │   ├── sequence_generator.py   # ← Uses data_loader output
    │   │   ↑
    │   │   └── trainer.py          # ← Orchestrates all components
    │       ↑
    │       ├── model_builder.py    # ← Uses config for architecture
    │       └── utils.py            # ← Helper functions (optional)
```

### **Import Dependencies**
```python
# config.py - ZERO IMPORTS (core configuration)
# No external dependencies

# data_loader.py
import config                       # ← Local config
from sklearn.preprocessing import   # ← Feature scaling

# sequence_generator.py  
import config                       # ← Local config
import numpy as np                  # ← Memory views

# model_builder.py
import config                       # ← Local config
import tensorflow as tf             # ← Model building

# trainer.py - MAIN ORCHESTRATOR
import config                       # ← Local config
from data_loader import TrainingDataLoader
from sequence_generator import MemoryEfficientDataLoader
from model_builder import DualWindowLSTMBuilder
from utils import monitor_memory_usage

# utils.py
import psutil                       # ← Optional memory monitoring
```

## 🎯 **CORE MODULES OVERVIEW**

### 1. **📄 config.py - Single Source of Truth**

#### **Responsibility:**
- ✅ **Centralized configuration** - wszystkie parametry w jednym miejscu
- ✅ **Parameter validation** - comprehensive validation functions
- ✅ **Filename generation** - consistent naming based on parameters
- ✅ **Helper functions** - configuration-related utilities

#### **Key Functions:**
```python
# Configuration
PAIR = "BTCUSDT"                    # Crypto pair selection
EPOCHS = 100                        # Training parameters
LSTM_UNITS = [128, 64, 32]         # Model architecture

# Helper functions
get_expected_filename()             # Generate expected input filename
get_model_output_dir()              # Generate output directory
validate_config()                   # Validate all parameters
print_config_summary()              # Display configuration overview
```

#### **Zero Dependencies:**
- ❌ **NO imports** from other modules
- ❌ **NO validation module** dependencies
- ✅ **Pure configuration** - standalone operation

### 2. **📄 data_loader.py - Explicit Path Loading**

#### **Responsibility:**
- ✅ **Explicit path configuration** - no auto-detection chaos
- ✅ **Filename parameter parsing** - extract parameters from filename
- ✅ **Feature scaling system** - zero data leakage prevention
- ✅ **Class balancing** - systematic undersampling algorithms
- ✅ **Date filtering** - optional date range support

#### **Key Classes:**
```python
class TrainingDataLoader:
    def __init__(self, explicit_path: str, pair: str)
    def load_training_data() -> Dict[str, Any]      # Main loading function
    def apply_class_balancing(features, labels)     # Balance classes
    def validate_file_exists() -> bool              # Check file existence
    def _split_features_labels(df) -> Tuple        # Split with scaling
```

#### **Innovation - Feature Scaling:**
```python
# ZERO DATA LEAKAGE - proper train/val splitting
def _split_features_labels(self, df, fit_scaler: bool = True):
    if config.ENABLE_FEATURE_SCALING:
        if fit_scaler and not self.scaler_fitted:
            # FIT scaler only on training data
            self.scaler = self._create_scaler()
            features = self.scaler.fit_transform(features)
            self.scaler_fitted = True
        elif self.scaler_fitted:
            # TRANSFORM validation data with train scaler
            features = self.scaler.transform(features)
```

### 3. **📄 sequence_generator.py - Memory Efficiency**

#### **Responsibility:**
- ✅ **Memory-efficient generators** - 2-3GB instead of 81GB+
- ✅ **Numpy memory views** - zero-copy data access
- ✅ **Pre-computed labels** - eliminates competitive labeling
- ✅ **Chronological splitting** - proper train/validation split
- ✅ **Batch generation** - on-demand sequence loading

#### **Key Classes:**
```python
class MemoryEfficientDataLoader:
    def create_generators_from_arrays(features, labels)     # Main generator creation
    def generate_sequences_with_labels()                    # Core generator function
    def _chronological_split(features, labels)             # Time-aware splitting
    def test_generator(generator, num_batches)              # Generator validation
```

#### **Memory Innovation:**
```python
# NUMPY MEMORY VIEWS - zero copy
def generate_sequences_with_labels(self):
    for i in range(self.total_sequences):
        start_idx = self.indices[i]
        # Memory view - NO COPY!
        sequence = self.features[start_idx:start_idx + self.window_size]
        label = self.labels[start_idx + self.window_size - 1]
        yield sequence, label
```

### 4. **📄 model_builder.py - LSTM Architecture**

#### **Responsibility:**
- ✅ **LSTM model building** - crypto-optimized architecture  
- ✅ **Production callbacks** - checkpointing, early stopping, LR reduction
- ✅ **GPU optimization** - memory growth, auto-fallback to CPU
- ✅ **Model persistence** - .keras format for TensorFlow 2.10+
- ✅ **Memory estimation** - predict training memory requirements

#### **Key Classes:**
```python
class DualWindowLSTMBuilder:
    def build_model(compile_model: bool = True) -> Model    # Build LSTM architecture
    def create_callbacks(model_output_dir: str) -> list     # Production callbacks
    def save_model(model, save_path, format='saved_model')  # Model persistence
    def setup_gpu_memory() -> bool                          # GPU optimization
```

#### **Architecture Design:**
```python
# CRYPTO-OPTIMIZED LSTM STACK
inputs = layers.Input(shape=(120, 8))                      # 120 timesteps, 8 features
x = layers.LSTM(128, return_sequences=True)(inputs)        # First LSTM layer
x = layers.LSTM(64, return_sequences=True)(x)              # Second LSTM layer  
x = layers.LSTM(32, return_sequences=False)(x)             # Final LSTM layer
x = layers.Dense(32, activation='relu')(x)                 # Dense layer 1
x = layers.Dropout(0.3)(x)                                 # Regularization
x = layers.Dense(16, activation='relu')(x)                 # Dense layer 2
x = layers.Dropout(0.3)(x)                                 # Regularization
outputs = layers.Dense(3, activation='softmax')(x)         # 3-class output
```

### 5. **📄 trainer.py - Main Pipeline**

#### **Responsibility:**
- ✅ **Training orchestration** - complete training workflow
- ✅ **Component integration** - coordinates all modules
- ✅ **Error handling** - comprehensive error messages
- ✅ **Confusion matrix** - post-training analysis
- ✅ **CLI interface** - parameter override support

#### **Key Classes:**
```python
class StandaloneTrainer:
    def __init__()                                          # Initialize trainer
    def initialize_components()                             # Setup all components
    def load_and_validate_data() -> Dict[str, Any]         # Data loading pipeline
    def create_generators(features, labels) -> Tuple       # Generator creation
    def build_model()                                       # Model building
    def train_model(train_gen, val_gen, callbacks)         # Training execution
    def save_model_and_metadata(data_info)                 # Save results
    def generate_confusion_matrix_report()                 # Post-training analysis
    def run_training()                                      # Main training pipeline
```

#### **Complete Workflow:**
```python
def run_training(self):
    # 1. Initialize all components
    self.initialize_components()
    
    # 2. Load and validate data
    data_info = self.load_and_validate_data()
    
    # 3. Create memory-efficient generators
    train_gen, val_gen = self.create_generators(data_info['features'], data_info['labels'])
    
    # 4. Build model
    self.build_model()
    
    # 5. Setup callbacks
    callbacks_list = self.setup_callbacks()
    
    # 6. Train model
    self.train_model(train_gen, val_gen, callbacks_list)
    
    # 7. Save model and metadata
    self.save_model_and_metadata(data_info)
    
    # 8. Generate confusion matrix
    self.generate_confusion_matrix_report()
```

### 6. **📄 utils.py - Helper Functions**

#### **Responsibility:**
- ✅ **Memory monitoring** - real-time memory usage tracking
- ✅ **Data validation** - OHLCV data quality checks
- ✅ **File operations** - directory management
- ✅ **Memory calculations** - estimate memory requirements
- ✅ **System utilities** - optional psutil integration

#### **Key Functions:**
```python
validate_data_quality(df, pair) -> bool                    # Data quality validation
calculate_memory_usage(data_shape, dtype) -> Dict          # Memory calculations
get_system_memory_info() -> Dict                           # System memory info
monitor_memory_usage(process_name) -> None                 # Real-time monitoring
chronological_split(df, train_ratio) -> Tuple             # Time-aware splitting
safe_memory_cleanup()                                      # Garbage collection
```

## 🔄 **DATA FLOW WORKFLOW**

### **Training Pipeline Flow**
```
1. CONFIG LOADING
   config.py → Load parameters, validate configuration

2. DATA LOADING  
   data_loader.py → Load training-ready file, parse filename parameters
   ↓
   Validate file exists: BTCUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather
   ↓
   Apply feature scaling (fit on train, transform on train/val)
   ↓
   Apply class balancing (systematic undersampling or class weights)

3. SEQUENCE GENERATION
   sequence_generator.py → Create memory-efficient generators
   ↓
   Chronological split (train/validation)
   ↓
   Generate sequences using numpy views (zero-copy)

4. MODEL BUILDING
   model_builder.py → Build LSTM architecture from config
   ↓
   Setup production callbacks (checkpointing, early stopping, LR reduction)
   ↓
   Configure GPU optimization

5. TRAINING EXECUTION
   trainer.py → Execute training pipeline
   ↓
   Train model with generators
   ↓
   Monitor progress with callbacks

6. RESULTS SAVING
   trainer.py → Save model, scaler, metadata
   ↓
   Generate confusion matrix report
   ↓
   Save to outputs/models/{PAIR}/
```

### **File Format Specifications**

#### **Input File Format (Feather)**
```python
# Expected columns in training-ready file:
FEATURE_COLUMNS = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200', 
    'volume_to_ma1440', 'volume_to_ma43200'
]

LABEL_COLUMNS = [
    'label_0',  # SHORT probability (one-hot encoded)
    'label_1',  # HOLD probability
    'label_2'   # LONG probability
]

# Optional datetime index for date filtering
```

#### **Output File Formats**
```python
# Model: .keras format (TensorFlow 2.10+)
model_BTCUSDT_FW120_SL050_TP100.keras

# Scaler: .pkl format (pickle)
scaler_BTCUSDT_FW120_SL050_TP100.pkl

# Metadata: .json format
metadata_BTCUSDT_FW120_SL050_TP100.json
```

## 🐳 **DOCKER ENVIRONMENT STRUCTURE**

### **Docker Paths Configuration**
```python
# EXPLICIT DOCKER PATHS (no relative paths)
TRAINING_DATA_PATH = "/freqtrade/user_data/trening2/inputs/"    # Input data
OUTPUT_BASE_PATH = "/freqtrade/user_data/trening2/outputs/"     # Output artifacts

# Path validation
if not TRAINING_DATA_PATH.startswith('/'):
    raise ValueError("TRAINING_DATA_PATH must be absolute Docker path")
```

### **Container Volume Mapping**
```yaml
# docker-compose.yml structure
volumes:
  - ./ft_bot_docker_compose/user_data:/freqtrade/user_data
  
# Results in:
# Host: ft_bot_docker_compose/user_data/trening2/
# Container: /freqtrade/user_data/trening2/
```

## 🔧 **DEPENDENCY MANAGEMENT**

### **External Dependencies**
```python
# REQUIRED (core functionality)
tensorflow>=2.10.0          # Model building and training
pandas>=1.5.0               # Data manipulation
numpy>=1.21.0               # Numerical computing
pyarrow>=10.0.0             # Feather file support

# REQUIRED (feature scaling)  
scikit-learn>=1.1.0         # StandardScaler, RobustScaler, MinMaxScaler

# OPTIONAL (enhanced functionality)
psutil                      # Memory monitoring (graceful fallback if missing)
```

### **Internal Dependencies**
```python
# ZERO validation module dependencies
# All functionality is self-contained
# Clean separation of concerns
```

---

**🎯 KLUCZOWE ZALETY STRUKTURY V3:**
- ✅ **Modular Design** - każdy plik ma jasną odpowiedzialność
- ✅ **Zero Circular Dependencies** - clean import hierarchy  
- ✅ **Docker-Optimized** - explicit paths, production-ready
- ✅ **Memory-Efficient** - generators, views, lazy loading
- ✅ **Config-Driven** - single source of truth
- ✅ **Production-Ready** - error handling, monitoring, validation

**📈 NEXT:** [03_Config.md](./03_Config.md) - Szczegółowa dokumentacja konfiguracji 