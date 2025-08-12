# 📖 DATA LOADER MODUŁU TRENUJĄCEGO V3 (data_loader.py)

## 🎯 PRZEGLĄD DATA LOADER

**data_loader.py** (641 linii) to standalone moduł ładowania danych, który eliminuje chaos auto-detection V2 na rzecz **explicit path configuration**. Zawiera zaawansowany system feature scaling z zero data leakage oraz sophisticated class balancing.

### ✨ **Kluczowe Innowacje V3 Data Loader**
- ✅ **Explicit Path Loading** - no auto-detection chaos
- ✅ **Filename Parameter Parsing** - extract & validate parameters from filename
- ✅ **Zero Data Leakage Scaling** - fit scaler only on train data
- ✅ **Advanced Class Balancing** - systematic undersampling & class weights
- ✅ **Date Range Filtering** - optional temporal data filtering
- ✅ **Hard Fail Error Handling** - clear error messages with solutions

## 🏗️ **ARCHITECTURE OVERVIEW**

### **TrainingDataLoader Class**
```python
class TrainingDataLoader:
    """
    🎯 STANDALONE DATA LOADER V3
    
    Core functionality:
    - Load training-ready files from explicit path
    - Parse parameters from filename
    - Validate parameter compatibility
    - Hard fail on missing files
    - Zero auto-detection or validation module dependencies
    - FEATURE SCALING with proper train/val handling
    """
```

### **Class Initialization**
```python
def __init__(self, explicit_path: str, pair: str):
    """
    🎯 STANDALONE DATA LOADER V3
    
    Args:
        explicit_path: Exact path to training-ready files
        pair: Crypto pair (from config.py)
    """
    self.data_path = explicit_path
    self.pair = pair
    self.expected_filename = self._generate_expected_filename()
    self.full_file_path = os.path.join(self.data_path, self.expected_filename)
    
    # Feature scaling state
    self.scaler = None
    self.scaler_fitted = False
    self.scaling_stats = {}
```

## 📁 **EXPLICIT PATH LOADING SYSTEM**

### 1. **Filename Generation & Validation**

#### **Expected Filename Pattern**
```python
def _generate_expected_filename(self) -> str:
    """Generate expected filename pattern from config"""
    return f"{self.pair}_TF-1m__FW-{config.FUTURE_WINDOW:03d}__SL-{int(config.LONG_SL_PCT*100):03d}__TP-{int(config.LONG_TP_PCT*100):03d}__training_ready.feather"

# Example output:
# "BTCUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather"
```

#### **Filename Pattern Breakdown**
```
BTCUSDT_TF-1m__FW-120__SL-050__TP-100__training_ready.feather
^^^^^^^               ^^^     ^^^     ^^^
  │                    │       │       │
  │                    │       │       └─ Take Profit: 1.0% (100/100)
  │                    │       └─ Stop Loss: 0.5% (050/100)
  │                    └─ Future Window: 120 bars
  └─ Crypto Pair: BTCUSDT
```

#### **Parameter Parsing**
```python
def _parse_filename_parameters(self, filename: str) -> dict:
    """Parse parameters from training-ready filename"""
    pattern = r'(\w+)_TF-1m__FW-(\d+)__SL-(\d+)__TP-(\d+)__training_ready\.feather'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid training-ready filename format: {filename}")
        
    return {
        'pair': match.group(1),
        'future_window': int(match.group(2)),
        'stop_loss': int(match.group(3)) / 100.0,    # Convert to percentage
        'take_profit': int(match.group(4)) / 100.0   # Convert to percentage
    }
```

### 2. **Hard Fail File Validation**

#### **File Existence Check**
```python
def validate_file_exists(self) -> bool:
    """Hard fail validation - explicit path only"""
    if not os.path.exists(self.full_file_path):
        print(f"❌ TRAINING-READY FILE NOT FOUND:")
        print(f"   Expected path: {self.full_file_path}")
        print(f"   Expected filename: {self.expected_filename}")
        print(f"   Data directory: {self.data_path}")
        print(f"   Crypto pair: {self.pair}")
        print("")
        print(f"💡 SOLUTION:")
        print(f"   1. Verify TRAINING_DATA_PATH in config.py")
        print(f"   2. Ensure validation module generated training-ready files")
        print(f"   3. Check crypto pair spelling in config.py")
        return False
    return True
```

#### **Parameter Compatibility Validation**
```python
def _validate_parameters_compatibility(self, file_params: dict) -> bool:
    """Validate filename parameters against config"""
    errors = []
    
    # Check pair match
    if file_params['pair'] != self.pair:
        errors.append(f"Pair mismatch: config={self.pair}, file={file_params['pair']}")
    
    # Check TP/SL/FW match (critical for label compatibility!)
    if file_params['future_window'] != config.FUTURE_WINDOW:
        errors.append(f"Future window mismatch: config={config.FUTURE_WINDOW}, file={file_params['future_window']}")
    
    if abs(file_params['stop_loss'] - config.LONG_SL_PCT) > 0.01:
        errors.append(f"Stop loss mismatch: config={config.LONG_SL_PCT}, file={file_params['stop_loss']}")
        
    if abs(file_params['take_profit'] - config.LONG_TP_PCT) > 0.01:
        errors.append(f"Take profit mismatch: config={config.LONG_TP_PCT}, file={file_params['take_profit']}")
    
    if errors:
        print(f"❌ Parameter compatibility errors:")
        for error in errors:
            print(f"   - {error}")
        return False
        
    return True
```

## 📏 **FEATURE SCALING SYSTEM**

### 1. **Zero Data Leakage Architecture**

#### **Scaler Creation & Management**
```python
def _create_scaler(self):
    """Create scaler based on config"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for feature scaling")
        
    if config.SCALER_TYPE == "standard":
        return StandardScaler()         # Zero mean, unit variance
    elif config.SCALER_TYPE == "robust":
        return RobustScaler()           # Median, IQR (robust to outliers)
    elif config.SCALER_TYPE == "minmax":
        return MinMaxScaler()           # Scale to [0, 1] range
    else:
        raise ValueError(f"Unknown scaler type: {config.SCALER_TYPE}")
```

#### **Critical: Zero Data Leakage Implementation**
```python
def _split_features_labels(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Split DataFrame into features and labels WITH OPTIONAL SCALING"""
    
    # Expected feature columns (must match validation module)
    feature_columns = [
        'high_change', 'low_change', 'close_change', 'volume_change',
        'price_to_ma1440', 'price_to_ma43200', 
        'volume_to_ma1440', 'volume_to_ma43200'
    ]
    
    # Expected label columns (onehot format)
    label_columns = ['label_0', 'label_1', 'label_2']
    
    # Extract features and labels
    features = df[feature_columns].values.astype(np.float32)
    labels = df[label_columns].values.astype(np.float32)
    
    # FEATURE SCALING LOGIC - ZERO DATA LEAKAGE!
    if config.ENABLE_FEATURE_SCALING:
        original_features = features.copy()
        
        if fit_scaler and not self.scaler_fitted:
            # FIRST CALL - fit scaler on this data (should be TRAIN data only!)
            self.scaler = self._create_scaler()
            features = self.scaler.fit_transform(features)
            self.scaler_fitted = True
            
            # Calculate and store scaling statistics
            self._calculate_scaling_stats(original_features, features, 'train')
            
            # Save scaler for future use
            self._save_scaler()
            
        elif self.scaler_fitted:
            # SUBSEQUENT CALLS - transform only (validation data)
            features = self.scaler.transform(features)
            self._calculate_scaling_stats(original_features, features, 'validation')
            
        else:
            # Try to load existing scaler
            if self._load_scaler():
                features = self.scaler.transform(features)
                self._calculate_scaling_stats(original_features, features, 'loaded')
            else:
                raise ValueError("Scaler not fitted and no saved scaler found")
    
    return features, labels, feature_columns
```

### 2. **Scaler Persistence System**

#### **Scaler Saving with Metadata**
```python
def _save_scaler(self):
    """Save fitted scaler to disk"""
    if not self.scaler or not self.scaler_fitted:
        return
        
    scaler_path = os.path.join(config.get_model_output_dir(), config.get_scaler_filename())
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Create comprehensive scaler package
    scaler_package = {
        'scaler': self.scaler,                     # Fitted scaler object
        'scaler_type': config.SCALER_TYPE,         # Scaler type for validation
        'scaling_stats': self.scaling_stats,       # Scaling statistics
        'feature_names': [                         # Feature names for validation
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200', 
            'volume_to_ma1440', 'volume_to_ma43200'
        ],
        'config_params': {                         # Config parameters for validation
            'pair': config.PAIR,
            'future_window': config.FUTURE_WINDOW,
            'stop_loss': config.LONG_SL_PCT,
            'take_profit': config.LONG_TP_PCT
        }
    }
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_package, f)
        
    if config.VERBOSE_LOGGING:
        print(f"   ✅ Scaler saved: {config.get_scaler_filename()}")
```

#### **Scaler Loading with Validation**
```python
def _load_scaler(self):
    """Load existing scaler from disk"""
    scaler_path = os.path.join(config.get_model_output_dir(), config.get_scaler_filename())
    
    if not os.path.exists(scaler_path):
        return False
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler_package = pickle.load(f)
            
        self.scaler = scaler_package['scaler']
        self.scaling_stats = scaler_package.get('scaling_stats', {})
        self.scaler_fitted = True
        
        if config.VERBOSE_LOGGING:
            print(f"   📂 Scaler loaded: {config.get_scaler_filename()}")
            print(f"   📊 Scaler type: {scaler_package.get('scaler_type', 'unknown')}")
            
        return True
        
    except Exception as e:
        if config.VERBOSE_LOGGING:
            print(f"   ⚠️ Failed to load scaler: {e}")
        return False
```

### 3. **Scaling Statistics & Validation**

#### **Comprehensive Scaling Statistics**
```python
def _calculate_scaling_stats(self, original_features: np.ndarray, scaled_features: np.ndarray, data_split: str):
    """Calculate and store detailed scaling statistics"""
    feature_names = [
        'high_change', 'low_change', 'close_change', 'volume_change',
        'price_to_ma1440', 'price_to_ma43200', 
        'volume_to_ma1440', 'volume_to_ma43200'
    ]
    
    stats = {
        'data_split': data_split,
        'original': {
            'mean': np.mean(original_features, axis=0).tolist(),
            'std': np.std(original_features, axis=0).tolist(),
            'min': np.min(original_features, axis=0).tolist(),
            'max': np.max(original_features, axis=0).tolist(),
        },
        'scaled': {
            'mean': np.mean(scaled_features, axis=0).tolist(),
            'std': np.std(scaled_features, axis=0).tolist(),
            'min': np.min(scaled_features, axis=0).tolist(),
            'max': np.max(scaled_features, axis=0).tolist(),
        },
        'feature_names': feature_names
    }
    
    self.scaling_stats[data_split] = stats
    
    if config.VALIDATE_SCALING_STATS and config.VERBOSE_LOGGING:
        print(f"   📊 Scaling stats ({data_split}):")
        print(f"      Original - Mean: {np.mean(stats['original']['mean']):.3f}, Std: {np.mean(stats['original']['std']):.3f}")
        print(f"      Scaled   - Mean: {np.mean(stats['scaled']['mean']):.3f}, Std: {np.mean(stats['scaled']['std']):.3f}")
```

## ⚖️ **CLASS BALANCING SYSTEM**

### 1. **Class Balancing Overview**

#### **Main Balancing Function**
```python
def apply_class_balancing(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply class balancing based on configuration
    
    Args:
        features: Input features array
        labels: Labels array (categorical)
        
    Returns:
        Balanced features and labels
    """
    if not config.ENABLE_CLASS_BALANCING or config.CLASS_BALANCING_METHOD == "none":
        return features, labels
        
    print(f"\n⚖️ APPLYING CLASS BALANCING...")
    print(f"   Method: {config.CLASS_BALANCING_METHOD}")
    
    # Convert labels to numerical if needed
    if labels.ndim > 1:
        labels_1d = np.argmax(labels, axis=1)  # Convert onehot to class indices
    else:
        labels_1d = labels.copy()
    
    # Count classes
    unique_classes, class_counts = np.unique(labels_1d, return_counts=True)
    class_names = ['SHORT', 'HOLD', 'LONG']
    
    print(f"   📊 Original class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        pct = (count / len(labels_1d)) * 100
        print(f"      {class_names[cls]}: {count:,} ({pct:.1f}%)")
    
    if config.CLASS_BALANCING_METHOD == "systematic_undersampling":
        return self._systematic_undersampling(features, labels, labels_1d, unique_classes, class_counts)
    elif config.CLASS_BALANCING_METHOD == "class_weights":
        # Class weights are applied in model training, not here
        print(f"   💡 Class weights will be applied during training")
        return features, labels
    else:
        raise ValueError(f"Unknown balancing method: {config.CLASS_BALANCING_METHOD}")
```

### 2. **Systematic Undersampling Algorithm**

#### **Advanced Undersampling Implementation**
```python
def _systematic_undersampling(self, features: np.ndarray, labels: np.ndarray, labels_1d: np.ndarray, 
                            unique_classes: np.ndarray, class_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply systematic undersampling - reduce majority classes to match minority class
    Uses systematic sampling (every N-th sample) to preserve diversity
    """
    print(f"   🎯 Applying systematic undersampling...")
    
    # Find target size (smallest class or minimum threshold)
    min_class_size = np.min(class_counts)
    target_size = max(min_class_size, config.UNDERSAMPLING_MIN_SAMPLES)
    
    if target_size < min_class_size:
        print(f"   ⚠️ Target size ({config.UNDERSAMPLING_MIN_SAMPLES:,}) larger than minority class ({min_class_size:,})")
        print(f"   📉 Using minority class size: {min_class_size:,}")
        target_size = min_class_size
    
    print(f"   🎯 Target samples per class: {target_size:,}")
    
    # Collect balanced indices
    balanced_indices = []
    np.random.seed(config.UNDERSAMPLING_SEED)
    
    for cls in unique_classes:
        class_indices = np.where(labels_1d == cls)[0]
        class_size = len(class_indices)
        class_name = ['SHORT', 'HOLD', 'LONG'][cls]
        
        if class_size <= target_size:
            # Take all samples from minority classes
            selected_indices = class_indices
            print(f"   ✅ {class_name}: {len(selected_indices):,}/{class_size:,} (all samples)")
        else:
            # SYSTEMATIC SAMPLING for majority classes
            step = class_size // target_size
            if step < 1:
                step = 1
            
            # Start from random offset to avoid bias
            start_offset = np.random.randint(0, step) if step > 1 else 0
            selected_indices = class_indices[start_offset::step][:target_size]
            
            print(f"   📉 {class_name}: {len(selected_indices):,}/{class_size:,} (every {step}-th sample)")
        
        balanced_indices.extend(selected_indices)
    
    # Convert to array and sort to maintain TEMPORAL ORDER
    balanced_indices = np.array(balanced_indices)
    balanced_indices = np.sort(balanced_indices)  # ← CRITICAL for time series!
    
    # Apply balancing
    balanced_features = features[balanced_indices]
    balanced_labels = labels[balanced_indices]
    
    # Verify results and report
    if balanced_labels.ndim > 1:
        balanced_labels_1d = np.argmax(balanced_labels, axis=1)
    else:
        balanced_labels_1d = balanced_labels
        
    unique_balanced, balanced_counts = np.unique(balanced_labels_1d, return_counts=True)
    
    print(f"   📊 Balanced class distribution:")
    total_balanced = len(balanced_labels_1d)
    for cls, count in zip(unique_balanced, balanced_counts):
        pct = (count / total_balanced) * 100
        class_name = ['SHORT', 'HOLD', 'LONG'][cls]
        print(f"      {class_name}: {count:,} ({pct:.1f}%)")
    
    reduction_pct = ((len(features) - len(balanced_features)) / len(features) * 100)
    print(f"   📈 Dataset size: {len(features):,} → {len(balanced_features):,} (reduction: {reduction_pct:.1f}%)")
    
    return balanced_features, balanced_labels
```

#### **Key Algorithm Features:**
1. **Target Size Calculation:** `max(min_class_size, UNDERSAMPLING_MIN_SAMPLES)`
2. **Systematic Sampling:** Every N-th sample (preserves diversity)
3. **Random Offset:** Prevents bias in systematic sampling
4. **Temporal Order Preservation:** Sort indices after selection
5. **Safety Limits:** Minimum samples per class protection
6. **Comprehensive Reporting:** Before/after distribution analysis

## 📅 **DATE RANGE FILTERING**

### **Flexible Date Filtering System**
```python
def _filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame by date range from config"""
    if not config.ENABLE_DATE_FILTER:
        if config.VERBOSE_LOGGING:
            print(f"   📅 Date filtering: DISABLED")
        return df
        
    if not config.START_DATE or not config.END_DATE:
        if config.VERBOSE_LOGGING:
            print(f"   📅 Date filtering: No dates specified")
        return df
        
    # METHOD 1: Check if datetime index exists (training-ready files use datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        from datetime import datetime
        start_dt = datetime.strptime(config.START_DATE, '%Y-%m-%d')
        end_dt = datetime.strptime(config.END_DATE, '%Y-%m-%d')
        
        # Filter by date range using datetime index
        original_count = len(df)
        df_filtered = df[
            (df.index >= start_dt) & 
            (df.index <= end_dt)
        ].copy()
        
        if config.VERBOSE_LOGGING:
            print(f"   📅 Date filtering: {config.START_DATE} to {config.END_DATE}")
            print(f"   📊 Filtered: {original_count:,} → {len(df_filtered):,} samples")
            if len(df_filtered) > 0:
                print(f"   📅 Actual range: {df_filtered.index.min()} to {df_filtered.index.max()}")
            
        if len(df_filtered) == 0:
            raise ValueError(f"No data found in date range {config.START_DATE} to {config.END_DATE}")
            
        return df_filtered
        
    # METHOD 2: Fallback - Check if timestamp column exists
    elif 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        start_dt = datetime.strptime(config.START_DATE, '%Y-%m-%d')
        end_dt = datetime.strptime(config.END_DATE, '%Y-%m-%d')
        
        df_filtered = df[
            (df['timestamp'] >= start_dt) & 
            (df['timestamp'] <= end_dt)
        ].reset_index(drop=True)
        
        # Logging and validation...
        
    else:
        print(f"⚠️ No datetime index or timestamp column - skipping date filter")
        return df
```

## 📊 **MAIN DATA LOADING PIPELINE**

### **Complete Loading Workflow**
```python
def load_training_data(self) -> Dict[str, Any]:
    """
    📁 MAIN DATA LOADING FUNCTION
    Complete pipeline: validate → load → parse → filter → scale → balance
    """
    print(f"\n📁 LOADING TRAINING DATA...")
    
    # 1. Validate file exists (hard fail with clear guidance)
    if not self.validate_file_exists():
        raise FileNotFoundError(f"Training-ready file not found: {self.expected_filename}")
    
    # 2. Load and validate data
    try:
        df = pd.read_feather(self.full_file_path)
        print(f"   ✅ File loaded: {len(df):,} samples")
    except Exception as e:
        raise IOError(f"Error loading feather file: {e}")
    
    # 3. Parse and validate filename parameters
    file_params = self._parse_filename_parameters(self.expected_filename)
    if not self._validate_parameters_compatibility(file_params):
        raise ValueError("Parameter compatibility validation failed")
    
    # 4. Apply date filtering (optional)
    df = self._filter_by_date_range(df)
    
    # 5. Split features and labels WITH SCALING
    features, labels, feature_columns = self._split_features_labels(df, fit_scaler=True)
    
    # 6. Apply class balancing
    features, labels = self.apply_class_balancing(features, labels)
    
    # 7. Return comprehensive data info
    return {
        'features': features,
        'labels': labels,
        'feature_columns': feature_columns,
        'samples_count': len(features),
        'features_shape': features.shape,
        'labels_shape': labels.shape,
        'label_format': 'onehot',
        'file_params': file_params,
        'compatibility_validated': True,
        'scaling_enabled': config.ENABLE_FEATURE_SCALING,
        'scaler_fitted': self.scaler_fitted,
        'scaling_stats': self.scaling_stats if hasattr(self, 'scaling_stats') else {},
        'balancing_applied': config.ENABLE_CLASS_BALANCING,
        'balancing_method': config.CLASS_BALANCING_METHOD if config.ENABLE_CLASS_BALANCING else 'none'
    }
```

## 📋 **EXPECTED DATA FORMAT**

### **Input File Structure**
```python
# Expected columns in training-ready .feather file:
REQUIRED_FEATURE_COLUMNS = [
    'high_change',      # Price change features
    'low_change',
    'close_change',
    'volume_change',    # Volume change feature
    'price_to_ma1440',  # Moving average ratio features  
    'price_to_ma43200',
    'volume_to_ma1440', # Volume MA ratio features
    'volume_to_ma43200'
]

REQUIRED_LABEL_COLUMNS = [
    'label_0',  # SHORT probability (one-hot encoded)
    'label_1',  # HOLD probability
    'label_2'   # LONG probability
]

# Optional datetime index for date filtering
# Format: pandas DatetimeIndex or 'timestamp' column
```

### **Data Validation Checks**
```python
# Validate columns exist
missing_features = [col for col in feature_columns if col not in df.columns]
missing_labels = [col for col in label_columns if col not in df.columns]

if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")
if missing_labels:
    raise ValueError(f"Missing label columns: {missing_labels}")

# Validate data types and ranges
features = df[feature_columns].values.astype(np.float32)
labels = df[label_columns].values.astype(np.float32)
```

## 🎯 **USAGE EXAMPLES**

### **Example 1: Basic Loading**
```python
# Initialize data loader
data_loader = TrainingDataLoader(
    explicit_path="/freqtrade/user_data/trening2/inputs/",
    pair="BTCUSDT"
)

# Load training data
data_info = data_loader.load_training_data()

# Access loaded data
features = data_info['features']        # (N, 8) numpy array
labels = data_info['labels']           # (N, 3) numpy array (onehot)
samples_count = data_info['samples_count']
```

### **Example 2: With Feature Scaling**
```python
# config.py settings:
ENABLE_FEATURE_SCALING = True
SCALER_TYPE = "robust"
SCALER_FIT_ONLY_TRAIN = True

# Data loader automatically:
# 1. Fits scaler on loaded data (train split)
# 2. Saves scaler to disk
# 3. Provides scaling statistics
```

### **Example 3: With Class Balancing**
```python
# config.py settings:
ENABLE_CLASS_BALANCING = True
CLASS_BALANCING_METHOD = "systematic_undersampling"
UNDERSAMPLING_MIN_SAMPLES = 50000

# Data loader automatically:
# 1. Analyzes class distribution
# 2. Applies systematic undersampling
# 3. Reports before/after statistics
```

---

**🎯 KLUCZOWE ZALETY DATA LOADER V3:**
- ✅ **Explicit Path Configuration** - no auto-detection chaos
- ✅ **Zero Data Leakage** - proper scaler fitting on train only
- ✅ **Parameter Validation** - filename compatibility checking
- ✅ **Advanced Class Balancing** - systematic undersampling preserving temporal order
- ✅ **Hard Fail Error Handling** - clear error messages with solutions
- ✅ **Comprehensive Statistics** - detailed scaling and balancing reports
- ✅ **Production Ready** - scaler persistence, date filtering, validation

**📈 NEXT:** [05_Sequence_generator.md](./05_Sequence_generator.md) - Memory-efficient generators documentation 