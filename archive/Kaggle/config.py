"""
🎯 STANDALONE TRAINING MODULE V3 - KAGGLE CONFIGURATION
100% Standalone | Config-driven | Kaggle-optimized
"""

# =============================================================================
# TRAINING CONFIGURATION - KAGGLE ENVIRONMENT
# =============================================================================

# CRYPTO PAIR
PAIR = "BTCUSDT"  # Explicit pair selection

# DATA PATHS (RunPod environment paths)
TRAINING_DATA_PATH = "/workspace/crypto/validation_and_labeling/output/"  # RunPod path to training-ready files
OUTPUT_BASE_PATH = "/workspace/crypto/Kaggle/output/"  # RunPod output directory

# TRAINING PARAMETERS
EPOCHS = 100
BATCH_SIZE = 8192
VALIDATION_SPLIT = 0.1
TRAIN_SPLIT = 0.9  # Added missing variable

# CALLBACKS CONFIGURATION
EARLY_STOPPING_PATIENCE = 15         # Number of epochs with no improvement after which training will be stopped
REDUCE_LR_PATIENCE = 5               # Number of epochs with no improvement after which learning rate will be reduced
REDUCE_LR_FACTOR = 0.5               # Factor by which the learning rate will be reduced
MIN_LEARNING_RATE = 1e-7             # Lower bound on the learning rate

# CLASS BALANCING CONFIGURATION
ENABLE_CLASS_BALANCING = True                    # Enable/disable class balancing
CLASS_BALANCING_METHOD = "systematic_undersampling"  # "systematic_undersampling", "class_weights", "focal_loss", "none"

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

# FOCAL LOSS SETTINGS (if using focal_loss method)
FOCAL_LOSS_GAMMA = 2.0  # Focusing parameter (e.g., 2.0). Higher values focus more on hard examples.
FOCAL_LOSS_ALPHA_WEIGHTS = [3.0, 0.1, 3.0] # EXTREME weighting to fight HOLD bias - 30x stronger SHORT/LONG focus

# DATE RANGE CONTROL (Optional - if None, uses full dataset)
START_DATE = "2021-01-01"  # Format: YYYY-MM-DD or None for full range
END_DATE = "2025-05-30"    # Format: YYYY-MM-DD or None for full range
ENABLE_DATE_FILTER = True  # Set to False to disable date filtering

# FEATURE SCALING CONFIGURATION
ENABLE_FEATURE_SCALING = True     # Enable/disable feature scaling
SCALER_TYPE = "robust"            # "standard", "robust", "minmax"
SCALER_FIT_ONLY_TRAIN = True      # Fit scaler only on train data (prevent data leakage)
VALIDATE_SCALING_STATS = True     # Print scaling statistics for validation

# MODEL PARAMETERS
SEQUENCE_LENGTH = 120  # Time window for LSTM
LSTM_UNITS = [ 128, 64, 32]  # Dodanie 4. warstwy + większe jednostki
DENSE_UNITS = [ 64, 32]      # 3 warstwy Dense + większe
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00001

# === CONFIDENCE THRESHOLDING ===
USE_CONFIDENCE_THRESHOLDING: bool = True     # Włącz/wyłącz confidence thresholding
CONFIDENCE_THRESHOLD_SHORT: float = 0.47    # 47% pewności dla SHORT
CONFIDENCE_THRESHOLD_LONG: float = 0.47     # 47% pewności dla LONG  
CONFIDENCE_THRESHOLD_HOLD: float = 0.3      # 45% wystarczy dla HOLD


# TRADING PARAMETERS (will be validated against filename)
LONG_TP_PCT = 1.0    # Take profit %
LONG_SL_PCT = 0.5    # Stop loss %
SHORT_TP_PCT = 1.0   # Take profit %
SHORT_SL_PCT = 0.5   # Stop loss %

FUTURE_WINDOW = 120  # Future bars for labeling

# OUTPUT SETTINGS
OVERWRITE_EXISTING = True  # Overwrite model if exists
SAVE_METADATA = True       # Save training metadata JSON
VERBOSE_LOGGING: bool = True             # Detailed logging (turn off for production)

# MODEL ARCHITECTURE
INPUT_FEATURES = 8    # Number of technical indicators
OUTPUT_CLASSES = 3    # SHORT, HOLD, LONG

# FEATURE NAMES (for sequence generator compatibility)
FEATURES = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200', 
    'volume_to_ma1440', 'volume_to_ma43200'
]

# SEQUENCE GENERATOR COMPATIBILITY
WINDOW_SIZE = SEQUENCE_LENGTH  # Alias for backward compatibility

# === DIAGNOSTICS & LOGGING ===
SAVE_VALIDATION_PREDICTIONS: bool = True      # Save validation set predictions to a CSV file

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_expected_filename():
    """Generate expected training-ready filename based on config"""
    return f"{PAIR}_TF-1m__FW-{FUTURE_WINDOW:03d}__SL-{int(LONG_SL_PCT*100):03d}__TP-{int(LONG_TP_PCT*100):03d}__training_ready.feather"

def get_model_output_dir():
    """Get output directory for this crypto pair"""
    return f"{OUTPUT_BASE_PATH}models/{PAIR}/"

def get_model_filename():
    """Generate model filename with parameters"""
    return f"model_{PAIR}_FW{FUTURE_WINDOW:03d}_SL{int(LONG_SL_PCT*100):03d}_TP{int(LONG_TP_PCT*100):03d}.keras"

def get_scaler_filename():
    """Generate scaler filename with parameters"""
    return f"scaler_{PAIR}_FW{FUTURE_WINDOW:03d}_SL{int(LONG_SL_PCT*100):03d}_TP{int(LONG_TP_PCT*100):03d}.pkl"

def get_metadata_filename():
    """Generate metadata filename with parameters"""
    return f"metadata_{PAIR}_FW{FUTURE_WINDOW:03d}_SL{int(LONG_SL_PCT*100):03d}_TP{int(LONG_TP_PCT*100):03d}.json"

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Validate paths (allow both absolute and relative for Kaggle/RunPod compatibility)
    if not (TRAINING_DATA_PATH.startswith('/') or TRAINING_DATA_PATH.startswith('.') or TRAINING_DATA_PATH.startswith('input') or TRAINING_DATA_PATH.startswith('Kaggle')):
        errors.append("TRAINING_DATA_PATH must be absolute path or relative Kaggle path")
    
    # Validate pair
    if not PAIR or len(PAIR) < 6:
        errors.append("PAIR must be valid crypto pair (e.g. BTCUSDT)")
    
    # Validate training params
    if EPOCHS <= 0:
        errors.append("EPOCHS must be positive")
    
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    # Validate callbacks params
    if EARLY_STOPPING_PATIENCE <= 0:
        errors.append("EARLY_STOPPING_PATIENCE must be positive")
    
    if REDUCE_LR_PATIENCE <= 0:
        errors.append("REDUCE_LR_PATIENCE must be positive")
    
    if not (0 < REDUCE_LR_FACTOR < 1):
        errors.append(f"REDUCE_LR_FACTOR must be between 0 and 1, got {REDUCE_LR_FACTOR}")
    
    if MIN_LEARNING_RATE <= 0:
        errors.append("MIN_LEARNING_RATE must be positive")
    
    # Validate date range
    if ENABLE_DATE_FILTER:
        if START_DATE and END_DATE:
            try:
                from datetime import datetime
                start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
                end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
                if start_dt >= end_dt:
                    errors.append(f"START_DATE must be before END_DATE")
            except ValueError as e:
                errors.append(f"Invalid date format: {e}")
        elif START_DATE and not END_DATE:
            errors.append("END_DATE required when START_DATE is set")
        elif END_DATE and not START_DATE:
            errors.append("START_DATE required when END_DATE is set")
    
    # Validate model params
    if SEQUENCE_LENGTH <= 0:
        errors.append("SEQUENCE_LENGTH must be positive")
    
    # Validate TP/SL params
    if not (0 < LONG_TP_PCT <= 100):
        errors.append(f"LONG_TP_PCT must be 0-100%, got {LONG_TP_PCT}")
    if not (0 < LONG_SL_PCT <= 100):
        errors.append(f"LONG_SL_PCT must be 0-100%, got {LONG_SL_PCT}")
    
    # Validate LSTM units
    if not LSTM_UNITS or len(LSTM_UNITS) == 0:
        errors.append("LSTM_UNITS cannot be empty")
    
    # Validate dropout
    if not (0 <= DROPOUT_RATE <= 1):
        errors.append(f"DROPOUT_RATE must be 0-1, got {DROPOUT_RATE}")
    
    # Validate scaler configuration
    if ENABLE_FEATURE_SCALING:
        valid_scalers = ["standard", "robust", "minmax"]
        if SCALER_TYPE not in valid_scalers:
            errors.append(f"SCALER_TYPE must be one of {valid_scalers}, got {SCALER_TYPE}")
    
    # Validate confidence thresholding
    if not (0 <= CONFIDENCE_THRESHOLD_SHORT <= 1):
        errors.append(f"CONFIDENCE_THRESHOLD_SHORT must be 0-1, got {CONFIDENCE_THRESHOLD_SHORT}")
    if not (0 <= CONFIDENCE_THRESHOLD_LONG <= 1):
        errors.append(f"CONFIDENCE_THRESHOLD_LONG must be 0-1, got {CONFIDENCE_THRESHOLD_LONG}")
    if not (0 <= CONFIDENCE_THRESHOLD_HOLD <= 1):
        errors.append(f"CONFIDENCE_THRESHOLD_HOLD must be 0-1, got {CONFIDENCE_THRESHOLD_HOLD}")
    
    if USE_CONFIDENCE_THRESHOLDING:
        max_threshold = max(CONFIDENCE_THRESHOLD_SHORT,
                           CONFIDENCE_THRESHOLD_LONG,
                           CONFIDENCE_THRESHOLD_HOLD)
        if max_threshold > 0.9:
            errors.append(f"Maximum confidence threshold too high: {max_threshold:.2f} (may reject all predictions)")
    
    return errors

def print_config_summary():
    """Print configuration summary"""
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
    print(f"   Reduce LR Patience: {REDUCE_LR_PATIENCE}")
    print(f"   LR Reduction Factor: {REDUCE_LR_FACTOR}")
    print("")
    print(f"📅 Date Range:")
    if ENABLE_DATE_FILTER and START_DATE and END_DATE:
        print(f"   Start: {START_DATE}")
        print(f"   End: {END_DATE}")
        print(f"   Filtering: ENABLED")
    else:
        print(f"   Range: FULL DATASET")
        print(f"   Filtering: DISABLED")
    print("")
    print(f"📏 Feature Scaling:")
    if ENABLE_FEATURE_SCALING:
        print(f"   Enabled: YES")
        print(f"   Scaler Type: {SCALER_TYPE}")
        print(f"   Fit Only Train: {SCALER_FIT_ONLY_TRAIN}")
        print(f"   Validation Stats: {VALIDATE_SCALING_STATS}")
    else:
        print(f"   Enabled: NO")
    print("")
    print(f"⚖️ Class Balancing:")
    if ENABLE_CLASS_BALANCING:
        print(f"   Enabled: YES")
        print(f"   Method: {CLASS_BALANCING_METHOD}")
        if CLASS_BALANCING_METHOD == "systematic_undersampling":
            print(f"   Min Samples: {UNDERSAMPLING_MIN_SAMPLES:,}")
            print(f"   Preserve Ratio: {UNDERSAMPLING_PRESERVE_RATIO}")
            print(f"   Seed: {UNDERSAMPLING_SEED}")
        elif CLASS_BALANCING_METHOD == "class_weights":
            print(f"   Weight Method: {CLASS_WEIGHT_METHOD}")
            if CLASS_WEIGHT_METHOD == "manual":
                print(f"   Manual Weights: {MANUAL_CLASS_WEIGHTS}")
    else:
        print(f"   Enabled: NO")
    print("")
    print(f"🧠 Model:")
    print(f"   Sequence Length: {SEQUENCE_LENGTH}")
    print(f"   LSTM Units: {LSTM_UNITS}")
    print(f"   Dense Units: {DENSE_UNITS}")
    print(f"   Dropout: {DROPOUT_RATE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print("")
    print(f"🎯 Confidence Thresholding:")
    print(f"   Enabled: {USE_CONFIDENCE_THRESHOLDING}")
    if USE_CONFIDENCE_THRESHOLDING:
        print(f"   SHORT threshold: {CONFIDENCE_THRESHOLD_SHORT:.1%}")
        print(f"   LONG threshold: {CONFIDENCE_THRESHOLD_LONG:.1%}")
        print(f"   HOLD threshold: {CONFIDENCE_THRESHOLD_HOLD:.1%}")
    print("")
    print(f"💰 Trading:")
    print(f"   TP: {LONG_TP_PCT}% | SL: {LONG_SL_PCT}% | FW: {FUTURE_WINDOW}")
    print("")
    print(f"💾 Output:")
    print(f"   Model: {get_model_filename()}")
    print(f"   Scaler: {get_scaler_filename()}")
    print(f"   Metadata: {get_metadata_filename()}")
    print("=" * 60)

# =============================================================================
# LEGACY COMPATIBILITY HELPERS
# =============================================================================

def get_input_shape():
    """Get input shape for model"""
    return (SEQUENCE_LENGTH, INPUT_FEATURES)

def get_output_classes():
    """Get number of output classes"""
    return OUTPUT_CLASSES

def get_label_shape():
    """Get label shape (sparse format for V4)"""
    return ()  # Sparse shape: 1D array, no second dimension

if __name__ == "__main__":
    print_config_summary()
    errors = validate_config()
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("\n✅ Configuration Valid")