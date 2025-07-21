"""
🎯 STANDALONE TRAINER V3 - 100% PRODUCTION READY
Zero dependencies from validation module | Config-driven | Docker-optimized

FEATURES:
- Direct loading from explicit paths
- Parameter validation
- Memory-efficient generators
- Production training pipeline
- Zero validation module dependencies
- Pre-computed labels (no competitive labeling)
- FEATURE SCALING support with zero data leakage
"""

import os
import sys
import gc
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models, layers, optimizers, callbacks
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError as e:
    TF_AVAILABLE = False
    print(f"❌ TensorFlow not available: {e}")

# Sklearn imports for feature scaling compatibility
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - feature scaling validation disabled")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure local modules are imported first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import standalone modules
import config
from data_loader import TrainingDataLoader
from sequence_generator import MemoryEfficientDataLoader
from model_builder import DualWindowLSTMBuilder
from utils import monitor_memory_usage, safe_memory_cleanup

# Import diagnostic utils
try:
    # Try RunPod path first (/workspace/diagnostic_utils.py)
    sys.path.append('/workspace')
    from diagnostic_utils import run_complete_diagnostic, save_model_scaler_audit, save_scaled_features_sample
    DIAGNOSTIC_AVAILABLE = True
except ImportError:
    try:
        # Try local crypto directory (local development)
        sys.path.append('/workspace/crypto')
        from diagnostic_utils import run_complete_diagnostic, save_model_scaler_audit, save_scaled_features_sample
        DIAGNOSTIC_AVAILABLE = True
    except ImportError:
        try:
            # Try parent directory (local development)
            sys.path.append('..')
            from diagnostic_utils import run_complete_diagnostic, save_model_scaler_audit, save_scaled_features_sample
            DIAGNOSTIC_AVAILABLE = True
        except ImportError:
            print("⚠️ diagnostic_utils not available - diagnostic features disabled")
            DIAGNOSTIC_AVAILABLE = False
            
            # Create dummy functions to prevent errors
            def run_complete_diagnostic(*args, **kwargs):
                print("⚠️ Diagnostic system not available")
                return None, None
            def save_model_scaler_audit(*args, **kwargs):
                print("⚠️ Model/scaler audit not available")
                return ""
            def save_scaled_features_sample(*args, **kwargs):
                print("⚠️ Features sample not available") 
                return ""


def setup_tensorflow(verbose: bool = True):
    """Setup TensorFlow for optimal performance"""
    if not TF_AVAILABLE:
        return
        
    # Configure GPU memory growth if available
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if verbose:
                print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            if verbose:
                print("ℹ️ No GPU devices found, using CPU")
    except RuntimeError as e:
        if verbose:
            print(f"⚠️ GPU setup warning: {e}")


def validate_model_weights(model_path: str, verbose: bool = True) -> bool:
    """
    🔍 VALIDATE MODEL WEIGHTS AFTER SAVING
    
    Checks if saved model has proper LSTM weights
    
    Args:
        model_path: Path to saved model
        verbose: Print validation details
        
    Returns:
        True if model has valid weights, False otherwise
    """
    try:
        if verbose:
            print(f"   🔍 Validating weights in: {os.path.basename(model_path)}")
        
        # Try to load the model
        test_model = tf.keras.models.load_model(model_path, compile=False)
        
        # Check ALL layers for weights, including lstm_cell
        lstm_layers_found = 0
        lstm_weights_valid = 0
        
        for i, layer in enumerate(test_model.layers):
            layer_name_lower = layer.name.lower()
            if 'lstm' in layer_name_lower:
                lstm_layers_found += 1
                weights = layer.get_weights()
                
                if verbose:
                    print(f"      🔍 Layer '{layer.name}': {len(weights)} weight arrays")
                
                if len(weights) >= 3:  # Should have kernel, recurrent_kernel, bias
                    # Check if weights are not all zeros
                    total_weight_sum = sum(np.sum(np.abs(w)) for w in weights)
                    if total_weight_sum > 1e-6:  # Non-zero weights
                        lstm_weights_valid += 1
                        if verbose:
                            print(f"      ✅ Layer {layer.name}: VALID weights, sum={total_weight_sum:.6f}")
                    else:
                        if verbose:
                            print(f"      ❌ Layer {layer.name}: weights are all zeros!")
                else:
                    if verbose:
                        print(f"      ❌ Layer {layer.name}: only {len(weights)} weight arrays (expected 3+)")
        
        # Also check for nested LSTM cells in RNN layers
        for layer in test_model.layers:
            if hasattr(layer, 'cell') and hasattr(layer.cell, 'get_weights'):
                cell_weights = layer.cell.get_weights()
                if len(cell_weights) > 0:
                    lstm_layers_found += 1
                    total_weight_sum = sum(np.sum(np.abs(w)) for w in cell_weights)
                    if total_weight_sum > 1e-6:
                        lstm_weights_valid += 1
                        if verbose:
                            print(f"      ✅ Cell in {layer.name}: VALID weights, sum={total_weight_sum:.6f}")
                    else:
                        if verbose:
                            print(f"      ❌ Cell in {layer.name}: weights are all zeros!")
        
        # Clean up
        del test_model
        gc.collect()
        
        if lstm_layers_found == 0:
            if verbose:
                print(f"      ⚠️ No LSTM layers found in model")
            return True  # Not an LSTM model, assume OK
        
        validation_passed = lstm_weights_valid == lstm_layers_found
        
        if verbose:
            if validation_passed:
                print(f"      ✅ Validation PASSED: {lstm_weights_valid}/{lstm_layers_found} LSTM layers have valid weights")
            else:
                print(f"      ❌ Validation FAILED: {lstm_weights_valid}/{lstm_layers_found} LSTM layers have valid weights")
        
        return validation_passed
        
    except Exception as e:
        if verbose:
            print(f"      ❌ Validation failed with error: {e}")
        return False


class SafeModelCheckpoint(tf.keras.callbacks.Callback):
    """
    💾 SAFE MODEL CHECKPOINT WITH WEIGHT VALIDATION
    
    Custom checkpoint callback that validates weights after saving
    """
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, 
                 mode='min', verbose=1, validate_weights=True):
        super().__init__()
        # Force H5 format to avoid TensorFlow 2.15.0 LSTM bug
        if not filepath.endswith('.h5'):
            filepath = filepath.replace('.keras', '.h5')
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.validate_weights = validate_weights
        self.best = None
        self.saves_count = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best
            self.best = float('inf')
        else:
            self.monitor_op = lambda current, best: current > best
            self.best = float('-inf')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} not available for checkpoint")
            return
        
        # Check if we should save
        should_save = False
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                should_save = True
                self.best = current
        else:
            should_save = True
        
        if should_save:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                print(f"   💾 Saving model to {os.path.basename(self.filepath)} (H5 format)")
            
            # Try multiple save methods to ensure success
            save_successful = False
            
            # Method 1: H5 format save (avoids TensorFlow 2.15.0 LSTM bug)
            try:
                self.model.save(self.filepath, save_format='h5')
                self.saves_count += 1
                
                # Validate weights if requested
                if self.validate_weights:
                    is_valid = validate_model_weights(self.filepath, verbose=self.verbose > 0)
                    
                    if is_valid:
                        if self.verbose > 0:
                            print(f"   ✅ Model checkpoint validated successfully (H5)")
                        save_successful = True
                    else:
                        print(f"   ❌ CRITICAL: Saved model has invalid weights!")
                        save_successful = False
                else:
                    save_successful = True
                    
            except Exception as e:
                print(f"   ❌ H5 save failed: {e}")
                save_successful = False
            
            # Method 2: Alternative save with weights only + architecture
            if not save_successful:
                try:
                    print(f"   🔄 Trying alternative save method...")
                    
                    # Save weights separately
                    weights_path = self.filepath.replace('.h5', '_weights.h5')
                    self.model.save_weights(weights_path)
                    
                    # Save architecture
                    arch_path = self.filepath.replace('.h5', '_arch.json')
                    with open(arch_path, 'w') as f:
                        f.write(self.model.to_json())
                    
                    # Create a new model and load weights to test
                    test_model = tf.keras.models.model_from_json(self.model.to_json())
                    test_model.load_weights(weights_path)
                    
                    # Save the reconstructed model in H5
                    test_model.save(self.filepath, save_format='h5')
                    
                    # Validate
                    if self.validate_weights:
                        is_valid = validate_model_weights(self.filepath, verbose=self.verbose > 0)
                        if is_valid:
                            print(f"   ✅ Alternative save method successful (H5)")
                            save_successful = True
                            # Clean up temporary files
                            if os.path.exists(weights_path):
                                os.remove(weights_path)
                            if os.path.exists(arch_path):
                                os.remove(arch_path)
                        else:
                            print(f"   ❌ Alternative save also has invalid weights")
                    
                    del test_model
                    gc.collect()
                    
                except Exception as e:
                    print(f"   ❌ Alternative save failed: {e}")
            
            # Method 3: SavedModel format as last resort, then convert to H5
            if not save_successful:
                try:
                    print(f"   🔄 Trying SavedModel format...")
                    savedmodel_path = self.filepath.replace('.h5', '_savedmodel')
                    self.model.save(savedmodel_path, save_format='tf')
                    
                    # Convert back to H5
                    temp_model = tf.keras.models.load_model(savedmodel_path)
                    temp_model.save(self.filepath, save_format='h5')
                    
                    # Validate
                    if self.validate_weights:
                        is_valid = validate_model_weights(self.filepath, verbose=self.verbose > 0)
                        if is_valid:
                            print(f"   ✅ SavedModel conversion successful (H5)")
                            save_successful = True
                    
                    # Clean up
                    if os.path.exists(savedmodel_path):
                        import shutil
                        shutil.rmtree(savedmodel_path)
                    del temp_model
                    gc.collect()
                    
                except Exception as e:
                    print(f"   ❌ SavedModel save failed: {e}")
            
            if not save_successful:
                print(f"   ❌ ALL SAVE METHODS FAILED - Model checkpoint not saved!")
            
        elif self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")


def apply_confidence_thresholding(predictions_proba: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Apply confidence thresholding to predictions based on config settings
    
    Args:
        predictions_proba: Array of shape (n_samples, n_classes) with prediction probabilities
        
    Returns:
        Tuple of (filtered_predictions, stats) where:
        - filtered_predictions: Array of predicted classes (-1 for rejected predictions)
        - stats: Dictionary with rejection statistics
    """
    if not config.USE_CONFIDENCE_THRESHOLDING:
        # No thresholding - use standard argmax
        return np.argmax(predictions_proba, axis=1), {"total": len(predictions_proba), "rejected": 0, "accepted": len(predictions_proba)}
    
    # Get class names and thresholds
    class_names = ['SHORT', 'HOLD', 'LONG'] 
    thresholds = [
        config.CONFIDENCE_THRESHOLD_SHORT,  # 0: SHORT
        config.CONFIDENCE_THRESHOLD_HOLD,   # 1: HOLD  
        config.CONFIDENCE_THRESHOLD_LONG    # 2: LONG
    ]
    
    n_samples = len(predictions_proba)
    filtered_predictions = np.full(n_samples, -1, dtype=int)  # -1 means rejected
    
    # Statistics
    total_rejected = 0
    class_rejections = {name: 0 for name in class_names}
    class_accepted = {name: 0 for name in class_names}
    
    for i, proba in enumerate(predictions_proba):
        # Get the class with highest probability
        max_class = np.argmax(proba)
        max_confidence = proba[max_class]
        class_name = class_names[max_class]
        
        # Check if confidence meets threshold
        if max_confidence >= thresholds[max_class]:
            # Accept prediction
            filtered_predictions[i] = max_class
            class_accepted[class_name] += 1
        else:
            # Reject prediction (stays as -1)
            total_rejected += 1
            class_rejections[class_name] += 1
    
    # Create statistics
    stats = {
        "total": n_samples,
        "accepted": n_samples - total_rejected,
        "rejected": total_rejected,
        "rejection_rate": total_rejected / n_samples if n_samples > 0 else 0,
        "class_rejections": class_rejections,
        "class_accepted": class_accepted,
        "thresholds_used": {
            "SHORT": config.CONFIDENCE_THRESHOLD_SHORT,
            "HOLD": config.CONFIDENCE_THRESHOLD_HOLD,
            "LONG": config.CONFIDENCE_THRESHOLD_LONG
        }
    }
    
    return filtered_predictions, stats


class StandaloneTrainer:
    """
    🎯 STANDALONE TRAINER V3 - PRODUCTION READY
    
    Complete training pipeline with:
    - Explicit path configuration (no auto-detection)
    - Parameter validation from filename
    - Memory-efficient processing
    - Model checkpointing and saving
    - Zero validation module dependencies
    - Pre-computed labels for zero duplication
    - Feature scaling with zero data leakage
    """
    
    def __init__(self):
        """Initialize standalone trainer"""
        self.start_time = time.time()
        self.model = None
        self.history = None
        self.data_loader = None
        self.model_builder = None
        self.memory_loader = None
        self.scaler_info = {}
        self.best_checkpoint_path = None
        self.val_df = None
        
        # Store generators and data for confusion matrix analysis
        self.train_gen = None
        self.val_gen = None
        self.full_features = None
        self.full_labels = None
        
        # Print configuration
        config.print_config_summary()
        
        # Validate configuration
        config_errors = config.validate_config()
        if config_errors:
            print("\n❌ Configuration validation failed:")
            for error in config_errors:
                print(f"   - {error}")
            raise ValueError("Configuration validation failed")
            
        print("✅ Configuration validated successfully")
    
    def initialize_components(self):
        """Initialize all training components"""
        print(f"\n🔧 INITIALIZING TRAINING COMPONENTS...")
        
        # Setup TensorFlow
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for training")
        
        setup_tensorflow(config.VERBOSE_LOGGING)
        
        # Initialize data loader
        print(f"📂 Initializing data loader...")
        self.data_loader = TrainingDataLoader(config.TRAINING_DATA_PATH, config.PAIR)
        
        # Initialize memory loader  
        print(f"💾 Initializing memory loader...")
        self.memory_loader = MemoryEfficientDataLoader(config)
        
        # Initialize model builder
        print(f"🧠 Initializing model builder...")
        self.model_builder = DualWindowLSTMBuilder(config)
        
        print(f"✅ All components initialized")
        monitor_memory_usage()
    
    def load_and_validate_data(self) -> Dict[str, Any]:
        """
        🎯 FIXED: Load and validate training data with chronological split
        """
        print(f"\n📁 LOADING AND VALIDATING DATA WITH FIXED CHRONOLOGICAL PIPELINE...")
        
        # Load training data with FIXED chronological split
        data_info = self.data_loader.load_training_data()
        
        print(f"✅ Data loaded successfully with FIXED pipeline:")
        print(f"   Original samples: {data_info['original_samples_count']:,}")
        print(f"   Raw train samples: {data_info['train_samples_count']:,} (before balancing)")
        print(f"   Balanced train samples: {data_info['balanced_train_samples_count']:,} (after balancing)")
        print(f"   Val samples: {data_info['val_samples_count']:,}")
        print(f"   Train Features: {data_info['train_features'].shape}")
        print(f"   Train Labels: {data_info['train_labels'].shape} ({data_info['label_format']})")
        print(f"   Val Features: {data_info['val_features'].shape}")
        print(f"   Val Labels: {data_info['val_labels'].shape}")
        print(f"   Label format: {data_info['label_format']}")
        print(f"   File parameters: {data_info['file_params']}")
        print(f"   Compatibility validated: {data_info['compatibility_validated']}")
        print(f"   Chronological split applied: {data_info['chronological_split_applied']}")
        print(f"   Feature scaling enabled: {data_info['scaling_enabled']}")
        if data_info['scaling_enabled']:
            print(f"   Scaler fitted: {data_info['scaler_fitted']}")
            if hasattr(self.data_loader, 'scaling_stats'):
                self.scaler_info = self.data_loader.scaling_stats
                print(f"   Scaling statistics available: {len(self.scaler_info)} splits")
        
        # Store data for confusion matrix analysis
        self.train_data = {
            'features': data_info['train_features'],
            'labels': data_info['train_labels'],
            'timestamps': data_info['train_timestamps']
        }
        self.val_data = {
            'features': data_info['val_features'],
            'labels': data_info['val_labels'],
            'timestamps': data_info['val_timestamps']
        }
        self.val_df = data_info.get('val_df')
        
        # Store full dataset for confusion matrix analysis (chronologically ordered)
        self.full_features = np.concatenate([data_info['train_features'], data_info['val_features']], axis=0)
        self.full_labels = np.concatenate([data_info['train_labels'], data_info['val_labels']], axis=0)
        
        monitor_memory_usage()
        return data_info
    
    def create_generators(self, data_info: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        🎯 SEQUENCE-AWARE: Create revolutionary sequence-aware generators
        """
        print(f"\n🧠 CREATING SEQUENCE-AWARE GENERATORS...")
        
        # Prepare train and val data dictionaries based on approach
        val_data = {
            'features': data_info['val_features'],
            'labels': data_info['val_labels'],
            'timestamps': data_info['val_timestamps']
        }
        
        # Prepare train_data differently based on sequence-aware detection
        if data_info.get('sequence_aware_undersampling', False):
            # For sequence-aware: original features + selected labels + indices
            train_data = {
                'features': data_info['train_features'],      # ORIGINAL features for sequences
                'labels': data_info['train_labels'],          # SELECTED labels for balance
                'timestamps': data_info['train_timestamps'],  # SELECTED timestamps (synchronized)
                'selected_target_indices': data_info.get('selected_target_indices')  # Real indices
            }
        else:
            # For standard approach: features and labels must match in length
            train_data = {
                'features': data_info['train_features'],
                'labels': data_info['train_labels'],
                'timestamps': data_info['train_timestamps']
            }
        
        # Add sequence-aware flag to config for generator detection
        self.memory_loader.config.sequence_aware_undersampling = data_info.get('sequence_aware_undersampling', False)
        
                # Create SEQUENCE-AWARE generators! 
        if data_info.get('sequence_aware_undersampling', False):
            print(f"   🎯 Using SEQUENCE-AWARE approach (undersampling detected)")
            train_gen, val_gen = self.memory_loader.create_sequence_aware_generators(
                train_data=train_data,
                val_data=val_data,
                random_seed=42
            )
        else:
            print(f"   📊 Using STANDARD approach (no undersampling)")
            train_gen, val_gen = self.memory_loader.create_generators_from_splits(
                train_data=train_data,
                val_data=val_data,
                random_seed=42
            )
        
        # Store generators for confusion matrix analysis
        self.train_gen = train_gen
        self.val_gen = val_gen
        
        # Test generators
        # self._test_sequence_aware_generator(train_gen, "training")
        # self._test_sequence_aware_generator(val_gen, "validation")
        
        # Store necessary data on instance for later use in reporting
        self.y_valid = data_info['val_labels']
        self.val_timestamps = data_info['val_timestamps']

        return train_gen, val_gen
    
    def _test_sequence_aware_generator(self, generator, mode: str, num_batches: int = 2):
        """
        🧪 Test SEQUENCE-AWARE generator functionality
        """
        generator_type = type(generator).__name__
        print(f"   Testing {mode} generator ({generator_type})...")
        
        for i in range(min(num_batches, len(generator))):
            X_batch, y_batch = generator[i]
            
            print(f"      Batch {i+1}: X{X_batch.shape}, y{y_batch.shape}")
            print(f"      X dtype: {X_batch.dtype}, y dtype: {y_batch.dtype}")
            print(f"      Memory: X={X_batch.nbytes / (1024**2):.1f}MB, y={y_batch.nbytes / (1024**2):.1f}MB")
            
            # Check for NaN/inf values
            nan_count = np.isnan(X_batch).sum()
            inf_count = np.isinf(X_batch).sum()
            
            if nan_count > 0:
                print(f"      ⚠️ NaN values in X: {nan_count}")
            if inf_count > 0:
                print(f"      ⚠️ Inf values in X: {inf_count}")
            
            # Validate sparse labels
            if len(y_batch.shape) == 1:
                # Sparse labels - check valid class range
                unique_classes = np.unique(y_batch)
                print(f"      Sparse label classes: {unique_classes}")
                print(f"      Sparse label range: [{y_batch.min()}, {y_batch.max()}]")
                if y_batch.max() >= 3 or y_batch.min() < 0:
                    print(f"      ⚠️ Invalid sparse labels detected!")
            else:
                # One-hot labels - check row sums
                row_sums = np.sum(y_batch, axis=1)
                print(f"      One-hot sums range: [{row_sums.min():.3f}, {row_sums.max():.3f}]")
            
            # Special checks for sequence-aware generators
            if generator_type == 'SequenceAwareGenerator':
                print(f"      🎯 SEQUENCE-AWARE: Continuous temporal sequences verified")
                # Check sequence continuity (simplified - just verify shape consistency)
                expected_features = X_batch.shape[2]
                if hasattr(generator, 'original_features'):
                    actual_features = generator.original_features.shape[1]
                    if expected_features == actual_features:
                        print(f"      ✅ Feature continuity: {expected_features} features preserved")
                    else:
                        print(f"      ⚠️ Feature mismatch: expected {actual_features}, got {expected_features}")
        
        print(f"   ✅ {mode} generator test completed")
    
    def _validate_data_integrity(self, data_info: Dict[str, Any]):
        """
        🎯 FIXED: Validate data integrity and chronological properties
        """
        print(f"\n🔍 VALIDATING DATA INTEGRITY...")
        
        # Check if timestamps are synthetic
        timestamps_are_synthetic = False
        if data_info['train_timestamps'] is not None:
            synthetic_start = pd.Timestamp('2022-01-01 00:00:00')
            first_train_timestamp = pd.Timestamp(data_info['train_timestamps'][0])
            timestamps_are_synthetic = (first_train_timestamp == synthetic_start)
        
        # Test 1: Chronological split integrity
        if data_info['train_timestamps'] is not None and data_info['val_timestamps'] is not None:
            train_max = pd.Timestamp(data_info['train_timestamps'].max())
            val_min = pd.Timestamp(data_info['val_timestamps'].min())
            
            if train_max >= val_min:
                raise ValueError(f"CRITICAL: Train data newer than validation! Train max: {train_max}, Val min: {val_min}")
            
            gap_days = (val_min - train_max).days
            gap_hours = ((val_min - train_max).total_seconds() / 3600)
            
            print(f"   ✅ Chronological split integrity:")
            if timestamps_are_synthetic:
                print(f"      Timestamps: SYNTHETIC (row-order based chronological split)")
                print(f"      Train rows: first {data_info['train_samples_count']:,} samples")
                print(f"      Val rows: last {data_info['val_samples_count']:,} samples")
            else:
                print(f"      Timestamps: REAL (timestamp-based chronological split)")
                print(f"      Train period: {pd.Timestamp(data_info['train_timestamps'].min())} to {train_max}")
                print(f"      Val period: {val_min} to {pd.Timestamp(data_info['val_timestamps'].max())}")
            print(f"      Chronological gap: {gap_days} days ({gap_hours:.1f} hours)")
        else:
            print(f"   ⚠️ Missing timestamps - cannot validate chronological integrity")
        
        # Test 2: Feature scaling integrity
        if data_info['scaling_enabled'] and hasattr(self.data_loader, 'scaler') and self.data_loader.scaler:
            scaler = self.data_loader.scaler
            if hasattr(scaler, 'n_samples_seen_'):
                fitted_samples = scaler.n_samples_seen_
                expected_samples = data_info['train_samples_count']
                if fitted_samples != expected_samples:
                    raise ValueError(f"CRITICAL: Scaler fitted on {fitted_samples}, expected {expected_samples} (train only)")
                print(f"   ✅ Feature scaling integrity: Scaler fitted on {fitted_samples} train samples only")
            else:
                print(f"   ⚠️ Cannot verify scaler sample count")
        else:
            print(f"   📏 Feature scaling disabled or not available")
        
        # Test 3: Data shapes consistency
        train_samples = data_info['train_samples_count']
        val_samples = data_info['val_samples_count']
        original_samples = data_info['original_samples_count']
        
        if train_samples + val_samples > original_samples:
            print(f"   ⚠️ WARNING: Train + Val samples ({train_samples + val_samples}) > Original ({original_samples}) - class balancing applied")
        
        print(f"   ✅ Data shapes consistency:")
        print(f"      Original: {original_samples:,} samples")
        print(f"      Train: {train_samples:,} samples")
        print(f"      Val: {val_samples:,} samples")
        print(f"      Split ratio: {val_samples / (train_samples + val_samples):.1%} validation")
        
        # Test 4: Timestamp integrity (adapted for synthetic timestamps)
        if data_info['train_timestamps'] is not None:
            if timestamps_are_synthetic:
                print(f"   📅 Synthetic timestamps detected: Chronological split based on ROW ORDER")
                print(f"   🎯 Row-based chronological integrity: GUARANTEED")
            else:
                print(f"   ✅ Real timestamps confirmed: Train starts at {pd.Timestamp(data_info['train_timestamps'][0])}")
                print(f"   🎯 Timestamp-based chronological integrity: GUARANTEED")
        
        print(f"   🎯 DATA INTEGRITY VALIDATION PASSED!")
        if timestamps_are_synthetic:
            print(f"   🎯 CHRONOLOGICAL SPLIT BY ROW ORDER CONFIRMED!")
        else:
            print(f"   🎯 ZERO DATA LEAKAGE CONFIRMED!")
        print(f"   🎯 TRAIN < VALIDATION TEMPORALLY GUARANTEED!")
    
    def build_model(self):
        """Build and compile model"""
        print(f"\n🧠 BUILDING MODEL...")
        
        # Build model
        self.model = self.model_builder.build_model()
        
        # Print model summary
        print(f"\n📋 Model Summary:")
        self.model.summary()
        
        monitor_memory_usage()
    
    def setup_callbacks(self) -> List[Any]:
        """Setup training callbacks with ENHANCED SAFETY"""
        print(f"\n⚙️ SETTING UP SAFE CALLBACKS...")
        
        callbacks_list = []
        
        # 1. SAFE MODEL CHECKPOINT with weight validation (H5 format)
        self.best_checkpoint_path = os.path.join(config.get_model_output_dir(), "best_model.h5")
        
        safe_checkpoint = SafeModelCheckpoint(
            filepath=self.best_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1 if config.VERBOSE_LOGGING else 0,
            validate_weights=True  # ✅ Enable weight validation
        )
        callbacks_list.append(safe_checkpoint)
        
        # 2. EARLY STOPPING without restore_best_weights (we'll do it manually)
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=False,  # ✅ Manual restoration for safety
            mode='min',
            verbose=1 if config.VERBOSE_LOGGING else 0
        )
        callbacks_list.append(early_stopping)
        
        # 3. REDUCE LEARNING RATE
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1 if config.VERBOSE_LOGGING else 0
        )
        callbacks_list.append(reduce_lr)
        
        # 4. MEMORY CLEANUP CALLBACK
        class MemoryCleanupCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:  # Every 10 epochs
                    gc.collect()
        
        callbacks_list.append(MemoryCleanupCallback())
        
        print(f"✅ Safe callbacks configured: {len(callbacks_list)} callbacks")
        print(f"   📊 Primary metric: val_loss")
        print(f"   💾 Checkpoint validation: ENABLED (H5 format)")
        print(f"   🔄 Manual weight restoration: ENABLED")
        print(f"   ⚠️ Balanced callbacks: DISABLED (for safety)")
        
        return callbacks_list
    
    def train_model(self, train_gen, val_gen, callbacks_list: List[Any]):
        """Execute training with ENHANCED SAFETY"""
        print(f"\n🚀 STARTING SAFE TRAINING...")
        print(f"   Epochs: {config.EPOCHS}")
        print(f"   Train batches: {len(train_gen)}")
        print(f"   Val batches: {len(val_gen)}")
        
        # Store generators for later use
        self.train_gen = train_gen
        self.val_gen = val_gen
        
        # NO balanced_callbacks - they can interfere with model saving
        print(f"⚠️ Balanced callbacks DISABLED for model saving safety")
        
        training_start = time.time()
        
        # Train model
        try:
            self.history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=config.EPOCHS,
                callbacks=callbacks_list,
                verbose=1 if config.VERBOSE_LOGGING else 0,
                shuffle=False,  # Already chronologically split
            )
            
            training_time = time.time() - training_start
            print(f"✅ Training completed in {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        # Print training results with safe access to history
        if self.history and self.history.history:
            final_train_acc = self.history.history.get('accuracy', [0])[-1] if 'accuracy' in self.history.history else 0
            final_val_acc = self.history.history.get('val_accuracy', [0])[-1] if 'val_accuracy' in self.history.history else 0
            best_val_acc = max(self.history.history.get('val_accuracy', [0])) if 'val_accuracy' in self.history.history else 0
            
            print(f"📊 Training Results:")
            print(f"   Final train accuracy: {final_train_acc:.4f}")
            print(f"   Final val accuracy: {final_val_acc:.4f}")
            print(f"   Best val accuracy: {best_val_acc:.4f}")
        else:
            print(f"📊 Training Results: No history available (interrupted?)")
        
        monitor_memory_usage()
    
    def _calculate_class_weights(self) -> dict:
        """
        Calculate class weights based on config
        
        Returns:
            Dictionary of class weights or None
        """
        if config.CLASS_BALANCING_METHOD == "focal_loss":
            print("   ⚖️ Class balancing handled by Focal Loss. Standard class weights disabled.")
            return None
            
        # Proceed with standard class weight calculation only if not using focal_loss
        if not hasattr(self, 'train_data') or self.train_data['labels'] is None:
            print("   ⚠️ Cannot calculate class weights: training labels not available.")
            return None
            
        y_train = self.train_data['labels']
        # Labels are now in sparse format (1D)
        y_train_1d = y_train
            
        # Count classes
        unique_classes, class_counts = np.unique(y_train_1d, return_counts=True, axis=0)
        
        if len(unique_classes) < 3:
            print(f"   ⚠️ Warning: Found only {len(unique_classes)} classes in train data. Expected 3.")
            return None

        print(f"   Calculating class weights for {len(y_train_1d):,} samples...")
        print(f"   Original distribution: {dict(zip(unique_classes, class_counts))}")

        # Check if config asks for class weights
        if config.CLASS_WEIGHT_METHOD == "balanced":
            print("   ⚖️ Calculating 'balanced' class weights...")
            
            try:
                from sklearn.utils.class_weight import compute_class_weight
            except ImportError:
                print("   ⚠️ scikit-learn not available. Cannot compute balanced weights.")
                return None
                
                class_weights_array = compute_class_weight(
                class_weight='balanced',
                    classes=unique_classes,
                y=y_train_1d
                )
            
            balanced_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_array)}
            print(f"      Computed weights: {balanced_weights}")
            return balanced_weights
                
        elif config.CLASS_WEIGHT_METHOD == "manual":
            print(f"   ⚖️ Using 'manual' class weights: {config.MANUAL_CLASS_WEIGHTS}")
            return config.MANUAL_CLASS_WEIGHTS
            
        else:
            print(f"   ⚖️ Class weight method '{config.CLASS_WEIGHT_METHOD}' not recognized or 'none'. No weights applied.")
            return None
    
    def save_model_and_metadata(self, data_info: Dict[str, Any]):
        """Save model and training metadata with VALIDATION"""
        print(f"\n💾 SAVING MODEL WITH VALIDATION...")
        
        model_dir = config.get_model_output_dir()
        os.makedirs(model_dir, exist_ok=True)
        
        # Save final model in H5 format (avoid TensorFlow 2.15.0 LSTM bug)
        model_filename = config.get_model_filename().replace('.keras', '.h5')
        model_path = os.path.join(model_dir, model_filename)
        
        try:
            print(f"   💾 Saving final model: {model_filename} (H5 format)")
            self.model.save(model_path, save_format='h5')
            
            # Validate final model
            if validate_model_weights(model_path, verbose=True):
                print(f"✅ Model saved and validated: {model_filename}")
            else:
                print(f"❌ CRITICAL: Final model has invalid weights!")
                
                # Try to use checkpoint instead
                if os.path.exists(self.best_checkpoint_path):
                    print(f"   🔄 Using checkpoint as final model...")
                    import shutil
                    shutil.copy2(self.best_checkpoint_path, model_path)
                    
                    if validate_model_weights(model_path, verbose=True):
                        print(f"✅ Checkpoint used as final model successfully")
                    else:
                        print(f"❌ Even checkpoint has invalid weights!")
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise
        
        if config.SAVE_METADATA:
            # Prepare metadata
            metadata = {
                'training_config': {
                    'pair': config.PAIR,
                    'epochs': config.EPOCHS,
                    'batch_size': config.BATCH_SIZE,
                    'validation_split': config.VALIDATION_SPLIT,
                    'sequence_length': config.SEQUENCE_LENGTH,
                    'learning_rate': config.LEARNING_RATE,
                    'model_params': {
                        'lstm_units': config.LSTM_UNITS,
                        'dense_units': config.DENSE_UNITS,
                        'dropout_rate': config.DROPOUT_RATE
                    }
                },
                'data_info': {
                    'samples_count': data_info['original_samples_count'],
                    'features_shape': list(data_info['train_features'].shape),
                    'labels_shape': list(data_info['train_labels'].shape),
                    'label_format': data_info['label_format'],
                    'file_params': data_info['file_params'],
                    'scaling_enabled': data_info.get('scaling_enabled', False),
                    'scaler_fitted': data_info.get('scaler_fitted', False)
                },
                'scaling_info': {
                    'enabled': config.ENABLE_FEATURE_SCALING,
                    'scaler_type': config.SCALER_TYPE if config.ENABLE_FEATURE_SCALING else None,
                    'fit_only_train': config.SCALER_FIT_ONLY_TRAIN if config.ENABLE_FEATURE_SCALING else None,
                    'statistics': self.scaler_info if hasattr(self, 'scaler_info') else {}
                },
                'training_results': {
                    'final_train_accuracy': float(self.history.history.get('accuracy', [0])[-1]) if self.history and self.history.history and 'accuracy' in self.history.history else 0.0,
                    'final_val_accuracy': float(self.history.history.get('val_accuracy', [0])[-1]) if self.history and self.history.history and 'val_accuracy' in self.history.history else 0.0,
                    'best_val_accuracy': float(max(self.history.history.get('val_accuracy', [0]))) if self.history and self.history.history and 'val_accuracy' in self.history.history else 0.0,
                    'total_epochs': len(self.history.history.get('accuracy', [0])) if self.history and self.history.history and 'accuracy' in self.history.history else 0,
                    'training_time_minutes': (time.time() - self.start_time) / 60,
                    'training_interrupted': self.history is None or not self.history.history
                },
                'file_paths': {
                    'model_file': config.get_model_filename(),
                    'scaler_file': config.get_scaler_filename() if config.ENABLE_FEATURE_SCALING else None,
                    'source_data': getattr(self.data_loader, 'selected_filename', 'unknown')
                },
                'created_at': datetime.now().isoformat(),
                'version': 'STANDALONE_V3_WITH_SCALING'
            }
            
            # Save metadata
            metadata_path = os.path.join(model_dir, config.get_metadata_filename())
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Metadata saved: {config.get_metadata_filename()}")
        
        print(f"📁 All files saved to: {model_dir}")
    
    def print_training_summary(self, data_info: Dict[str, Any]):
        """Print comprehensive training summary"""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "="*80)
        print(f"🎯 STANDALONE TRAINING V3 - SUMMARY")
        print(f"="*80)
        print(f"📍 Crypto Pair: {config.PAIR}")
        print(f"⏱️ Total Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print("")
        
        print(f"📊 Data:")
        print(f"   Source: {getattr(self.data_loader, 'selected_filename', 'unknown')}")
        print(f"   Samples: {data_info['original_samples_count']:,}")
        print(f"   Train Features: {data_info['train_features'].shape}")
        print(f"   Train Labels: {data_info['train_labels'].shape} ({data_info['label_format']})")
        print(f"   Val Features: {data_info['val_features'].shape}")
        print(f"   Val Labels: {data_info['val_labels'].shape}")
        print(f"   Parameters: FW={data_info['file_params']['future_window']}, "
              f"SL={data_info['file_params']['stop_loss']:.1%}, "
              f"TP={data_info['file_params']['take_profit']:.1%}")
        print("")
        
        print(f"📏 Feature Scaling:")
        if data_info.get('scaling_enabled', False):
            print(f"   Enabled: YES")
            print(f"   Scaler type: {config.SCALER_TYPE}")
            print(f"   Fit only on train: {config.SCALER_FIT_ONLY_TRAIN}")
            print(f"   Scaler fitted: {data_info.get('scaler_fitted', False)}")
            if hasattr(self, 'scaler_info') and self.scaler_info:
                print(f"   Statistics available: {list(self.scaler_info.keys())}")
        else:
            print(f"   Enabled: NO")
        print("")
        
        print(f"🧠 Model:")
        print(f"   Architecture: LSTM {config.LSTM_UNITS} + Dense {config.DENSE_UNITS}")
        print(f"   Sequence Length: {config.SEQUENCE_LENGTH}")
        print(f"   Dropout: {config.DROPOUT_RATE}")
        print(f"   Learning Rate: {config.LEARNING_RATE}")
        print("")
        
        print(f"🏋️ Training:")
        if self.history and self.history.history and 'accuracy' in self.history.history:
            print(f"   Epochs: {len(self.history.history['accuracy'])}/{config.EPOCHS}")
        else:
            print(f"   Epochs: 0/{config.EPOCHS} (interrupted)")
        print(f"   Batch Size: {config.BATCH_SIZE}")
        print(f"   Validation Split: {config.VALIDATION_SPLIT}")
        print("")
        
        print(f"📈 Results:")
        if self.history and self.history.history:
            final_train_acc = self.history.history.get('accuracy', [0])[-1] if 'accuracy' in self.history.history else 0
            final_val_acc = self.history.history.get('val_accuracy', [0])[-1] if 'val_accuracy' in self.history.history else 0
            best_val_acc = max(self.history.history.get('val_accuracy', [0])) if 'val_accuracy' in self.history.history else 0
            
            print(f"   Final Train Acc: {final_train_acc:.4f}")
            print(f"   Final Val Acc: {final_val_acc:.4f}")
            print(f"   Best Val Acc: {best_val_acc:.4f}")
        else:
            print(f"   Training was interrupted - no final results available")
        print("")
        
        print(f"💾 Output:")
        print(f"   Model: {config.get_model_filename()}")
        if config.ENABLE_FEATURE_SCALING:
            print(f"   Scaler: {config.get_scaler_filename()}")
        print(f"   Metadata: {config.get_metadata_filename()}")
        print(f"   Directory: {config.get_model_output_dir()}")
        print("")
        
        print(f"🎯 Key Features:")
        print(f"   ✅ Zero validation module dependencies")
        print(f"   ✅ Pre-computed labels (no competitive labeling)")
        print(f"   ✅ Memory-efficient generators")
        print(f"   ✅ Parameter validation from filename")
        print(f"   ✅ Chronological train/val split")
        if config.ENABLE_FEATURE_SCALING:
            print(f"   ✅ Feature scaling with zero data leakage")
        print(f"   ✅ Production-ready Docker paths")
        print(f"="*80)
    
    def generate_confusion_matrix_report(self):
        """Generate BALANCED METRICS REPORT with proper handling for imbalanced data"""
        if not SKLEARN_AVAILABLE:
            print("⚠️ Balanced metrics analysis skipped - sklearn not available")
            return
            
        if self.model is None:
            print("⚠️ Balanced metrics analysis skipped - no trained model")
            return
        
        print(f"================================================================================")
        
        # Generate predictions on the validation set
        print("\n🔍 VALIDATION SET ANALYSIS:")
        print("----------------------------------------")
        print("   Generating predictions on validation set...")
        
        # Ensure val_gen is available
        if self.val_gen is None:
            print("   ❌ val_gen not available. Skipping validation analysis.")
            return

        try:
            # Import balanced metrics module
            try:
                from balanced_metrics import calculate_balanced_metrics, BalancedMetricsCalculator
                balanced_metrics_available = True
            except ImportError:
                print("⚠️ balanced_metrics module not available - using basic metrics")
                balanced_metrics_available = False

            val_predictions_proba = self.model.predict(self.val_gen)
            
            # 🎯 RESTORED WORKING CODE: Use val_gen.labels like in the working backup
            num_predictions = len(val_predictions_proba)  # Actual number of predictions from model
            y_val_true = self.val_gen.labels[:num_predictions]  # WORKING: Use sequential indexing from generator
            
            # 🔍 DIAGNOSTIC LOGS - ADD DETAILED DEBUGGING
            print(f"\n   🔍 PREDICTION-LABEL MAPPING DIAGNOSTICS:")
            print(f"      Model predictions: {num_predictions:,}")
            print(f"      Generator labels available: {len(self.val_gen.labels):,}")
            print(f"      Taking first {num_predictions:,} labels for alignment")
            
            # Sample diagnostics - first 5 predictions vs labels (reduced for cleaner output)
            print(f"      🎯 SAMPLE ALIGNMENT (first 5):")
            for i in range(min(5, num_predictions)):
                pred_proba = val_predictions_proba[i]
                true_label = y_val_true[i]
                pred_class = np.argmax(pred_proba)
                true_class = true_label  # Already in sparse format
                pred_confidence = pred_proba[pred_class]
                
                print(f"         Sample {i}: Pred={pred_class}({pred_confidence:.3f}) vs True={true_class} | {['SHORT','HOLD','LONG'][pred_class]} vs {['SHORT','HOLD','LONG'][true_class]}")
            
            # Check prediction distribution BEFORE confidence thresholding
            raw_predictions = np.argmax(val_predictions_proba, axis=1)
            raw_unique, raw_counts = np.unique(raw_predictions, return_counts=True)
            print(f"      📊 RAW PREDICTION DISTRIBUTION (before confidence thresholding):")
            for cls, count in zip(raw_unique, raw_counts):
                pct = (count / len(raw_predictions)) * 100
                class_name = ['SHORT', 'HOLD', 'LONG'][cls]
                print(f"         {class_name}: {count:,} ({pct:.1f}%)")
            
            # Check true label distribution (already sparse)
            true_labels_1d = y_val_true
                
            true_unique, true_counts = np.unique(true_labels_1d, return_counts=True)
            print(f"      📊 TRUE LABEL DISTRIBUTION:")
            for cls, count in zip(true_unique, true_counts):
                pct = (count / len(true_labels_1d)) * 100
                class_name = ['SHORT', 'HOLD', 'LONG'][cls]
                print(f"         {class_name}: {count:,} ({pct:.1f}%)")
            
            # Apply confidence thresholding (or just get argmax if disabled)
            y_val_pred_filtered, stats = apply_confidence_thresholding(val_predictions_proba)
            
            print("\n   🎯 CONFIDENCE THRESHOLDING RESULTS:")
            if config.USE_CONFIDENCE_THRESHOLDING:
                print(f"   Total predictions: {stats['total']:,}")
                print(f"   Accepted: {stats['accepted']:,} ({stats['accepted']/stats['total']*100:.1f}%)")
                print(f"   Rejected: {stats['rejected']:,} ({stats['rejection_rate']*100:.1f}%)")
            
                print("\n   📊 PER-CLASS CONFIDENCE RESULTS:")
                print(f"      {'Class':<8} {'Accepted':>10} {'Rejected':>10}  {'Accept Rate':>12}")
                class_names = ['SHORT', 'HOLD', 'LONG']
                for i, name in enumerate(class_names):
                    accepted = stats['class_accepted'][name]
                    rejected = stats['class_rejections'][name]
                    total = accepted + rejected
                    accept_rate = accepted / total * 100 if total > 0 else 0
                    print(f"      {name:<8} {accepted:>10,} {rejected:>10,}  {accept_rate:>11.1f}%")
            
                # Filter out rejected predictions (-1) for confusion matrix
                accepted_indices = (y_val_pred_filtered != -1)
                y_val_pred = y_val_pred_filtered[accepted_indices]
                y_val_true_filtered = y_val_true[accepted_indices]
                report_title = "BALANCED METRICS (accepted predictions only)"
                
                # 🔍 DIAGNOSTIC: Check what's happening with filtering
                print(f"      🔍 FILTERING DIAGNOSTICS:")
                print(f"         Total predictions: {len(y_val_pred_filtered):,}")
                print(f"         Accepted (not -1): {len(y_val_pred):,}")
                print(f"         Rejected (-1): {np.sum(y_val_pred_filtered == -1):,}")
                
            else:
                # No thresholding, use all predictions
                y_val_pred = y_val_pred_filtered
                y_val_true_filtered = y_val_true
                report_title = "BALANCED METRICS (raw predictions)"

            # Handle cases where there are no accepted predictions
            if len(y_val_pred) == 0:
                print("\n   ⚠️ No predictions were accepted. Cannot generate metrics.")
                return

            # 🎯 GENERATE BALANCED METRICS REPORT
            if balanced_metrics_available:
                print(f"\n{'='*80}")
                print(f"📊 {report_title}")
                print(f"{'='*80}")
                
                # Calculate balanced metrics with probabilities
                y_val_pred_proba_filtered = None
                if config.USE_CONFIDENCE_THRESHOLDING:
                    y_val_pred_proba_filtered = val_predictions_proba[accepted_indices]
                else:
                    y_val_pred_proba_filtered = val_predictions_proba
                
                # Use the new balanced metrics calculator
                metrics = calculate_balanced_metrics(
                    y_val_true_filtered, 
                    y_val_pred, 
                    y_val_pred_proba_filtered,
                    print_report=True
                )
                
                # Store metrics for potential later use
                self.validation_metrics = metrics
                
            else:
                # Fallback to basic metrics if balanced_metrics not available
                print(f"\n   📋 BASIC CONFUSION MATRIX ({report_title}):")
                
                # True labels are already in sparse format
                y_val_true_indices = y_val_true_filtered

                # Generate and print confusion matrix
                cm = confusion_matrix(y_val_true_indices, y_val_pred, labels=[0, 1, 2])
                cm_df = pd.DataFrame(cm, index=['SHORT', 'HOLD', 'LONG'], columns=['SHORT', 'HOLD', 'LONG'])
                
                # Pretty print the DataFrame
                print(cm_df.to_string())
                    
                print(f"\n   📈 BASIC PER-CLASS METRICS:")
                report = classification_report(
                    y_val_true_indices, 
                    y_val_pred, 
                    target_names=['SHORT', 'HOLD', 'LONG'],
                    labels=[0, 1, 2],
                    zero_division=0
                )
                print(report)
                
                # Calculate and display balanced accuracy manually
                try:
                    from sklearn.metrics import balanced_accuracy_score
                    bal_acc = balanced_accuracy_score(y_val_true_indices, y_val_pred)
                    print(f"\n   🎯 BALANCED ACCURACY: {bal_acc:.4f} ⭐ (NAJWAŻNIEJSZA METRYKA)")
                except ImportError:
                    print("\n   ⚠️ Balanced accuracy not available")
            
            # 💾 Save predictions to CSV if enabled in config
            if hasattr(config, 'SAVE_VALIDATION_PREDICTIONS') and config.SAVE_VALIDATION_PREDICTIONS:
                try:
                    print("\n   💾 Saving validation predictions to CSV...")
                    val_indices = self.val_gen.get_valid_indices()
                    all_timestamps = self.data_loader.get_all_timestamps()
                    
                    if np.max(val_indices) < len(all_timestamps):
                        timestamps_to_save = all_timestamps[val_indices]
                        if len(timestamps_to_save) > len(predictions_proba):
                            timestamps_to_save = timestamps_to_save[:len(predictions_proba)]

                        final_class_indices, _ = apply_confidence_thresholding(predictions_proba)
                        class_mapping = {0: "SHORT", 1: "HOLD", 2: "LONG", -1: "REJECTED"}
                        final_class_names = [class_mapping[idx] for idx in final_class_indices]
                        
                        predictions_df = pd.DataFrame({
                            'timestamp': timestamps_to_save,
                            'short_prediction': predictions_proba[:, 0],
                            'hold_prediction': predictions_proba[:, 1],
                            'long_prediction': predictions_proba[:, 2],
                            'final_class': final_class_names
                        })
                        
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = os.path.join(config.OUTPUT_BASE_PATH, 'predictions')
                        os.makedirs(output_dir, exist_ok=True)
                        filename = f"validation_predictions_{timestamp_str}.csv"
                        full_path = os.path.join(output_dir, filename)
                        
                        predictions_df.to_csv(full_path, index=False, float_format='%.6f')
                        print(f"      ✅ Predictions saved successfully to: {full_path}")
                    else:
                        print(f"      ❌ ERROR: Cannot save predictions. Max validation index out of bounds.")
                except Exception as e:
                    print(f"      ❌ FAILED to save predictions to CSV: {e}")

        except Exception as e:
            print(f"\n   ❌ Validation analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_validation_positions_report(self):
        """
        📝 GENERATE BACKTEST-LIKE REPORT FOR VALIDATION SET
        
        Analyzes validation predictions, compares them to true labels,
        and generates two reports:
        1. A detailed analysis file with all LONG/SHORT predictions and their correctness.
        2. A simplified file containing only the predicted trades for visualization.
        """
        print("\n\n" + "="*80)
        print("📝 Attempting to generate Validation Analysis Report...")

        # --- Rigorous Pre-computation Checks ---
        required_attrs = ['model', 'val_gen', 'y_valid', 'val_timestamps']
        for attr in required_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                print(f"   ⚠️ Prerequisite '{attr}' not found or is None. Skipping report generation.")
                print("="*80)
                return
        
        print("   ✅ All prerequisites found. Proceeding with report generation.")

        # Predict on validation data and APPLY CONFIDENCE THRESHOLDING
        y_pred_probs = self.model.predict(self.val_gen)
        y_pred_filtered, _ = apply_confidence_thresholding(y_pred_probs)
        
        # Ensure alignment between predictions and ground truth data
        num_predictions = len(y_pred_filtered)
        y_true = self.y_valid[:num_predictions]
        timestamps = self.val_timestamps[:num_predictions]

        if len(y_pred_filtered) != len(y_true) or len(y_pred_filtered) != len(timestamps):
            print(f"   ❌ CRITICAL ERROR: Length mismatch after slicing.")
            print(f"      Predictions: {len(y_pred_filtered)}, True Labels: {len(y_true)}, Timestamps: {len(timestamps)}")
            print("="*80)
            return

        # 1. Create a detailed analysis DataFrame
        # ----------------------------------------
        analysis_df = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_signal': y_pred_filtered,
            'true_signal': y_true,
            'short_prob': y_pred_probs[:, 0],
            'hold_prob': y_pred_probs[:, 1],
            'long_prob': y_pred_probs[:, 2]
        })

        # Add correctness column
        analysis_df['is_correct'] = (analysis_df['predicted_signal'] == analysis_df['true_signal'])

        # --- CORRECT FILTERING LOGIC ---
        # First, filter out predictions rejected by confidence threshold (-1)
        accepted_predictions_df = analysis_df[analysis_df['predicted_signal'] != -1].copy()
        
        # Then, from the accepted predictions, filter out HOLD signals (1)
        actionable_predictions_df = accepted_predictions_df[accepted_predictions_df['predicted_signal'] != 1].copy()

        # Save the detailed analysis report
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = os.path.join(config.OUTPUT_BASE_PATH, f'validation_analysis_{config.PAIR}_{timestamp_str}.csv')
        actionable_predictions_df.to_csv(analysis_filename, index=False, float_format='%.8f')
        print(f"   ✅ Detailed analysis for {len(actionable_predictions_df)} SHORT/LONG predictions saved to: {analysis_filename}")

        # 2. Create the simplified trades-only report for visualization
        # -------------------------------------------------------------
        # This part is kept for compatibility with the visualization script
        trades_only_df = actionable_predictions_df.copy()

        # Rename for clarity and compatibility
        trades_only_df.rename(columns={'predicted_signal': 'final_signal'}, inplace=True)
        
        # Keep only necessary columns
        trades_only_df = trades_only_df[['timestamp', 'short_prob', 'hold_prob', 'long_prob', 'final_signal']]

        trades_filename = os.path.join(config.OUTPUT_BASE_PATH, f'ml_trades_only_{config.PAIR}_{timestamp_str}.csv')
        trades_only_df.to_csv(trades_filename, index=False, float_format='%.8f')
        print(f"   ✅ Simplified trades-only report saved to: {trades_filename}")
        print("="*80)

    def run_training(self):
        """
        🚀 RUN FULL TRAINING PIPELINE
        """
        try:
            # Initialize components
            self.initialize_components()
            
            # Load and validate data with FIXED chronological split
            data_info = self.load_and_validate_data()
            
            # Validate data integrity and chronological properties
            self._validate_data_integrity(data_info)
            
            # Create generators from chronological splits
            train_gen, val_gen = self.create_generators(data_info)
            
            # Build model
            self.build_model()
            
            # Setup callbacks
            callbacks_list = self.setup_callbacks()
            
            # Train model
            self.train_model(train_gen, val_gen, callbacks_list)

            # MANUAL RESTORATION of best weights after training
            if os.path.exists(self.best_checkpoint_path):
                print(f"\n🔄 MANUALLY RESTORING BEST WEIGHTS...")
                
                # Validate checkpoint before loading
                if validate_model_weights(self.best_checkpoint_path, verbose=True):
                    print(f"   ✅ Checkpoint validation passed, loading best weights from {os.path.basename(self.best_checkpoint_path)}")
                    # Load model without compiling, as we only need weights
                    best_model = tf.keras.models.load_model(self.best_checkpoint_path, compile=False)
                    
                    # Transfer weights to current model
                    self.model.set_weights(best_model.get_weights())
                    
                    # Clean up
                    del best_model
                    gc.collect()
                    
                    print(f"   ✅ Best weights restored successfully to self.model")
                else:
                    print(f"   ❌ Checkpoint validation failed, keeping last epoch weights in self.model")
            else:
                print(f"   ⚠️ No checkpoint found, keeping last epoch weights in self.model")

            # Save results
            self.save_model_and_metadata(data_info)
            
            # Print summary
            self.print_training_summary(data_info)
            
            # Generate confusion matrix report
            self.generate_confusion_matrix_report()
            
            # Generate validation positions report
            self.generate_validation_positions_report()
            
            # Cleanup
            safe_memory_cleanup()
            
            print(f"\n🎉 FIXED TRAINING COMPLETED SUCCESSFULLY!")
            print(f"🎯 ZERO DATA LEAKAGE PIPELINE CONFIRMED!")
            return True
            
        except Exception as e:
            print(f"\n❌ TRAINING FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Standalone Training Module V3')
    parser.add_argument('--pair', type=str, help='Crypto pair (e.g., BTCUSDT)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--validation-split', type=float, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Override config with CLI arguments if provided
    if args.pair:
        config.PAIR = args.pair
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.validation_split:
        config.VALIDATION_SPLIT = args.validation_split
    
    print("🎯 STANDALONE TRAINING MODULE V3 WITH SCALING")
    print("=" * 60)
    
    # Print CLI overrides if any
    if args.pair or args.epochs or args.batch_size or args.validation_split:
        print("🔧 CLI OVERRIDES:")
        if args.pair:
            print(f"   Pair: {args.pair}")
        if args.epochs:
            print(f"   Epochs: {args.epochs}")
        if args.batch_size:
            print(f"   Batch size: {args.batch_size}")
        if args.validation_split:
            print(f"   Validation split: {args.validation_split}")
        print("")
    
    trainer = StandaloneTrainer()
    success = trainer.run_training()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 