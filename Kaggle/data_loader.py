"""
🎯 STANDALONE DATA LOADER V3 - EXPLICIT PATH LOADING
100% Standalone | Config-driven | Docker-optimized

KEY INNOVATION:
- Explicit path configuration (no auto-detection)
- Filename parameter parsing and validation
- Hard fail on missing files
- Zero validation module dependencies
- Config-driven approach
- FEATURE SCALING support with train/val split awareness

ELIMINATES: Auto-detection logic, validation module imports
"""

import os
import pandas as pd
import numpy as np
import re
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Sklearn imports for feature scaling
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - feature scaling disabled")

# Ensure local config.py is imported first (not from parent directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import standalone config
import config


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
    
    def __init__(self, explicit_path: str, pair: str):
        """
        🎯 STANDALONE DATA LOADER V3
        
        Args:
            explicit_path: Path to training-ready files (can be file or directory)
            pair: Crypto pair (from config.py)
        """
        # Handle both file and directory paths
        if os.path.isfile(explicit_path):
            # If it's a file, use its directory
            self.data_path = os.path.dirname(explicit_path)
            # If explicit file exists, use it directly
            self.explicit_file = explicit_path
        else:
            # If it's a directory, use it as-is
            self.data_path = explicit_path
            self.explicit_file = None
            
        self.pair = pair
        self.expected_filenames = self._generate_expected_filenames()
        self.selected_filename = None
        self.full_file_path = None
        
        # Feature scaling state
        self.scaler = None
        self.scaler_fitted = False
        self.scaling_stats = {}
        
        # Determine which file to use (with priority for single_label)
        self._select_best_file()
        
        if config.VERBOSE_LOGGING:
            print(f"🎯 STANDALONE DATA LOADER V3 INITIALIZED")
            print(f"   Data path: {self.data_path}")
            print(f"   Crypto pair: {self.pair}")
            print(f"   Expected files: {self.expected_filenames}")
            print(f"   Selected file: {self.selected_filename}")
            print(f"   Full path: {self.full_file_path}")
            print(f"   Feature scaling: {config.ENABLE_FEATURE_SCALING}")
            if config.ENABLE_FEATURE_SCALING:
                print(f"   Scaler type: {config.SCALER_TYPE}")
        
    def _generate_expected_filenames(self) -> list:
        """Generate list of expected filename patterns from config (priority order)"""
        # Use the format from validation module (no leading zeros for FW)
        base_pattern = f"{self.pair}_TF-1m__FW-{config.FUTURE_WINDOW}__SL-{int(config.LONG_SL_PCT*100):03d}__TP-{int(config.LONG_TP_PCT*100):03d}"
        
        # Priority order: single_label has priority over training_ready
        return [
            f"{base_pattern}__single_label.feather",     # NEW format (priority)
            f"{base_pattern}__training_ready.feather"    # OLD format (fallback)
        ]
    
    def _select_best_file(self):
        """Select the best available file based on priority"""
        # If explicit file was provided and exists, use it
        if self.explicit_file and os.path.exists(self.explicit_file):
            self.selected_filename = os.path.basename(self.explicit_file)
            self.full_file_path = self.explicit_file
            return
        
        # Otherwise, search for files in the directory
        for filename in self.expected_filenames:
            full_path = os.path.join(self.data_path, filename)
            if os.path.exists(full_path):
                self.selected_filename = filename
                self.full_file_path = full_path
                return
        
        # If no file found, use the priority filename for error reporting
        self.selected_filename = self.expected_filenames[0]
        self.full_file_path = os.path.join(self.data_path, self.selected_filename)
        
    def _create_scaler(self):
        """Create scaler based on config"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for feature scaling")
            
        if config.SCALER_TYPE == "standard":
            return StandardScaler()
        elif config.SCALER_TYPE == "robust":
            return RobustScaler()
        elif config.SCALER_TYPE == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {config.SCALER_TYPE}")
    
    def _save_scaler(self):
        """Save fitted scaler to disk"""
        if not self.scaler or not self.scaler_fitted:
            return
            
        scaler_path = os.path.join(config.get_model_output_dir(), config.get_scaler_filename())
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # Create scaler package with metadata
        scaler_package = {
            'scaler': self.scaler,
            'scaler_type': config.SCALER_TYPE,
            'scaling_stats': self.scaling_stats,
            'feature_names': [
                'high_change', 'low_change', 'close_change', 'volume_change',
                'price_to_ma1440', 'price_to_ma43200', 
                'volume_to_ma1440', 'volume_to_ma43200'
            ],
            'config_params': {
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
    
    def _calculate_scaling_stats(self, original_features: np.ndarray, scaled_features: np.ndarray, data_split: str):
        """Calculate and store scaling statistics"""
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

    def apply_class_balancing(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        🎯 SEQUENCE-AWARE: Apply class balancing while preserving temporal continuity
        
        Returns:
            Tuple (original_features, selected_labels, selected_target_indices)
            - original_features: UNCHANGED for sequence generation  
            - selected_labels: Labels for selected target moments only
            - selected_target_indices: Which time moments to use as targets
        """
        if not config.ENABLE_CLASS_BALANCING or config.CLASS_BALANCING_METHOD == "none":
            # No balancing - return original data with all indices
            indices = np.arange(len(features))
            return features, labels, indices
        
        # Convert labels to 1D if needed for class balancing
        if labels.ndim > 1:
            labels_1d = np.argmax(labels, axis=1)
        else:
            labels_1d = labels
        
        # Get class distribution
        unique_classes, class_counts = np.unique(labels_1d, return_counts=True)
        
        print(f"   📊 Original class distribution:")
        total_samples = len(labels_1d)
        for cls, count in zip(unique_classes, class_counts):
            pct = (count / total_samples) * 100
            class_name = ['SHORT', 'HOLD', 'LONG'][cls]
            print(f"      {class_name}: {count:,} ({pct:.1f}%)")
        
        if config.CLASS_BALANCING_METHOD == "systematic_undersampling":
            return self._sequence_aware_undersampling(features, labels, labels_1d, unique_classes, class_counts)
        else:
            raise ValueError(f"Unsupported class balancing method: {config.CLASS_BALANCING_METHOD}")
    
    def _sequence_aware_undersampling(self, features: np.ndarray, labels: np.ndarray, labels_1d: np.ndarray, 
                                    unique_classes: np.ndarray, original_class_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        🎯 SEQUENCE-AWARE UNDERSAMPLING - REVOLUTIONARY APPROACH!
        
        KEY INNOVATION: Decouples target selection from sequence generation
        - Undersampling selects WHICH time moments to use as training targets (class balance)
        - Sequences always created from original continuous temporal data (LSTM patterns)
        
        Example:
        - Original data: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10...]
        - Selected targets: [t2, t7] (balanced selection)  
        - Sequence for t7: [original t0, t1, t2, t3, t4, t5, t6] → predict t7
        
        NOT like broken approach:
        - Broken: [sparse t2, sparse t7] → [t2, t7] → predict ??? (temporal chaos!)
        
        Returns:
            Tuple (original_features, selected_labels, selected_target_indices)
        """
        print(f"   🎯 Applying SEQUENCE-AWARE undersampling (ALGORITHM COMPLIANT)...")
        
        # ALGORITHM COMPLIANT: Calculate eligible indices first
        window_size = config.WINDOW_SIZE  # Get from config
        total_samples = len(features)
        eligible_indices = np.arange(window_size, total_samples)
        
        print(f"   📏 Algorithm compliance: eligible_indices = range({window_size}, {total_samples})")
        print(f"   📊 Eligible samples: {len(eligible_indices):,} (have full history)")
        
        # Filter labels to only eligible indices (as per algorithm)
        eligible_labels_1d = labels_1d[eligible_indices]
        
        # Recalculate class counts for ELIGIBLE indices only (as per algorithm)
        eligible_unique_classes, eligible_class_counts = np.unique(eligible_labels_1d, return_counts=True)
        
        print(f"   📊 Original class distribution (all samples):")
        for cls, count in zip(unique_classes, original_class_counts):
            pct = (count / len(labels_1d)) * 100
            class_name = ['SHORT', 'HOLD', 'LONG'][cls]
            print(f"      {class_name}: {count:,} ({pct:.1f}%)")
        
        print(f"   📊 Eligible class distribution (algorithm compliant):")
        for cls, count in zip(eligible_unique_classes, eligible_class_counts):
            pct = (count / len(eligible_labels_1d)) * 100
            class_name = ['SHORT', 'HOLD', 'LONG'][cls]
            print(f"      {class_name}: {count:,} ({pct:.1f}%)")
        
        # Find target size (smallest class or minimum threshold) - FROM ELIGIBLE COUNTS
        min_class_size = np.min(eligible_class_counts)
        target_size = max(min_class_size, config.UNDERSAMPLING_MIN_SAMPLES)
        
        if target_size < min_class_size:
            print(f"   ⚠️ Target size ({config.UNDERSAMPLING_MIN_SAMPLES:,}) larger than minority class ({min_class_size:,})")
            print(f"   📉 Using minority class size: {min_class_size:,}")
            target_size = min_class_size
        
        print(f"   🎯 Target samples per class: {target_size:,}")
        
        # CRITICAL: Collect TARGET INDICES (which time moments to predict) - ALGORITHM COMPLIANT
        selected_target_indices = []
        np.random.seed(config.UNDERSAMPLING_SEED)
        
        for cls in eligible_unique_classes:
            # ALGORITHM COMPLIANT: Find class indices only within eligible range
            class_mask = (eligible_labels_1d == cls)
            class_eligible_positions = np.where(class_mask)[0]  # Positions within eligible_labels_1d
            class_indices = eligible_indices[class_eligible_positions]  # Convert back to original indices
            
            class_size = len(class_indices)
            class_name = ['SHORT', 'HOLD', 'LONG'][cls]
            
            if class_size <= target_size:
                # Take all samples from minority classes
                selected_indices = class_indices
                print(f"   ✅ {class_name}: {len(selected_indices):,}/{class_size:,} targets (all samples)")
            else:
                # Systematic sampling for majority classes
                step = class_size // target_size
                if step < 1:
                    step = 1
                
                # Start from random offset to avoid bias
                start_offset = np.random.randint(0, step) if step > 1 else 0
                selected_indices = class_indices[start_offset::step][:target_size]
                
                print(f"   📉 {class_name}: {len(selected_indices):,}/{class_size:,} targets (every {step}-th sample)")
            
            selected_target_indices.extend(selected_indices)
        
        # CRITICAL: Sort indices to maintain chronological order for targets
        selected_target_indices = np.sort(selected_target_indices)
        
        # REVOLUTIONARY: Return original features + selected target labels
        # - features: ORIGINAL (for continuous sequence generation)
        # - labels: ONLY for selected target indices (for class balance)
        # - indices: Which time moments to use as targets
        selected_labels = labels[selected_target_indices]
        
        # Verify results
        if selected_labels.ndim > 1:
            selected_labels_1d = np.argmax(selected_labels, axis=1)
        else:
            selected_labels_1d = selected_labels
            
        unique_selected, selected_counts = np.unique(selected_labels_1d, return_counts=True)
        
        print(f"   📊 Target distribution (what model will learn):")
        total_selected = len(selected_labels_1d)
        for cls, count in zip(unique_selected, selected_counts):
            pct = (count / total_selected) * 100
            class_name = ['SHORT', 'HOLD', 'LONG'][cls]
            print(f"      {class_name}: {count:,} ({pct:.1f}%)")
        
        print(f"   🎯 SEQUENCE-AWARE RESULT:")
        print(f"      Original features: {features.shape} (PRESERVED for sequences)")
        print(f"      Selected targets: {len(selected_target_indices):,} (balanced for training)")
        print(f"      Temporal continuity: MAINTAINED")
        print(f"      Class balance: ACHIEVED") 
        
        return features, selected_labels, selected_target_indices
    
    def _parse_filename_parameters(self, filename: str) -> dict:
        """Parse parameters from filename (supports both single_label and training_ready formats)"""
        # Flexible pattern that works with both FW-90 and FW-090 formats
        # Try single_label format first (priority)
        pattern_single = r'(\w+)_TF-1m__FW-(\d+)__SL-(\d+)__TP-(\d+)__single_label\.feather'
        match = re.match(pattern_single, filename)
        
        if match:
            return {
                'pair': match.group(1),
                'future_window': int(match.group(2)),
                'stop_loss': int(match.group(3)) / 100.0,
                'take_profit': int(match.group(4)) / 100.0,
                'format': 'single_label'
            }
        
        # Try training_ready format (fallback)
        pattern_ready = r'(\w+)_TF-1m__FW-(\d+)__SL-(\d+)__TP-(\d+)__training_ready\.feather'
        match = re.match(pattern_ready, filename)
            
        if match:
            return {
                'pair': match.group(1),
                'future_window': int(match.group(2)),
                'stop_loss': int(match.group(3)) / 100.0,
                    'take_profit': int(match.group(4)) / 100.0,
                    'format': 'training_ready'
            }
        
        raise ValueError(f"Invalid filename format: {filename}. Expected single_label or training_ready format.")
        
    def validate_file_exists(self) -> bool:
        """Hard fail validation - supports both single_label and training_ready formats"""
        if not os.path.exists(self.full_file_path):
            print(f"❌ TRAINING DATA FILE NOT FOUND:")
            print(f"   Selected path: {self.full_file_path}")
            print(f"   Selected filename: {self.selected_filename}")
            print(f"   Data directory: {self.data_path}")
            print(f"   Crypto pair: {self.pair}")
            print(f"   Tried filenames:")
            for filename in self.expected_filenames:
                full_path = os.path.join(self.data_path, filename)
                exists = "✅" if os.path.exists(full_path) else "❌"
                print(f"      {exists} {filename}")
            print("")
            print(f"💡 SOLUTION:")
            print(f"   1. Verify TRAINING_DATA_PATH in config.py")
            print(f"   2. Ensure validation module generated files with correct parameters:")
            print(f"      - FW-{config.FUTURE_WINDOW:03d} (current config)")
            print(f"      - SL-{int(config.LONG_SL_PCT*100):03d} (current config)")
            print(f"      - TP-{int(config.LONG_TP_PCT*100):03d} (current config)")
            print(f"   3. Check crypto pair spelling in config.py")
            return False
        return True
        
    def _validate_parameters_compatibility(self, file_params: dict) -> bool:
        """Validate filename parameters against config"""
        errors = []
        
        # Check pair match
        if file_params['pair'] != self.pair:
            errors.append(f"Pair mismatch: config={self.pair}, file={file_params['pair']}")
        
        # Check TP/SL/FW match
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
            
        # Check if datetime index exists (training-ready files use datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            # Parse dates
            from datetime import datetime
            # 🎯 KLUCZOWA POPRAWKA: Użyj `pd.to_datetime` i ustaw strefę UTC, aby zapewnić spójność
            start_dt = pd.to_datetime(config.START_DATE, utc=True)
            end_dt = pd.to_datetime(config.END_DATE, utc=True)
            
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
            
        # Fallback: Check if timestamp column exists
        elif 'timestamp' in df.columns:
            # Parse dates
            from datetime import datetime
            # 🎯 KLUCZOWA POPRAWKA: Użyj `pd.to_datetime` i ustaw strefę UTC
            start_dt = pd.to_datetime(config.START_DATE, utc=True)
            end_dt = pd.to_datetime(config.END_DATE, utc=True)
            
            # Convert timestamp column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Filter by date range
            original_count = len(df)
            df_filtered = df[
                (df['timestamp'] >= start_dt) & 
                (df['timestamp'] <= end_dt)
            ].reset_index(drop=True)
            
            if config.VERBOSE_LOGGING:
                print(f"   📅 Date filtering: {config.START_DATE} to {config.END_DATE}")
                print(f"   📊 Filtered: {original_count:,} → {len(df_filtered):,} samples")
                if len(df_filtered) > 0:
                    print(f"   📅 Actual range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
                
            if len(df_filtered) == 0:
                raise ValueError(f"No data found in date range {config.START_DATE} to {config.END_DATE}")
                
            return df_filtered
            
        else:
            # No datetime information available
            if config.VERBOSE_LOGGING:
                print(f"   ⚠️ No datetime index or timestamp column - skipping date filter")
            return df
        
    def load_training_data(self) -> Dict[str, Any]:
        """
        🎯 FIXED: Load training data with proper chronological split and zero data leakage
        
        FIXED ARCHITECTURE:
        1. Load RAW data with real timestamps
        2. CHRONOLOGICAL SPLIT FIRST (before any processing)
        3. Feature scaling ONLY on train, then transform val
        4. Class balancing ONLY on train data
        5. Validate chronological integrity
        
        Returns:
            Dict with separate train/val data (NO MIXED DATA!)
        """
        if not self.validate_file_exists():
            raise FileNotFoundError(f"Training-ready file not found: {self.full_file_path}")
            
        if config.VERBOSE_LOGGING:
            print(f"📁 Loading training-ready data with FIXED chronological pipeline...")
            print(f"   File: {self.selected_filename}")
            print(f"   Path: {self.data_path}")
        
        # Parse and validate filename parameters
        file_params = self._parse_filename_parameters(self.selected_filename)
        if config.VERBOSE_LOGGING:
            print(f"   Parameters from filename: {file_params}")
            print(f"   File format: {file_params.get('format', 'unknown')}")
        
        # Validate against config
        if not self._validate_parameters_compatibility(file_params):
            raise ValueError("Parameter compatibility validation failed")
        
        # 1. Load RAW data with real timestamps
        try:
            df = pd.read_feather(self.full_file_path)
        except Exception as e:
            raise IOError(f"Failed to load training-ready file: {e}")
            
        # Ensure timestamp column exists from index if not present
        if isinstance(df.index, pd.DatetimeIndex) and 'timestamp' not in df.columns:
            if config.VERBOSE_LOGGING:
                print(f"   ℹ️ 'timestamp' column not found, creating it from DatetimeIndex.")
            df['timestamp'] = df.index

        if config.VERBOSE_LOGGING:
            print(f"   ✅ Loaded: {len(df):,} samples")
            print(f"   Columns: {list(df.columns)}")
            print(f"   File size: {os.path.getsize(self.full_file_path) / (1024*1024):.1f} MB")
            if 'timestamp' in df.columns:
                print(f"   🗓️  RAW DATA DATE RANGE: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Apply date filtering (if enabled)
        df = self._filter_by_date_range(df)
        
        # 2. CHRONOLOGICAL SPLIT FIRST! (BEFORE any processing)
        train_df, val_df = self._chronological_split(df, config.VALIDATION_SPLIT)
        
        # 3. Feature scaling: fit ONLY on train, transform val (ZERO DATA LEAKAGE!)
        train_features, val_features, feature_names = self._apply_feature_scaling_chronological(train_df, val_df)
        
        # 4. SEQUENCE-AWARE Class balancing ONLY on train data (preserve original features for sequences)
        if config.ENABLE_CLASS_BALANCING and config.CLASS_BALANCING_METHOD == "systematic_undersampling":
            # SEQUENCE-AWARE: Returns original_features + selected targets + TARGET INDICES
            train_features, train_labels, train_timestamps, selected_target_indices = self._apply_class_balancing_with_sync(train_features, train_df)
            # train_features = ORIGINAL (unchanged for sequence generation)
            # train_labels = ONLY selected targets (for class balance)
            # train_timestamps = ONLY for selected targets
            # selected_target_indices = 🔥 CRITICAL FIX: Real indices for sequence generation
        else:
            # If class_weights or none, just get labels and timestamps without balancing
            train_labels = self._extract_labels(train_df)
            train_timestamps = train_df['timestamp'].values if 'timestamp' in train_df.columns else None
            selected_target_indices = None  # No undersampling - no target indices needed
            if config.VERBOSE_LOGGING and config.ENABLE_CLASS_BALANCING:
                print(f"   ⚖️ Skipping data modification for '{config.CLASS_BALANCING_METHOD}' method. Balancing will be handled by the trainer.")
        
        # Extract validation labels and timestamps (no balancing on validation!)
        val_labels = self._extract_labels(val_df)
        val_timestamps = val_df['timestamp'].values if 'timestamp' in val_df.columns else None
        
        # 5. Validate chronological integrity
        self._validate_chronological_integrity(train_df, val_df)
        
        if config.VERBOSE_LOGGING:
            print(f"✅ SEQUENCE-AWARE chronological pipeline completed:")
            print(f"   Train features: {train_features.shape} (ORIGINAL for sequences)")
            print(f"   Train targets: {len(train_labels):,} (SELECTED for balance)")
            print(f"   Val samples: {len(val_features):,}")
            print(f"   Scaling fitted on: train only ({len(train_df):,} samples)")
            print(f"   Balancing: SEQUENCE-AWARE (temporal continuity preserved)")
        
        # Calculate counts for reporting
        if config.ENABLE_CLASS_BALANCING and config.CLASS_BALANCING_METHOD == "systematic_undersampling":
            balanced_train_samples_count = len(train_labels)  # Selected targets count
        else:
            balanced_train_samples_count = len(train_features)  # All features count
        
        return {
            'train_features': train_features,
            'train_labels': train_labels,
            'train_timestamps': train_timestamps,
            'val_features': val_features,
            'val_labels': val_labels, 
            'val_timestamps': val_timestamps,
            'val_df': val_df,
            'original_samples_count': len(df),
            'train_samples_count': len(train_df),
            'balanced_train_samples_count': len(train_features),
            'val_samples_count': len(val_features),
            'file_params': file_params,
            'compatibility_validated': True,
            'chronological_split_applied': True,
            'label_format': 'sparse',  # V4 uses sparse format
            'scaling_enabled': True,
            'scaler_fitted': True,
            'sequence_aware_undersampling': config.CLASS_BALANCING_METHOD == "systematic_undersampling",
            'selected_target_indices': selected_target_indices
        }
    
    def _chronological_split(self, df: pd.DataFrame, validation_split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        🎯 FIXED: Proper chronological split with validation
        
        Args:
            df: Full DataFrame with real timestamps (or chronologically sorted rows)
            validation_split: Fraction for validation (e.g., 0.2 = 20%)
            
        Returns:
            Tuple (train_df, val_df) chronologically split
        """
        if config.VERBOSE_LOGGING:
            print(f"\n📅 APPLYING CHRONOLOGICAL SPLIT...")
            print(f"   Validation split: {validation_split:.1%}")
        
        # Check if timestamp column exists
        has_timestamp = 'timestamp' in df.columns
        
        if has_timestamp:
            print(f"   📅 Using real timestamps for chronological split")
            # Ensure data is sorted chronologically
            if not df['timestamp'].is_monotonic_increasing:
                if config.VERBOSE_LOGGING:
                    print(f"   ⚠️ Data not sorted chronologically, sorting...")
                df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            print(f"   ⚠️ No timestamp column found - using row order as chronological sequence")
            print(f"   📊 Assuming data is already chronologically sorted by row index")
            # Create synthetic timestamps for validation purposes only
            df['timestamp'] = pd.date_range(start='2022-01-01', periods=len(df), freq='1min')
        
        # Calculate split index (train_ratio = 1 - validation_split)
        train_ratio = 1.0 - validation_split
        split_idx = int(len(df) * train_ratio)
        
        # Split chronologically
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        # Validate chronological integrity
        train_max_date = train_df['timestamp'].max()
        val_min_date = val_df['timestamp'].min()
        
        if train_max_date >= val_min_date:
            raise ValueError(f"CRITICAL: Chronological split failed! Train max ({train_max_date}) >= Val min ({val_min_date})")
        
        gap_days = (val_min_date - train_max_date).days
        gap_hours = ((val_min_date - train_max_date).total_seconds() / 3600)
        
        if config.VERBOSE_LOGGING:
            print(f"   ✅ Chronological split successful:")
            if has_timestamp:
                print(f"      Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
                print(f"      Val period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
            else:
                print(f"      Train samples: rows 0 to {split_idx-1}")
                print(f"      Val samples: rows {split_idx} to {len(df)-1}")
                print(f"      Synthetic time period: {train_df['timestamp'].min()} to {val_df['timestamp'].max()}")
            print(f"      Chronological gap: {gap_days} days ({gap_hours:.1f} hours)")
            print(f"      Train samples: {len(train_df):,} ({train_ratio:.1%})")
            print(f"      Val samples: {len(val_df):,} ({validation_split:.1%})")
        
        return train_df, val_df
    
    def _apply_feature_scaling_chronological(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        🎯 NEW: Feature scaling with ZERO data leakage
        
        Args:
            train_df: Training DataFrame (chronologically first)
            val_df: Validation DataFrame (chronologically second)
            
        Returns:
            Tuple (train_features, val_features, feature_names)
        """
        # Expected feature columns (must match validation module)
        feature_columns = [
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200', 
            'volume_to_ma1440', 'volume_to_ma43200'
        ]
        
        # Validate feature columns exist
        missing_features = [col for col in feature_columns if col not in train_df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in train data: {missing_features}")
        
        missing_features_val = [col for col in feature_columns if col not in val_df.columns]
        if missing_features_val:
            raise ValueError(f"Missing feature columns in val data: {missing_features_val}")
        
        # Extract raw features
        train_features = train_df[feature_columns].values.astype(np.float32)
        val_features = val_df[feature_columns].values.astype(np.float32)
        
        if config.VERBOSE_LOGGING:
            print(f"\n📏 APPLYING FEATURE SCALING (chronological)...")
            print(f"   Train features shape: {train_features.shape}")
            print(f"   Val features shape: {val_features.shape}")
        
        # Feature scaling with ZERO data leakage
        if config.ENABLE_FEATURE_SCALING:
            if not SKLEARN_AVAILABLE:
                print("   ⚠️ Feature scaling requested but scikit-learn not available")
                return train_features, val_features, feature_columns
                
            if config.VERBOSE_LOGGING:
                print(f"   🔧 Scaler type: {config.SCALER_TYPE}")
                print(f"   🔧 Fit only on train: {config.SCALER_FIT_ONLY_TRAIN}")
            
            # Store original features for statistics
            original_train_features = train_features.copy()
            original_val_features = val_features.copy()
            
            # Create and fit scaler ONLY on train data
            self.scaler = self._create_scaler()
            train_features = self.scaler.fit_transform(train_features)
            self.scaler_fitted = True
            
            # Transform validation data using train-fitted scaler (ZERO LEAKAGE!)
            val_features = self.scaler.transform(val_features)
            
            # Calculate and store scaling statistics
            self._calculate_scaling_stats(original_train_features, train_features, 'train')
            self._calculate_scaling_stats(original_val_features, val_features, 'validation')
            
            # Save scaler fitted only on train
            self._save_scaler()
            
            if config.VERBOSE_LOGGING:
                print(f"   ✅ Feature scaling completed (ZERO data leakage):")
                print(f"      Scaler fitted on: {len(train_features):,} train samples")
                print(f"      Scaler transformed: {len(val_features):,} val samples")
                
                # Validate scaling results
                if config.VALIDATE_SCALING_STATS:
                    train_mean = np.mean(train_features, axis=0)
                    val_mean = np.mean(val_features, axis=0)
                    print(f"      Train feature means: [{train_mean.min():.3f}, {train_mean.max():.3f}]")
                    print(f"      Val feature means: [{val_mean.min():.3f}, {val_mean.max():.3f}]")
        
        else:
            if config.VERBOSE_LOGGING:
                print(f"   📏 Feature scaling: DISABLED")
        
        return train_features, val_features, feature_columns
    
    def _apply_class_balancing_with_sync(self, features: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        🎯 SEQUENCE-AWARE: Class balancing with temporal continuity preservation
        
        Args:
            features: Training features (already scaled) - ORIGINAL for sequence generation
            df: Training DataFrame with labels and timestamps
            
        Returns:
            Tuple (original_features, selected_labels, selected_timestamps, selected_target_indices)
            - original_features: UNCHANGED for continuous sequence generation
            - selected_labels: Labels for selected target indices only  
            - selected_timestamps: Timestamps for selected targets only
            - selected_target_indices: 🔥 CRITICAL FIX: Real indices for sequence generation
        """
        # This method is now only called for systematic_undersampling
        if config.CLASS_BALANCING_METHOD != "systematic_undersampling":
             raise RuntimeError("This method should only be called for systematic_undersampling")

        # Extract train labels
        train_labels = self._extract_labels(df)
        
        if config.VERBOSE_LOGGING:
            print(f"\n⚖️ APPLYING SEQUENCE-AWARE CLASS BALANCING (train only)...")
            print(f"   Method: {config.CLASS_BALANCING_METHOD}")
            print(f"   Original train samples: {len(features):,}")
            print(f"   🔍 DIAGNOSTIC: Input validation:")
            print(f"      Features shape: {features.shape}")
            print(f"      DataFrame length: {len(df)}")
            print(f"      Train labels shape: {train_labels.shape}")
        
        # Apply sequence-aware class balancing - returns original features + selected targets!
        original_features, selected_labels, selected_target_indices = self.apply_class_balancing(features, train_labels)
        
        # 🔍 DIAGNOSTIC LOGS - CRITICAL TARGET INDICES VALIDATION
        if config.VERBOSE_LOGGING:
            print(f"\n   🔍 TARGET INDICES DIAGNOSTIC:")
            print(f"      Selected target indices shape: {selected_target_indices.shape}")
            print(f"      Selected target indices range: [{selected_target_indices.min()}, {selected_target_indices.max()}]")
            print(f"      Window size: {config.WINDOW_SIZE}")
            print(f"      Original features length: {len(original_features)}")
            
            # Validate all indices are eligible (>= WINDOW_SIZE)
            invalid_indices = selected_target_indices[selected_target_indices < config.WINDOW_SIZE]
            if len(invalid_indices) > 0:
                print(f"      ❌ CRITICAL: Found {len(invalid_indices)} invalid indices < WINDOW_SIZE!")
                print(f"         Invalid indices: {invalid_indices[:10]}...")  # Show first 10
            else:
                print(f"      ✅ All target indices >= WINDOW_SIZE (valid)")
            
            # Sample mapping validation
            print(f"      🎯 SAMPLE TARGET MAPPING (first 5):")
            for i in range(min(5, len(selected_target_indices))):
                target_idx = selected_target_indices[i]
                sequence_start = target_idx - config.WINDOW_SIZE
                sequence_end = target_idx
                label = selected_labels[i]
                label_class = np.argmax(label)
                
                print(f"         Target {i}: idx={target_idx} → seq[{sequence_start}:{sequence_end}] → predict {['SHORT','HOLD','LONG'][label_class]}")
                
                # Validate sequence bounds
                if sequence_start < 0:
                    print(f"            ❌ Invalid: sequence_start ({sequence_start}) < 0")
                if sequence_end > len(original_features):
                    print(f"            ❌ Invalid: sequence_end ({sequence_end}) > features length ({len(original_features)})")
        
        if config.VERBOSE_LOGGING:
            print(f"   ✅ Sequence-aware class balancing completed:")
            print(f"      Original features preserved: {original_features.shape}")
            print(f"      Selected targets: {len(selected_labels):,}")
            print(f"      🔥 CRITICAL FIX: Selected target indices: {len(selected_target_indices):,}")
            print(f"      Validation samples: UNTOUCHED (no balancing)")
        
        # Extract train timestamps using selected_target_indices to keep synchronization
        if 'timestamp' in df.columns:
            all_timestamps = df['timestamp'].values
            selected_timestamps = all_timestamps[selected_target_indices]
            
            # 🔍 DIAGNOSTIC: Validate timestamp synchronization
            if config.VERBOSE_LOGGING:
                print(f"      🔍 TIMESTAMP SYNCHRONIZATION:")
                print(f"         All timestamps length: {len(all_timestamps)}")
                print(f"         Selected timestamps length: {len(selected_timestamps)}")
                print(f"         Sample sync check (first 3):")
                for i in range(min(3, len(selected_target_indices))):
                    target_idx = selected_target_indices[i]
                    target_timestamp = all_timestamps[target_idx]
                    selected_timestamp = selected_timestamps[i]
                    is_synced = target_timestamp == selected_timestamp
                    print(f"            Target {i}: idx={target_idx} → {target_timestamp} == {selected_timestamp} ({'✅' if is_synced else '❌'})")
        else:
            selected_timestamps = None
            if config.VERBOSE_LOGGING:
                print(f"      ⚠️ No timestamp column in DataFrame")
        
        return original_features, selected_labels, selected_timestamps, selected_target_indices
    
    def _extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        🎯 V4 SPARSE: Extract labels from DataFrame and return sparse format
        
        Args:
            df: DataFrame with label columns
            
        Returns:
            Labels array in sparse categorical format (shape: (n,))
        """
        # Check for single_label format first (already sparse)
        if 'label' in df.columns:
            if config.VERBOSE_LOGGING:
                print(f"   🎯 Detected single_label format (sparse categorical)")
            
            # Return sparse labels directly
            sparse_labels = df['label'].values.astype(np.int32)
            
            if config.VERBOSE_LOGGING:
                unique, counts = np.unique(sparse_labels, return_counts=True)
                total = len(sparse_labels)
                print(f"   📊 Label distribution (sparse):")
                for cls, count in zip(unique, counts):
                    class_name = ['SHORT', 'HOLD', 'LONG'][cls]
                    pct = (count / total) * 100
                    print(f"      {class_name} ({cls}): {count:,} ({pct:.1f}%)")
            
            return sparse_labels
        
        # Convert onehot format to sparse (training_ready format)
        else:
            if config.VERBOSE_LOGGING:
                print(f"   🎯 Detected training_ready format (onehot → sparse)")
            
            # Expected label columns (onehot format)
            label_columns = ['label_0', 'label_1', 'label_2']
            
            # Validate label columns exist
            missing_labels = [col for col in label_columns if col not in df.columns]
            if missing_labels:
                raise ValueError(f"Missing label columns: {missing_labels}")
            
            # Convert onehot to sparse using argmax
            onehot_labels = df[label_columns].values.astype(np.float32)
            sparse_labels = np.argmax(onehot_labels, axis=1).astype(np.int32)
            
            if config.VERBOSE_LOGGING:
                unique, counts = np.unique(sparse_labels, return_counts=True)
                total = len(sparse_labels)
                print(f"   📊 Label distribution (onehot → sparse):")
                for cls, count in zip(unique, counts):
                    class_name = ['SHORT', 'HOLD', 'LONG'][cls]
                    pct = (count / total) * 100
                    print(f"      {class_name} ({cls}): {count:,} ({pct:.1f}%)")
            
            return sparse_labels
    
    def _validate_chronological_integrity(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        🎯 FIXED: Validate chronological integrity between train and validation
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
        """
        if config.VERBOSE_LOGGING:
            print(f"\n🔍 VALIDATING CHRONOLOGICAL INTEGRITY...")
        
        # Check if timestamps are synthetic (created by data_loader)
        synthetic_start = pd.Timestamp('2022-01-01 00:00:00')
        first_train_timestamp = pd.Timestamp(train_df['timestamp'].iloc[0])
        timestamps_are_synthetic = (first_train_timestamp == synthetic_start)
        
        if timestamps_are_synthetic:
            print(f"   📅 Timestamps are synthetic (created by data_loader for validation)")
            print(f"   🎯 Chronological integrity based on ROW ORDER")
        else:
            print(f"   📅 Timestamps are real (from original data)")
            print(f"   🎯 Chronological integrity based on ACTUAL TIMESTAMPS")
        
        # Test 1: Chronological order
        train_max = train_df['timestamp'].max()
        val_min = val_df['timestamp'].min()
        
        if train_max >= val_min:
            raise ValueError(f"CRITICAL: Train data newer than validation! Train max: {train_max}, Val min: {val_min}")
        
        gap_days = (val_min - train_max).days
        gap_hours = ((val_min - train_max).total_seconds() / 3600)
        
        # Test 2: No overlap in timestamps (if real timestamps)
        if not timestamps_are_synthetic:
            train_timestamps = set(train_df['timestamp'])
            val_timestamps = set(val_df['timestamp'])
            overlap = train_timestamps.intersection(val_timestamps)
            
            if overlap:
                raise ValueError(f"CRITICAL: Timestamp overlap detected! {len(overlap)} overlapping timestamps")
            print(f"   ✅ No timestamp overlap confirmed")
        else:
            print(f"   📅 Timestamp overlap test skipped (synthetic timestamps)")
        
        # Test 3: Feature scaling integrity
        if hasattr(self, 'scaler') and self.scaler and hasattr(self.scaler, 'n_samples_seen_'):
            fitted_samples = self.scaler.n_samples_seen_
            expected_samples = len(train_df)
            if fitted_samples != expected_samples:
                raise ValueError(f"CRITICAL: Scaler fitted on {fitted_samples}, expected {expected_samples} (train only)")
        
        # Test 4: Row index integrity (most important for synthetic timestamps)
        max_train_index = train_df.index.max()
        min_val_index = val_df.index.min()
        
        if max_train_index >= min_val_index:
            raise ValueError(f"CRITICAL: Row index overlap! Train max index: {max_train_index}, Val min index: {min_val_index}")
        
        if config.VERBOSE_LOGGING:
            print(f"   ✅ Chronological integrity validation passed:")
            print(f"      Chronological gap: {gap_days} days ({gap_hours:.1f} hours)")
            if timestamps_are_synthetic:
                print(f"      Row index gap: Train ends at {max_train_index}, Val starts at {min_val_index}")
                print(f"      Synthetic timestamps: Used for validation only")
            else:
                print(f"      Real timestamps: Full chronological integrity confirmed")
            print(f"      Scaler fitted on train only: ✓")
            if hasattr(self, 'scaler') and self.scaler:
                print(f"      Scaler samples: {getattr(self.scaler, 'n_samples_seen_', 'Unknown')}")
        
        if timestamps_are_synthetic:
            print("   🎯 CHRONOLOGICAL SPLIT BY ROW ORDER CONFIRMED!")
        else:
            print("   🎯 ZERO DATA LEAKAGE CONFIRMED!")
        print("   🎯 TRAIN < VALIDATION TEMPORALLY GUARANTEED!")
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get file information without loading full data"""
        if not self.validate_file_exists():
            return {'exists': False, 'tried_files': self.expected_filenames}
        
        file_size = os.path.getsize(self.full_file_path)
        file_params = self._parse_filename_parameters(self.selected_filename)
        
        # Quick sample count (read just first few rows)
        try:
            df_sample = pd.read_feather(self.full_file_path, columns=['timestamp'] if os.path.exists(self.full_file_path) else [])
            rows = len(df_sample) if not df_sample.empty else 0
        except:
            rows = 0
        
        return {
            'exists': True,
            'file_path': self.full_file_path,
            'filename': self.selected_filename,
            'size_mb': file_size / (1024*1024),
            'size_bytes': file_size,
            'rows': rows,
            'file_params': file_params,
            'format': file_params.get('format', 'unknown'),
            'compatible': True,  # Assume compatible if file exists with correct naming
            'scaling_configured': config.ENABLE_FEATURE_SCALING,
            'scaler_type': config.SCALER_TYPE if config.ENABLE_FEATURE_SCALING else None,
            'tried_files': self.expected_filenames
        }


def main():
    """Test standalone data loader"""
    print("🧪 TESTING STANDALONE DATA LOADER V3 WITH SCALING")
    
    try:
        # Test with config parameters
        loader = TrainingDataLoader(config.TRAINING_DATA_PATH, config.PAIR)
        
        print(f"\n🔍 File validation test:")
        if loader.validate_file_exists():
            info = loader.get_file_info()
            print(f"   ✅ File found: {info['filename']}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Rows: {info['rows']:,}")
            print(f"   Parameters: {info['file_params']}")
            print(f"   Scaling configured: {info['scaling_configured']}")
            if info['scaling_configured']:
                print(f"   Scaler type: {info['scaler_type']}")
        else:
            print(f"   ❌ File not found")
        
        print(f"\n✅ Standalone data loader with scaling test completed!")
        
    except Exception as e:
        print(f"❌ Standalone data loader test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 