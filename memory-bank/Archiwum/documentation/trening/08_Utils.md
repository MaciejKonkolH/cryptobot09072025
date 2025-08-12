# 📖 UTILS MODUŁU TRENUJĄCEGO V3 (utils.py)

## 🎯 PRZEGLĄD UTILS

**utils.py** (422 linie) to zbiór helper functions i monitoring utilities wspierających główne komponenty modułu trenującego V3. Zawiera funkcje walidacji danych, monitorowania pamięci, operacji na plikach oraz systemowych utility z graceful fallback.

### ✨ **Kluczowe Funkcjonalności V3 Utils**
- ✅ **Data Validation** - OHLCV data quality checks
- ✅ **Memory Monitoring** - real-time memory usage tracking
- ✅ **File Operations** - directory management, safe file handling
- ✅ **System Utilities** - optional psutil integration with fallback
- ✅ **Memory Calculations** - estimate memory requirements
- ✅ **Error Handling** - graceful degradation when dependencies missing

## 📊 **DATA VALIDATION UTILITIES**

### **OHLCV Data Quality Validation**
```python
def validate_data_quality(df: pd.DataFrame, pair: str, verbose: bool = True) -> bool:
    """
    📊 COMPREHENSIVE DATA QUALITY VALIDATION
    
    Validates OHLCV data integrity for crypto trading:
    - Missing values detection
    - Invalid price relationships (High < Low, etc.)
    - Zero/negative volume detection
    - Duplicate timestamps
    - Data type validation
    """
    
    if verbose:
        print(f"\n📊 VALIDATING DATA QUALITY FOR {pair}...")
    
    validation_errors = []
    warnings = []
    
    # 1. Basic structure validation
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_errors.append(f"Missing required columns: {missing_columns}")
    
    # 2. Missing values check
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        for col, count in missing_values.items():
            if count > 0:
                validation_errors.append(f"Missing values in {col}: {count}")
    
    # 3. Invalid price relationships
    invalid_high_low = (df['high'] < df['low']).sum()
    if invalid_high_low > 0:
        validation_errors.append(f"Invalid High < Low relationships: {invalid_high_low}")
    
    invalid_prices = (
        (df['close'] > df['high']) | 
        (df['close'] < df['low']) |
        (df['open'] > df['high']) | 
        (df['open'] < df['low'])
    ).sum()
    if invalid_prices > 0:
        validation_errors.append(f"Invalid OHLC relationships: {invalid_prices}")
    
    # 4. Zero/negative values
    zero_negative_prices = (
        (df['open'] <= 0) | 
        (df['high'] <= 0) | 
        (df['low'] <= 0) | 
        (df['close'] <= 0)
    ).sum()
    if zero_negative_prices > 0:
        validation_errors.append(f"Zero/negative prices: {zero_negative_prices}")
    
    zero_volume = (df['volume'] == 0).sum()
    if zero_volume > 0:
        warnings.append(f"Zero volume bars: {zero_volume}")
    
    negative_volume = (df['volume'] < 0).sum()
    if negative_volume > 0:
        validation_errors.append(f"Negative volume: {negative_volume}")
    
    # 5. Duplicate timestamps (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            validation_errors.append(f"Duplicate timestamps: {duplicates}")
    
    # 6. Extreme outliers detection
    for col in ['open', 'high', 'low', 'close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > len(df) * 0.01:  # More than 1% outliers
            warnings.append(f"Potential outliers in {col}: {outliers} ({outliers/len(df)*100:.1f}%)")
    
    # Report results
    if verbose:
        print(f"   📈 Dataset: {len(df):,} samples")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"   📅 Period: {df.index.min()} to {df.index.max()}")
        
        if validation_errors:
            print(f"   ❌ Validation ERRORS ({len(validation_errors)}):")
            for error in validation_errors:
                print(f"      - {error}")
        
        if warnings:
            print(f"   ⚠️ Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"      - {warning}")
        
        if not validation_errors and not warnings:
            print(f"   ✅ Data quality: EXCELLENT")
        elif not validation_errors:
            print(f"   ✅ Data quality: GOOD (with warnings)")
        else:
            print(f"   ❌ Data quality: POOR (errors found)")
    
    return len(validation_errors) == 0
```

### **Feature Data Validation**
```python
def validate_feature_data(features: np.ndarray, feature_names: list, verbose: bool = True) -> bool:
    """
    📊 VALIDATE FEATURE DATA INTEGRITY
    
    Validates processed feature arrays:
    - NaN/Inf detection
    - Feature range validation
    - Statistical distribution analysis
    """
    
    if verbose:
        print(f"\n📊 VALIDATING FEATURE DATA...")
    
    validation_passed = True
    
    # Basic shape validation
    expected_features = 8
    if features.shape[1] != expected_features:
        print(f"   ❌ Invalid feature count: expected {expected_features}, got {features.shape[1]}")
        validation_passed = False
    
    # NaN/Inf detection
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    
    if nan_count > 0:
        print(f"   ❌ NaN values found: {nan_count}")
        validation_passed = False
    
    if inf_count > 0:
        print(f"   ❌ Inf values found: {inf_count}")
        validation_passed = False
    
    # Per-feature analysis
    if verbose and validation_passed:
        print(f"   📊 Feature statistics:")
        for i, name in enumerate(feature_names):
            feature_data = features[:, i]
            print(f"      {name}:")
            print(f"         Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
            print(f"         Mean: {feature_data.mean():.4f}, Std: {feature_data.std():.4f}")
    
    if verbose:
        if validation_passed:
            print(f"   ✅ Feature validation: PASSED")
        else:
            print(f"   ❌ Feature validation: FAILED")
    
    return validation_passed
```

## 💾 **MEMORY MONITORING UTILITIES**

### **Real-Time Memory Monitoring**
```python
def monitor_memory_usage(process_name: str = "Training Process", 
                        interval: int = 10, 
                        max_monitoring_time: int = 3600) -> None:
    """
    💾 REAL-TIME MEMORY MONITORING
    
    Features:
    - Real-time memory usage tracking
    - Memory leak detection
    - Alert system for high memory usage
    - Graceful fallback if psutil unavailable
    """
    
    try:
        import psutil
        
        print(f"\n💾 STARTING MEMORY MONITORING...")
        print(f"   Process: {process_name}")
        print(f"   Interval: {interval} seconds")
        print(f"   Max monitoring time: {max_monitoring_time} seconds")
        
        process = psutil.Process()
        start_time = time.time()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = initial_memory
        memory_history = []
        
        print(f"   📊 Initial memory: {initial_memory:.1f} MB")
        
        while (time.time() - start_time) < max_monitoring_time:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                memory_history.append({
                    'timestamp': time.time(),
                    'memory_mb': current_memory,
                    'increase_mb': memory_increase
                })
                
                # Track maximum memory
                if current_memory > max_memory:
                    max_memory = current_memory
                
                # Alert for high memory usage
                if current_memory > 8000:  # 8GB alert
                    print(f"   🚨 HIGH MEMORY USAGE: {current_memory:.1f} MB")
                
                # Alert for significant memory leaks
                if memory_increase > 1000:  # 1GB increase
                    print(f"   ⚠️ MEMORY LEAK DETECTED: +{memory_increase:.1f} MB from start")
                
                # Periodic reporting
                if len(memory_history) % 6 == 0:  # Every 60 seconds (if interval=10)
                    print(f"   📊 Memory: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
                
                time.sleep(interval)
                
            except psutil.NoSuchProcess:
                print(f"   ⚠️ Process terminated - stopping memory monitoring")
                break
            except KeyboardInterrupt:
                print(f"   🛑 Memory monitoring interrupted by user")
                break
        
        # Final report
        print(f"\n💾 MEMORY MONITORING SUMMARY:")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Maximum memory: {max_memory:.1f} MB")
        print(f"   Total increase: {max_memory - initial_memory:.1f} MB")
        print(f"   Monitoring duration: {(time.time() - start_time)/60:.1f} minutes")
        
    except ImportError:
        print(f"   ⚠️ psutil not available - memory monitoring disabled")
        print(f"   💡 Install psutil for memory monitoring: pip install psutil")
```

### **Memory Usage Calculation**
```python
def calculate_memory_usage(data_shape: tuple, dtype: str = 'float32') -> Dict[str, float]:
    """
    💾 CALCULATE MEMORY USAGE FOR DATA STRUCTURES
    
    Estimates memory requirements for arrays and data structures
    """
    
    # Bytes per element based on dtype
    dtype_sizes = {
        'float32': 4,
        'float64': 8,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'bool': 1
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 4)  # Default to float32
    total_elements = np.prod(data_shape)
    total_bytes = total_elements * bytes_per_element
    
    # Convert to different units
    memory_info = {
        'total_elements': int(total_elements),
        'bytes_per_element': bytes_per_element,
        'total_bytes': int(total_bytes),
        'kilobytes': total_bytes / 1024,
        'megabytes': total_bytes / 1024 / 1024,
        'gigabytes': total_bytes / 1024 / 1024 / 1024,
        'data_shape': data_shape,
        'dtype': dtype
    }
    
    return memory_info

def get_system_memory_info() -> Dict[str, float]:
    """
    💾 GET SYSTEM MEMORY INFORMATION
    
    Returns current system memory usage with graceful fallback
    """
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        
        memory_info = {
            'total_gb': memory.total / 1024 / 1024 / 1024,
            'available_gb': memory.available / 1024 / 1024 / 1024,
            'used_gb': memory.used / 1024 / 1024 / 1024,
            'percentage_used': memory.percent,
            'free_gb': memory.free / 1024 / 1024 / 1024
        }
        
        return memory_info
        
    except ImportError:
        # Graceful fallback when psutil unavailable
        return {
            'total_gb': 'unknown',
            'available_gb': 'unknown',
            'used_gb': 'unknown',
            'percentage_used': 'unknown',
            'free_gb': 'unknown',
            'note': 'psutil not available'
        }
```

## 📁 **FILE OPERATIONS UTILITIES**

### **Safe Directory Management**
```python
def safe_create_directory(directory_path: str, verbose: bool = True) -> bool:
    """
    📁 SAFE DIRECTORY CREATION
    
    Creates directory with proper error handling and permissions
    """
    
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            if verbose:
                print(f"   📁 Directory created: {directory_path}")
        else:
            if verbose:
                print(f"   📁 Directory exists: {directory_path}")
        
        # Verify directory is writable
        test_file = os.path.join(directory_path, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            if verbose:
                print(f"   ✅ Directory writable: {directory_path}")
            return True
            
        except PermissionError:
            if verbose:
                print(f"   ❌ Directory not writable: {directory_path}")
            return False
            
    except Exception as e:
        if verbose:
            print(f"   ❌ Failed to create directory: {e}")
        return False

def cleanup_temp_files(temp_directory: str, pattern: str = "temp_*", verbose: bool = True) -> int:
    """
    📁 CLEANUP TEMPORARY FILES
    
    Removes temporary files matching pattern with safety checks
    """
    
    if not os.path.exists(temp_directory):
        if verbose:
            print(f"   📁 Temp directory doesn't exist: {temp_directory}")
        return 0
    
    import glob
    
    temp_files = glob.glob(os.path.join(temp_directory, pattern))
    removed_count = 0
    
    if verbose and temp_files:
        print(f"   🧹 Cleaning {len(temp_files)} temporary files...")
    
    for temp_file in temp_files:
        try:
            if os.path.isfile(temp_file):
                # Safety check - only remove files, not directories
                file_size = os.path.getsize(temp_file)
                os.remove(temp_file)
                removed_count += 1
                
                if verbose:
                    print(f"      Removed: {os.path.basename(temp_file)} ({file_size} bytes)")
                    
        except Exception as e:
            if verbose:
                print(f"      Failed to remove {temp_file}: {e}")
    
    if verbose:
        print(f"   ✅ Cleanup completed: {removed_count} files removed")
    
    return removed_count
```

### **File Size and Disk Space Utilities**
```python
def get_file_size_info(file_path: str) -> Dict[str, any]:
    """
    📁 GET COMPREHENSIVE FILE SIZE INFORMATION
    """
    
    if not os.path.exists(file_path):
        return {'exists': False, 'error': 'File not found'}
    
    try:
        stat_info = os.stat(file_path)
        size_bytes = stat_info.st_size
        
        file_info = {
            'exists': True,
            'size_bytes': size_bytes,
            'size_kb': size_bytes / 1024,
            'size_mb': size_bytes / 1024 / 1024,
            'size_gb': size_bytes / 1024 / 1024 / 1024,
            'modified_timestamp': stat_info.st_mtime,
            'modified_datetime': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'is_file': os.path.isfile(file_path),
            'is_directory': os.path.isdir(file_path)
        }
        
        return file_info
        
    except Exception as e:
        return {'exists': True, 'error': f'Cannot access file info: {e}'}

def check_disk_space(directory_path: str, required_gb: float = 1.0) -> Dict[str, any]:
    """
    📁 CHECK AVAILABLE DISK SPACE
    
    Verifies sufficient disk space for operations
    """
    
    try:
        if hasattr(os, 'statvfs'):  # Unix/Linux
            statvfs = os.statvfs(directory_path)
            available_bytes = statvfs.f_frsize * statvfs.f_available
        else:  # Windows
            import shutil
            total, used, available_bytes = shutil.disk_usage(directory_path)
        
        available_gb = available_bytes / 1024 / 1024 / 1024
        sufficient_space = available_gb >= required_gb
        
        return {
            'available_gb': available_gb,
            'required_gb': required_gb,
            'sufficient_space': sufficient_space,
            'available_bytes': available_bytes
        }
        
    except Exception as e:
        return {
            'error': f'Cannot check disk space: {e}',
            'sufficient_space': False
        }
```

## 🔄 **TIME SERIES UTILITIES**

### **Chronological Data Splitting**
```python
def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8, 
                       validate_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    🔄 CHRONOLOGICAL DATA SPLITTING FOR TIME SERIES
    
    Critical for time series - prevents data leakage
    """
    
    if validate_split:
        print(f"\n🔄 CHRONOLOGICAL DATA SPLITTING...")
        print(f"   Train ratio: {train_ratio:.1%}")
        print(f"   Total samples: {len(df):,}")
    
    # Calculate split point
    split_index = int(len(df) * train_ratio)
    
    # Chronological split
    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()
    
    if validate_split:
        print(f"   Training set: {len(train_df):,} samples ({len(train_df)/len(df):.1%})")
        print(f"   Validation set: {len(val_df):,} samples ({len(val_df)/len(df):.1%})")
        
        # Report date ranges if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"   Train period: {train_df.index.min()} to {train_df.index.max()}")
            print(f"   Validation period: {val_df.index.min()} to {val_df.index.max()}")
        
        # Validate no overlap
        if isinstance(df.index, pd.DatetimeIndex):
            if train_df.index.max() >= val_df.index.min():
                print(f"   ⚠️ Warning: Potential temporal overlap detected")
        
        print(f"   ✅ Chronological split completed")
    
    return train_df, val_df
```

## 🧹 **MEMORY CLEANUP UTILITIES**

### **Safe Memory Cleanup**
```python
def safe_memory_cleanup(verbose: bool = True) -> Dict[str, any]:
    """
    🧹 SAFE MEMORY CLEANUP AND GARBAGE COLLECTION
    
    Forces garbage collection and reports memory impact
    """
    
    import gc
    
    if verbose:
        print(f"\n🧹 PERFORMING MEMORY CLEANUP...")
    
    # Get initial memory usage
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        psutil_available = True
    except ImportError:
        initial_memory = None
        psutil_available = False
    
    # Force garbage collection
    collected_objects = []
    for generation in range(3):  # Python has 3 generations
        collected = gc.collect()
        collected_objects.append(collected)
        if verbose and collected > 0:
            print(f"   🧹 Generation {generation}: {collected} objects collected")
    
    # Get final memory usage
    if psutil_available:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = initial_memory - final_memory
    else:
        final_memory = None
        memory_freed = None
    
    cleanup_info = {
        'total_collected': sum(collected_objects),
        'collected_by_generation': collected_objects,
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_freed_mb': memory_freed,
        'gc_stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
    }
    
    if verbose:
        print(f"   📊 Cleanup summary:")
        print(f"      Objects collected: {sum(collected_objects)}")
        if memory_freed is not None:
            if memory_freed > 0:
                print(f"      Memory freed: {memory_freed:.1f} MB")
            else:
                print(f"      Memory freed: {abs(memory_freed):.1f} MB (negative = increased)")
        print(f"   ✅ Memory cleanup completed")
    
    return cleanup_info
```

## 🎯 **USAGE EXAMPLES**

### **Example 1: Data Validation**
```python
# Validate OHLCV data quality
df = pd.read_feather("BTCUSDT_data.feather")
is_valid = validate_data_quality(df, "BTCUSDT", verbose=True)

# Validate feature arrays
features = np.random.random((10000, 8))
feature_names = ['high_change', 'low_change', 'close_change', 'volume_change',
                'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200']
is_valid = validate_feature_data(features, feature_names, verbose=True)
```

### **Example 2: Memory Monitoring**
```python
# Start memory monitoring in background
import threading

monitor_thread = threading.Thread(
    target=monitor_memory_usage,
    args=("LSTM Training", 30, 7200),  # 30s interval, 2h max
    daemon=True
)
monitor_thread.start()

# Calculate memory requirements
data_shape = (1000000, 8)  # 1M samples, 8 features
memory_info = calculate_memory_usage(data_shape, 'float32')
print(f"Estimated memory: {memory_info['megabytes']:.1f} MB")
```

### **Example 3: File Operations**
```python
# Safe directory creation
output_dir = "/freqtrade/user_data/trening2/outputs/models/BTCUSDT/"
success = safe_create_directory(output_dir, verbose=True)

# Cleanup temporary files
temp_dir = "/freqtrade/user_data/trening2/temp/"
cleaned = cleanup_temp_files(temp_dir, "temp_*.npy", verbose=True)

# Check disk space
disk_info = check_disk_space(output_dir, required_gb=5.0)
if disk_info['sufficient_space']:
    print("Sufficient disk space available")
```

---

**🎯 KLUCZOWE ZALETY UTILS V3:**
- ✅ **Comprehensive Validation** - OHLCV and feature data quality checks
- ✅ **Memory Monitoring** - real-time tracking with psutil integration
- ✅ **Graceful Fallback** - works even when optional dependencies missing
- ✅ **File Safety** - safe directory and file operations
- ✅ **Time Series Aware** - chronological splitting utilities
- ✅ **Memory Management** - cleanup and garbage collection utilities
- ✅ **Production Ready** - error handling and verbose reporting

**📈 NEXT:** [09_Algorytmy_i_integracje.md](./09_Algorytmy_i_integracje.md) - Detailed algorithms and integration patterns 