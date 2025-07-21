"""
🛠️ UTILS V2 - MEMORY EFFICIENT MODULE
Helper functions and utilities for memory-efficient training system

Functions:
- Data validation and quality checks
- Memory usage monitoring (optional psutil)
- File I/O operations
- Chronological data splitting
- OHLCV data loading
- Directory management
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import gc

# Add TensorFlow import for Focal Loss
try:
    import tensorflow.keras.backend as K
    import tensorflow as tf
except ImportError:
    # Handle cases where TensorFlow is not installed
    K = None
    tf = None

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - memory monitoring will be limited")


def validate_data_quality(df: pd.DataFrame, pair: str) -> bool:
    """
    🔍 VALIDATE DATA QUALITY
    Sprawdza kompletność i jakość danych OHLCV
    
    Args:
        df: DataFrame z danymi OHLCV
        pair: Nazwa pary (do logów)
        
    Returns:
        bool: True jeśli dane są poprawne
    """
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    print(f"🔍 Validating data quality for {pair}...")
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    
    # Check data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"❌ Column {col} is not numeric")
            return False
    
    # Check for NaN values
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        print(f"❌ NaN values found:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"   {col}: {count} NaN values")
        return False
    
    # Check for negative values in price/volume
    for col in numeric_columns:
        if (df[col] < 0).any():
            print(f"❌ Negative values found in {col}")
            return False
    
    # Check OHLC logic
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    
    if invalid_ohlc > 0:
        print(f"❌ Invalid OHLC logic in {invalid_ohlc} rows")
        return False
    
    # Check chronological order
    if not df['timestamp'].is_monotonic_increasing:
        print(f"❌ Data is not in chronological order")
        return False
    
    print(f"✅ Data quality validation passed:")
    print(f"   📊 Rows: {len(df):,}")
    print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   ⏱️ Time span: {df['timestamp'].max() - df['timestamp'].min()}")
    
    return True


def calculate_memory_usage(data_shape: Tuple[int, ...], dtype=np.float32) -> Dict[str, float]:
    """
    💾 CALCULATE MEMORY USAGE
    Oblicza wymagania pamięciowe dla danych o podanym kształcie
    
    Args:
        data_shape: Kształt danych (np. sequences, timesteps, features)
        dtype: Typ danych (domyślnie float32)
        
    Returns:
        Dict z wynikami w różnych jednostkach
    """
    # Calculate total elements
    total_elements = np.prod(data_shape)
    
    # Calculate bytes based on dtype
    if dtype == np.float32:
        bytes_per_element = 4
    elif dtype == np.float64:
        bytes_per_element = 8
    elif dtype == np.int8:
        bytes_per_element = 1
    elif dtype == np.int32:
        bytes_per_element = 4
    else:
        bytes_per_element = 4  # Default
    
    total_bytes = total_elements * bytes_per_element
    
    return {
        'total_elements': int(total_elements),
        'bytes': int(total_bytes),
        'kb': total_bytes / 1024,
        'mb': total_bytes / (1024 ** 2),
        'gb': total_bytes / (1024 ** 3),
        'dtype': str(dtype),
        'shape': data_shape
    }


def get_system_memory_info() -> Dict[str, float]:
    """
    📊 GET SYSTEM MEMORY INFO
    Pobiera informacje o pamięci systemowej (jeśli psutil dostępne)
    
    Returns:
        Dict z informacjami o pamięci
    """
    if not PSUTIL_AVAILABLE:
        return {
            'total_gb': 32.0,      # Default assumption
            'available_gb': 20.0,  # Conservative estimate
            'used_gb': 12.0,       # Conservative estimate
            'percent_used': 37.5,  # Conservative estimate
            'free_gb': 20.0,       # Conservative estimate
            'psutil_available': False
        }
    
    try:
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'used_gb': memory.used / (1024 ** 3),
            'percent_used': memory.percent,
            'free_gb': memory.free / (1024 ** 3),
            'psutil_available': True
        }
    except Exception as e:
        print(f"⚠️ Could not get memory info: {e}")
        return {
            'total_gb': 32.0,
            'available_gb': 20.0,
            'used_gb': 12.0,
            'percent_used': 37.5,
            'free_gb': 20.0,
            'psutil_available': False
        }


def monitor_memory_usage(process_name: str = "current") -> None:
    """
    📈 MONITOR MEMORY USAGE
    Wyświetla bieżące zużycie pamięci (jeśli psutil dostępne)
    
    Args:
        process_name: Nazwa procesu do monitorowania
    """
    if not PSUTIL_AVAILABLE:
        print(f"💾 Memory Usage Report (limited - psutil not available):")
        print(f"   Process: Unknown")
        print(f"   System: Estimated 32GB total, ~12GB used")
        print(f"   Available: Estimated ~20GB")
        return
    
    try:
        # Current process memory
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # System memory
        system_info = get_system_memory_info()
        
        print(f"💾 Memory Usage Report:")
        print(f"   Process: {process_memory_mb:.1f}MB")
        print(f"   System: {system_info['used_gb']:.1f}GB / {system_info['total_gb']:.1f}GB ({system_info['percent_used']:.1f}%)")
        print(f"   Available: {system_info['available_gb']:.1f}GB")
        
        # Memory pressure warning
        if system_info['percent_used'] > 80:
            print(f"⚠️ HIGH MEMORY USAGE: {system_info['percent_used']:.1f}%")
        elif system_info['percent_used'] > 90:
            print(f"🚨 CRITICAL MEMORY USAGE: {system_info['percent_used']:.1f}%")
            
    except Exception as e:
        print(f"⚠️ Could not monitor memory: {e}")


def setup_output_directories(base_path: str, subdirs: List[str]) -> None:
    """
    📁 SETUP OUTPUT DIRECTORIES
    Tworzy strukturę katalogów wyjściowych
    
    Args:
        base_path: Ścieżka bazowa
        subdirs: Lista podkatalogów do utworzenia
    """
    directories_created = []
    
    # Create base directory
    os.makedirs(base_path, exist_ok=True)
    directories_created.append(base_path)
    
    # Create subdirectories
    for subdir in subdirs:
        full_path = os.path.join(base_path, subdir)
        os.makedirs(full_path, exist_ok=True)
        directories_created.append(full_path)
    
    print(f"📁 Created {len(directories_created)} directories:")
    for directory in directories_created:
        print(f"   ✅ {directory}")


def create_focal_loss(alpha: list = [0.75, 0.25, 0.75], gamma: float = 2.0) -> callable:
    """
    🔥 CREATE FOCAL LOSS FUNCTION - FIXED IMPLEMENTATION
    
    Creates a focal loss function, a powerful tool for handling class imbalance.
    It dynamically scales the cross-entropy loss, down-weighting easy-to-classify
    examples and focusing on hard-to-classify ones.

    Args:
        alpha (list): A list of balancing factors for each class [SHORT, HOLD, LONG].
                      Values > 0.33 emphasize the class, < 0.33 de-emphasize it.
                      Default [0.75, 0.25, 0.75] emphasizes SHORT/LONG over HOLD.
        gamma (float): The focusing parameter. Higher values give more weight to
                       hard-to-classify examples. A value of 0 makes it equivalent
                       to standard categorical cross-entropy.

    Returns:
        A callable Keras loss function.
    """
    if K is None or tf is None:
        raise ImportError("TensorFlow and Keras backend are required for Focal Loss.")

    def focal_loss(y_true, y_pred):
        """
        FIXED Focal loss calculation.
        y_true: A one-hot-encoded tensor of true labels.
        y_pred: A tensor of predicted probabilities.
        """
        # Convert alpha to a tensor
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

        # Clip predictions to prevent log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Calculate cross-entropy for each class
        cross_entropy = -y_true * K.log(y_pred)
        
        # FIXED: Apply alpha weighting correctly - multiply by corresponding alpha for true class
        # alpha_tensor has shape [3], y_true has shape [batch, 3]  
        # We need to select the alpha for the true class
        alpha_weights = K.sum(y_true * alpha_tensor, axis=-1, keepdims=True)
        
        # Calculate focal term: (1 - p_t)^gamma where p_t is prediction for true class
        true_class_pred = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_term = K.pow(1 - true_class_pred, gamma)
        
        # Calculate focal loss: -alpha_t * (1-p_t)^gamma * log(p_t)
        focal_loss_value = alpha_weights * focal_term * K.sum(cross_entropy, axis=-1, keepdims=True)
        
        # Return scalar loss per sample
        return K.squeeze(focal_loss_value, axis=-1)

    return focal_loss


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    📅 CHRONOLOGICAL SPLIT
    Dzieli dane chronologicznie (bez data leakage)
    
    Args:
        df: DataFrame z danymi posortowanymi chronologicznie
        train_ratio: Proporcja danych treningowych (0.8 = 80%)
        
    Returns:
        Tuple (train_df, val_df)
    """
    if not df['timestamp'].is_monotonic_increasing:
        print("⚠️ Warning: Data is not sorted chronologically, sorting...")
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"📅 Chronological split completed:")
    print(f"   📊 Train: {len(train_df):,} rows ({train_ratio:.0%})")
    print(f"   📊 Val: {len(val_df):,} rows ({1-train_ratio:.0%})")
    print(f"   📅 Train range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"   📅 Val range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    
    return train_df, val_df


def load_ohlcv_data(pair: str, start_date: str, end_date: str, data_path: str = "historic") -> pd.DataFrame:
    """
    📂 LOAD OHLCV DATA FROM VALIDATED FILES
    Ładuje przetworzone dane OHLCV z plików z modułu walidacji
    
    Args:
        pair: Para kryptowalutowa (np. 'BTC_USDT')
        start_date: Data rozpoczęcia (YYYY-MM-DD)
        end_date: Data zakończenia (YYYY-MM-DD)
        data_path: Ścieżka do danych (domyślnie 'historic')
        
    Returns:
        DataFrame z danymi OHLCV
    """
    print(f"📂 Loading OHLCV data for {pair}...")
    print(f"   📅 Date range: {start_date} to {end_date}")
    
    # Convert dates for filtering
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Path to validated data file
    validated_filename = f"{pair}_1m_validated.feather"
    file_path = os.path.join(data_path, validated_filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data path not found: {file_path}")
    
    print(f"   📁 Loading validated file: {file_path}")
    
    try:
        # Load validated data
        df = pd.read_feather(file_path)
        print(f"   ✅ Loaded {len(df):,} rows from validated file")
        
        # Convert timestamp if needed
        if 'timestamp' not in df.columns and 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("No timestamp or date column found in data")
        
        # Filter by exact date range
        df_filtered = df[
            (df['timestamp'] >= start_dt) & 
            (df['timestamp'] <= end_dt)
        ].reset_index(drop=True)
        
        print(f"📊 Data loading completed:")
        print(f"   Total rows: {len(df_filtered):,}")
        print(f"   Date range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
        print(f"   Columns: {list(df_filtered.columns)}")
        
        return df_filtered
        
    except Exception as e:
        print(f"   ❌ Error loading {file_path}: {e}")
        raise


def safe_memory_cleanup():
    """
    🧹 SAFE MEMORY CLEANUP
    Bezpieczne czyszczenie pamięci
    """
    print("🧹 Performing memory cleanup...")
    
    # Get memory before cleanup (if available)
    if PSUTIL_AVAILABLE:
        memory_before = get_system_memory_info()
    
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory after cleanup (if available)
    if PSUTIL_AVAILABLE:
        memory_after = get_system_memory_info()
        memory_freed_mb = (memory_before['used_gb'] - memory_after['used_gb']) * 1024
        
        print(f"   🗑️ Garbage collected: {collected} objects")
        print(f"   💾 Memory freed: {memory_freed_mb:.1f}MB")
        print(f"   📊 Memory usage: {memory_after['percent_used']:.1f}%")
    else:
        print(f"   🗑️ Garbage collected: {collected} objects")
        print(f"   💾 Memory monitoring limited (psutil not available)")


def estimate_sequence_memory(total_sequences: int, window_size: int, features: int) -> Dict[str, float]:
    """
    🧮 ESTIMATE SEQUENCE MEMORY
    Oszacowuje wymagania pamięciowe dla sekwencji
    
    Args:
        total_sequences: Liczba sekwencji
        window_size: Rozmiar okna czasowego
        features: Liczba cech
        
    Returns:
        Dict z oszacowaniami pamięci
    """
    # Calculate X array (sequences, timesteps, features)
    X_shape = (total_sequences, window_size, features)
    X_memory = calculate_memory_usage(X_shape, np.float32)
    
    # Calculate y array (sequences,)
    y_shape = (total_sequences,)
    y_memory = calculate_memory_usage(y_shape, np.int8)
    
    # Total memory
    total_gb = X_memory['gb'] + y_memory['gb']
    
    return {
        'total_sequences': total_sequences,
        'X_memory_gb': X_memory['gb'],
        'y_memory_gb': y_memory['gb'],
        'total_memory_gb': total_gb,
        'X_shape': X_shape,
        'y_shape': y_shape,
        'feasible': total_gb < 30.0  # Assume 30GB limit
    }


if __name__ == "__main__":
    # Test utilities
    print("🧪 TESTING UTILITIES")
    print(f"💻 psutil available: {PSUTIL_AVAILABLE}")
    
    print("\n💾 System Memory Info:")
    memory_info = get_system_memory_info()
    for key, value in memory_info.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n📊 Memory Usage Monitoring:")
    monitor_memory_usage()
    
    print("\n🧮 Memory Estimation Example:")
    estimation = estimate_sequence_memory(100000, 180, 8)
    for key, value in estimation.items():
        print(f"   {key}: {value}")
    
    print("\n🧹 Memory Cleanup Test:")
    safe_memory_cleanup()
    
    print("\n✅ Utilities test completed!") 