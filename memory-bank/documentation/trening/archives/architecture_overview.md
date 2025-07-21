# Architektura Dwuokiennego Systemu Trenowania

*Wersja: 2.0.0 | Data: 24 maja 2025*

## 📋 Przegląd Architektury

### Główne Komponenty
```
user_data/training/
├── config/                          ← Centralna konfiguracja
├── core/                            ← Główne komponenty
│   ├── data_loaders/               ← Ładowanie i przetwarzanie danych
│   ├── sequence_builders/          ← Tworzenie sekwencji ML
│   └── models/                     ← Modele LSTM i ewaluacja
├── scripts/                        ← Skrypty wykonawcze
├── outputs/                        ← Wyniki treningu
└── test_implementation.py         ← Testy jednostkowe
```

## 🔄 Przepływ Danych

### 1. Data Pipeline
```
.feather files → EnhancedFeatherLoader → Features DataFrame → DualWindowSequenceBuilder → Training Sequences
     ↓                    ↓                      ↓                       ↓                        ↓
Raw OHLCV data    Smart buffering      8 technical features    Temporal separation     (X, y) arrays
(timestamps)      (33 days buffer)     (price/volume ratios)   (historical/future)     (60, 8) shapes
```

### 2. Training Pipeline
```
Training Sequences → Class Balancing → LSTM Model → Callbacks → Best Model
       ↓                    ↓             ↓           ↓            ↓
(n_samples,60,8)      Weighted classes   134K params  Monitoring   .keras file
(n_samples,)          {0:14.7,1:0.4,2:14.4}  3 layers    Early stop   + metadata
```

### 3. Evaluation Pipeline
```
Best Model → Validation Data → Predictions → Trading Metrics → Artifacts
    ↓              ↓              ↓             ↓               ↓
.keras file    (11509,60,8)    [0,1,2] classes  F1 scores    JSON reports
134K params    validation set   softmax output  Precision/Recall  + config
```

## 🧩 Szczegółowa Architektura Komponentów

### TrainingConfig (Centralna Konfiguracja)
```python
@dataclass
class TrainingConfig:
    # === TEMPORAL WINDOWS ===
    WINDOW_SIZE: int = 60          # Historical window (model input)
    FUTURE_WINDOW: int = 60        # Future window (label verification)
    
    # === LABELING ===  
    LONG_TP_PCT: float = 0.007     # Take Profit threshold
    LONG_SL_PCT: float = 0.007     # Stop Loss threshold
    
    # === FEATURES ===
    MA_FAST_PERIODS: int = 1440    # 24h moving average
    MA_SLOW_PERIODS: int = 43200   # 30d moving average
    FEATURE_COLUMNS: List[str]     # 8 technical features
    
    # === MODEL ===
    LSTM_UNITS: List[int] = [128, 64, 32]
    DENSE_UNITS: List[int] = [32, 16]
    DROPOUT_RATE: float = 0.2
    
    # === TRAINING ===
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0005

# Funkcjonalności:
✅ Automatyczna walidacja parametrów
✅ Obliczanie wymaganych bufferów
✅ Zapis/odczyt do JSON
✅ Integracja z Freqtrade config
```

### EnhancedFeatherLoader (Inteligentne Ładowanie)
```python
class EnhancedFeatherLoader:
    """
    KLUCZOWE INNOWACJE:
    - Automatyczne obliczanie buffera (33 dni)
    - Timezone-aware data handling
    - Smart feature engineering
    - Data validation
    """
    
    def load_training_data(pair, start_date, end_date):
        # 1. BUFFER CALCULATION
        ma_buffer = 30 days          # For MA43200 (30-day average)
        window_buffer = 1 day        # For 60+60 minute windows  
        safety_buffer = 2 days       # Safety margin
        total_buffer = 33 days       # Total required
        
        # 2. EXTENDED DATE RANGE
        extended_start = start_date - 33 days
        extended_end = end_date + 1 day
        
        # 3. LOAD RAW DATA
        raw_df = load_feather_files(pair, extended_start, extended_end)
        
        # 4. COMPUTE FEATURES
        features_df = compute_8_features(raw_df)
        
        return features_df

# Feature Engineering (8 cech):
features = [
    'high_change',      # (high - open) / open
    'low_change',       # (low - open) / open  
    'close_change',     # (close - open) / open
    'volume_change',    # volume.pct_change()
    'price_to_ma1440',  # close / MA(close, 1440)
    'price_to_ma43200', # close / MA(close, 43200)
    'volume_to_ma1440', # volume / MA(volume, 1440)
    'volume_to_ma43200' # volume / MA(volume, 43200)
]
```

### DualWindowSequenceBuilder (Separacja Czasowa)
```python
class DualWindowSequenceBuilder:
    """
    KLUCZOWA INNOWACJA: Eliminacja data leakage
    
    Timeline dla świecy i:
    [i-60:i] ← Historical Window (model input)
    [i] ← Prediction Point (decision moment)
    [i+1:i+61] ← Future Window (label verification)
    """
    
    def create_training_sequences(df):
        X_sequences = []  # Model inputs
        y_labels = []     # Verified labels
        
        for i in range(WINDOW_SIZE, len(df) - FUTURE_WINDOW):
            # 1. HISTORICAL FEATURES (model sees this)
            historical_window = df.iloc[i-60:i][FEATURE_COLUMNS]
            # Shape: (60, 8) - 60 candles × 8 features
            
            # 2. CURRENT POINT (decision moment)
            current_price = df.iloc[i]['close']
            timestamp = df.iloc[i].name
            
            # 3. FUTURE VERIFICATION (model doesn't see this)
            future_window = df.iloc[i+1:i+61]
            label = verify_tp_sl_signal(current_price, future_window)
            
            X_sequences.append(historical_window.values)
            y_labels.append(label)
        
        return np.array(X_sequences), np.array(y_labels)
    
    def verify_tp_sl_signal(base_price, future_candles):
        """
        Hybrydowa klasyfikacja na podstawie TP/SL:
        
        LONG signal (2): TP hit & SL not hit
        SHORT signal (0): TP hit & SL not hit  
        HOLD signal (1): Everything else
        """
        long_tp = base_price * (1 + LONG_TP_PCT)    # +0.7%
        long_sl = base_price * (1 - LONG_SL_PCT)    # -0.7%
        short_tp = base_price * (1 - SHORT_TP_PCT)  # -0.7%
        short_sl = base_price * (1 + SHORT_SL_PCT)  # +0.7%
        
        # Check future candles
        long_tp_hit = (future_candles['high'] >= long_tp).any()
        long_sl_hit = (future_candles['low'] <= long_sl).any()
        short_tp_hit = (future_candles['low'] <= short_tp).any()
        short_sl_hit = (future_candles['high'] >= short_sl).any()
        
        # Hybrid classification
        if long_tp_hit and not long_sl_hit and not (short_tp_hit and not short_sl_hit):
            return 2  # LONG
        elif short_tp_hit and not short_sl_hit and not (long_tp_hit and not long_sl_hit):
            return 0  # SHORT
        else:
            return 1  # HOLD
```

### DualWindowLSTM (Model Architecture)
```python
def build_dual_window_lstm_model(config):
    """
    ARCHITECTURE OVERVIEW:
    
    Input Layer: (None, 60, 8)
         ↓
    LSTM(128) + Dropout(0.2) + RecurrentDropout(0.2) + L2(1e-4)
         ↓
    BatchNormalization()
         ↓
    LSTM(64) + Dropout(0.2) + RecurrentDropout(0.2) + L2(1e-4)
         ↓
    BatchNormalization()
         ↓
    LSTM(32) + Dropout(0.2) + RecurrentDropout(0.2) + L2(1e-4)
         ↓
    BatchNormalization()
         ↓
    Dense(32) + ReLU + Dropout(0.3) + L2(1e-4)
         ↓
    Dense(16) + ReLU + Dropout(0.3) + L2(1e-4)
         ↓
    Dense(3) + Softmax → [SHORT, HOLD, LONG]
    
    PARAMETERS:
    - Total: 134,499
    - Trainable: 134,051
    - Optimizer: RMSprop(lr=0.0005, decay=1e-6)
    - Loss: sparse_categorical_crossentropy
    """
    
    # Regularization Strategy:
    ✅ Dropout: Prevents overfitting in dense layers
    ✅ Recurrent Dropout: Prevents overfitting in LSTM
    ✅ L2 Regularization: Weight decay for all layers
    ✅ BatchNormalization: Stabilizes training
    ✅ Early Stopping: Prevents overtraining
    ✅ ReduceLROnPlateau: Adaptive learning rate
```

## 🎯 Training Pipeline (Orkiestracja)

### DualWindowTrainingPipeline
```python
class DualWindowTrainingPipeline:
    """
    COMPLETE ORCHESTRATION:
    1. Data Loading with buffers
    2. Sequence Building with temporal separation  
    3. Class Balancing for imbalanced data
    4. Model Training with callbacks
    5. Evaluation with trading metrics
    6. Artifact Management
    """
    
    def run_training(pair, start_date, end_date):
        # 1. LOAD DATA (with 33-day buffer)
        df = enhanced_loader.load_training_data(pair, start_date, end_date)
        
        # 2. CREATE SEQUENCES (temporal separation)
        sequences = sequence_builder.create_training_sequences(df)
        X, y = sequences['X'], sequences['y']  # (n_samples, 60, 8), (n_samples,)
        
        # 3. TIME-AWARE SPLIT (chronological)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 4. CLASS BALANCING (for imbalanced data)
        class_weights = calculate_balanced_weights(y_train)
        # Example: {0: 14.7, 1: 0.4, 2: 14.4} for SHORT/HOLD/LONG
        
        # 5. BUILD MODEL
        model = build_dual_window_lstm_model(config)
        
        # 6. SETUP CALLBACKS
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=8),
            ModelCheckpoint(save_best_only=True)
        ]
        
        # 7. TRAIN MODEL
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # 8. EVALUATE
        evaluation = ModelEvaluator(config).evaluate_model(model, X_val, y_val)
        
        # 9. SAVE ARTIFACTS
        save_model_artifacts(model, config, evaluation, artifacts_dir)
        
        return {
            'model': model,
            'history': history.history,
            'evaluation': evaluation,
            'artifacts_dir': artifacts_dir
        }
```

## 📊 Advanced Evaluation System

### ModelEvaluator (Trading-Focused Metrics)
```python
class ModelEvaluator:
    """
    TRADING-SPECIFIC EVALUATION:
    
    Standard ML Metrics:
    - Accuracy, Precision, Recall, F1
    - Confusion Matrix
    - Classification Report
    
    Trading-Specific Metrics:
    - SHORT Precision/Recall/F1
    - LONG Precision/Recall/F1  
    - Trading Signals Average F1 ⭐ (KLUCZOWA METRYKA)
    - Signal Distribution Analysis
    """
    
    def evaluate_model(model, X_test, y_test):
        # 1. BASIC PREDICTIONS
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 2. STANDARD METRICS
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # 3. TRADING METRICS (key innovation)
        short_tp = conf_matrix[0, 0]  # True positives for SHORT
        short_fp = conf_matrix[1, 0] + conf_matrix[2, 0]  # False positives
        short_fn = conf_matrix[0, 1] + conf_matrix[0, 2]  # False negatives
        
        short_precision = short_tp / (short_tp + short_fp) if (short_tp + short_fp) > 0 else 0
        short_recall = short_tp / (short_tp + short_fn) if (short_tp + short_fn) > 0 else 0
        short_f1 = 2 * (short_precision * short_recall) / (short_precision + short_recall) if (short_precision + short_recall) > 0 else 0
        
        # Same for LONG...
        
        # 4. TRADING SIGNALS AVERAGE F1 ⭐
        trading_f1_avg = (short_f1 + long_f1) / 2
        
        return {
            'test_accuracy': accuracy,
            'trading_metrics': {
                'trading_f1_avg': trading_f1_avg,  # KLUCZOWA METRYKA
                'short_precision': short_precision,
                'short_recall': short_recall,
                'long_precision': long_precision,
                'long_recall': long_recall
            }
        }
```

## 🔧 Advanced Features

### Class Balancing Strategy
```python
def calculate_class_weights(y):
    """
    ADVANCED BALANCING for severely imbalanced data:
    
    1. Standard sklearn balanced weights
    2. Additional boosting for minority classes (<10%)
    3. Extreme imbalance detection (>90% one class)
    """
    
    # Standard balanced weights
    weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    
    # Detect extreme imbalance
    class_ratios = [np.sum(y == cls) / len(y) for cls in unique_classes]
    max_ratio = max(class_ratios)
    
    if max_ratio > 0.9:  # One class dominates >90%
        for cls, ratio in enumerate(class_ratios):
            if ratio < 0.1:  # Minority class <10%
                weights[cls] *= 2.0  # Double the weight
    
    return {cls: weight for cls, weight in enumerate(weights)}

# Example output:
# {0: 14.7, 1: 0.4, 2: 14.4}  for SHORT/HOLD/LONG
# HOLD gets low weight (dominant class)
# SHORT/LONG get high weights (minority classes)
```

### Timezone-Aware Data Handling
```python
def handle_timezone_issues(df, start_date, end_date):
    """
    ROBUST TIMEZONE HANDLING:
    
    Problem: .feather files may have timezone-aware datetime
    Solution: Convert all to timezone-naive for consistent comparison
    """
    
    # Convert index to timezone-naive if needed
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    
    # Ensure comparison dates are timezone-naive
    start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
    end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
    
    # Safe filtering
    mask = (df.index >= start_naive) & (df.index <= end_naive)
    return df[mask]
```

## 🎭 Deployment Architecture

### Model Artifacts Structure
```
outputs/models/training_BTC_USDT_YYYYMMDD_HHMMSS/
├── dual_window_lstm_model.keras     ← TensorFlow model (134K params)
├── training_config.json             ← Complete configuration
├── evaluation_results.json          ← All metrics and confusion matrix
└── model_metadata.json              ← Model info and timestamps

# + Checkpoint model:
outputs/models/best_model_BTC_USDT_YYYYMMDD_YYYYMMDD.keras
```

### Production Integration
```python
# Loading for prediction:
model = tf.keras.models.load_model('path/to/model.keras')
config = TrainingConfig.from_config_file('path/to/config.json')

# Prepare data:
sequence_builder = DualWindowSequenceBuilder(config)
sequences = sequence_builder.create_prediction_sequences(current_df)

# Predict:
predictions = model.predict(sequences['X'])
predicted_classes = np.argmax(predictions, axis=1)

# Interpret: 0=SHORT, 1=HOLD, 2=LONG
```

---

## 🎯 Key Architectural Advantages

### 1. **Temporal Separation** 
✅ Eliminates data leakage completely  
✅ Realistic trading simulation  
✅ Model never sees future data

### 2. **Modular Design**
✅ Each component has single responsibility  
✅ Easy to test and maintain  
✅ Configurable and extensible

### 3. **Production-Ready**
✅ Comprehensive error handling  
✅ Automatic data validation  
✅ Complete artifact management

### 4. **Trading-Focused**
✅ TP/SL based labeling  
✅ Trading-specific metrics  
✅ Class balancing for imbalanced signals

### 5. **Robust Data Pipeline**
✅ Automatic buffer calculation  
✅ Timezone-aware handling  
✅ Smart feature engineering

---

*📐 Architecture Status: ✅ Production-ready  
🔄 Last Updated: 24 maja 2025  
📊 Performance: Trading F1 0.171 (baseline)* 