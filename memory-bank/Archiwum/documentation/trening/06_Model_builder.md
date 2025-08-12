# 📖 MODEL BUILDER MODUŁU TRENUJĄCEGO V3 (model_builder.py)

## 🎯 PRZEGLĄD MODEL BUILDER

**model_builder.py** (503 linie) to zaawansowany moduł budowy architektury LSTM zoptymalizowanej dla crypto trading. Zawiera production-ready callbacks, GPU optimization z automatic CPU fallback oraz TensorFlow 2.10+ compatibility.

### ✨ **Kluczowe Innowacje V3 Model Builder**
- ✅ **Crypto-Optimized LSTM Architecture** - progressive reduction design
- ✅ **Production Callbacks** - checkpointing, early stopping, LR reduction
- ✅ **GPU Optimization** - memory growth control, automatic CPU fallback
- ✅ **TensorFlow 2.10+ Compatibility** - .keras SavedModel format
- ✅ **Memory Estimation** - predict training memory requirements
- ✅ **Model Persistence** - comprehensive model saving with metadata

## 🧠 **LSTM ARCHITECTURE DESIGN**

### **Crypto-Optimized Architecture**
```python
def build_model(compile_model: bool = True) -> Model:
    """
    🧠 BUILD CRYPTO-OPTIMIZED LSTM MODEL
    
    Architecture designed specifically for cryptocurrency trading:
    - Progressive LSTM reduction: 128 → 64 → 32
    - Dense feature extraction: 32 → 16
    - Dropout regularization: 0.3
    - Softmax classification: 3 classes (SHORT, HOLD, LONG)
    """
    
    # INPUT LAYER - Time sequences
    inputs = layers.Input(
        shape=(config.SEQUENCE_LENGTH, 8),  # (120, 8) - 120 timesteps, 8 features
        name='lstm_input'
    )
    
    # LSTM STACK - Progressive reduction
    x = inputs
    for i, units in enumerate(config.LSTM_UNITS):  # [128, 64, 32]
        return_sequences = (i < len(config.LSTM_UNITS) - 1)  # Last layer: False
        
        x = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=config.DROPOUT_RATE,
            recurrent_dropout=config.DROPOUT_RATE,
            name=f'lstm_layer_{i+1}_{units}units'
        )(x)
        
        print(f"   🧠 LSTM Layer {i+1}: {units} units (return_sequences={return_sequences})")
    
    # DENSE STACK - Feature extraction
    for i, units in enumerate(config.DENSE_UNITS):  # [32, 16]
        x = layers.Dense(
            units=units,
            activation='relu',
            name=f'dense_layer_{i+1}_{units}units'
        )(x)
        
        x = layers.Dropout(
            rate=config.DROPOUT_RATE,
            name=f'dropout_{i+1}'
        )(x)
        
        print(f"   📊 Dense Layer {i+1}: {units} units + dropout ({config.DROPOUT_RATE})")
    
    # OUTPUT LAYER - Softmax classification
    outputs = layers.Dense(
        units=3,
        activation='softmax',
        name='trading_decisions'
    )(x)
    
    print(f"   🎯 Output Layer: 3 units (SHORT, HOLD, LONG) + softmax")
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CryptoTradingLSTM_V3')
    
    # Compile model
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',  # For one-hot encoded labels
            metrics=['accuracy', 'categorical_accuracy']
        )
    
    return model
```

### **Architecture Breakdown**
```
INPUT: (batch_size, 120, 8)
    ↓
LSTM-128 → return_sequences=True  → (batch_size, 120, 128)
    ↓
LSTM-64  → return_sequences=True  → (batch_size, 120, 64)
    ↓
LSTM-32  → return_sequences=False → (batch_size, 32)
    ↓
Dense-32 + ReLU + Dropout(0.3)   → (batch_size, 32)
    ↓
Dense-16 + ReLU + Dropout(0.3)   → (batch_size, 16)
    ↓
Dense-3  + Softmax               → (batch_size, 3) [SHORT, HOLD, LONG]
```

### **Architecture Justification**
```python
# 🎯 DESIGN DECISIONS EXPLAINED

# 1. PROGRESSIVE LSTM REDUCTION (128 → 64 → 32)
# - Captures multi-scale temporal patterns
# - Reduces overfitting through parameter reduction
# - Optimized for crypto market volatility

# 2. RETURN_SEQUENCES Strategy
# - First/middle layers: return_sequences=True (full temporal info)
# - Last layer: return_sequences=False (final decision vector)

# 3. DENSE LAYERS (32 → 16)
# - Extract high-level features from LSTM output
# - Compress information for final classification
# - Prevent overfitting with progressive reduction

# 4. DROPOUT REGULARIZATION (0.3)
# - Applied to both LSTM and Dense layers
# - Prevents overfitting on training data
# - Improves generalization to unseen market conditions

# 5. SOFTMAX OUTPUT
# - Probability distribution over 3 trading actions
# - Enables confidence-based decision making
# - Compatible with categorical_crossentropy loss
```

## 🏗️ **MODEL BUILDING CLASS**

### **DualWindowLSTMBuilder Class**
```python
class DualWindowLSTMBuilder:
    """
    🏗️ PRODUCTION-READY LSTM MODEL BUILDER
    
    Features:
    - Crypto-optimized architecture
    - GPU optimization with CPU fallback  
    - Production callbacks
    - Memory estimation
    - Model persistence
    """
    
    def __init__(self):
        self.model = None
        self.training_history = None
        self.gpu_available = self.setup_gpu_memory()
```

### **GPU Optimization System**
```python
def setup_gpu_memory(self) -> bool:
    """
    🖥️ SETUP GPU OPTIMIZATION
    
    Features:
    - Memory growth control (prevent allocation of all GPU memory)
    - Automatic CPU fallback if GPU unavailable
    - GPU detection and configuration
    """
    try:
        import tensorflow as tf
        
        # Get list of available GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"   🖥️ GPU Configuration:")
                print(f"      Available GPUs: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"      GPU {i}: {gpu.name}")
                print(f"      Memory growth: ENABLED")
                
                # Test GPU availability
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0]])
                    result = tf.matmul(test_tensor, test_tensor)
                
                print(f"   ✅ GPU test successful")
                return True
                
            except RuntimeError as e:
                print(f"   ⚠️ GPU setup failed: {e}")
                print(f"   🔄 Falling back to CPU")
                return False
        else:
            print(f"   💻 No GPU detected - using CPU")
            return False
            
    except ImportError:
        print(f"   ⚠️ TensorFlow GPU support not available")
        return False
```

### **Memory Estimation System**
```python
def estimate_model_memory(self, batch_size: int = 256) -> Dict[str, float]:
    """
    💾 ESTIMATE MODEL MEMORY REQUIREMENTS
    
    Predicts memory usage for training to prevent OOM errors
    """
    
    # Model parameters estimation
    lstm_params = 0
    for units in config.LSTM_UNITS:
        # LSTM has 4 gates, each with weights and biases
        lstm_params += units * (8 + units + 1) * 4  # Simplified estimation
    
    dense_params = 0
    prev_units = config.LSTM_UNITS[-1]
    for units in config.DENSE_UNITS:
        dense_params += prev_units * units + units  # weights + biases
        prev_units = units
    
    # Output layer
    output_params = config.DENSE_UNITS[-1] * 3 + 3
    
    total_params = lstm_params + dense_params + output_params
    
    # Memory estimations (in MB)
    model_memory = total_params * 4 / 1024 / 1024  # 4 bytes per float32
    
    # Training memory (gradients, optimizer states, activations)
    training_memory = model_memory * 4  # Rough estimate: 4x model size
    
    # Batch memory
    sequence_memory = batch_size * config.SEQUENCE_LENGTH * 8 * 4 / 1024 / 1024
    
    total_estimated = model_memory + training_memory + sequence_memory
    
    memory_info = {
        'model_parameters': total_params,
        'model_memory_mb': model_memory,
        'training_memory_mb': training_memory,
        'batch_memory_mb': sequence_memory,
        'total_estimated_mb': total_estimated,
        'total_estimated_gb': total_estimated / 1024,
        'batch_size': batch_size
    }
    
    print(f"   💾 Memory estimation:")
    print(f"      Model parameters: {total_params:,}")
    print(f"      Model memory: {model_memory:.1f} MB")
    print(f"      Training memory: {training_memory:.1f} MB")
    print(f"      Batch memory: {sequence_memory:.1f} MB")
    print(f"      Total estimated: {total_estimated:.1f} MB ({total_estimated/1024:.2f} GB)")
    
    return memory_info
```

## 📋 **PRODUCTION CALLBACKS**

### **Comprehensive Callback System**
```python
def create_callbacks(self, model_output_dir: str) -> list:
    """
    📋 CREATE PRODUCTION-READY CALLBACKS
    
    Callbacks included:
    - ModelCheckpoint: Save best model during training
    - EarlyStopping: Stop training when validation stops improving
    - ReduceLROnPlateau: Reduce learning rate when stuck
    - TensorBoard: Training visualization and monitoring
    """
    
    callbacks_list = []
    
    # 1. MODEL CHECKPOINT - Save best model
    checkpoint_path = os.path.join(model_output_dir, "best_model_checkpoint.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',          # Monitor validation accuracy
        mode='max',                      # Save when val_accuracy increases
        save_best_only=True,            # Only save if better than previous best
        save_weights_only=False,        # Save full model
        verbose=1,
        save_format='keras'             # TensorFlow 2.10+ format
    )
    callbacks_list.append(checkpoint_callback)
    print(f"   💾 ModelCheckpoint: {checkpoint_path}")
    
    # 2. EARLY STOPPING - Prevent overfitting
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=config.EARLY_STOPPING_PATIENCE,  # Wait N epochs
        verbose=1,
        restore_best_weights=True       # Restore best weights when stopping
    )
    callbacks_list.append(early_stopping_callback)
    print(f"   ⏹️ EarlyStopping: patience={config.EARLY_STOPPING_PATIENCE}")
    
    # 3. LEARNING RATE REDUCTION - Dynamic LR adjustment
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=config.REDUCE_LR_FACTOR,        # Multiply LR by this factor
        patience=config.REDUCE_LR_PATIENCE,    # Wait N epochs before reducing
        min_lr=config.MIN_LEARNING_RATE,       # Don't go below this LR
        verbose=1
    )
    callbacks_list.append(reduce_lr_callback)
    print(f"   📉 ReduceLROnPlateau: factor={config.REDUCE_LR_FACTOR}, patience={config.REDUCE_LR_PATIENCE}")
    
    # 4. TENSORBOARD - Training visualization
    tensorboard_dir = os.path.join(model_output_dir, "logs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,               # Log weight histograms every epoch
        write_graph=True,               # Visualize model graph
        update_freq='epoch',            # Update logs every epoch
        profile_batch=2,                # Profile batch 2 for performance analysis
        embeddings_freq=1               # Log embeddings every epoch
    )
    callbacks_list.append(tensorboard_callback)
    print(f"   📊 TensorBoard: {tensorboard_dir}")
    
    # 5. CUSTOM CALLBACK - Memory monitoring (optional)
    if PSUTIL_AVAILABLE:
        memory_monitor_callback = MemoryMonitorCallback()
        callbacks_list.append(memory_monitor_callback)
        print(f"   💾 MemoryMonitor: enabled")
    
    return callbacks_list
```

### **Custom Memory Monitor Callback**
```python
class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor memory usage during training
    """
    
    def __init__(self):
        super().__init__()
        self.memory_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.memory_history.append({
                'epoch': epoch,
                'memory_mb': memory_mb,
                'val_accuracy': logs.get('val_accuracy', 0),
                'val_loss': logs.get('val_loss', 0)
            })
            
            if epoch % 10 == 0:  # Print every 10 epochs
                print(f"   💾 Memory usage: {memory_mb:.1f} MB")
                
        except ImportError:
            pass  # psutil not available
```

## 💾 **MODEL PERSISTENCE SYSTEM**

### **Advanced Model Saving**
```python
def save_model(self, model: Model, save_path: str, format: str = 'saved_model') -> bool:
    """
    💾 SAVE MODEL WITH COMPREHENSIVE METADATA
    
    Supports multiple formats:
    - 'saved_model': TensorFlow SavedModel format (recommended)
    - 'keras': Keras .keras format (TensorFlow 2.10+)
    - 'h5': Legacy HDF5 format (for compatibility)
    """
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if format == 'saved_model':
            # TensorFlow SavedModel format (production recommended)
            model.save(save_path, save_format='tf')
            print(f"   ✅ Model saved (SavedModel): {save_path}")
            
        elif format == 'keras':
            # Keras .keras format (TensorFlow 2.10+)
            if not save_path.endswith('.keras'):
                save_path += '.keras'
            model.save(save_path, save_format='keras')
            print(f"   ✅ Model saved (Keras): {save_path}")
            
        elif format == 'h5':
            # Legacy HDF5 format
            if not save_path.endswith('.h5'):
                save_path += '.h5'
            model.save(save_path, save_format='h5')
            print(f"   ✅ Model saved (H5): {save_path}")
            
        else:
            raise ValueError(f"Unknown save format: {format}")
        
        # Save model architecture summary
        summary_path = save_path.replace('.keras', '_summary.txt').replace('.h5', '_summary.txt')
        if format == 'saved_model':
            summary_path = os.path.join(save_path, 'model_summary.txt')
            
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print(f"   📄 Model summary saved: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to save model: {e}")
        return False
```

### **Model Loading with Validation**
```python
def load_model(self, model_path: str) -> Model:
    """
    📂 LOAD MODEL WITH VALIDATION
    
    Automatically detects format and loads appropriately
    Validates model architecture matches expected configuration
    """
    
    try:
        print(f"   📂 Loading model from: {model_path}")
        
        # Auto-detect format and load
        if os.path.isdir(model_path):
            # SavedModel format
            model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model loaded (SavedModel format)")
            
        elif model_path.endswith('.keras'):
            # Keras format
            model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model loaded (Keras format)")
            
        elif model_path.endswith('.h5'):
            # HDF5 format
            model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model loaded (H5 format)")
            
        else:
            raise ValueError(f"Unknown model format: {model_path}")
        
        # Validate model architecture
        self._validate_loaded_model(model)
        
        return model
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise
```

### **Model Architecture Validation**
```python
def _validate_loaded_model(self, model: Model) -> None:
    """
    Validate loaded model matches expected architecture
    """
    
    # Check input shape
    expected_input_shape = (None, config.SEQUENCE_LENGTH, 8)
    actual_input_shape = model.input_shape
    
    if actual_input_shape != expected_input_shape:
        print(f"   ⚠️ Input shape mismatch: expected {expected_input_shape}, got {actual_input_shape}")
    
    # Check output shape
    expected_output_shape = (None, 3)
    actual_output_shape = model.output_shape
    
    if actual_output_shape != expected_output_shape:
        print(f"   ⚠️ Output shape mismatch: expected {expected_output_shape}, got {actual_output_shape}")
    
    # Check total parameters (rough validation)
    total_params = model.count_params()
    print(f"   📊 Model parameters: {total_params:,}")
    
    # Print model summary for verification
    print(f"   📋 Model architecture validation:")
    model.summary()
```

## 🎯 **USAGE EXAMPLES**

### **Example 1: Basic Model Building**
```python
# Initialize model builder
model_builder = DualWindowLSTMBuilder()

# Build model
model = model_builder.build_model(compile_model=True)

# Print model summary
model.summary()

# Estimate memory requirements
memory_info = model_builder.estimate_model_memory(batch_size=256)
```

### **Example 2: Training with Callbacks**
```python
# Create production callbacks
callbacks = model_builder.create_callbacks(model_output_dir)

# Train model with callbacks
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=config.EPOCHS,
    callbacks=callbacks,  # Production callbacks applied
    verbose=1
)

# Save final model
model_builder.save_model(model, save_path, format='keras')
```

### **Example 3: Model Loading and Inference**
```python
# Load trained model
loaded_model = model_builder.load_model("path/to/model.keras")

# Make predictions
predictions = loaded_model.predict(sequences)

# Convert to trading decisions
decisions = np.argmax(predictions, axis=1)
# 0=SHORT, 1=HOLD, 2=LONG
```

---

**🎯 KLUCZOWE ZALETY MODEL BUILDER V3:**
- ✅ **Crypto-Optimized Architecture** - designed for trading patterns
- ✅ **Production Callbacks** - comprehensive training monitoring
- ✅ **GPU Optimization** - memory growth control, CPU fallback
- ✅ **Memory Estimation** - prevent OOM errors before training
- ✅ **Multiple Save Formats** - SavedModel, Keras, H5 compatibility
- ✅ **Architecture Validation** - ensure model integrity
- ✅ **TensorBoard Integration** - advanced training visualization

**📈 NEXT:** [07_Trainer.md](./07_Trainer.md) - Main training pipeline and orchestration 