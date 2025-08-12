# 📖 TRAINER MODUŁU TRENUJĄCEGO V3 (trainer.py)

## 🎯 PRZEGLĄD TRAINER

**trainer.py** (724 linie) to główny orchestrator modułu trenującego V3 - **complete training pipeline** który integruje wszystkie komponenty w jeden spójny workflow. Zawiera standalone training process, confusion matrix generation, metadata saving oraz CLI interface.

### ✨ **Kluczowe Zalety V3 Trainer**
- ✅ **Complete Training Pipeline** - full end-to-end workflow
- ✅ **Component Integration** - orchestrates all V3 modules
- ✅ **Standalone Operation** - zero dependencies on validation module
- ✅ **Confusion Matrix Analysis** - post-training model evaluation
- ✅ **CLI Interface** - parameter override support
- ✅ **Error Handling** - comprehensive error messages with solutions

## 🏗️ **TRAINER ARCHITECTURE**

### **StandaloneTrainer Class**
```python
class StandaloneTrainer:
    """
    🎯 STANDALONE TRAINING ORCHESTRATOR V3
    
    Complete training pipeline:
    1. Initialize all components (data_loader, sequence_generator, model_builder)
    2. Load and validate training data
    3. Create memory-efficient generators
    4. Build LSTM model with production callbacks
    5. Execute training with monitoring
    6. Save model, scaler, and metadata
    7. Generate confusion matrix analysis
    """
    
    def __init__(self):
        self.data_loader = None
        self.sequence_generator = None
        self.model_builder = None
        self.model = None
        self.training_history = None
        self.model_output_dir = config.get_model_output_dir()
```

### **Complete Training Workflow**
```python
def run_training(self):
    """
    🚀 MAIN TRAINING PIPELINE
    
    Complete workflow from data loading to model saving
    """
    try:
        print(f"\n🚀 STARTING STANDALONE TRAINING...")
        config.print_config_summary()
        
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
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        raise
```

## 🔧 **COMPONENT INITIALIZATION**

### **Initialize All Components**
```python
def initialize_components(self):
    """
    🔧 INITIALIZE ALL TRAINING COMPONENTS
    
    Sets up:
    - TrainingDataLoader (explicit path loading)
    - MemoryEfficientDataLoader (sequence generation)
    - DualWindowLSTMBuilder (model building)
    """
    print(f"\n🔧 INITIALIZING COMPONENTS...")
    
    # 1. Data Loader - explicit path loading
    self.data_loader = TrainingDataLoader(
        explicit_path=config.TRAINING_DATA_PATH,
        pair=config.PAIR
    )
    print(f"   ✅ TrainingDataLoader initialized")
    
    # 2. Model Builder - LSTM architecture
    self.model_builder = DualWindowLSTMBuilder()
    print(f"   ✅ DualWindowLSTMBuilder initialized")
    
    # 3. Create output directory
    os.makedirs(self.model_output_dir, exist_ok=True)
    print(f"   📁 Output directory: {self.model_output_dir}")
    
    print(f"   🎯 All components initialized successfully!")
```

## 📁 **DATA LOADING & VALIDATION**

### **Complete Data Loading Pipeline**
```python
def load_and_validate_data(self) -> Dict[str, Any]:
    """
    📁 LOAD AND VALIDATE TRAINING DATA
    
    Complete pipeline:
    1. File existence validation (hard fail with guidance)
    2. Data loading and parsing
    3. Parameter compatibility validation
    4. Feature scaling (zero data leakage)
    5. Class balancing (systematic undersampling)
    """
    print(f"\n📁 LOADING AND VALIDATING DATA...")
    
    # Load training data (complete pipeline)
    data_info = self.data_loader.load_training_data()
    
    # Validate data integrity
    features = data_info['features']
    labels = data_info['labels']
    
    print(f"   📊 Data validation:")
    print(f"      Features shape: {features.shape}")
    print(f"      Labels shape: {labels.shape}")
    print(f"      Samples count: {data_info['samples_count']:,}")
    print(f"      Feature scaling: {data_info['scaling_enabled']}")
    print(f"      Class balancing: {data_info['balancing_applied']}")
    
    # Validate data quality
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Features contain NaN or Inf values")
    if np.isnan(labels).any() or np.isinf(labels).any():
        raise ValueError("Labels contain NaN or Inf values")
    
    # Validate shapes
    expected_features_shape = (data_info['samples_count'], 8)
    expected_labels_shape = (data_info['samples_count'], 3)
    
    if features.shape != expected_features_shape:
        raise ValueError(f"Invalid features shape: expected {expected_features_shape}, got {features.shape}")
    if labels.shape != expected_labels_shape:
        raise ValueError(f"Invalid labels shape: expected {expected_labels_shape}, got {labels.shape}")
    
    print(f"   ✅ Data validation passed!")
    return data_info
```

## 🎛️ **GENERATOR CREATION**

### **Memory-Efficient Generator Setup**
```python
def create_generators(self, features: np.ndarray, labels: np.ndarray) -> Tuple[Any, Any]:
    """
    🎛️ CREATE MEMORY-EFFICIENT GENERATORS
    
    Features:
    - Chronological train/validation split
    - Numpy memory views (zero-copy)
    - Generator validation and testing
    """
    print(f"\n🎛️ CREATING GENERATORS...")
    
    # Configure generator parameters
    config_params = {
        'sequence_length': config.SEQUENCE_LENGTH,
        'batch_size': config.BATCH_SIZE,
        'validation_split': config.VALIDATION_SPLIT
    }
    
    # Import sequence generator
    from sequence_generator import create_generators_from_arrays
    
    # Create generators (handles chronological split internally)
    train_gen, val_gen = create_generators_from_arrays(
        features=features,
        labels=labels,
        config_params=config_params
    )
    
    print(f"   ✅ Generators created successfully!")
    print(f"      Training sequences: {len(train_gen):,}")
    print(f"      Validation sequences: {len(val_gen):,}")
    
    return train_gen, val_gen
```

## 🧠 **MODEL BUILDING**

### **LSTM Model Construction**
```python
def build_model(self):
    """
    🧠 BUILD LSTM MODEL
    
    Features:
    - Crypto-optimized architecture
    - Memory estimation
    - GPU optimization with CPU fallback
    """
    print(f"\n🧠 BUILDING MODEL...")
    
    # Estimate memory requirements
    memory_info = self.model_builder.estimate_model_memory(batch_size=config.BATCH_SIZE)
    
    # Build model
    self.model = self.model_builder.build_model(compile_model=True)
    
    # Print model summary
    print(f"\n   📋 MODEL ARCHITECTURE:")
    self.model.summary()
    
    print(f"   ✅ Model built successfully!")
    return self.model
```

## 📋 **CALLBACK SETUP**

### **Production Callbacks Configuration**
```python
def setup_callbacks(self) -> list:
    """
    📋 SETUP PRODUCTION CALLBACKS
    
    Callbacks:
    - ModelCheckpoint: Save best model
    - EarlyStopping: Prevent overfitting
    - ReduceLROnPlateau: Dynamic learning rate
    - TensorBoard: Training visualization
    - MemoryMonitor: Memory usage tracking (optional)
    """
    print(f"\n📋 SETTING UP CALLBACKS...")
    
    callbacks_list = self.model_builder.create_callbacks(self.model_output_dir)
    
    print(f"   ✅ Callbacks configured: {len(callbacks_list)} callbacks")
    return callbacks_list
```

## 🏋️ **MODEL TRAINING**

### **Training Execution**
```python
def train_model(self, train_gen, val_gen, callbacks_list: list):
    """
    🏋️ EXECUTE MODEL TRAINING
    
    Features:
    - Memory-efficient generator feeding
    - Production callback monitoring
    - Training progress tracking
    - Error handling with recovery guidance
    """
    print(f"\n🏋️ STARTING MODEL TRAINING...")
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_gen) // config.BATCH_SIZE
    validation_steps = len(val_gen) // config.BATCH_SIZE
    
    print(f"   📊 Training configuration:")
    print(f"      Epochs: {config.EPOCHS}")
    print(f"      Batch size: {config.BATCH_SIZE}")
    print(f"      Steps per epoch: {steps_per_epoch}")
    print(f"      Validation steps: {validation_steps}")
    print(f"      Callbacks: {len(callbacks_list)}")
    
    try:
        # Execute training
        self.training_history = self.model.fit(
            train_gen,
            epochs=config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1,
            shuffle=False  # Generators handle shuffling internally
        )
        
        print(f"   ✅ Training completed successfully!")
        
        # Print final metrics
        final_loss = self.training_history.history['loss'][-1]
        final_val_loss = self.training_history.history['val_loss'][-1]
        final_accuracy = self.training_history.history['accuracy'][-1]
        final_val_accuracy = self.training_history.history['val_accuracy'][-1]
        
        print(f"   📊 Final metrics:")
        print(f"      Training loss: {final_loss:.4f}")
        print(f"      Validation loss: {final_val_loss:.4f}")
        print(f"      Training accuracy: {final_accuracy:.4f}")
        print(f"      Validation accuracy: {final_val_accuracy:.4f}")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        print(f"   💡 Troubleshooting:")
        print(f"      1. Check memory usage (reduce batch size if OOM)")
        print(f"      2. Verify generator integrity")
        print(f"      3. Check GPU availability")
        print(f"      4. Validate data shapes and types")
        raise
```

## 💾 **MODEL & METADATA SAVING**

### **Comprehensive Model Persistence**
```python
def save_model_and_metadata(self, data_info: Dict[str, Any]):
    """
    💾 SAVE MODEL AND COMPREHENSIVE METADATA
    
    Saves:
    - Model (.keras format)
    - Scaler (.pkl format) 
    - Training metadata (.json format)
    - Training history
    - Configuration parameters
    """
    print(f"\n💾 SAVING MODEL AND METADATA...")
    
    # 1. Save model
    model_path = os.path.join(self.model_output_dir, config.get_model_filename())
    success = self.model_builder.save_model(self.model, model_path, format='keras')
    
    if not success:
        raise RuntimeError("Failed to save model")
    
    # 2. Save metadata
    if config.SAVE_METADATA:
        self._save_training_metadata(data_info)
    
    # 3. Save training history
    self._save_training_history()
    
    print(f"   ✅ Model and metadata saved successfully!")
```

### **Training Metadata Generation**
```python
def _save_training_metadata(self, data_info: Dict[str, Any]):
    """
    Save comprehensive training metadata to JSON
    """
    metadata = {
        # Configuration parameters
        'config': {
            'pair': config.PAIR,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'sequence_length': config.SEQUENCE_LENGTH,
            'lstm_units': config.LSTM_UNITS,
            'dense_units': config.DENSE_UNITS,
            'dropout_rate': config.DROPOUT_RATE,
            'learning_rate': config.LEARNING_RATE,
            'validation_split': config.VALIDATION_SPLIT,
            'future_window': config.FUTURE_WINDOW,
            'long_tp_pct': config.LONG_TP_PCT,
            'long_sl_pct': config.LONG_SL_PCT
        },
        
        # Data information
        'data': {
            'samples_count': data_info['samples_count'],
            'features_shape': list(data_info['features_shape']),
            'labels_shape': list(data_info['labels_shape']),
            'feature_columns': data_info['feature_columns'],
            'label_format': data_info['label_format'],
            'scaling_enabled': data_info['scaling_enabled'],
            'scaler_type': config.SCALER_TYPE if config.ENABLE_FEATURE_SCALING else None,
            'balancing_applied': data_info['balancing_applied'],
            'balancing_method': data_info['balancing_method']
        },
        
        # Model information
        'model': {
            'total_parameters': self.model.count_params(),
            'trainable_parameters': sum([np.prod(v.shape) for v in self.model.trainable_variables]),
            'optimizer': 'Adam',
            'loss_function': 'categorical_crossentropy',
            'metrics': ['accuracy', 'categorical_accuracy']
        },
        
        # Training results
        'training': {
            'completed_epochs': len(self.training_history.history['loss']),
            'final_training_loss': float(self.training_history.history['loss'][-1]),
            'final_validation_loss': float(self.training_history.history['val_loss'][-1]),
            'final_training_accuracy': float(self.training_history.history['accuracy'][-1]),
            'final_validation_accuracy': float(self.training_history.history['val_accuracy'][-1]),
            'best_validation_accuracy': float(max(self.training_history.history['val_accuracy']))
        },
        
        # System information
        'system': {
            'timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'python_version': sys.version,
            'gpu_available': self.model_builder.gpu_available
        },
        
        # File paths
        'files': {
            'model_file': config.get_model_filename(),
            'scaler_file': config.get_scaler_filename(),
            'training_data_file': config.get_expected_filename()
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(self.model_output_dir, config.get_metadata_filename())
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   📄 Metadata saved: {config.get_metadata_filename()}")
```

## 📊 **CONFUSION MATRIX ANALYSIS**

### **Post-Training Model Evaluation**
```python
def generate_confusion_matrix_report(self):
    """
    📊 GENERATE CONFUSION MATRIX AND ANALYSIS REPORT
    
    Features:
    - Model evaluation on validation data
    - Confusion matrix visualization
    - Per-class performance metrics
    - Trading signal analysis
    """
    print(f"\n📊 GENERATING CONFUSION MATRIX REPORT...")
    
    try:
        # Load validation data for evaluation
        data_info = self.data_loader.load_training_data()
        
        # Create generators for evaluation
        train_gen, val_gen = self.create_generators(data_info['features'], data_info['labels'])
        
        # Collect validation predictions
        print(f"   🔍 Collecting validation predictions...")
        
        val_predictions = []
        val_true_labels = []
        
        # Iterate through validation generator
        for sequences, true_labels in val_gen:
            # Make predictions
            predictions = self.model.predict(sequences, verbose=0)
            
            val_predictions.extend(predictions)
            val_true_labels.extend(true_labels)
            
            # Break after enough samples for evaluation
            if len(val_predictions) >= 10000:  # Evaluate on 10K samples
                break
        
        # Convert to numpy arrays
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)
        
        # Convert to class indices
        pred_classes = np.argmax(val_predictions, axis=1)
        true_classes = np.argmax(val_true_labels, axis=1)
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(true_classes, pred_classes)
        class_names = ['SHORT', 'HOLD', 'LONG']
        
        # Create detailed report
        report = {
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'classification_report': classification_report(true_classes, pred_classes, 
                                                         target_names=class_names, 
                                                         output_dict=True),
            'sample_count': len(pred_classes),
            'accuracy': float(np.sum(pred_classes == true_classes) / len(pred_classes))
        }
        
        # Print confusion matrix
        print(f"\n   📊 CONFUSION MATRIX:")
        print(f"      Samples evaluated: {len(pred_classes):,}")
        print(f"      Overall accuracy: {report['accuracy']:.4f}")
        print(f"\n      Predicted →")
        print(f"      True ↓     SHORT   HOLD    LONG")
        for i, class_name in enumerate(class_names):
            row = f"      {class_name:4s}     "
            for j in range(3):
                row += f"{cm[i,j]:6d}  "
            print(row)
        
        # Per-class metrics
        print(f"\n   📈 PER-CLASS METRICS:")
        for class_name in class_names:
            metrics = report['classification_report'][class_name]
            print(f"      {class_name}:")
            print(f"         Precision: {metrics['precision']:.4f}")
            print(f"         Recall:    {metrics['recall']:.4f}")
            print(f"         F1-Score:  {metrics['f1-score']:.4f}")
        
        # Save confusion matrix report
        cm_report_path = os.path.join(self.model_output_dir, "confusion_matrix_report.json")
        with open(cm_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n   ✅ Confusion matrix report saved: confusion_matrix_report.json")
        
    except Exception as e:
        print(f"   ⚠️ Failed to generate confusion matrix: {e}")
        print(f"   💡 Continuing without confusion matrix analysis...")
```

## 🖥️ **CLI INTERFACE**

### **Command Line Interface**
```python
def main():
    """
    🖥️ MAIN FUNCTION WITH CLI SUPPORT
    
    Supports:
    - Parameter override via command line
    - Configuration testing
    - Training execution
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone Training Module V3')
    parser.add_argument('--pair', type=str, help='Override crypto pair (e.g., ETHUSDT)')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--config-test', action='store_true', help='Test configuration only')
    
    args = parser.parse_args()
    
    # Apply CLI overrides
    if args.pair:
        config.PAIR = args.pair
        print(f"   🔧 CLI Override: PAIR = {args.pair}")
    
    if args.epochs:
        config.EPOCHS = args.epochs
        print(f"   🔧 CLI Override: EPOCHS = {args.epochs}")
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        print(f"   🔧 CLI Override: BATCH_SIZE = {args.batch_size}")
    
    # Validate configuration
    config_errors = config.validate_config()
    if config_errors:
        print(f"❌ Configuration errors:")
        for error in config_errors:
            print(f"   - {error}")
        return
    
    # Configuration test mode
    if args.config_test:
        print(f"✅ Configuration test passed!")
        config.print_config_summary()
        return
    
    # Execute training
    trainer = StandaloneTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()
```

## 🎯 **USAGE EXAMPLES**

### **Example 1: Basic Training**
```bash
# Basic training with default configuration
python trainer.py
```

### **Example 2: CLI Parameter Override**
```bash
# Override parameters via command line
python trainer.py --pair ETHUSDT --epochs 50 --batch-size 128
```

### **Example 3: Configuration Testing**
```bash
# Test configuration without training
python trainer.py --config-test
```

### **Example 4: Programmatic Usage**
```python
# Use trainer programmatically
trainer = StandaloneTrainer()
trainer.run_training()
```

---

**🎯 KLUCZOWE ZALETY TRAINER V3:**
- ✅ **Complete Pipeline** - end-to-end training workflow
- ✅ **Component Integration** - orchestrates all V3 modules
- ✅ **Standalone Operation** - zero validation module dependencies
- ✅ **Error Handling** - comprehensive error messages with solutions
- ✅ **CLI Interface** - parameter override and testing support
- ✅ **Confusion Matrix** - post-training model evaluation
- ✅ **Comprehensive Metadata** - full training documentation

**📈 NEXT:** [08_Utils.md](./08_Utils.md) - Helper functions and monitoring utilities 