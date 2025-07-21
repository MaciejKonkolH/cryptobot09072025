"""
🤖 MODEL BUILDER V2 - MEMORY EFFICIENT MODULE
LSTM Model Architecture for crypto trading predictions

ARCHITECTURE:
- Multi-layer LSTM stack (128, 64, 32)
- Dense layers (32, 16) 
- Dropout for regularization
- Output: 3 classes (SHORT, HOLD, LONG)
- Memory optimized for training

Compatible with:
- TensorFlow/Keras
- Memory-efficient generators
- SavedModel format
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
    print(f"✅ TensorFlow {tf.__version__} available")
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not available - model building disabled")
    # Create dummy classes for testing
    class Model:
        pass
    class keras:
        class callbacks:
            class Callback:
                pass

# Import standalone config (no TrainingConfig class)
import config
from utils import monitor_memory_usage, safe_memory_cleanup, create_focal_loss


class DualWindowLSTMBuilder:
    """
    🤖 DUAL WINDOW LSTM BUILDER
    
    Builds memory-efficient LSTM model for crypto trading
    - Configurable architecture from TrainingConfig
    - Memory optimized
    - Compatible with generators
    """
    
    def __init__(self, config):
        self.config = config
        
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available - model building will fail")
    
    def build_model(self, compile_model: bool = True) -> Optional[Model]:
        """
        🏗️ BUILD LSTM MODEL
        
        Args:
            compile_model: Whether to compile the model
            
        Returns:
            Compiled Keras model or None if TF not available
        """
        if not TF_AVAILABLE:
            print("❌ Cannot build model - TensorFlow not available")
            return None
        
        print(f"🏗️ Building LSTM model...")
        print(f"   Input shape: {self.config.get_input_shape()}")
        print(f"   Output classes: {self.config.get_output_classes()}")
        
        # Input layer
        inputs = layers.Input(shape=self.config.get_input_shape(), name='sequence_input')
        
        # LSTM stack with dropout
        x = inputs
        for i, units in enumerate(self.config.LSTM_UNITS):
            return_sequences = (i < len(self.config.LSTM_UNITS) - 1)  # Return sequences except last layer
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                name=f'lstm_{i+1}_{units}'
            )(x)
            
            print(f"   Added LSTM layer {i+1}: {units} units, return_sequences={return_sequences}")
        
        # Dense stack
        for i, units in enumerate(self.config.DENSE_UNITS):
            x = layers.Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}_{units}'
            )(x)
            
            # Add dropout after each dense layer
            x = layers.Dropout(self.config.DROPOUT_RATE, name=f'dropout_{i+1}')(x)
            
            print(f"   Added Dense layer {i+1}: {units} units + dropout")
        
        # Output layer (3 classes: SHORT, HOLD, LONG)
        outputs = layers.Dense(
            units=self.config.get_output_classes(),
            activation='softmax',
            name='prediction_output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='DualWindowLSTM')
        
        # Print model summary
        print(f"\n📊 Model Architecture:")
        model.summary()
        
        # Calculate model parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        print(f"\n📈 Model Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        
        # Compile model if requested
        if compile_model:
            self._compile_model(model)
        
        return model
    
    def _compile_model(self, model: Model) -> None:
        """
        ⚙️ COMPILE MODEL
        Configure optimizer, loss, and metrics
        """
        print(f"\n⚙️ Compiling model...")
        
        # Determine the loss function based on config
        loss_function = 'sparse_categorical_crossentropy'
        if self.config.CLASS_BALANCING_METHOD == 'focal_loss':
            print(f"   🔥 Using Focal Loss")
            loss_function = create_focal_loss(
                gamma=self.config.FOCAL_LOSS_GAMMA,
                alpha=self.config.FOCAL_LOSS_ALPHA_WEIGHTS
            )
            print(f"      Gamma: {self.config.FOCAL_LOSS_GAMMA}, Alpha (weights): {self.config.FOCAL_LOSS_ALPHA_WEIGHTS}")
        else:
            print(f"   Loss: sparse_categorical_crossentropy")

        print(f"   Optimizer: Adam")
        print(f"   Learning rate: {self.config.LEARNING_RATE}")
        print(f"   Metrics: accuracy")
        
        # Optimizer
        optimizer = optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=['accuracy']
        )
        
        print(f"   ✅ Model compiled successfully")
    
    def create_callbacks(self, model_output_dir: str) -> list:
        """
        📋 CREATE TRAINING CALLBACKS
        
        Args:
            model_output_dir: Directory to save model checkpoints
            
        Returns:
            List of Keras callbacks
        """
        if not TF_AVAILABLE:
            return []
        
        print(f"📋 Creating training callbacks...")
        
        callbacks_list = []
        
        # Create output directory
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 1. Model Checkpoint - save best model
        checkpoint_path = os.path.join(model_output_dir, 'best_model.keras')
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint_callback)
        print(f"   ✅ ModelCheckpoint: {checkpoint_path}")
        
        # 2. Early Stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callbacks_list.append(early_stopping)
        print(f"   ✅ EarlyStopping: patience=10")
        
        # 3. Reduce Learning Rate on Plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        print(f"   ✅ ReduceLROnPlateau: factor=0.5, patience=5")
        
        # 4. TensorBoard (optional, if logs directory exists)
        tensorboard_dir = os.path.join(model_output_dir, 'logs')
        try:
            os.makedirs(tensorboard_dir, exist_ok=True)
            tensorboard = callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False
            )
            callbacks_list.append(tensorboard)
            print(f"   ✅ TensorBoard: {tensorboard_dir}")
        except Exception as e:
            print(f"   ⚠️ TensorBoard setup failed: {e}")
        
        print(f"   📋 Created {len(callbacks_list)} callbacks")
        
        return callbacks_list
    
    def save_model(self, model: Model, save_path: str, format: str = 'saved_model') -> str:
        """
        💾 SAVE MODEL
        
        Args:
            model: Trained Keras model
            save_path: Path to save model
            format: 'saved_model' or 'h5'
            
        Returns:
            str: Final save path
        """
        if not TF_AVAILABLE:
            print("❌ Cannot save model - TensorFlow not available")
            return ""
        
        print(f"💾 Saving model...")
        print(f"   Format: {format}")
        print(f"   Path: {save_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if format == 'saved_model':
            # SavedModel format (recommended)
            model.save(save_path)
            final_path = save_path
        elif format == 'h5':
            # H5 format (legacy)
            h5_path = save_path + '.h5' if not save_path.endswith('.h5') else save_path
            model.save(h5_path)
            final_path = h5_path
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Calculate model size
        if os.path.exists(final_path):
            if os.path.isdir(final_path):
                # SavedModel directory
                size_bytes = sum(os.path.getsize(os.path.join(root, file)) 
                               for root, dirs, files in os.walk(final_path) 
                               for file in files)
            else:
                # H5 file
                size_bytes = os.path.getsize(final_path)
            
            size_mb = size_bytes / (1024 * 1024)
            print(f"   💾 Model saved: {size_mb:.1f}MB")
        
        print(f"   ✅ Model saved to: {final_path}")
        
        return final_path
    
    def load_model(self, model_path: str) -> Optional[Model]:
        """
        📂 LOAD MODEL
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded Keras model or None
        """
        if not TF_AVAILABLE:
            print("❌ Cannot load model - TensorFlow not available")
            return None
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        print(f"📂 Loading model from: {model_path}")
        
        try:
            model = keras.models.load_model(model_path)
            print(f"   ✅ Model loaded successfully")
            
            # Print model info
            total_params = model.count_params()
            print(f"   📊 Parameters: {total_params:,}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def get_model_info(self, model: Model) -> Dict[str, Any]:
        """
        📊 GET MODEL INFO
        
        Args:
            model: Keras model
            
        Returns:
            Dict with model information
        """
        if not TF_AVAILABLE or model is None:
            return {}
        
        info = {
            'name': model.name,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'layers': len(model.layers),
            'optimizer': model.optimizer.__class__.__name__ if hasattr(model, 'optimizer') else None,
            'loss': model.loss if hasattr(model, 'loss') else None
        }
        
        return info


class MemoryOptimizedTrainer:
    """
    🏃 MEMORY OPTIMIZED TRAINER
    Helper class for memory-efficient training
    """
    
    def __init__(self, config):
        self.config = config
    
    def setup_gpu_memory(self) -> bool:
        """
        🖥️ SETUP GPU MEMORY
        Configure GPU memory growth to avoid OOM
        """
        if not TF_AVAILABLE:
            return False
        
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
                return True
            else:
                print("⚠️ No GPUs found - using CPU")
                return False
        except Exception as e:
            print(f"⚠️ GPU setup failed: {e}")
            return False
    
    def train_model(self, model, train_generator, val_generator, epochs: int):
        """
        🏋️ TRAIN MODEL
        Train model with memory optimization
        
        Args:
            model: Compiled Keras model
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs to train
            
        Returns:
            Training history or None if TF not available
        """
        if not TF_AVAILABLE:
            print("❌ Cannot train model - TensorFlow not available")
            return None
            
        print(f"🏋️ Training model for {epochs} epochs...")
        
        try:
            # Setup GPU memory
            self.setup_gpu_memory()
            
            # Train model
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                verbose=1
            )
            
            print(f"✅ Training completed successfully")
            return history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def estimate_training_memory(self, train_batches: int, val_batches: int) -> Dict[str, float]:
        """
        🧮 ESTIMATE TRAINING MEMORY
        Estimate memory requirements for training
        """
        from utils import calculate_memory_usage
        
        # Batch memory
        batch_shape = (self.config.BATCH_SIZE, self.config.WINDOW_SIZE, len(self.config.FEATURES))
        batch_memory = calculate_memory_usage(batch_shape, np.float32)
        
        # Model memory (rough estimate)
        model_params = (
            sum(self.config.LSTM_UNITS) * self.config.WINDOW_SIZE * len(self.config.FEATURES) +
            sum(self.config.DENSE_UNITS) * sum(self.config.LSTM_UNITS) +
            self.config.get_output_classes() * self.config.DENSE_UNITS[-1]
        )
        model_memory_gb = model_params * 4 / (1024**3)  # float32
        
        # Total estimate
        training_memory_gb = batch_memory['gb'] * 2 + model_memory_gb  # *2 for gradients
        
        return {
            'batch_memory_gb': batch_memory['gb'],
            'model_memory_gb': model_memory_gb,
            'training_memory_gb': training_memory_gb,
            'train_batches': train_batches,
            'val_batches': val_batches,
            'feasible': training_memory_gb < self.config.MAX_MEMORY_GB
        }


def main():
    """Test model builder"""
    print("🧪 TESTING MODEL BUILDER")
    
    # Create mock config for testing
    class MockConfig:
        SEQUENCE_LENGTH = 120
        INPUT_FEATURES = 8
        LSTM_UNITS = [128, 64, 32]
        DENSE_UNITS = [32, 16]
        DROPOUT_RATE = 0.3
        LEARNING_RATE = 0.001
        
        def get_input_shape(self):
            return (self.SEQUENCE_LENGTH, self.INPUT_FEATURES)
        
        def get_output_classes(self):
            return 3
    
    config = MockConfig()
    
    # Create builder
    builder = DualWindowLSTMBuilder(config)
    
    print(f"\n🔧 Configuration:")
    print(f"   TensorFlow available: {TF_AVAILABLE}")
    print(f"   Input shape: {config.get_input_shape()}")
    print(f"   Output classes: {config.get_output_classes()}")
    print(f"   LSTM units: {config.LSTM_UNITS}")
    print(f"   Dense units: {config.DENSE_UNITS}")
    
    if TF_AVAILABLE:
        print(f"\n🏗️ Building test model...")
        model = builder.build_model(compile_model=True)
        
        if model:
            print(f"\n📊 Model Info:")
            info = builder.get_model_info(model)
            for key, value in info.items():
                print(f"   {key}: {value}")
            
            print(f"\n💡 Model building test completed!")
        else:
            print(f"\n❌ Model building failed!")
    else:
        print(f"\n💡 Model builder ready (TensorFlow not available for testing)")


if __name__ == "__main__":
    main() 