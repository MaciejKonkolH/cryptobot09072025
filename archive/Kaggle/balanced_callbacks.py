"""
📊 BALANCED CALLBACKS MODULE
Custom TensorFlow callbacks dla lepszych metryk podczas trenowania

CALLBACKS:
- BalancedAccuracyCallback - oblicza balanced accuracy na końcu każdej epoki
- DetailedMetricsCallback - szczegółowe metryki per-klasa
- ClassDistributionCallback - monitoruje rozkład predykcji

ELIMINUJE: Mylące val_accuracy z TensorFlow
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional

try:
    from sklearn.metrics import balanced_accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - balanced callbacks disabled")


class BalancedAccuracyCallback(tf.keras.callbacks.Callback):
    """
    📊 BALANCED ACCURACY CALLBACK
    
    Oblicza balanced accuracy na końcu każdej epoki
    - Zastępuje mylące val_accuracy
    - Pokazuje prawdziwą wydajność na niezbalansowanych danych
    - Loguje do TensorBoard i konsoli
    """
    
    def __init__(self, validation_data=None, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.verbose = verbose
        self.balanced_accuracies = []
        
    def on_epoch_end(self, epoch, logs=None):
        if not SKLEARN_AVAILABLE:
            return
            
        logs = logs or {}
        
        # Oblicz balanced accuracy dla danych walidacyjnych
        if self.validation_data is not None:
            try:
                # Pobierz dane walidacyjne
                if hasattr(self.validation_data, 'labels'):
                    # Generator z atrybutem labels
                    y_true = self.validation_data.labels
                    y_pred_proba = self.model.predict(self.validation_data, verbose=0)
                    
                    # Ogranicz do rzeczywistej liczby predykcji
                    num_predictions = len(y_pred_proba)
                    y_true = y_true[:num_predictions]
                    
                else:
                    # Standardowy format (X, y)
                    X_val, y_true = self.validation_data
                    y_pred_proba = self.model.predict(X_val, verbose=0)
                
                # Konwertuj prawdopodobieństwa na klasy
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # Oblicz balanced accuracy
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                
                # Dodaj do logów
                logs['val_balanced_accuracy'] = balanced_acc
                self.balanced_accuracies.append(balanced_acc)
                
                # Wydrukuj jeśli verbose
                if self.verbose > 0:
                    print(f" - val_balanced_accuracy: {balanced_acc:.4f} ⭐")
                
            except Exception as e:
                if self.verbose > 0:
                    print(f" - balanced_accuracy calculation failed: {e}")


class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    """
    📈 DETAILED METRICS CALLBACK
    
    Oblicza szczegółowe metryki per-klasa co N epok
    - Precision, Recall, F1 per klasa
    - Confusion matrix
    - Class distribution analysis
    """
    
    def __init__(self, validation_data=None, frequency=5, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.frequency = frequency  # Co ile epok pokazywać szczegóły
        self.verbose = verbose
        self.class_names = ['SHORT', 'HOLD', 'LONG']
        
    def on_epoch_end(self, epoch, logs=None):
        if not SKLEARN_AVAILABLE:
            return
            
        # Pokazuj szczegóły co N epok
        if (epoch + 1) % self.frequency != 0:
            return
            
        if self.validation_data is None:
            return
            
        try:
            # Pobierz dane walidacyjne
            if hasattr(self.validation_data, 'labels'):
                y_true = self.validation_data.labels
                y_pred_proba = self.model.predict(self.validation_data, verbose=0)
                num_predictions = len(y_pred_proba)
                y_true = y_true[:num_predictions]
            else:
                X_val, y_true = self.validation_data
                y_pred_proba = self.model.predict(X_val, verbose=0)
            
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            if self.verbose > 0:
                print(f"\n📊 DETAILED METRICS - EPOCH {epoch + 1}")
                print("=" * 50)
                
                # Balanced accuracy
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                print(f"🎯 Balanced Accuracy: {balanced_acc:.4f}")
                
                # Per-class metrics
                report = classification_report(
                    y_true, y_pred, 
                    target_names=self.class_names,
                    labels=[0, 1, 2],
                    zero_division=0,
                    output_dict=True
                )
                
                print(f"\n📈 Per-Class Metrics:")
                print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
                print("-" * 45)
                
                for i, class_name in enumerate(self.class_names):
                    if str(i) in report:
                        precision = report[str(i)]['precision']
                        recall = report[str(i)]['recall']
                        f1 = report[str(i)]['f1-score']
                        print(f"{class_name:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
                
                # Class distribution
                pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
                true_unique, true_counts = np.unique(y_true, return_counts=True)
                
                print(f"\n📊 Class Distribution:")
                print(f"{'Class':<8} {'True':>8} {'Pred':>8} {'Bias':>8}")
                print("-" * 35)
                
                total_samples = len(y_true)
                for i, class_name in enumerate(self.class_names):
                    true_count = true_counts[true_unique == i][0] if i in true_unique else 0
                    pred_count = pred_counts[pred_unique == i][0] if i in pred_unique else 0
                    
                    true_pct = (true_count / total_samples) * 100
                    pred_pct = (pred_count / total_samples) * 100
                    bias = pred_pct - true_pct
                    
                    print(f"{class_name:<8} {true_pct:>7.1f}% {pred_pct:>7.1f}% {bias:>+7.1f}")
                
                print("=" * 50)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"\n❌ Detailed metrics calculation failed: {e}")


class ClassDistributionCallback(tf.keras.callbacks.Callback):
    """
    📈 CLASS DISTRIBUTION CALLBACK
    
    Monitoruje jak zmienia się rozkład predykcji podczas trenowania
    - Wykrywa czy model zaczyna faworyzować jedną klasę
    - Ostrzega przed overfittingiem do dominującej klasy
    """
    
    def __init__(self, validation_data=None, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.verbose = verbose
        self.class_names = ['SHORT', 'HOLD', 'LONG']
        self.distribution_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            return
            
        try:
            # Pobierz dane walidacyjne
            if hasattr(self.validation_data, 'labels'):
                y_true = self.validation_data.labels
                y_pred_proba = self.model.predict(self.validation_data, verbose=0)
                num_predictions = len(y_pred_proba)
                y_true = y_true[:num_predictions]
            else:
                X_val, y_true = self.validation_data
                y_pred_proba = self.model.predict(X_val, verbose=0)
            
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Oblicz rozkład predykcji
            pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
            total_samples = len(y_pred)
            
            distribution = {}
            for i, class_name in enumerate(self.class_names):
                count = pred_counts[pred_unique == i][0] if i in pred_unique else 0
                percentage = (count / total_samples) * 100
                distribution[class_name] = percentage
            
            self.distribution_history.append(distribution)
            
            # Sprawdź czy jest problem z rozkładem
            max_class_pct = max(distribution.values())
            min_class_pct = min(distribution.values())
            
            # Ostrzeżenie jeśli jedna klasa dominuje za bardzo
            if max_class_pct > 90.0 and self.verbose > 0:
                dominant_class = max(distribution, key=distribution.get)
                print(f" ⚠️ WARNING: {dominant_class} dominates predictions ({max_class_pct:.1f}%)")
            
            # Ostrzeżenie jeśli jakaś klasa znika
            if min_class_pct < 1.0 and self.verbose > 0:
                rare_class = min(distribution, key=distribution.get)
                print(f" ⚠️ WARNING: {rare_class} rarely predicted ({min_class_pct:.1f}%)")
            
            # Dodaj do logów
            if logs is not None:
                for class_name, pct in distribution.items():
                    logs[f'val_pred_pct_{class_name}'] = pct
                    
        except Exception as e:
            if self.verbose > 0:
                print(f" - distribution monitoring failed: {e}")


def create_balanced_callbacks(validation_data=None, 
                            detailed_metrics_frequency=5,
                            verbose=1) -> List[tf.keras.callbacks.Callback]:
    """
    🎯 FACTORY FUNCTION - STWÓRZ ZBALANSOWANE CALLBACKS
    
    Args:
        validation_data: Dane walidacyjne (generator lub tuple)
        detailed_metrics_frequency: Co ile epok pokazywać szczegółowe metryki
        verbose: Poziom szczegółowości logów
        
    Returns:
        Lista callbacków do użycia w model.fit()
    """
    callbacks = []
    
    if SKLEARN_AVAILABLE:
        # Balanced accuracy callback (każda epoka)
        callbacks.append(BalancedAccuracyCallback(
            validation_data=validation_data,
            verbose=verbose
        ))
        
        # Detailed metrics callback (co N epok)
        callbacks.append(DetailedMetricsCallback(
            validation_data=validation_data,
            frequency=detailed_metrics_frequency,
            verbose=verbose
        ))
        
        # Class distribution monitoring
        callbacks.append(ClassDistributionCallback(
            validation_data=validation_data,
            verbose=verbose
        ))
    else:
        print("⚠️ Balanced callbacks not available - sklearn required")
    
    return callbacks


if __name__ == "__main__":
    # Test callbacków
    print("🧪 TESTING BALANCED CALLBACKS")
    
    # Symulacja danych
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Niezbalansowane dane
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.15, 0.70, 0.15])
    
    # Prosty model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Test callbacków
    callbacks = create_balanced_callbacks(
        validation_data=(X, y),
        detailed_metrics_frequency=2,
        verbose=1
    )
    
    print(f"✅ Created {len(callbacks)} balanced callbacks")
    
    # Krótki trening testowy
    if len(callbacks) > 0:
        print("🏋️ Testing callbacks with short training...")
        model.fit(X, y, epochs=3, validation_data=(X, y), 
                 callbacks=callbacks, verbose=1)
        print("✅ Callbacks test completed!")
    else:
        print("⚠️ No callbacks to test") 