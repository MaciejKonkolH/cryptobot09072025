"""
📊 BALANCED METRICS MODULE
Lepsze metryki dla niezbalansowanych danych kryptowalutowych

METRYKI:
- Balanced Accuracy (średnia z recall per klasa)
- Per-class Precision, Recall, F1-Score
- Macro/Micro F1-Score
- Confusion Matrix Analysis
- Class-wise Performance Report

ELIMINUJE: Mylące accuracy dla niezbalansowanych danych
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    from sklearn.metrics import (
        confusion_matrix, 
        classification_report,
        precision_recall_fscore_support,
        balanced_accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - balanced metrics disabled")


class BalancedMetricsCalculator:
    """
    📊 CALCULATOR ZBALANSOWANYCH METRYK
    
    Oblicza metryki odpowiednie dla niezbalansowanych danych:
    - Balanced Accuracy zamiast zwykłej accuracy
    - Per-class metrics (precision, recall, f1)
    - Macro/Micro averaging
    - Szczegółowa analiza confusion matrix
    """
    
    def __init__(self):
        self.class_names = ['SHORT', 'HOLD', 'LONG']
        self.class_labels = [0, 1, 2]
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        🎯 OBLICZ WSZYSTKIE ZBALANSOWANE METRYKI
        
        Args:
            y_true: Prawdziwe etykiety (sparse format)
            y_pred: Przewidziane etykiety (sparse format)
            y_pred_proba: Prawdopodobieństwa predykcji (opcjonalne)
            
        Returns:
            Dict z wszystkimi metrykami
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        metrics = {}
        
        # 1. BALANCED ACCURACY (najważniejsza metryka!)
        metrics['balanced_accuracy'] = self._calculate_balanced_accuracy(y_true, y_pred)
        
        # 2. PER-CLASS METRICS
        per_class_metrics = self._calculate_per_class_metrics(y_true, y_pred)
        metrics.update(per_class_metrics)
        
        # 3. MACRO/MICRO F1
        macro_micro_metrics = self._calculate_macro_micro_metrics(y_true, y_pred)
        metrics.update(macro_micro_metrics)
        
        # 4. CONFUSION MATRIX ANALYSIS
        cm_analysis = self._analyze_confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix_analysis'] = cm_analysis
        
        # 5. CLASS DISTRIBUTION ANALYSIS
        distribution_analysis = self._analyze_class_distributions(y_true, y_pred)
        metrics['distribution_analysis'] = distribution_analysis
        
        # 6. CONFIDENCE ANALYSIS (jeśli dostępne)
        if y_pred_proba is not None:
            confidence_analysis = self._analyze_prediction_confidence(y_pred_proba, y_pred)
            metrics['confidence_analysis'] = confidence_analysis
        
        return metrics
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        🎯 BALANCED ACCURACY - KLUCZOWA METRYKA!
        
        Balanced Accuracy = średnia z recall każdej klasy
        - Nie faworyzuje dominującej klasy (HOLD)
        - Pokazuje czy model radzi sobie ze wszystkimi klasami
        """
        return float(balanced_accuracy_score(y_true, y_pred))
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        📊 METRYKI PER-KLASA
        
        Precision, Recall, F1-Score dla każdej klasy osobno
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_labels, zero_division=0
        )
        
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = float(precision[i])
            metrics[f'recall_{class_name}'] = float(recall[i])
            metrics[f'f1_{class_name}'] = float(f1[i])
            metrics[f'support_{class_name}'] = int(support[i])
        
        return metrics
    
    def _calculate_macro_micro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        🔄 MACRO/MICRO AVERAGING
        
        - Macro: średnia z metryk każdej klasy (równe traktowanie klas)
        - Micro: globalna metryka (faworyzuje częste klasy)
        """
        # Macro averaging (równe wagi dla klas)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Micro averaging (wagi proporcjonalne do support)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        return {
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro)
        }
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
        """
        🔍 ANALIZA CONFUSION MATRIX
        
        Szczegółowa analiza błędów modelu
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.class_labels)
        
        # Confusion matrix jako DataFrame
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        
        # Analiza błędów
        total_samples = np.sum(cm)
        correct_predictions = np.trace(cm)
        
        # Per-class accuracy (diagonal / row sum)
        per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            if cm[i].sum() > 0:
                per_class_accuracy[class_name] = cm[i, i] / cm[i].sum()
            else:
                per_class_accuracy[class_name] = 0.0
        
        # Najczęstsze błędy
        common_errors = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    error_rate = cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
                    common_errors.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'error_rate': float(error_rate)
                    })
        
        # Sortuj błędy po częstości
        common_errors.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_df': cm_df.to_dict(),
            'total_samples': int(total_samples),
            'correct_predictions': int(correct_predictions),
            'overall_accuracy': float(correct_predictions / total_samples) if total_samples > 0 else 0.0,
            'per_class_accuracy': per_class_accuracy,
            'common_errors': common_errors[:5]  # Top 5 błędów
        }
    
    def _analyze_class_distributions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
        """
        📈 ANALIZA ROZKŁADÓW KLAS
        
        Porównanie rozkładu prawdziwych vs przewidzianych etykiet
        """
        # Rozkład prawdziwych etykiet
        true_unique, true_counts = np.unique(y_true, return_counts=True)
        true_distribution = {}
        total_true = len(y_true)
        
        for label, count in zip(true_unique, true_counts):
            if label in self.class_labels:
                class_name = self.class_names[label]
                true_distribution[class_name] = {
                    'count': int(count),
                    'percentage': float(count / total_true * 100)
                }
        
        # Rozkład przewidzianych etykiet
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        pred_distribution = {}
        total_pred = len(y_pred)
        
        for label, count in zip(pred_unique, pred_counts):
            if label in self.class_labels:
                class_name = self.class_names[label]
                pred_distribution[class_name] = {
                    'count': int(count),
                    'percentage': float(count / total_pred * 100)
                }
        
        # Bias analysis - czy model faworyzuje pewne klasy?
        bias_analysis = {}
        for class_name in self.class_names:
            true_pct = true_distribution.get(class_name, {}).get('percentage', 0)
            pred_pct = pred_distribution.get(class_name, {}).get('percentage', 0)
            bias = pred_pct - true_pct
            bias_analysis[class_name] = {
                'true_percentage': true_pct,
                'predicted_percentage': pred_pct,
                'bias': bias,
                'bias_interpretation': self._interpret_bias(bias)
            }
        
        return {
            'true_distribution': true_distribution,
            'predicted_distribution': pred_distribution,
            'bias_analysis': bias_analysis
        }
    
    def _interpret_bias(self, bias: float) -> str:
        """Interpretacja bias dla klasy"""
        if abs(bias) < 2.0:
            return "balanced"
        elif bias > 2.0:
            return "over-predicted"
        else:
            return "under-predicted"
    
    def _analyze_prediction_confidence(self, y_pred_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
        """
        🎯 ANALIZA PEWNOŚCI PREDYKCJI
        
        Analiza jak pewny jest model swoich predykcji
        """
        # Średnia pewność per klasa
        confidence_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Próbki przewidziane jako ta klasa
            class_mask = (y_pred == i)
            if np.any(class_mask):
                # Pewność dla tej klasy
                class_confidences = y_pred_proba[class_mask, i]
                confidence_per_class[class_name] = {
                    'mean_confidence': float(np.mean(class_confidences)),
                    'std_confidence': float(np.std(class_confidences)),
                    'min_confidence': float(np.min(class_confidences)),
                    'max_confidence': float(np.max(class_confidences)),
                    'samples_count': int(np.sum(class_mask))
                }
            else:
                confidence_per_class[class_name] = {
                    'mean_confidence': 0.0,
                    'std_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0,
                    'samples_count': 0
                }
        
        # Globalna analiza pewności
        max_confidences = np.max(y_pred_proba, axis=1)
        
        return {
            'confidence_per_class': confidence_per_class,
            'global_confidence': {
                'mean': float(np.mean(max_confidences)),
                'std': float(np.std(max_confidences)),
                'min': float(np.min(max_confidences)),
                'max': float(np.max(max_confidences))
            }
        }
    
    def print_balanced_report(self, metrics: Dict[str, any], title: str = "BALANCED METRICS REPORT") -> None:
        """
        📋 WYDRUKUJ RAPORT ZBALANSOWANYCH METRYK
        
        Czytelny raport z najważniejszymi metrykami
        """
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}")
        
        # 1. KLUCZOWE METRYKI
        print(f"\n🎯 KLUCZOWE METRYKI (dla niezbalansowanych danych):")
        print(f"   Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f} ⭐ (NAJWAŻNIEJSZA)")
        print(f"   Macro F1-Score:    {metrics.get('f1_macro', 0):.4f}")
        print(f"   Micro F1-Score:    {metrics.get('f1_micro', 0):.4f}")
        
        # 2. PER-CLASS PERFORMANCE
        print(f"\n📊 WYDAJNOŚĆ PER-KLASA:")
        print(f"   {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print(f"   {'-'*58}")
        
        for class_name in self.class_names:
            precision = metrics.get(f'precision_{class_name}', 0)
            recall = metrics.get(f'recall_{class_name}', 0)
            f1 = metrics.get(f'f1_{class_name}', 0)
            support = metrics.get(f'support_{class_name}', 0)
            
            print(f"   {class_name:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10,}")
        
        # 3. CONFUSION MATRIX
        if 'confusion_matrix_analysis' in metrics:
            cm_analysis = metrics['confusion_matrix_analysis']
            print(f"\n🔍 CONFUSION MATRIX:")
            
            cm = np.array(cm_analysis['confusion_matrix'])
            cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
            print(cm_df.to_string())
            
            print(f"\n   Per-class Accuracy:")
            for class_name, accuracy in cm_analysis['per_class_accuracy'].items():
                print(f"      {class_name}: {accuracy:.4f}")
        
        # 4. BIAS ANALYSIS
        if 'distribution_analysis' in metrics:
            dist_analysis = metrics['distribution_analysis']
            print(f"\n⚖️ BIAS ANALYSIS:")
            print(f"   {'Class':<8} {'True %':>8} {'Pred %':>8} {'Bias':>8} {'Status':>15}")
            print(f"   {'-'*55}")
            
            for class_name in self.class_names:
                if class_name in dist_analysis['bias_analysis']:
                    bias_info = dist_analysis['bias_analysis'][class_name]
                    true_pct = bias_info['true_percentage']
                    pred_pct = bias_info['predicted_percentage']
                    bias = bias_info['bias']
                    status = bias_info['bias_interpretation']
                    
                    print(f"   {class_name:<8} {true_pct:>7.1f}% {pred_pct:>7.1f}% {bias:>+7.1f} {status:>15}")
        
        # 5. NAJCZĘSTSZE BŁĘDY
        if 'confusion_matrix_analysis' in metrics:
            cm_analysis = metrics['confusion_matrix_analysis']
            if cm_analysis['common_errors']:
                print(f"\n❌ NAJCZĘSTSZE BŁĘDY:")
                for i, error in enumerate(cm_analysis['common_errors'][:3], 1):
                    true_class = error['true_class']
                    pred_class = error['predicted_class']
                    count = error['count']
                    rate = error['error_rate']
                    print(f"   {i}. {true_class} → {pred_class}: {count:,} błędów ({rate:.1%})")
        
        print(f"{'='*80}")


def calculate_balanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None,
                              print_report: bool = True) -> Dict[str, any]:
    """
    🎯 FUNKCJA POMOCNICZA - OBLICZ I WYDRUKUJ ZBALANSOWANE METRYKI
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Przewidziane etykiety  
        y_pred_proba: Prawdopodobieństwa (opcjonalne)
        print_report: Czy wydrukować raport
        
    Returns:
        Dict z metrykami
    """
    calculator = BalancedMetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    if print_report and 'error' not in metrics:
        calculator.print_balanced_report(metrics)
    
    return metrics


if __name__ == "__main__":
    # Test z przykładowymi danymi
    print("🧪 TESTING BALANCED METRICS")
    
    # Symulacja niezbalansowanych danych (jak w rzeczywistości)
    np.random.seed(42)
    n_samples = 1000
    
    # Niezbalansowany rozkład: 70% HOLD, 15% SHORT, 15% LONG
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.15, 0.70, 0.15])
    
    # Model który faworyzuje HOLD
    y_pred = np.random.choice([0, 1, 2], size=n_samples, p=[0.10, 0.80, 0.10])
    
    # Test metryk
    metrics = calculate_balanced_metrics(y_true, y_pred, print_report=True)
    
    print(f"\n✅ Balanced metrics test completed!") 