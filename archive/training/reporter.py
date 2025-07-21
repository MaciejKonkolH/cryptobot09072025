"""
Moduł odpowiedzialny za generowanie szczegółowego raportu końcowego.
"""
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from training import config as cfg

# Mapowanie nazw klas na indeksy liczbowe dla progów z pliku konfiguracyjnego
CLASS_NAME_TO_INT_MAPPING = {
    'SHORT': 0,
    'HOLD': 1,
    'LONG': 2
}

class TrainingReporter:
    """
    Generuje i wyświetla kompleksowy raport podsumowujący proces treningu.
    """
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
        self.history = trainer_instance.history
        self.test_results = trainer_instance.evaluation_results
        
        # Przechowujemy surowe prawdopodobieństwa do analizy
        self.raw_probabilities = trainer_instance.raw_probabilities
        
        # Pobieramy prawdziwe etykiety bezpośrednio z instancji Trainera
        self.true_labels = self.trainer.true_labels

    def print_summary(self):
        """Drukuje kompletny raport końcowy."""
        print("\n" + "=" * 80)
        print("🎯                           TRENING - RAPORT KOŃCOWY                           🎯")
        print("=" * 80)
        
        self._print_data_summary()
        self._print_model_summary()
        self._print_training_results()
        self._print_class_weights()
        
        # Pętla po zdefiniowanych w configu progach
        for i, thresholds in enumerate(cfg.MULTIPLE_THRESHOLDS_LIST):
            # Konwertujemy klucze-stringi ('SHORT') na klucze-inty (0)
            try:
                int_thresholds = {CLASS_NAME_TO_INT_MAPPING[k.upper()]: v for k, v in thresholds.items()}
            except KeyError as e:
                print(f"\n[BŁĄD] Nieprawidłowy klucz w MULTIPLE_THRESHOLDS_LIST w config.py: {e}. Użyj 'SHORT', 'HOLD', 'LONG'.")
                continue

            self._print_evaluation_summary(i + 1, int_thresholds)

    def _print_data_summary(self):
        """Drukuje podsumowanie danych."""
        start_time = self.trainer.start_time
        end_time = self.trainer.end_time
        total_time = timedelta(seconds=end_time - start_time)
        
        report = [
            "📊 DANE:",
            f"   Plik źródłowy: {cfg.INPUT_FILENAME}",
            f"   Całkowity czas: {str(total_time)}",
            f"   Rozmiary zbiorów:",
            f"     - Treningowy: {self.trainer.train_df.shape}",
            f"     - Walidacyjny: {self.trainer.val_df.shape}",
            f"     - Testowy: {self.trainer.test_df.shape}"
        ]
        print("\n".join(report))

    def _print_model_summary(self):
        """Drukuje podsumowanie dotyczące modelu."""
        print("🧠 MODEL:")
        
        # Pobieramy informacje o architekturze z samego modelu, a nie z configu
        architecture_summary = []
        if hasattr(self.trainer.model, 'layers'):
            for layer in self.trainer.model.layers:
                layer_type = layer.__class__.__name__
                if layer_type not in architecture_summary:
                    architecture_summary.append(layer_type)
        
        architecture_str = " -> ".join(architecture_summary)
        
        lines = [
            f"   Architektura: {architecture_str}",
            f"   Liczba parametrów: {self.trainer.model.count_params():,}",
            f"   Zapisany w: {cfg.MODEL_FILENAME}"
        ]
        # Poprawka: Usunięto wadliwe wywołanie, zastępując je prostą pętlą.
        for line in lines:
            print(line)
        
    def _print_training_results(self):
        """Drukuje wyniki treningu."""
        best_epoch = np.argmin(self.history.history['val_loss']) + 1
        final_val_loss = self.history.history['val_loss'][-1]
        best_val_loss = self.history.history['val_loss'][best_epoch - 1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        best_val_acc = self.history.history['val_accuracy'][best_epoch - 1]

        report = [
            "\n🏋️ WYNIKI TRENINGU:",
            f"   Liczba epok: {len(self.history.epoch)} / {cfg.EPOCHS}",
            f"   Najlepsza epoka (wg val_loss): {best_epoch}",
            f"   Końcowa strata walidacyjna: {final_val_loss:.4f}",
            f"   Najlepsza strata walidacyjna: {best_val_loss:.4f}",
            f"   Końcowa dokładność walidacyjna: {final_val_acc:.4f}",
            f"   Najlepsza dokładność walidacyjna: {best_val_acc:.4f}"
        ]
        print("\n".join(report))

    def _print_class_weights(self):
        """Drukuje informacje o użytych wagach klas i metodzie balansowania."""
        print("\n⚖️ BALANSOWANIE I WAGI KLAS:")
        
        # Sprawdzamy, czy jakakolwiek metoda była włączona
        if not cfg.ENABLE_CLASS_BALANCING and not cfg.ENABLE_CLASS_WEIGHTS:
            print("   Metoda: Brak (trening na niezbalansowanych danych)")
            return

        if cfg.ENABLE_CLASS_BALANCING:
            print("   - Metoda balansowania: Dynamiczny Undersampling (Callback)")
        
        if cfg.ENABLE_CLASS_WEIGHTS:
            weights = self.trainer.class_weight_dict
            if weights:
                # Formatujemy wagi dla czytelnego wyświetlenia
                weights_str = f"S={weights.get(0, 'N/A'):.2f}, H={weights.get(1, 'N/A'):.2f}, L={weights.get(2, 'N/A'):.2f}"
                print(f"   - Metoda wag: Włączona (wagi: {weights_str})")
            else:
                # Ten przypadek nie powinien wystąpić, jeśli logika jest spójna, ale to dobre zabezpieczenie
                print("   - Metoda wag: Włączona, ale wagi nie zostały zdefiniowane")
        else:
            print("   - Metoda wag: Wyłączona w konfiguracji")


    def _apply_thresholding_and_get_metrics(self, thresholds: dict):
        """
        Prywatna metoda do filtrowania predykcji i obliczania metryk dla danych progów.
        """
        # Filtrowanie predykcji
        final_predictions = []
        accepted_indices = []
        
        # Pobieramy maksymalne prawdopodobieństwo i odpowiadającą mu klasę
        predicted_classes = np.argmax(self.raw_probabilities, axis=1)
        max_probabilities = np.max(self.raw_probabilities, axis=1)

        for i, (p_class, p_max) in enumerate(zip(predicted_classes, max_probabilities)):
            if p_max >= thresholds.get(p_class, 0):
                final_predictions.append(p_class)
                accepted_indices.append(i)
        
        final_predictions = np.array(final_predictions)
        accepted_true_labels = self.true_labels[accepted_indices]
        
        # Statystyki thresholdingu
        total_predictions = len(self.raw_probabilities)
        accepted_count = len(final_predictions)
        rejected_count = total_predictions - accepted_count
        acceptance_rate = (accepted_count / total_predictions) * 100 if total_predictions > 0 else 0

        # Obliczenie metryk
        report_dict = classification_report(accepted_true_labels, final_predictions, labels=[0, 1, 2], target_names=['SHORT (0)', 'HOLD (1)', 'LONG (2)'], output_dict=True, zero_division=0)
        cm = confusion_matrix(accepted_true_labels, final_predictions, labels=[0, 1, 2])
        
        return {
            "report_dict": report_dict,
            "confusion_matrix": cm,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "total_predictions": total_predictions,
            "acceptance_rate": acceptance_rate
        }

    def _print_evaluation_summary(self, scenario_num: int, thresholds: dict):
        """Drukuje podsumowanie ewaluacji dla danego scenariusza progów."""
        
        metrics = self._apply_thresholding_and_get_metrics(thresholds)
        
        print("\n" + "-" * 80)
        print(f"🧪 EWALUACJA NA ZBIORZE TESTOWYM (Scenariusz #{scenario_num})")
        print("-" * 80)
        
        print("THRESHOLDING - FILTROWANIE PREDYKCJI:")
        threshold_str = f"SHORT={thresholds.get(0, 'N/A')}, HOLD={thresholds.get(1, 'N/A')}, LONG={thresholds.get(2, 'N/A')}"
        print(f"   Próg pewności: {threshold_str}")
        print(f"   Zaakceptowano: {metrics['accepted_count']:,} / {metrics['total_predictions']:,} ({metrics['acceptance_rate']:.2f}%)")
        print(f"   Odrzucono: {metrics['rejected_count']:,} / {metrics['total_predictions']:,} ({100 - metrics['acceptance_rate']:.2f}%)")

        # --- ZABEZPIECZENIE ---
        # Drukuj szczegółowe metryki tylko jeśli co najmniej jedna predykcja została zaakceptowana.
        if metrics['accepted_count'] > 0:
            print("\nWYNIKI DLA ZAAKCEPTOWANYCH PREDYKCJI:")
            
            report = metrics['report_dict']
            print(f"{'':<12}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}")
            print("-" * 62)
            for name, values in report.items():
                if name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                p, r, f1, s = values['precision'], values['recall'], values['f1-score'], values['support']
                print(f"{name:<12}{p:>12.4f}{r:>12.4f}{f1:>12.4f}{s:>12,}")
            
            print("-" * 62)
            acc = report['accuracy']
            macro_avg = report['macro avg']
            weighted_avg = report['weighted avg']
            
            print(f"{'accuracy':<12}{'':>12}{'':>12}{acc:>12.4f}{report['macro avg']['support']:>12,}")
            print(f"{'macro avg':<12}{macro_avg['precision']:>12.4f}{macro_avg['recall']:>12.4f}{macro_avg['f1-score']:>12.4f}{macro_avg['support']:>12,}")
            print(f"{'weighted avg':<12}{weighted_avg['precision']:>12.4f}{weighted_avg['recall']:>12.4f}{weighted_avg['f1-score']:>12.4f}{weighted_avg['support']:>12,}")

            print("\nMacierz pomyłek (dla zaakceptowanych predykcji):")
            cm = metrics['confusion_matrix']
            print(f"{'':>12}{'SHORT (0)':>12}{'HOLD (1)':>12}{'LONG (2)':>12}")
            print(f"{'SHORT (0)':>12}{cm[0, 0]:>12,}{cm[0, 1]:>12,}{cm[0, 2]:>12,}")
            print(f"{'HOLD (1)':>12}{cm[1, 0]:>12,}{cm[1, 1]:>12,}{cm[1, 2]:>12,}")
            print(f"{'LONG (2)':>12}{cm[2, 0]:>12,}{cm[2, 1]:>12,}{cm[2, 2]:>12,}")
        else:
            print("\nŻadna predykcja nie spełniła progów pewności. Szczegółowe metryki nie są dostępne.")
        
        if scenario_num == len(cfg.MULTIPLE_THRESHOLDS_LIST):
            print("=" * 80)