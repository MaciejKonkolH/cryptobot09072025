"""
Moduł odpowiedzialny za generowanie szczegółowego raportu końcowego.
"""
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from training import config as cfg

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
        
        # Stosujemy próg pewności (jeśli włączony)
        if cfg.ENABLE_CONFIDENCE_THRESHOLDING:
            self.filtered_predictions, self.threshold_stats = self._apply_confidence_thresholding()
            self.final_predictions = self.filtered_predictions[self.filtered_predictions != -1]
            self.final_true_labels = trainer_instance.true_labels[self.filtered_predictions != -1]
        else:
            self.filtered_predictions = trainer_instance.predictions
            self.threshold_stats = None
            self.final_predictions = trainer_instance.predictions
            self.final_true_labels = trainer_instance.true_labels

    def print_summary(self):
        """Drukuje pełny, sformatowany raport końcowy."""
        print(self._format_header("TRENING - RAPORT KOŃCOWY"))
        print(self._format_section_data())
        print(self._format_section_model())
        print(self._format_section_training_results())
        if self.threshold_stats:
            print(self._format_section_thresholding_stats())
        print(self._format_section_test_evaluation())
        print(self._format_footer())

    def _format_header(self, title: str) -> str:
        """Formatuje główny nagłówek raportu."""
        border = "=" * 80
        return f"\n{border}\n🎯 {title:^76} 🎯\n{border}"

    def _format_footer(self) -> str:
        """Formatuje stopkę raportu."""
        return "=" * 80 + "\n"
    
    def _format_section_data(self) -> str:
        """Formatuje sekcję z informacjami o danych."""
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
        return "\n".join(report)

    def _format_section_model(self) -> str:
        """Formatuje sekcję z informacjami o architekturze modelu."""
        report = [
            "\n🧠 MODEL:",
            f"   Architektura: LSTM {cfg.LSTM_UNITS} + Dense {cfg.DENSE_UNITS}",
            f"   Długość sekwencji: {cfg.SEQUENCE_LENGTH}",
            f"   Dropout: {cfg.DROPOUT_RATE}",
            f"   Learning Rate: {cfg.LEARNING_RATE}"
        ]
        return "\n".join(report)

    def _format_section_training_results(self) -> str:
        """Formatuje sekcję z wynikami treningu."""
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
        return "\n".join(report)

    def _format_section_test_evaluation(self) -> str:
        """Formatuje sekcję z ewaluacją na zbiorze testowym."""
        report = [
            "\n🧪 EWALUACJA NA ZBIORZE TESTOWYM:",
            f"   Strata: {self.test_results['test_loss']:.4f}",
            f"   Dokładność: {self.test_results['test_accuracy']:.4f}",
        ]

        # Generowanie raportu klasyfikacji
        class_names = ['SHORT (0)', 'HOLD (1)', 'LONG (2)']
        class_report = classification_report(
            self.final_true_labels, 
            self.final_predictions, 
            target_names=class_names,
            digits=4
        )
        report.append("\n" + class_report)
        
        # Generowanie macierzy pomyłek
        cm = confusion_matrix(self.final_true_labels, self.final_predictions)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        report.append("\nMacierz pomyłek (dla zaakceptowanych predykcji):")
        report.append(str(cm_df))

        return "\n".join(report)

    def _apply_confidence_thresholding(self) -> tuple[np.ndarray, dict]:
        """Stosuje próg pewności do surowych predykcji modelu."""
        thresholds = np.array([
            cfg.CONFIDENCE_THRESHOLDS.get(0, 0), # SHORT
            cfg.CONFIDENCE_THRESHOLDS.get(1, 0), # HOLD
            cfg.CONFIDENCE_THRESHOLDS.get(2, 0)  # LONG
        ])

        # Pobierz klasę z najwyższym prawdopodobieństwem
        predicted_classes = np.argmax(self.raw_probabilities, axis=1)
        # Pobierz odpowiadające jej prawdopodobieństwo
        max_probabilities = np.max(self.raw_probabilities, axis=1)
        # Pobierz próg dla tej klasy
        class_thresholds = thresholds[predicted_classes]

        # Predykcja jest akceptowana, jeśli jej prawdopodobieństwo >= próg dla tej klasy
        accepted_mask = max_probabilities >= class_thresholds
        
        final_predictions = np.full_like(predicted_classes, -1) # -1 dla odrzuconych
        final_predictions[accepted_mask] = predicted_classes[accepted_mask]

        stats = {
            'total_predictions': len(self.raw_probabilities),
            'accepted': np.sum(accepted_mask),
            'rejected': np.sum(~accepted_mask)
        }
        return final_predictions, stats

    def _format_section_thresholding_stats(self) -> str:
        """Formatuje sekcję ze statystykami filtrowania predykcji."""
        total = self.threshold_stats['total_predictions']
        accepted = self.threshold_stats['accepted']
        rejected = self.threshold_stats['rejected']
        
        report = [
            "\nTHRESHOLDING - FILTROWANIE PREDYKCJI:",
            f"   Próg pewności: SHORT={cfg.CONFIDENCE_THRESHOLDS[0]}, HOLD={cfg.CONFIDENCE_THRESHOLDS[1]}, LONG={cfg.CONFIDENCE_THRESHOLDS[2]}",
            f"   Zaakceptowano: {accepted:,} / {total:,} ({accepted/total:.2%})",
            f"   Odrzucono: {rejected:,} / {total:,} ({rejected/total:.2%})"
        ]
        return "\n".join(report) 