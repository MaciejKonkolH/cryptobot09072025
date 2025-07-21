"""
Moduł odpowiedzialny za generowanie szczegółowego raportu końcowego dla Multi-Output modelu.
Skupia się na metrykach skuteczności SHORT i LONG zgodnie z wymaganiami użytkownika.
"""
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from training2 import config as cfg

class MultiOutputTrainingReporter:
    """
    Generuje i wyświetla kompleksowy raport podsumowujący proces treningu Multi-Output.
    Skupia się na skuteczności klas PROFIT_SHORT i PROFIT_LONG.
    """
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
        self.evaluation_results = trainer_instance.evaluation_results
        
    def print_summary(self):
        """Drukuje kompletny raport końcowy dla Multi-Output modelu."""
        print("\n" + "=" * 80)
        print("🎯                    MULTI-OUTPUT TRENING - RAPORT KOŃCOWY                    🎯")
        print("=" * 80)
        
        self._print_data_summary()
        self._print_model_summary()
        self._print_multioutput_results()
        self._print_short_long_comparison()
        
        print("=" * 80)
        print("🏁                         KONIEC RAPORTU                                    🏁")
        print("=" * 80)

    def _print_data_summary(self):
        """Drukuje podsumowanie danych."""
        print("\n📊 PODSUMOWANIE DANYCH")
        print("-" * 50)
        print(f"Plik wejściowy: {cfg.INPUT_FILENAME}")
        print(f"Liczba cech: {len(cfg.FEATURES)}")
        print(f"Liczba poziomów TP/SL: {len(cfg.LABEL_COLUMNS)}")
        
        if hasattr(self.trainer, 'X_train'):
            print(f"Dane treningowe: {len(self.trainer.X_train):,} próbek")
            print(f"Dane walidacyjne: {len(self.trainer.X_val):,} próbek") 
            print(f"Dane testowe: {len(self.trainer.X_test):,} próbek")

    def _print_model_summary(self):
        """Drukuje podsumowanie modelu."""
        print("\n🤖 PODSUMOWANIE MODELU")
        print("-" * 50)
        print("Typ modelu: Multi-Output XGBoost")
        print(f"Liczba wyjść: {len(cfg.LABEL_COLUMNS)}")
        print("Poziomy TP/SL:")
        for i, desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            print(f"  {i+1}. {desc}")

    def _print_multioutput_results(self):
        """Drukuje wyniki dla każdego poziomu TP/SL."""
        print("\n📈 WYNIKI DLA KAŻDEGO POZIOMU TP/SL")
        print("=" * 80)
        
        for label_col, results in self.evaluation_results.items():
            level_desc = results['level_desc']
            accuracy = results['accuracy']
            class_report = results['classification_report']
            
            print(f"\n🎯 {level_desc}")
            print("-" * 50)
            print(f"Accuracy: {accuracy:.4f}")
            
            # Wyświetl metryki dla klas PROFIT_SHORT i PROFIT_LONG
            for class_idx in cfg.FOCUS_CLASSES:
                class_name = cfg.CLASS_LABELS[class_idx]
                if str(class_idx) in class_report:
                    metrics = class_report[str(class_idx)]
                    print(f"{class_name:>15}: P={metrics['precision']:.3f}, "
                          f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
            # Wyświetl macierz pomyłek w kompaktowej formie
            conf_matrix = results['confusion_matrix']
            print(f"\nMacierz pomyłek (skrót):")
            print(f"  PROFIT_SHORT: {conf_matrix[0][0]} prawidłowych z {sum(conf_matrix[0])}")
            print(f"  PROFIT_LONG:  {conf_matrix[2][2]} prawidłowych z {sum(conf_matrix[2])}")

    def _print_short_long_comparison(self):
        """Porównuje skuteczność SHORT vs LONG na wszystkich poziomach."""
        print("\n⚖️  PORÓWNANIE SKUTECZNOŚCI SHORT vs LONG")
        print("=" * 80)
        
        # Przygotuj tabelę porównawczą
        comparison_data = []
        
        for label_col, results in self.evaluation_results.items():
            level_desc = results['level_desc']
            class_report = results['classification_report']
            
            short_metrics = class_report.get('0', {})  # PROFIT_SHORT
            long_metrics = class_report.get('2', {})   # PROFIT_LONG
            
            comparison_data.append({
                'Poziom': level_desc,
                'SHORT_Precision': short_metrics.get('precision', 0),
                'SHORT_Recall': short_metrics.get('recall', 0),
                'SHORT_F1': short_metrics.get('f1-score', 0),
                'LONG_Precision': long_metrics.get('precision', 0),
                'LONG_Recall': long_metrics.get('recall', 0),
                'LONG_F1': long_metrics.get('f1-score', 0)
            })
        
        # Wyświetl tabelę
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False, float_format='%.3f'))
        
        # Znajdź najlepszy poziom dla SHORT i LONG
        best_short_idx = df_comparison['SHORT_F1'].idxmax()
        best_long_idx = df_comparison['LONG_F1'].idxmax()
        
        print(f"\n🏆 NAJLEPSZE WYNIKI:")
        print(f"SHORT: {df_comparison.iloc[best_short_idx]['Poziom']} "
              f"(F1={df_comparison.iloc[best_short_idx]['SHORT_F1']:.3f})")
        print(f"LONG:  {df_comparison.iloc[best_long_idx]['Poziom']} "
              f"(F1={df_comparison.iloc[best_long_idx]['LONG_F1']:.3f})")

    def save_detailed_report(self, filepath):
        """Zapisuje szczegółowy raport do pliku."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("MULTI-OUTPUT TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dane podstawowe
            f.write(f"Plik wejściowy: {cfg.INPUT_FILENAME}\n")
            f.write(f"Liczba cech: {len(cfg.FEATURES)}\n")
            f.write(f"Poziomy TP/SL: {len(cfg.LABEL_COLUMNS)}\n\n")
            
            # Wyniki dla każdego poziomu
            for label_col, results in self.evaluation_results.items():
                f.write(f"\n{results['level_desc']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                
                # Szczegółowy raport klasyfikacyjny
                class_report = results['classification_report']
                f.write("\nClassification Report:\n")
                for class_idx, class_name in cfg.CLASS_LABELS.items():
                    if str(class_idx) in class_report:
                        metrics = class_report[str(class_idx)]
                        f.write(f"{class_name}: P={metrics['precision']:.3f}, "
                               f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n")
        
        print(f"Szczegółowy raport zapisany: {filepath}")