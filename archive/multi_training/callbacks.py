"""
Moduł z niestandardowymi callbackami Keras.
"""
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback

class BalancedUndersamplingCallback(Callback):
    """
    Callback, który przed każdą epoką wykonuje undersampling dla klasy
    większościowej (HOLD) i tasuje indeksy treningowe.
    """
    def __init__(self, generator):
        """
        Inicjalizuje callback.

        Args:
            generator: Instancja generatora danych treningowych.
                       Callback będzie modyfikował jego `target_indices`.
        """
        super().__init__()
        self.generator = generator
        # Zapisujemy oryginalne indeksy, aby móc do nich wracać w każdej epoce
        self.original_indices = np.copy(generator.target_indices)
        self.original_labels = np.copy(generator.labels)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Funkcja wywoływana na początku każdej epoki.
        """
        # Pobranie etykiet tylko dla oryginalnych indeksów treningowych
        train_labels = self.original_labels[self.original_indices]

        # Identyfikacja indeksów dla każdej z klas
        hold_indices = self.original_indices[train_labels == 1]
        long_indices = self.original_indices[train_labels == 2]
        short_indices = self.original_indices[train_labels == 0]

        # Ustalenie liczby próbek do wylosowania (średnia z klas mniejszościowych)
        n_samples = int((len(long_indices) + len(short_indices)) / 2)
        
        # Jeśli klasa HOLD jest mniejsza niż docelowa liczba próbek, bierzemy wszystkie
        n_samples = min(n_samples, len(hold_indices))

        # Losowanie indeksów z klasy HOLD
        sampled_hold_indices = np.random.choice(hold_indices, size=n_samples, replace=False)

        # Stworzenie nowej, zbalansowanej puli indeksów
        new_indices = np.concatenate([sampled_hold_indices, long_indices, short_indices])
        
        # Tasowanie indeksów, aby model nie uczył się w określonej kolejności
        np.random.shuffle(new_indices)

        # Podmiana indeksów w generatorze na tę jedną epokę
        self.generator.target_indices = new_indices

        print(f"\nEpoch {epoch+1}: Rebalancing training data. "
              f"New training samples for this epoch: {len(new_indices)}. "
              f"({len(long_indices)} LONG, {len(short_indices)} SHORT, {len(sampled_hold_indices)} HOLD)")

    def on_epoch_end(self, epoch, logs=None):
        """
        Po zakończeniu epoki, przywracamy oryginalne indeksy.
        Nie jest to konieczne, bo `on_epoch_begin` zawsze bazuje na oryginałach,
        ale to dobra praktyka dla zachowania czystości.
        """
        self.generator.target_indices = self.original_indices 