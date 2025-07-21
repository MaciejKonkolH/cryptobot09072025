"""
Moduł zawierający generatory danych dla Keras, przystosowane do pracy z wieloma
aktywami (parami walutowymi) jednocześnie.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random

class BaseDataGenerator(tf.keras.utils.Sequence):
    """
    Abstrakcyjna klasa bazowa dla generatorów. Obsługuje wspólną logikę.
    """
    def __init__(self, list_of_dfs: list[pd.DataFrame], features: list[str], 
                 sequence_length: int, batch_size: int):
        self.list_of_dfs = list_of_dfs
        self.features = features
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # Przygotowujemy dane z góry, aby uniknąć powtarzania operacji
        self.data_sources = []
        for df in self.list_of_dfs:
            if len(df) >= self.sequence_length:
                self.data_sources.append({
                    'features': df[self.features].values,
                    'labels': df['label'].values,
                    'num_sequences': len(df) - self.sequence_length + 1
                })

        if not self.data_sources:
            raise ValueError("Żaden z dostarczonych zbiorów danych nie ma wystarczającej długości.")

    def __len__(self):
        """Zwraca całkowitą liczbę batchy na epokę."""
        total_sequences = sum(ds['num_sequences'] for ds in self.data_sources)
        return math.ceil(total_sequences / self.batch_size)

    def _get_sequence(self, source_idx, start_idx):
        """Pobiera pojedynczą sekwencję i etykietę."""
        end_idx = start_idx + self.sequence_length
        
        X = self.data_sources[source_idx]['features'][start_idx:end_idx]
        y = self.data_sources[source_idx]['labels'][end_idx - 1] # Etykieta z ostatniego kroku
        
        return X, y

class TrainingGenerator(BaseDataGenerator):
    """
    Generator do treningu. W każdej iteracji losowo wybiera źródło danych (aktyw),
    a następnie losuje z niego sekwencje do utworzenia batcha.
    Zapewnia to, że model uczy się na zróżnicowanych danych.
        """
    def __init__(self, list_of_dfs: list[pd.DataFrame], features: list[str], 
                 sequence_length: int, batch_size: int):
        super().__init__(list_of_dfs, features, sequence_length, batch_size)
        # Tablica prawdopodobieństw do losowania źródła danych, ważona ich wielkością
        total_sequences = sum(ds['num_sequences'] for ds in self.data_sources)
        self.source_probabilities = [ds['num_sequences'] / total_sequences for ds in self.data_sources]

    def __getitem__(self, index):
        """Generuje jeden, losowy batch danych."""
        batch_X = np.zeros((self.batch_size, self.sequence_length, len(self.features)), dtype=np.float32)
        batch_y = np.zeros(self.batch_size, dtype=np.int32)
        
        for i in range(self.batch_size):
            # 1. Losuj źródło danych (np. BTC lub ETH)
            source_idx = np.random.choice(len(self.data_sources), p=self.source_probabilities)
            
            # 2. Losuj punkt startowy sekwencji w tym źródle
            max_start_idx = self.data_sources[source_idx]['num_sequences'] - 1
            start_idx = random.randint(0, max_start_idx)
            
            batch_X[i], batch_y[i] = self._get_sequence(source_idx, start_idx)
            
        return batch_X, tf.keras.utils.to_categorical(batch_y, num_classes=3)

class ValidationGenerator(BaseDataGenerator):
    """
    Generator do walidacji i testowania. Działa w sposób deterministyczny.
    Przechodzi sekwencyjnie przez wszystkie dane z pierwszego źródła,
    następnie z drugiego itd. Zapewnia to powtarzalność wyników walidacji.
    """
    def __init__(self, list_of_dfs: list[pd.DataFrame], features: list[str], 
                 sequence_length: int, batch_size: int):
        super().__init__(list_of_dfs, features, sequence_length, batch_size)
        
        # Tworzymy jedną, ciągłą listę indeksów (source_idx, sequence_start_idx)
        self.global_indices = []
        for source_idx, source in enumerate(self.data_sources):
            for i in range(source['num_sequences']):
                self.global_indices.append((source_idx, i))

    def __len__(self):
        """Zwraca dokładną liczbę batchy potrzebną do pokrycia wszystkich danych."""
        return math.ceil(len(self.global_indices) / self.batch_size)

    def __getitem__(self, index):
        """Generuje jeden, deterministyczny batch danych."""
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.global_indices))
        
        current_batch_size = end - start
        
        batch_X = np.zeros((current_batch_size, self.sequence_length, len(self.features)), dtype=np.float32)
        batch_y = np.zeros(current_batch_size, dtype=np.int32)

        for i, global_i in enumerate(range(start, end)):
            source_idx, sequence_start_idx = self.global_indices[global_i]
            batch_X[i], batch_y[i] = self._get_sequence(source_idx, sequence_start_idx)

        return batch_X, tf.keras.utils.to_categorical(batch_y, num_classes=3) 