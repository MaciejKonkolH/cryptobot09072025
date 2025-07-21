"""
Hybrydowy generator danych sekwencyjnych.
"""
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence, to_categorical
from training import config as cfg

class DataGenerator(Sequence):
    """
    Generator, który wykonuje "hybrydowe" przygotowanie danych i opcjonalnie
    zarządza dynamicznym balansowaniem klas.
    1.  Przyjmuje gotowe, przeskalowane globalnie cechy statyczne.
    2.  Przyjmuje surowe dane OHLC.
    3.  W locie, dla każdej sekwencji, oblicza znormalizowane cechy `*_norm`.
    4.  Łączy oba zestawy, tworząc finalny tensor wejściowy dla modelu.
    5.  Na koniec każdej epoki (jeśli włączone), balansuje zbiór poprzez
        undersampling klasy większościowej i tasuje dane.
    """

    def __init__(self, scaled_static_features: np.ndarray, raw_dynamic_features: np.ndarray,
                 labels: np.ndarray, target_indices: np.ndarray, batch_size: int,
                 dynamic_feature_names: list, class_balancing: bool = False,
                 debug_save_sequence: bool = False):
        """
        Inicjalizuje generator hybrydowy.

        Args:
            scaled_static_features (np.ndarray): Macierz przeskalowanych cech statycznych.
            raw_dynamic_features (np.ndarray): Macierz surowych danych OHLC.
            labels (np.ndarray): Wektor wszystkich etykiet.
            target_indices (np.ndarray): Indeksy celów predykcji dla tego generatora.
            batch_size (int): Rozmiar paczki danych.
            dynamic_feature_names (list): Lista nazw surowych kolumn (OHLC).
            class_balancing (bool): Czy włączyć dynamiczne balansowanie klas.
            debug_save_sequence (bool): Czy zapisać pierwszą sekwencję do pliku CSV.
        """
        self.static_features = scaled_static_features
        self.dynamic_features = raw_dynamic_features
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = cfg.SEQUENCE_LENGTH
        self.class_balancing = class_balancing
        
        # Flagi do debugowania
        self.debug_save_sequence = debug_save_sequence
        
        # Przechowujemy oryginalne indeksy, aby mieć z czego próbkować w każdej epoce
        self.original_target_indices = np.copy(target_indices)
        # Ten atrybut będzie modyfikowany w każdej epoce przez on_epoch_end
        self.target_indices = np.copy(target_indices)

        # Dynamiczne i bezpieczne znajdowanie indeksów kolumn OHLC
        self.open_idx = dynamic_feature_names.index('open')
        self.high_idx = dynamic_feature_names.index('high')
        self.low_idx = dynamic_feature_names.index('low')
        self.close_idx = dynamic_feature_names.index('close')
        
        # Uruchamiamy raz na starcie, aby zainicjować stan (głównie w celu potasowania)
        self.on_epoch_end()

    def __len__(self) -> int:
        """Zwraca liczbę paczek (batchy) na epokę."""
        return int(np.ceil(len(self.target_indices) / self.batch_size))

    def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generuje jedną paczkę (batch) danych, wykonując normalizację w locie.
        Ta wersja jest zoptymalizowana pod kątem zużycia pamięci.
        """
        start_index = batch_idx * self.batch_size
        end_index = (batch_idx + 1) * self.batch_size
        batch_target_indices = self.target_indices[start_index:end_index]
        
        # Określenie rzeczywistego rozmiaru batcha (ważne dla ostatniego, mniejszego batcha)
        actual_batch_size = len(batch_target_indices)

        # Pre-alokacja pustych tablic NumPy dla batcha - klucz do oszczędności pamięci
        batch_X = np.empty((actual_batch_size, self.sequence_length, cfg.NUM_FEATURES), dtype=np.float32)
        batch_y = np.empty(actual_batch_size, dtype=np.int32)

        for i, target_idx in enumerate(batch_target_indices):
            seq_start = target_idx - self.sequence_length + 1
            seq_end = target_idx + 1
            
            dynamic_sequence = self.dynamic_features[seq_start:seq_end]
            
            open_anchor = dynamic_sequence[0, self.open_idx]
            if open_anchor == 0:
                open_anchor = 1e-9

            open_norm = ((dynamic_sequence[:, self.open_idx] - open_anchor) / open_anchor) * 100
            high_norm = ((dynamic_sequence[:, self.high_idx] - open_anchor) / open_anchor) * 100
            low_norm = ((dynamic_sequence[:, self.low_idx] - open_anchor) / open_anchor) * 100
            close_norm = ((dynamic_sequence[:, self.close_idx] - open_anchor) / open_anchor) * 100

            calculated_dynamic_features = np.stack([open_norm, high_norm, low_norm, close_norm], axis=1)
            static_sequence = self.static_features[seq_start:seq_end]

            batch_X[i] = np.concatenate([static_sequence, calculated_dynamic_features], axis=1)
            batch_y[i] = self.labels[target_idx]

        return batch_X, to_categorical(batch_y, num_classes=cfg.NUM_CLASSES)

    def _save_debug_sequence(self, sequence_data: np.ndarray, target_idx: int):
        """Zapisuje jedną, kompletną sekwencję do pliku CSV w celach diagnostycznych."""
        try:
            # Tworzenie DataFrame z finalnymi danymi, które trafiają do modelu
            column_names = cfg.STATIC_FEATURES + cfg.DYNAMIC_FEATURES_OUTPUT
            df = pd.DataFrame(sequence_data, columns=column_names)
            
            # Przygotowanie ścieżki zapisu
            output_path = os.path.join(cfg.REPORT_DIR, f"debug_sequence_target_{target_idx}.csv")
            os.makedirs(cfg.REPORT_DIR, exist_ok=True)
            
            # Zapis do CSV
            df.to_csv(output_path, index=False, float_format='%.8f')
            print(f"\n[DEBUG] Zapisano przykładową sekwencję do pliku: {output_path}")

        except Exception as e:
            print(f"\n[DEBUG] Błąd podczas zapisywania sekwencji: {e}")

    def on_epoch_end(self):
        """
        Metoda wywoływana na koniec każdej epoki przez Keras.
        Odpowiada za zbalansowanie i przetasowanie danych na następną epokę.
        """
        if self.class_balancing:
            # Pobieramy etykiety odpowiadające oryginalnym indeksom dla tego zbioru
            current_labels = self.labels[self.original_target_indices]

            # Identyfikujemy oryginalne indeksy dla każdej z klas
            hold_indices = self.original_target_indices[current_labels == 1]
            long_indices = self.original_target_indices[current_labels == 2]
            short_indices = self.original_target_indices[current_labels == 0]

            # Ustalamy docelową liczbę próbek dla klasy większościowej
            n_samples = int((len(long_indices) + len(short_indices)) / 2)
            n_samples = min(n_samples, len(hold_indices))

            # Losujemy 'n_samples' z klasy 'HOLD'
            sampled_hold_indices = np.random.choice(hold_indices, size=n_samples, replace=False)

            # Łączymy zbalansowaną klasę 'HOLD' z pozostałymi klasami
            self.target_indices = np.concatenate([sampled_hold_indices, long_indices, short_indices])
            
            print(f"\nRebalancing complete. New training samples for next epoch: {len(self.target_indices)}. "
                  f"({len(long_indices)} LONG, {len(short_indices)} SHORT, {len(sampled_hold_indices)} HOLD)")
        
            # Tasujemy dane TYLKO dla zbioru treningowego, aby poprawić generalizację.
            np.random.shuffle(self.target_indices)
        else:
            # DLA ZBIORÓW WALIDACYJNEGO/TESTOWEGO: Upewniamy się, że używamy oryginalnych, niepotasowanych indeksów,
            # aby zachować chronologiczną kolejność.
            self.target_indices = np.copy(self.original_target_indices) 