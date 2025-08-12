"""
Moduł do wczytywania i przygotowania danych z labeler4.
Dostosowany do nowego pipeline'u z feature_calculator_download2 i labeler4.
Obsługuje dane dla pojedynczej pary i automatyczne wykrywanie cech.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from training4 import config as cfg

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Klasa odpowiedzialna za wczytywanie i przygotowanie danych z labeler4.
    Dostosowana do obsługi wielu par z nowego pipeline'u.
    """
    
    def __init__(self, symbol=None):
        """
        Inicjalizuje loader danych.
        
        Args:
            symbol: Symbol pary (np. 'ETHUSDT'). Jeśli None, używa domyślnego z konfiguracji.
        """
        self.symbol = symbol
        self.scaler = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.available_features = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Wczytuje dane z pliku wyjściowego labeler4.
        
        Returns:
            pd.DataFrame: Dane z cechami i etykietami
        """
        # Określ ścieżkę do pliku
        if self.symbol:
            input_path = cfg.get_input_file_path(self.symbol)
        else:
            # Fallback - użyj pierwszej pary z konfiguracji
            input_path = cfg.get_input_file_path(cfg.PAIRS[0])
            logger.warning(f"Nie podano symbolu, używam domyślnej pary: {cfg.PAIRS[0]}")
        
        logger.info(f"Wczytywanie danych z: {input_path}")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Plik wejściowy nie istnieje: {input_path}")
        
        # Wczytaj dane
        df = pd.read_feather(input_path)
        logger.info(f"Wczytano: {len(df):,} wierszy, {len(df.columns)} kolumn")
        
        # Ustaw indeks na timestamp jeśli istnieje
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            logger.info("Ustawiono timestamp jako indeks")
        
        # Filtrowanie po dacie jeśli włączone
        if cfg.ENABLE_DATE_FILTER:
            logger.info(f"Filtrowanie danych do zakresu: {cfg.START_DATE} - {cfg.END_DATE}")
            df = df[(df.index >= pd.to_datetime(cfg.START_DATE)) & 
                   (df.index <= pd.to_datetime(cfg.END_DATE))]
            logger.info(f"Po filtrowaniu: {len(df):,} wierszy")
        
        # Automatyczne wykrywanie dostępnych cech
        self.available_features = cfg.detect_available_features(df.columns)
        logger.info(f"Wykryto {len(self.available_features)} dostępnych cech z {len(cfg.FEATURES)} oczekiwanych")
        
        # Sprawdź wymagane kolumny (etykiety)
        missing_label_cols = [col for col in cfg.LABEL_COLUMNS if col not in df.columns]
        if missing_label_cols:
            raise ValueError(f"Brakuje wymaganych kolumn etykiet: {missing_label_cols}")
        
        # Loguj informacje o etykietach
        logger.info("Rozkład etykiet:")
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            unique_labels = df[label_col].value_counts().sort_index()
            logger.info(f"  {cfg.TP_SL_LEVELS_DESC[i]} ({label_col}): {unique_labels.to_dict()}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame):
        """
        Przygotowuje dane do treningu: podział, skalowanie, balansowanie.
        
        Args:
            df: DataFrame z cechami i etykietami
        """
        logger.info("Przygotowywanie danych do treningu...")
        
        # Wybierz dostępne cechy i etykiety
        X = df[self.available_features]
        y = df[cfg.LABEL_COLUMNS]  # Multi-output: 15 kolumn etykiet
        
        self.feature_names = self.available_features
        logger.info(f"Liczba cech: {len(self.feature_names)}")
        logger.info(f"Liczba wyjść (poziomów TP/SL): {len(cfg.LABEL_COLUMNS)}")
        
        # Loguj grupy cech
        feature_groups = cfg.get_feature_groups()
        for group_name, group_features in feature_groups.items():
            available_in_group = [f for f in group_features if f in self.available_features]
            logger.info(f"  {group_name}: {len(available_in_group)}/{len(group_features)} cech")
        
        # Usuń wiersze z brakującymi danymi
        initial_rows = len(X)
        mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
        X = X[mask]
        y = y[mask]
        logger.info(f"Usunięto {initial_rows - len(X)} wierszy z brakującymi danymi")
        
        # Chronologiczny podział danych
        total_samples = len(X)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        # Podział X
        self.X_train = X.iloc[:train_size]
        self.X_val = X.iloc[train_size:train_size+val_size]
        self.X_test = X.iloc[train_size+val_size:]
        
        # Podział y (Multi-output)
        self.y_train = y.iloc[:train_size]
        self.y_val = y.iloc[train_size:train_size+val_size]
        self.y_test = y.iloc[train_size+val_size:]
        
        logger.info(f"Podział danych (chronologiczny):")
        logger.info(f"  Trening: {len(self.X_train):,} próbek")
        logger.info(f"  Walidacja: {len(self.X_val):,} próbek")
        logger.info(f"  Test: {len(self.X_test):,} próbek")
        
        # Sprawdź chronologię
        if hasattr(self.X_train, 'index'):
            logger.info("Zakresy czasowe:")
            logger.info(f"  Train: {self.X_train.index.min()} - {self.X_train.index.max()}")
            logger.info(f"  Val:   {self.X_val.index.min()} - {self.X_val.index.max()}")
            logger.info(f"  Test:  {self.X_test.index.min()} - {self.X_test.index.max()}")
        
        # Skalowanie cech
        self._scale_features()
        
        # Balansowanie klas jeśli włączone
        if cfg.ENABLE_CLASS_BALANCING:
            self._balance_classes()
        
        logger.info("Dane przygotowane pomyślnie.")
    
    def _scale_features(self):
        """Skaluje cechy używając RobustScaler."""
        logger.info("Skalowanie cech...")
        
        # Czyszczenie danych - zastąp inf/-inf wartościami NaN
        logger.info("Czyszczenie danych - usuwanie inf/-inf...")
        
        # Sprawdź i wyczyść dane treningowe
        inf_count_train = np.isinf(self.X_train).sum().sum()
        if inf_count_train > 0:
            logger.info(f"Znaleziono {inf_count_train} wartości inf/-inf w danych treningowych")
            self.X_train = self.X_train.replace([np.inf, -np.inf], np.nan)
        
        # Sprawdź i wyczyść dane walidacyjne
        inf_count_val = np.isinf(self.X_val).sum().sum()
        if inf_count_val > 0:
            logger.info(f"Znaleziono {inf_count_val} wartości inf/-inf w danych walidacyjnych")
            self.X_val = self.X_val.replace([np.inf, -np.inf], np.nan)
        
        # Sprawdź i wyczyść dane testowe
        inf_count_test = np.isinf(self.X_test).sum().sum()
        if inf_count_test > 0:
            logger.info(f"Znaleziono {inf_count_test} wartości inf/-inf w danych testowych")
            self.X_test = self.X_test.replace([np.inf, -np.inf], np.nan)
        
        # Wypełnij NaN medianą z danych treningowych
        if inf_count_train > 0 or inf_count_val > 0 or inf_count_test > 0:
            logger.info("Wypełnianie NaN medianą z danych treningowych...")
            train_median = self.X_train.median()
            
            self.X_train = self.X_train.fillna(train_median)
            self.X_val = self.X_val.fillna(train_median)
            self.X_test = self.X_test.fillna(train_median)
        
        self.scaler = RobustScaler()
        
        # Dopasuj skaler na danych treningowych
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        
        # Transformuj dane walidacyjne i testowe
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_names,
            index=self.X_val.index
        )
        
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        logger.info("Cechy przeskalowane za pomocą RobustScaler.")
    
    def _balance_classes(self):
        """Balansuje klasy w zbiorze treningowym."""
        logger.info("Balansowanie klas...")
        
        # Balansuj tylko na pierwszym poziomie TP/SL
        first_label_col = cfg.LABEL_COLUMNS[0]
        logger.info(f"Balansowanie na podstawie poziomu: {cfg.TP_SL_LEVELS_DESC[0]}")
        
        # Pobierz etykiety pierwszego poziomu
        y_first = self.y_train[first_label_col]
        
        # Znajdź indeksy dla każdej klasy
        class_indices = {}
        for class_label in [0, 1, 2]:  # LONG, SHORT, NEUTRAL
            class_indices[class_label] = y_first[y_first == class_label].index.tolist()
        
        # Sprawdź czy mamy wystarczająco próbek dla balansowania
        min_samples_per_class = 10  # Minimalna liczba próbek na klasę
        minority_classes = [cls for cls in [0, 1, 2] if len(class_indices[cls]) < len(class_indices[max(class_indices.keys(), key=lambda x: len(class_indices[x]))])]
        
        # Jeśli któraś klasa ma za mało próbek, nie balansuj
        if any(len(class_indices[cls]) < min_samples_per_class for cls in [0, 1, 2]):
            logger.info(f"Za mało próbek dla balansowania (minimum {min_samples_per_class} na klasę). Pomijam balansowanie.")
            return
        
        # Znajdź klasę większościową
        majority_class = max(class_indices.keys(), key=lambda x: len(class_indices[x]))
        minority_classes = [cls for cls in [0, 1, 2] if cls != majority_class]
        
        # Oblicz docelową liczbę próbek dla klasy większościowej
        minority_avg = np.mean([len(class_indices[cls]) for cls in minority_classes])
        target_majority = int(minority_avg * cfg.CLASS_WEIGHTS[majority_class])
        target_majority = min(target_majority, len(class_indices[majority_class]))
        
        # Upewnij się, że mamy wystarczająco próbek
        if target_majority < min_samples_per_class:
            logger.info(f"Docelowa liczba próbek ({target_majority}) jest za mała. Pomijam balansowanie.")
            return
        
        # Losuj próbki z klasy większościowej
        sampled_majority_indices = np.random.choice(
            class_indices[majority_class], 
            size=target_majority, 
            replace=False
        ).tolist()
        
        # Przygotuj wszystkie wybrane indeksy
        selected_indices = []
        for class_label in [0, 1, 2]:
            if class_label == majority_class:
                selected_indices.extend(sampled_majority_indices)
            else:
                # Dla klas mniejszościowych bierzemy wszystkie próbki
                selected_indices.extend(class_indices[class_label])
        
        # Przetasuj indeksy
        np.random.shuffle(selected_indices)
        
        # Wybierz dane po zbalansowanych indeksach
        self.X_train = self.X_train.loc[selected_indices].reset_index(drop=True)
        
        # Wybierz wszystkie etykiety po tych samych indeksach
        self.y_train = self.y_train.loc[selected_indices].reset_index(drop=True)
        
        logger.info(f"Po balansowaniu: {len(self.X_train)} próbek treningowych")
        
        # Loguj nowy rozkład klas dla każdego poziomu
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            unique_labels = self.y_train[label_col].value_counts().sort_index()
            logger.info(f"  {cfg.TP_SL_LEVELS_DESC[i]}: {unique_labels.to_dict()}")
    
    def get_data(self):
        """
        Zwraca przygotowane dane.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        """
        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
            self.scaler
        )
    
    def get_feature_info(self):
        """
        Zwraca informacje o cechach.
        
        Returns:
            dict: Informacje o cechach
        """
        return {
            'feature_names': self.feature_names,
            'available_features': self.available_features,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_groups': cfg.get_feature_groups()
        } 