"""
Moduł do wczytywania i przygotowania danych z labeler3.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from training3 import config as cfg
from training3.utils import balance_classes

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Klasa odpowiedzialna za wczytywanie i przygotowanie danych z labeler3.
    """
    
    def __init__(self):
        """Inicjalizuje loader danych."""
        self.scaler = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Wczytuje dane z pliku wyjściowego labeler3.
        
        Returns:
            pd.DataFrame: Dane z cechami i etykietami
        """
        logger.info(f"Wczytywanie danych z: {cfg.INPUT_FILE_PATH}")
        
        if not cfg.INPUT_FILE_PATH.exists():
            raise FileNotFoundError(f"Plik wejściowy nie istnieje: {cfg.INPUT_FILE_PATH}")
        
        # Wczytaj dane
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
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
        
        # Sprawdź wymagane kolumny
        required_cols = set(cfg.FEATURES + cfg.LABEL_COLUMNS)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Brakuje wymaganych kolumn: {missing_cols}")
        
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
        
        # Wybierz cechy i etykiety
        X = df[cfg.FEATURES]
        y = df[cfg.LABEL_COLUMNS]  # Multi-output: 5 kolumn etykiet
        
        self.feature_names = cfg.FEATURES
        logger.info(f"Liczba cech: {len(self.feature_names)}")
        logger.info(f"Liczba wyjść (poziomów TP/SL): {len(cfg.LABEL_COLUMNS)}")
        
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