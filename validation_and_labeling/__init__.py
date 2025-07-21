"""
Moduł validation_and_labeling

Moduł do przekształcania surowych danych OHLCV w gotowe do ML datasety
z 8 features technicznymi i etykietami competitive labeling.

Główne klasy:
- ValidationAndLabelingPipeline: Główny pipeline przetwarzania
- DataValidator: Walidacja i wypełnianie luk w danych
- FeatureCalculator: Obliczanie features technicznych
- CompetitiveLabeler: Algorytm competitive labeling
- FeatureQualityValidator: Walidacja jakości features

Uruchomienie:
    python main.py

Konfiguracja:
    Wszystkie parametry w pliku config.py
"""

__version__ = "1.0.0"
__author__ = "validation_and_labeling_module"

# Import głównych klas dla łatwego dostępu
from .main import ValidationAndLabelingPipeline, main
from .data_validator import DataValidator
from .feature_calculator import FeatureCalculator, FeatureQualityValidator
from .competitive_labeler import CompetitiveLabeler, LabelingStatistics
from . import config

# Definicja publicznego API
__all__ = [
    'ValidationAndLabelingPipeline',
    'DataValidator',
    'FeatureCalculator',
    'FeatureQualityValidator',
    'CompetitiveLabeler',
    'LabelingStatistics',
    'config',
    'main'
] 