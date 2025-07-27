"""
Moduł treningowy training3 - Multi-Output XGBoost dla 3-klasowych etykiet z labeler3.
"""

__version__ = "1.0.0"
__description__ = "Moduł treningowy Multi-Output XGBoost dla danych z labeler3"

from .config import *
from .utils import setup_logging
from .data_loader import DataLoader
from .model_builder import MultiOutputXGBoost
from .main import Trainer

__all__ = [
    'setup_logging',
    'DataLoader', 
    'MultiOutputXGBoost',
    'Trainer'
] 