"""
Moduł treningowy (training4).
Dostosowany do nowego pipeline'u z feature_calculator_download2 i labeler4.
Obsługuje trening dla pojedynczej pary i wszystkich par na raz.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Moduł treningowy Multi-Output XGBoost dla nowego pipeline'u"

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