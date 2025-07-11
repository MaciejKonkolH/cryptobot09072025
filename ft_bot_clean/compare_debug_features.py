#!/usr/bin/env python3
"""
Skrypt do porównywania features między modułem walidacji a FreqTrade
w celu zidentyfikowania przyczyn rozbieżności w predykcjach ML.

Autor: System Debug
Data: 2025-07-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format=
'
%(asctime)s - %(levelname)s - %(message)s
'
,
    handlers=[
        logging.FileHandler(f
'
feature_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log
'
),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureComparator:
    """Klasa do porównywania features między różnymi źródłami"""
    
    def __init__(self):
        self.validation_path = Path("../validation_and_labeling/output")
        self.freqtrade_debug_path = Path("user_data/debug_sequences")
        self.results = {}
