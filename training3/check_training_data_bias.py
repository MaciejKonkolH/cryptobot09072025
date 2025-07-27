"""
Skrypt do sprawdzenia czy nie ma błędu w danych treningowych faworyzującego SHORT.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Dodaj ścieżkę do modułu
sys.path.append(str(Path(__file__).parent))
import config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_bias():
    """Sprawdza czy nie ma błędu w danych faworyzującego SHORT."""
    
    logger.info("=== SPRAWDZANIE BIAS W DANYCH TRENINGOWYCH ===")
    
    # Wczytaj dane
    try:
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
        logger.info(f"Dane wczytane: {len(df):,} wierszy")
    except Exception as e:
        logger.error(f"Błąd wczytywania danych: {e}")
        return
    
    # Sprawdź kolumny z etykietami
    label_cols = cfg.LABEL_COLUMNS
    logger.info(f"Kolumny z etykietami: {label_cols}")
    
    # Sprawdź rozkład dla każdego poziomu
    for i, col in enumerate(label_cols):
        if col not in df.columns:
            logger.error(f"Brak kolumny: {col}")
            continue
            
        level_desc = cfg.TP_SL_LEVELS_DESC[i]
        logger.info(f"\n--- POZIOM {i+1}: {level_desc} ---")
        
        # Rozkład klas
        class_counts = df[col].value_counts().sort_index()
        total = len(df)
        
        logger.info(f"Rozkład klas:")
        logger.info(f"  LONG (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.2f}%)")
        logger.info(f"  SHORT (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.2f}%)")
        logger.info(f"  NEUTRAL (2): {class_counts.get(2, 0):,} ({class_counts.get(2, 0)/total*100:.2f}%)")
        
        # Sprawdź czy SHORT > LONG
        long_count = class_counts.get(0, 0)
        short_count = class_counts.get(1, 0)
        
        if short_count > long_count:
            diff = short_count - long_count
            ratio = short_count / long_count if long_count > 0 else float('inf')
            logger.warning(f"  ⚠️ SHORT > LONG: różnica = {diff:,}, stosunek = {ratio:.2f}x")
        else:
            diff = long_count - short_count
            ratio = long_count / short_count if short_count > 0 else float('inf')
            logger.info(f"  ✅ LONG >= SHORT: różnica = {diff:,}, stosunek = {ratio:.2f}x")
    
    # Sprawdź czy nie ma błędów w mapowaniu
    logger.info(f"\n--- SPRAWDZANIE MAPOWANIA KLAS ---")
    logger.info(f"Mapowanie z labeler3: {cfg.CLASS_LABELS}")
    
    # Sprawdź czy wszystkie wartości są poprawne
    for col in label_cols:
        if col in df.columns:
            unique_values = sorted(df[col].unique())
            logger.info(f"{col}: unikalne wartości = {unique_values}")
            
            if not all(val in [0, 1, 2] for val in unique_values):
                logger.error(f"  ❌ Nieprawidłowe wartości w {col}: {unique_values}")
            else:
                logger.info(f"  ✅ Wszystkie wartości poprawne")

if __name__ == "__main__":
    check_data_bias() 