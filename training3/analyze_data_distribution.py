"""
Analiza rozkładu danych w zbiorach train/val/test.
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

def analyze_data_distribution():
    """Analizuje rozkład danych w zbiorach train/val/test."""
    
    logger.info("=== ANALIZA ROZKŁADU DANYCH W ZBIORACH TRAIN/VAL/TEST ===")
    
    # Wczytaj dane
    try:
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
        logger.info(f"Dane wczytane: {len(df):,} wierszy")
    except Exception as e:
        logger.error(f"Błąd wczytywania danych: {e}")
        return
    
    # Ustaw indeks na timestamp jeśli istnieje
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info("Ustawiono timestamp jako indeks")
    
    # Wybierz cechy i etykiety
    X = df[cfg.FEATURES]
    y = df[cfg.LABEL_COLUMNS]
    
    # Usuń wiersze z brakującymi danymi
    initial_rows = len(X)
    mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
    X = X[mask]
    y = y[mask]
    logger.info(f"Po usunięciu brakujących danych: {len(X):,} wierszy")
    
    # Chronologiczny podział danych (identyczny jak w data_loader.py)
    total_samples = len(X)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    # Podział X
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size+val_size]
    X_test = X.iloc[train_size+val_size:]
    
    # Podział y (Multi-output)
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size+val_size]
    y_test = y.iloc[train_size+val_size:]
    
    logger.info(f"\n--- ROZKŁAD DANYCH W ZBIORACH ===")
    logger.info(f"Train: {len(X_train):,} próbek ({len(X_train)/total_samples*100:.1f}%)")
    logger.info(f"Val:   {len(X_val):,} próbek ({len(X_val)/total_samples*100:.1f}%)")
    logger.info(f"Test:  {len(X_test):,} próbek ({len(X_test)/total_samples*100:.1f}%)")
    
    # Zakresy czasowe
    logger.info(f"\n--- ZAKRESY CZASOWE ===")
    logger.info(f"Train: {X_train.index.min()} - {X_train.index.max()}")
    logger.info(f"Val:   {X_val.index.min()} - {X_val.index.max()}")
    logger.info(f"Test:  {X_test.index.min()} - {X_test.index.max()}")
    
    # Analiza rozkładu etykiet w każdym zbiorze
    logger.info(f"\n--- ROZKŁAD ETYKIET W ZBIORACH ===")
    
    for i, label_col in enumerate(cfg.LABEL_COLUMNS):
        level_desc = cfg.TP_SL_LEVELS_DESC[i]
        logger.info(f"\n{level_desc} ({label_col}):")
        
        # Train
        train_counts = y_train[label_col].value_counts().sort_index()
        train_total = len(y_train[label_col])
        logger.info(f"  Train: LONG={train_counts.get(0, 0):,} ({train_counts.get(0, 0)/train_total*100:.1f}%), "
                   f"SHORT={train_counts.get(1, 0):,} ({train_counts.get(1, 0)/train_total*100:.1f}%), "
                   f"NEUTRAL={train_counts.get(2, 0):,} ({train_counts.get(2, 0)/train_total*100:.1f}%)")
        
        # Val
        val_counts = y_val[label_col].value_counts().sort_index()
        val_total = len(y_val[label_col])
        logger.info(f"  Val:   LONG={val_counts.get(0, 0):,} ({val_counts.get(0, 0)/val_total*100:.1f}%), "
                   f"SHORT={val_counts.get(1, 0):,} ({val_counts.get(1, 0)/val_total*100:.1f}%), "
                   f"NEUTRAL={val_counts.get(2, 0):,} ({val_counts.get(2, 0)/val_total*100:.1f}%)")
        
        # Test
        test_counts = y_test[label_col].value_counts().sort_index()
        test_total = len(y_test[label_col])
        logger.info(f"  Test:  LONG={test_counts.get(0, 0):,} ({test_counts.get(0, 0)/test_total*100:.1f}%), "
                   f"SHORT={test_counts.get(1, 0):,} ({test_counts.get(1, 0)/test_total*100:.1f}%), "
                   f"NEUTRAL={test_counts.get(2, 0):,} ({test_counts.get(2, 0)/test_total*100:.1f}%)")
        
        # Sprawdź proporcje SHORT/LONG
        train_ratio = val_counts.get(1, 0) / val_counts.get(0, 1) if val_counts.get(0, 0) > 0 else float('inf')
        val_ratio = val_counts.get(1, 0) / val_counts.get(0, 1) if val_counts.get(0, 0) > 0 else float('inf')
        test_ratio = test_counts.get(1, 0) / test_counts.get(0, 1) if test_counts.get(0, 0) > 0 else float('inf')
        
        logger.info(f"  SHORT/LONG ratio: Train={train_ratio:.3f}, Val={val_ratio:.3f}, Test={test_ratio:.3f}")
        
        # Sprawdź czy są różnice
        ratios = [train_ratio, val_ratio, test_ratio]
        if max(ratios) - min(ratios) > 0.1:  # Różnica większa niż 10%
            logger.warning(f"  ⚠️ Znaczące różnice w proporcjach SHORT/LONG między zbiorami!")
    
    # Analiza data drift
    logger.info(f"\n--- ANALIZA DATA DRIFT ===")
    
    # Sprawdź czy rozkład NEUTRAL się zmienia
    for i, label_col in enumerate(cfg.LABEL_COLUMNS):
        level_desc = cfg.TP_SL_LEVELS_DESC[i]
        
        train_neutral_pct = y_train[label_col].value_counts().get(2, 0) / len(y_train[label_col]) * 100
        val_neutral_pct = y_val[label_col].value_counts().get(2, 0) / len(y_val[label_col]) * 100
        test_neutral_pct = y_test[label_col].value_counts().get(2, 0) / len(y_test[label_col]) * 100
        
        logger.info(f"{level_desc}:")
        logger.info(f"  NEUTRAL %: Train={train_neutral_pct:.1f}%, Val={val_neutral_pct:.1f}%, Test={test_neutral_pct:.1f}%")
        
        # Sprawdź drift
        drift_train_val = abs(train_neutral_pct - val_neutral_pct)
        drift_val_test = abs(val_neutral_pct - test_neutral_pct)
        
        if drift_train_val > 5 or drift_val_test > 5:
            logger.warning(f"  ⚠️ Data drift wykryty! Różnica > 5%")
    
    # Analiza chronologiczna - sprawdź czy nie ma systematycznych zmian
    logger.info(f"\n--- ANALIZA CHRONOLOGICZNA ===")
    
    # Podziel dane na okresy i sprawdź rozkład
    periods = 10
    period_size = len(X) // periods
    
    for i, label_col in enumerate(cfg.LABEL_COLUMNS):
        level_desc = cfg.TP_SL_LEVELS_DESC[i]
        logger.info(f"\n{level_desc} - rozkład w okresach:")
        
        ratios = []
        for p in range(periods):
            start_idx = p * period_size
            end_idx = start_idx + period_size if p < periods - 1 else len(X)
            
            period_y = y[label_col].iloc[start_idx:end_idx]
            period_counts = period_y.value_counts().sort_index()
            
            ratio = period_counts.get(1, 0) / period_counts.get(0, 1) if period_counts.get(0, 0) > 0 else float('inf')
            ratios.append(ratio)
            
            logger.info(f"  Okres {p+1}: SHORT/LONG={ratio:.3f}")
        
        # Sprawdź trend
        if len(ratios) >= 2:
            trend = np.polyfit(range(len(ratios)), ratios, 1)[0]
            logger.info(f"  Trend SHORT/LONG: {trend:.6f} (dodatni = więcej SHORT w czasie)")
            
            if abs(trend) > 0.01:
                logger.warning(f"  ⚠️ Wykryto trend w proporcji SHORT/LONG!")

if __name__ == "__main__":
    analyze_data_distribution() 