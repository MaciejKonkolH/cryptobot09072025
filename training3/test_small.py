"""
Skrypt testowy do sprawdzenia działania modułu training3 na małej próbce danych.
"""
import sys
import os
from pathlib import Path

# Dodaj ścieżkę projektu do sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from training3 import config as cfg
from training3.utils import setup_logging
from training3.data_loader import DataLoader

logger = setup_logging()

def test_data_loading():
    """Testuje wczytywanie danych."""
    logger.info("=== TEST WCZYTYWANIA DANYCH ===")
    
    try:
        # Wczytaj dane
        data_loader = DataLoader()
        df = data_loader.load_data()
        
        logger.info(f"Dane wczytane pomyślnie: {len(df):,} wierszy, {len(df.columns)} kolumn")
        
        # Sprawdź wymagane kolumny
        required_features = set(cfg.FEATURES)
        required_labels = set(cfg.LABEL_COLUMNS)
        
        available_features = set(df.columns)
        missing_features = required_features - available_features
        missing_labels = required_labels - available_features
        
        if missing_features:
            logger.error(f"Brakuje cech: {missing_features}")
            return False
        
        if missing_labels:
            logger.error(f"Brakuje etykiet: {missing_labels}")
            return False
        
        logger.info(f"Wszystkie wymagane kolumny obecne")
        logger.info(f"   - Cechy: {len(required_features)}")
        logger.info(f"   - Etykiety: {len(required_labels)}")
        
        # Sprawdź rozkład etykiet
        logger.info("Rozkład etykiet:")
        for label_col in cfg.LABEL_COLUMNS:
            unique_labels = df[label_col].value_counts().sort_index()
            logger.info(f"   {label_col}: {unique_labels.to_dict()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania danych: {e}")
        return False

def test_data_preparation():
    """Testuje przygotowanie danych."""
    logger.info("\n=== TEST PRZYGOTOWANIA DANYCH ===")
    
    try:
        # Wczytaj i przygotuj dane
        data_loader = DataLoader()
        df = data_loader.load_data()
        
        # Użyj tylko pierwszych 1000 wierszy do testu
        df_small = df.head(1000)
        logger.info(f"Używam małej próbki: {len(df_small)} wierszy")
        
        data_loader.prepare_data(df_small)
        
        # Pobierz przygotowane dane
        (X_train, X_val, X_test, 
         y_train, y_val, y_test, 
         scaler) = data_loader.get_data()
        
        logger.info(f"Dane przygotowane pomyślnie")
        logger.info(f"   - X_train: {X_train.shape}")
        logger.info(f"   - X_val: {X_val.shape}")
        logger.info(f"   - X_test: {X_test.shape}")
        logger.info(f"   - y_train: {y_train.shape}")
        logger.info(f"   - y_val: {y_val.shape}")
        logger.info(f"   - y_test: {y_test.shape}")
        logger.info(f"   - Scaler: {type(scaler).__name__}")
        
        # Sprawdź czy nie ma NaN
        if X_train.isnull().any().any():
            logger.error("Znaleziono NaN w X_train")
            return False
        
        if y_train.isnull().any().any():
            logger.error("Znaleziono NaN w y_train")
            return False
        
        logger.info("Brak wartości NaN w danych")
        
        # Sprawdź zakresy etykiet
        for col in y_train.columns:
            unique_vals = y_train[col].unique()
            if not all(val in [0, 1, 2] for val in unique_vals):
                logger.error(f"Nieprawidłowe wartości w {col}: {unique_vals}")
                return False
        
        logger.info("Wszystkie etykiety w zakresie [0, 1, 2]")
        
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas przygotowania danych: {e}")
        return False

def test_model_building():
    """Testuje budowanie modelu."""
    logger.info("\n=== TEST BUDOWANIA MODELU ===")
    
    try:
        from training3.model_builder import MultiOutputXGBoost
        
        # Utwórz model
        model = MultiOutputXGBoost()
        xgb_model = model.build_model()
        
        logger.info(f"Model zbudowany pomyślnie: {type(xgb_model).__name__}")
        
        # Sprawdź parametry
        params = xgb_model.get_params()
        logger.info(f"   - n_estimators: {params.get('n_estimators')}")
        logger.info(f"   - learning_rate: {params.get('learning_rate')}")
        logger.info(f"   - max_depth: {params.get('max_depth')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas budowania modelu: {e}")
        return False

def main():
    """Uruchamia wszystkie testy."""
    logger.info("ROZPOCZYNAM TESTY MODUŁU TRAINING3")
    logger.info("=" * 50)
    
    tests = [
        ("Wczytywanie danych", test_data_loading),
        ("Przygotowanie danych", test_data_preparation),
        ("Budowanie modelu", test_model_building)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Błąd w teście '{test_name}': {e}")
            results.append((test_name, False))
    
    # Podsumowanie
    logger.info("\n" + "=" * 50)
    logger.info("PODSUMOWANIE TESTÓW")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nWynik: {passed}/{total} testów przeszło")
    
    if passed == total:
        logger.info("WSZYSTKIE TESTY PRZESZŁY! Moduł gotowy do użycia.")
        return True
    else:
        logger.error("NIEKTÓRE TESTY NIE PRZESZŁY. Sprawdź błędy powyżej.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 