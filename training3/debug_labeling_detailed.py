"""
Szczegółowy debug algorytmu etykietowania.
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

def debug_labeling_detailed():
    """Szczegółowy debug algorytmu etykietowania."""
    
    logger.info("=== SZCZEGÓŁOWY DEBUG ALGORYTMU ETYKIETOWANIA ===")
    
    # Wczytaj dane
    try:
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
        logger.info(f"Dane wczytane: {len(df):,} wierszy")
    except Exception as e:
        logger.error(f"Błąd wczytywania danych: {e}")
        return
    
    # Analiza dla jednego poziomu
    tp_pct, sl_pct = 0.8, 0.2
    label_col = 'label_tp0p8_sl0p2'
    
    logger.info(f"\n--- DEBUG POZIOMU: TP={tp_pct}%, SL={sl_pct}% ---")
    
    # Pobierz dane OHLC
    ohlc_data = df[['high', 'low', 'close']].values
    labels = df[label_col].values
    
    # Sprawdź rozkład etykiet
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_stats = dict(zip(unique_labels, counts))
    
    logger.info("Rozkład etykiet w danych:")
    for label_num, count in label_stats.items():
        label_name = ['LONG', 'SHORT', 'NEUTRAL'][label_num]
        pct = count / len(labels) * 100
        logger.info(f"  {label_name} ({label_num}): {count:,} ({pct:.1f}%)")
    
    # Sprawdź czy SHORT > LONG
    if 1 in label_stats and 0 in label_stats:
        short_count = label_stats[1]
        long_count = label_stats[0]
        diff = short_count - long_count
        ratio = short_count / long_count if long_count > 0 else float('inf')
        logger.info(f"  SHORT vs LONG: różnica = {diff:,}, stosunek = {ratio:.2f}x")
    
    # Debug: sprawdź konkretne przypadki
    logger.info(f"\n--- DEBUG KONKRETNYCH PRZYPADKÓW ---")
    
    # Znajdź kilka przypadków SHORT i LONG
    short_indices = np.where(labels == 1)[0][:5]  # Pierwsze 5 SHORT
    long_indices = np.where(labels == 0)[0][:5]   # Pierwsze 5 LONG
    
    logger.info(f"Analiza pierwszych 5 przypadków SHORT:")
    for i, idx in enumerate(short_indices):
        if idx + 60 >= len(ohlc_data):
            continue
            
        entry_price = ohlc_data[idx, 2]
        long_tp_price = entry_price * (1 + tp_pct/100)
        long_sl_price = entry_price * (1 - sl_pct/100)
        short_tp_price = entry_price * (1 - tp_pct/100)
        short_sl_price = entry_price * (1 + sl_pct/100)
        
        logger.info(f"  SHORT {i+1} (indeks {idx}):")
        logger.info(f"    Entry: {entry_price:.2f}")
        logger.info(f"    LONG TP: {long_tp_price:.2f}, SL: {long_sl_price:.2f}")
        logger.info(f"    SHORT TP: {short_tp_price:.2f}, SL: {short_sl_price:.2f}")
        
        # Sprawdź przyszłe 60 minut
        long_result = None
        short_result = None
        
        for j in range(60):
            if idx + 1 + j >= len(ohlc_data):
                break
                
            future_high = ohlc_data[idx + 1 + j, 0]
            future_low = ohlc_data[idx + 1 + j, 1]
            
            # Sprawdź pozycję długą
            if long_result is None:
                if future_high >= long_tp_price:
                    long_result = 'TP'
                elif future_low <= long_sl_price:
                    long_result = 'SL'
            
            # Sprawdź pozycję krótką
            if short_result is None:
                if future_low <= short_tp_price:
                    short_result = 'TP'
                elif future_high >= short_sl_price:
                    short_result = 'SL'
            
            if long_result is not None and short_result is not None:
                break
        
        logger.info(f"    Wyniki: LONG={long_result}, SHORT={short_result}")
        
        # Sprawdź czy algorytm się zgadza
        if short_result == 'TP' and long_result != 'TP':
            expected_label = 1  # SHORT
        elif long_result == 'TP' and short_result != 'TP':
            expected_label = 0  # LONG
        elif long_result == 'TP' and short_result == 'TP':
            expected_label = 0  # LONG ma pierwszeństwo
        else:
            expected_label = 2  # NEUTRAL
        
        actual_label = labels[idx]
        logger.info(f"    Oczekiwana etykieta: {expected_label}, Rzeczywista: {actual_label}")
        
        if expected_label != actual_label:
            logger.error(f"    ❌ BŁĄD ALGORYTMU!")
        else:
            logger.info(f"    ✅ Algorytm poprawny")
    
    # Sprawdź czy nie ma błędu w mapowaniu
    logger.info(f"\n--- SPRAWDZENIE MAPOWANIA ---")
    
    # Sprawdź czy wszystkie wartości są poprawne
    invalid_labels = labels[~np.isin(labels, [0, 1, 2])]
    if len(invalid_labels) > 0:
        logger.error(f"Znaleziono nieprawidłowe etykiety: {np.unique(invalid_labels)}")
    else:
        logger.info("✅ Wszystkie etykiety są poprawne (0, 1, 2)")
    
    # Sprawdź czy nie ma błędu w konfiguracji
    logger.info(f"\n--- SPRAWDZENIE KONFIGURACJI ---")
    
    # Sprawdź czy TP/SL są symetryczne
    logger.info(f"TP: {tp_pct}%, SL: {sl_pct}%")
    logger.info(f"LONG TP: +{tp_pct}%, SL: -{sl_pct}%")
    logger.info(f"SHORT TP: -{tp_pct}%, SL: +{sl_pct}%")
    
    # Sprawdź czy poziomy są symetryczne
    if tp_pct == sl_pct * 4:  # TP = 4 * SL
        logger.info("✅ Poziomy są symetryczne")
    else:
        logger.warning("⚠️ Poziomy mogą nie być symetryczne")

if __name__ == "__main__":
    debug_labeling_detailed() 