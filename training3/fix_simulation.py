"""
Poprawiona symulacja algorytmu etykietowania.
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

def fix_simulation():
    """Poprawiona symulacja algorytmu etykietowania."""
    
    logger.info("=== POPRAWIONA SYMULACJA ALGORYTMU ===")
    
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
    
    logger.info(f"\n--- SYMULACJA POZIOMU: TP={tp_pct}%, SL={sl_pct}% ---")
    
    # Pobierz dane OHLC
    ohlc_data = df[['high', 'low', 'close']].values
    labels = df[label_col].values
    
    # Sprawdź rozkład rzeczywistych etykiet
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_stats = dict(zip(unique_labels, counts))
    
    logger.info("Rzeczywisty rozkład etykiet:")
    for label_num, count in label_stats.items():
        label_name = ['LONG', 'SHORT', 'NEUTRAL'][label_num]
        pct = count / len(labels) * 100
        logger.info(f"  {label_name} ({label_num}): {count:,} ({pct:.1f}%)")
    
    # Poprawiona symulacja - sprawdź więcej próbek
    sample_size = min(10000, len(ohlc_data) - 60)  # Zwiększ próbkę
    
    logger.info(f"\nSymulacja na {sample_size:,} próbkach:")
    
    long_count = 0
    short_count = 0
    neutral_count = 0
    
    # Sprawdź czy symulacja się zgadza z rzeczywistością
    correct_predictions = 0
    
    for i in range(sample_size):
        if i + 60 >= len(ohlc_data):
            break
            
        entry_price = ohlc_data[i, 2]
        long_tp_price = entry_price * (1 + tp_pct/100)
        long_sl_price = entry_price * (1 - sl_pct/100)
        short_tp_price = entry_price * (1 - tp_pct/100)
        short_sl_price = entry_price * (1 + sl_pct/100)
        
        long_result = None
        short_result = None
        
        for j in range(60):
            if i + 1 + j >= len(ohlc_data):
                break
                
            future_high = ohlc_data[i + 1 + j, 0]
            future_low = ohlc_data[i + 1 + j, 1]
            
            # LONG: TP przed SL
            if long_result is None:
                if future_high >= long_tp_price:
                    long_result = 'TP'
                elif future_low <= long_sl_price:
                    long_result = 'SL'
            
            # SHORT: TP przed SL
            if short_result is None:
                if future_low <= short_tp_price:
                    short_result = 'TP'
                elif future_high >= short_sl_price:
                    short_result = 'SL'
            
            if long_result is not None and short_result is not None:
                break
        
        # Logika decyzyjna
        if long_result == 'TP' and short_result != 'TP':
            simulated_label = 0  # LONG
            long_count += 1
        elif short_result == 'TP' and long_result != 'TP':
            simulated_label = 1  # SHORT
            short_count += 1
        elif long_result == 'TP' and short_result == 'TP':
            simulated_label = 0  # LONG ma pierwszeństwo
            long_count += 1
        else:
            simulated_label = 2  # NEUTRAL
            neutral_count += 1
        
        # Sprawdź czy symulacja się zgadza z rzeczywistością
        actual_label = labels[i]
        if simulated_label == actual_label:
            correct_predictions += 1
    
    total_samples = long_count + short_count + neutral_count
    logger.info(f"Symulacja:")
    logger.info(f"  LONG: {long_count} ({long_count/total_samples*100:.1f}%)")
    logger.info(f"  SHORT: {short_count} ({short_count/total_samples*100:.1f}%)")
    logger.info(f"  NEUTRAL: {neutral_count} ({neutral_count/total_samples*100:.1f}%)")
    
    accuracy = correct_predictions / total_samples * 100
    logger.info(f"  Dokładność symulacji: {accuracy:.1f}%")
    
    if accuracy < 95:
        logger.warning(f"⚠️ Symulacja nie zgadza się z rzeczywistością! Dokładność: {accuracy:.1f}%")
        
        # Sprawdź różnice
        logger.info(f"\n--- ANALIZA RÓŻNIC ---")
        
        # Sprawdź kilka przypadków gdzie symulacja się nie zgadza
        mismatch_count = 0
        for i in range(min(1000, sample_size)):
            if i + 60 >= len(ohlc_data):
                break
                
            entry_price = ohlc_data[i, 2]
            long_tp_price = entry_price * (1 + tp_pct/100)
            long_sl_price = entry_price * (1 - sl_pct/100)
            short_tp_price = entry_price * (1 - tp_pct/100)
            short_sl_price = entry_price * (1 + sl_pct/100)
            
            long_result = None
            short_result = None
            
            for j in range(60):
                if i + 1 + j >= len(ohlc_data):
                    break
                    
                future_high = ohlc_data[i + 1 + j, 0]
                future_low = ohlc_data[i + 1 + j, 1]
                
                # LONG: TP przed SL
                if long_result is None:
                    if future_high >= long_tp_price:
                        long_result = 'TP'
                    elif future_low <= long_sl_price:
                        long_result = 'SL'
                
                # SHORT: TP przed SL
                if short_result is None:
                    if future_low <= short_tp_price:
                        short_result = 'TP'
                    elif future_high >= short_sl_price:
                        short_result = 'SL'
                
                if long_result is not None and short_result is not None:
                    break
            
            # Logika decyzyjna
            if long_result == 'TP' and short_result != 'TP':
                simulated_label = 0  # LONG
            elif short_result == 'TP' and long_result != 'TP':
                simulated_label = 1  # SHORT
            elif long_result == 'TP' and short_result == 'TP':
                simulated_label = 0  # LONG ma pierwszeństwo
            else:
                simulated_label = 2  # NEUTRAL
            
            actual_label = labels[i]
            if simulated_label != actual_label:
                mismatch_count += 1
                if mismatch_count <= 5:  # Pokaż pierwsze 5 błędów
                    logger.info(f"  Błąd {mismatch_count}: Symulacja={simulated_label}, Rzeczywistość={actual_label}")
                    logger.info(f"    Entry: {entry_price:.2f}")
                    logger.info(f"    LONG: {long_result}, SHORT: {short_result}")
                    logger.info(f"    LONG TP: {long_tp_price:.2f}, SL: {long_sl_price:.2f}")
                    logger.info(f"    SHORT TP: {short_tp_price:.2f}, SL: {short_sl_price:.2f}")
        
        logger.info(f"  Łącznie błędów: {mismatch_count}")
    else:
        logger.info("✅ Symulacja zgadza się z rzeczywistością")

if __name__ == "__main__":
    fix_simulation() 