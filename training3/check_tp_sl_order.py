"""
Sprawdza kolejność sprawdzania TP vs SL w algorytmie.
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

def check_tp_sl_order():
    """Sprawdza kolejność sprawdzania TP vs SL."""
    
    logger.info("=== SPRAWDZENIE KOLEJNOŚCI TP vs SL ===")
    
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
    
    logger.info(f"\n--- ANALIZA KOLEJNOŚCI: TP={tp_pct}%, SL={sl_pct}% ---")
    
    # Pobierz dane OHLC
    ohlc_data = df[['high', 'low', 'close']].values
    labels = df[label_col].values
    
    # Sprawdź rozkład etykiet
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_stats = dict(zip(unique_labels, counts))
    
    logger.info("Rozkład etykiet:")
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
    
    # Sprawdź czy nie ma błędu w kolejności TP vs SL
    logger.info(f"\n--- SPRAWDZENIE KOLEJNOŚCI TP vs SL ---")
    
    # Symuluj algorytm z różnymi kolejnościami
    sample_size = min(1000, len(ohlc_data) - 60)
    
    # Test 1: Algorytm oryginalny (TP przed SL)
    logger.info("Test 1: TP przed SL (oryginalny)")
    long_tp_first = 0
    short_tp_first = 0
    neutral_first = 0
    
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
            long_tp_first += 1
        elif short_result == 'TP' and long_result != 'TP':
            short_tp_first += 1
        elif long_result == 'TP' and short_result == 'TP':
            long_tp_first += 1  # LONG ma pierwszeństwo
        else:
            neutral_first += 1
    
    total_samples = long_tp_first + short_tp_first + neutral_first
    logger.info(f"  LONG TP: {long_tp_first} ({long_tp_first/total_samples*100:.1f}%)")
    logger.info(f"  SHORT TP: {short_tp_first} ({short_tp_first/total_samples*100:.1f}%)")
    logger.info(f"  NEUTRAL: {neutral_first} ({neutral_first/total_samples*100:.1f}%)")
    
    if short_tp_first > long_tp_first:
        diff = short_tp_first - long_tp_first
        ratio = short_tp_first / long_tp_first if long_tp_first > 0 else float('inf')
        logger.warning(f"  ⚠️ SHORT > LONG: różnica = {diff}, stosunek = {ratio:.2f}x")
    else:
        diff = long_tp_first - short_tp_first
        ratio = long_tp_first / short_tp_first if short_tp_first > 0 else float('inf')
        logger.info(f"  ✅ LONG >= SHORT: różnica = {diff}, stosunek = {ratio:.2f}x")
    
    # Test 2: SL przed TP (sprawdź czy to zmienia wynik)
    logger.info(f"\nTest 2: SL przed TP (test)")
    long_sl_first = 0
    short_sl_first = 0
    neutral_sl_first = 0
    
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
            
            # LONG: SL przed TP
            if long_result is None:
                if future_low <= long_sl_price:
                    long_result = 'SL'
                elif future_high >= long_tp_price:
                    long_result = 'TP'
            
            # SHORT: SL przed TP
            if short_result is None:
                if future_high >= short_sl_price:
                    short_result = 'SL'
                elif future_low <= short_tp_price:
                    short_result = 'TP'
            
            if long_result is not None and short_result is not None:
                break
        
        # Logika decyzyjna
        if long_result == 'TP' and short_result != 'TP':
            long_sl_first += 1
        elif short_result == 'TP' and long_result != 'TP':
            short_sl_first += 1
        elif long_result == 'TP' and short_result == 'TP':
            long_sl_first += 1  # LONG ma pierwszeństwo
        else:
            neutral_sl_first += 1
    
    total_samples_sl = long_sl_first + short_sl_first + neutral_sl_first
    logger.info(f"  LONG TP: {long_sl_first} ({long_sl_first/total_samples_sl*100:.1f}%)")
    logger.info(f"  SHORT TP: {short_sl_first} ({short_sl_first/total_samples_sl*100:.1f}%)")
    logger.info(f"  NEUTRAL: {neutral_sl_first} ({neutral_sl_first/total_samples_sl*100:.1f}%)")
    
    if short_sl_first > long_sl_first:
        diff = short_sl_first - long_sl_first
        ratio = short_sl_first / long_sl_first if long_sl_first > 0 else float('inf')
        logger.warning(f"  ⚠️ SHORT > LONG: różnica = {diff}, stosunek = {ratio:.2f}x")
    else:
        diff = long_sl_first - short_sl_first
        ratio = long_sl_first / short_sl_first if short_sl_first > 0 else float('inf')
        logger.info(f"  ✅ LONG >= SHORT: różnica = {diff}, stosunek = {ratio:.2f}x")
    
    # Porównanie wyników
    logger.info(f"\n--- PORÓWNANIE WYNIKÓW ---")
    logger.info(f"TP przed SL: LONG={long_tp_first}, SHORT={short_tp_first}")
    logger.info(f"SL przed TP: LONG={long_sl_first}, SHORT={short_sl_first}")
    
    if short_tp_first != short_sl_first or long_tp_first != long_sl_first:
        logger.warning("⚠️ Kolejność TP vs SL wpływa na wyniki!")
    else:
        logger.info("✅ Kolejność TP vs SL nie wpływa na wyniki")

if __name__ == "__main__":
    check_tp_sl_order() 