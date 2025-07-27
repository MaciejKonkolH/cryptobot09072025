"""
Skrypt do analizy algorytmu etykietowania pod kątem asymetrii SHORT vs LONG.
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

def analyze_labeling_bias():
    """Analizuje potencjalny bias w algorytmie etykietowania."""
    
    logger.info("=== ANALIZA BIAS W ALGORYTMIE ETYKIETOWANIA ===")
    
    # Wczytaj dane
    try:
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
        logger.info(f"Dane wczytane: {len(df):,} wierszy")
    except Exception as e:
        logger.error(f"Błąd wczytywania danych: {e}")
        return
    
    # Sprawdź czy mamy dane OHLC
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        logger.error("Brak danych OHLC")
        return
    
    # Analiza dla jednego poziomu (TP: 0.8%, SL: 0.2%)
    tp_pct, sl_pct = 0.8, 0.2
    label_col = 'label_tp0p8_sl0p2'
    
    logger.info(f"\n--- ANALIZA POZIOMU: TP={tp_pct}%, SL={sl_pct}% ---")
    
    # Pobierz dane OHLC
    ohlc_data = df[['high', 'low', 'close']].values
    labels = df[label_col].values
    
    # Analiza statystyk cenowych
    logger.info("Statystyki cenowe:")
    logger.info(f"  Średnia zmiana ceny: {np.mean(np.diff(ohlc_data[:, 2])):.6f}")
    logger.info(f"  Odchylenie std zmiany ceny: {np.std(np.diff(ohlc_data[:, 2])):.6f}")
    
    # Sprawdź asymetrię w danych OHLC
    high_low_diff = ohlc_data[:, 0] - ohlc_data[:, 1]  # high - low
    logger.info(f"  Średnia różnica high-low: {np.mean(high_low_diff):.6f}")
    logger.info(f"  Odchylenie std high-low: {np.std(high_low_diff):.6f}")
    
    # Analiza dla każdej klasy
    for class_label, class_name in [(0, 'LONG'), (1, 'SHORT'), (2, 'NEUTRAL')]:
        mask = labels == class_label
        if np.sum(mask) == 0:
            continue
            
        class_ohlc = ohlc_data[mask]
        class_count = len(class_ohlc)
        
        logger.info(f"\n  {class_name} ({class_count:,} próbek):")
        
        # Średnie wartości OHLC dla tej klasy
        avg_high = np.mean(class_ohlc[:, 0])
        avg_low = np.mean(class_ohlc[:, 1])
        avg_close = np.mean(class_ohlc[:, 2])
        avg_high_low_diff = np.mean(class_ohlc[:, 0] - class_ohlc[:, 1])
        
        logger.info(f"    Średni HIGH: {avg_high:.6f}")
        logger.info(f"    Średni LOW: {avg_low:.6f}")
        logger.info(f"    Średni CLOSE: {avg_close:.6f}")
        logger.info(f"    Średnia różnica HIGH-LOW: {avg_high_low_diff:.6f}")
        
        # Sprawdź czy są różnice w zmienności
        price_changes = np.diff(class_ohlc[:, 2])
        logger.info(f"    Średnia zmiana ceny: {np.mean(price_changes):.6f}")
        logger.info(f"    Odchylenie std zmiany: {np.std(price_changes):.6f}")
    
    # Sprawdź czy nie ma systematycznej asymetrii w TP/SL
    logger.info(f"\n--- ANALIZA TP/SL ASYMETRII ---")
    
    # Symuluj algorytm etykietowania dla kilku próbek
    future_window = 60  # 60 minut
    sample_size = min(1000, len(ohlc_data) - future_window)
    
    long_tp_pct = tp_pct / 100
    long_sl_pct = sl_pct / 100
    short_tp_pct = tp_pct / 100
    short_sl_pct = sl_pct / 100
    
    long_tp_count = 0
    long_sl_count = 0
    short_tp_count = 0
    short_sl_count = 0
    neutral_count = 0
    
    for i in range(sample_size):
        if i + future_window >= len(ohlc_data):
            break
            
        entry_price = ohlc_data[i, 2]  # close price
        
        # Oblicz poziomy TP/SL
        long_tp_price = entry_price * (1 + long_tp_pct)
        long_sl_price = entry_price * (1 - long_sl_pct)
        short_tp_price = entry_price * (1 - short_tp_pct)
        short_sl_price = entry_price * (1 + short_sl_pct)
        
        # Sprawdź przyszłe 60 minut
        long_result = None
        short_result = None
        
        for j in range(future_window):
            if i + 1 + j >= len(ohlc_data):
                break
                
            future_high = ohlc_data[i + 1 + j, 0]
            future_low = ohlc_data[i + 1 + j, 1]
            
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
        
        # Logika decyzyjna
        if long_result == 'TP' and short_result != 'TP':
            long_tp_count += 1
        elif short_result == 'TP' and long_result != 'TP':
            short_tp_count += 1
        elif long_result == 'TP' and short_result == 'TP':
            long_tp_count += 1  # Long ma pierwszeństwo
        else:
            neutral_count += 1
    
    total_samples = long_tp_count + short_tp_count + neutral_count
    logger.info(f"Symulacja algorytmu na {total_samples} próbkach:")
    logger.info(f"  LONG TP: {long_tp_count} ({long_tp_count/total_samples*100:.1f}%)")
    logger.info(f"  SHORT TP: {short_tp_count} ({short_tp_count/total_samples*100:.1f}%)")
    logger.info(f"  NEUTRAL: {neutral_count} ({neutral_count/total_samples*100:.1f}%)")
    
    if short_tp_count > long_tp_count:
        diff = short_tp_count - long_tp_count
        ratio = short_tp_count / long_tp_count if long_tp_count > 0 else float('inf')
        logger.warning(f"  ⚠️ SHORT TP > LONG TP: różnica = {diff}, stosunek = {ratio:.2f}x")
    else:
        diff = long_tp_count - short_tp_count
        ratio = long_tp_count / short_tp_count if short_tp_count > 0 else float('inf')
        logger.info(f"  ✅ LONG TP >= SHORT TP: różnica = {diff}, stosunek = {ratio:.2f}x")

if __name__ == "__main__":
    analyze_labeling_bias() 