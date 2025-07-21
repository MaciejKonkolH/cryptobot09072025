"""
Skrypt diagnostyczny do sprawdzenia danych i procesu etykietowania.
"""
import pandas as pd
import numpy as np
import sys
import os

def debug_data():
    """Sprawdza dane wejściowe i proces etykietowania."""
    print("=== DIAGNOZA DANYCH I ETYKIETOWANIA ===\n")
    
    # 1. Sprawdź dane z feature_calculator_snapshot
    features_path = "../feature_calculator_snapshot/output/orderbook_ohlc_features.feather"
    print(f"1. Sprawdzanie danych z: {features_path}")
    print(f"   Sprawdzanie czy plik istnieje...")
    
    if not os.path.exists(features_path):
        print(f"   ❌ PLIK NIE ISTNIEJE: {features_path}")
        return
    else:
        print(f"   ✓ Plik istnieje")
    
    try:
        print("   Wczytywanie danych...")
        df = pd.read_feather(features_path)
        print(f"   ✓ Wczytano: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
        
        # Sprawdź kolumny OHLC
        print("   Sprawdzanie kolumn OHLC...")
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_cols = list(df.columns)
        print(f"   Pierwszych 10 kolumn: {available_cols[:10]}")
        
        missing_ohlc = [col for col in ohlc_cols if col not in df.columns]
        if missing_ohlc:
            print(f"   ❌ BRAKUJE KOLUMN OHLC: {missing_ohlc}")
            print(f"   Wszystkie kolumny: {available_cols}")
            return
        else:
            print(f"   ✓ Kolumny OHLC dostępne: {ohlc_cols}")
        
        # Sprawdź dane OHLC
        print("\n   Próbka danych OHLC:")
        ohlc_sample = df[ohlc_cols].head()
        print(ohlc_sample)
        
        print("\n   Statystyki OHLC:")
        ohlc_stats = df[ohlc_cols].describe()
        print(ohlc_stats)
        
        # Sprawdź czy ceny się zmieniają
        print("\n   Sprawdzanie zmienności cen...")
        price_changes = df['close'].diff().abs()
        print(f"   - Średnia zmiana: {price_changes.mean():.6f}")
        print(f"   - Max zmiana: {price_changes.max():.6f}")
        print(f"   - Wierszy z zerowymi zmianami: {(price_changes == 0).sum()}")
        
        # Sprawdź czy mamy NaN
        print(f"   - NaN w close: {df['close'].isna().sum()}")
        
        # 2. Sprawdź proces etykietowania ręcznie
        print(f"\n2. Test etykietowania dla pierwszych wierszy:")
        test_labeling(df.head(200))  # Zwiększono próbkę
        
    except Exception as e:
        print(f"   ❌ Błąd: {e}")
        import traceback
        traceback.print_exc()

def test_labeling(df_sample):
    """Testuje proces etykietowania na małej próbce."""
    # Parametry z config
    future_window = 120
    tp_pct = 2.0 / 100  # 2%
    sl_pct = 1.0 / 100  # 1%
    
    print(f"   Parametry: TP={tp_pct*100}%, SL={sl_pct*100}%, okno={future_window}")
    print(f"   Próbka danych: {len(df_sample)} wierszy")
    
    ohlc_data = df_sample[['high', 'low', 'close']].to_numpy()
    
    # Sprawdź kilka pierwszych wierszy
    labels_found = []
    
    for i in range(min(10, len(ohlc_data))):
        if i + future_window >= len(ohlc_data):
            print(f"   Wiersz {i}: SKIP (za mało danych w przyszłości, potrzeba {future_window}, mamy {len(ohlc_data)-i-1})")
            continue
            
        entry_price = ohlc_data[i, 2]  # close
        long_tp_price = entry_price * (1 + tp_pct)
        long_sl_price = entry_price * (1 - sl_pct)
        
        print(f"\n   Wiersz {i}:")
        print(f"   - Entry: {entry_price:.2f}")
        print(f"   - Long TP: {long_tp_price:.2f} (+{tp_pct*100}%)")
        print(f"   - Long SL: {long_sl_price:.2f} (-{sl_pct*100}%)")
        
        # Sprawdź przyszłe ceny w pełnym oknie
        future_slice = ohlc_data[i + 1 : i + 1 + future_window]
        if len(future_slice) == 0:
            print(f"   - BRAK danych przyszłych!")
            continue
            
        max_future_high = np.max(future_slice[:, 0])
        min_future_low = np.min(future_slice[:, 1])
        
        print(f"   - Max future high: {max_future_high:.2f}")
        print(f"   - Min future low: {min_future_low:.2f}")
        
        # Sprawdź czy TP/SL zostały osiągnięte
        tp_hit = max_future_high >= long_tp_price
        sl_hit = min_future_low <= long_sl_price
        
        print(f"   - TP hit: {tp_hit}")
        print(f"   - SL hit: {sl_hit}")
        
        if not tp_hit and not sl_hit:
            result = "TIMEOUT_HOLD"
            print(f"   - WYNIK: TIMEOUT_HOLD (brak TP/SL)")
        elif tp_hit and not sl_hit:
            result = "PROFIT_LONG"
            print(f"   - WYNIK: PROFIT_LONG")
        elif sl_hit and not tp_hit:
            result = "LOSS_LONG"
            print(f"   - WYNIK: LOSS_LONG")
        else:
            result = "BOTH_HIT"
            print(f"   - WYNIK: Oba zdarzenia (wymaga szczegółowej analizy)")
        
        labels_found.append(result)
    
    print(f"\n   PODSUMOWANIE TESTÓW:")
    from collections import Counter
    label_counts = Counter(labels_found)
    for label, count in label_counts.items():
        print(f"   - {label}: {count} przypadków")

if __name__ == "__main__":
    debug_data() 