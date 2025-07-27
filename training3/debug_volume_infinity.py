import pandas as pd
import numpy as np

print("🔍 DEBUGOWANIE PROBLEMU Z VOLUME_INFINITY")
print("=" * 60)

# Wczytaj dane
df = pd.read_feather("../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather")

print(f"📊 Wczytano: {len(df):,} wierszy")

# Znajdź wiersze z infinity w volume_change_norm
infinity_mask = np.isinf(df['volume_change_norm'])
infinity_rows = df[infinity_mask]

print(f"\n🔍 WIERSZE Z INFINITY W volume_change_norm:")
print(f"  Liczba wierszy: {len(infinity_rows)}")

if len(infinity_rows) > 0:
    print(f"\n📊 ANALIZA PROBLEMATYCZNYCH WIERSZY:")
    
    # Sprawdź wartości volume
    print(f"  Wartości volume:")
    for i, (idx, row) in enumerate(infinity_rows.iterrows()):
        print(f"    Wiersz {i+1}: volume={row['volume']}, volume_change_norm={row['volume_change_norm']}")
    
    # Sprawdź sąsiednie wiersze
    print(f"\n🔍 ANALIZA SĄSIEDNICH WIERSZY:")
    for i, (idx, row) in enumerate(infinity_rows.head(3).iterrows()):
        print(f"\n  Wiersz {i+1} (indeks {idx}):")
        
        # Znajdź indeks w DataFrame
        df_idx = df.index.get_loc(idx)
        
        # Sprawdź poprzedni wiersz
        if df_idx > 0:
            prev_idx = df.index[df_idx - 1]
            prev_row = df.loc[prev_idx]
            print(f"    Poprzedni: volume={prev_row['volume']}")
        
        # Sprawdź aktualny wiersz
        print(f"    Aktualny:  volume={row['volume']}")
        
        # Sprawdź następny wiersz
        if df_idx < len(df) - 1:
            next_idx = df.index[df_idx + 1]
            next_row = df.loc[next_idx]
            print(f"    Następny:  volume={next_row['volume']}")
        
        # Oblicz pct_change ręcznie
        if df_idx > 0:
            prev_volume = df.iloc[df_idx - 1]['volume']
            curr_volume = row['volume']
            
            if prev_volume == 0:
                print(f"    PROBLEM: poprzedni volume=0, aktualny volume={curr_volume}")
                print(f"    pct_change = ({curr_volume} - 0) / 0 = INFINITY")
            else:
                pct_change = (curr_volume - prev_volume) / prev_volume
                print(f"    pct_change = ({curr_volume} - {prev_volume}) / {prev_volume} = {pct_change}")

# Sprawdź ogólne statystyki volume
print(f"\n📊 STATYSTYKI VOLUME:")
print(f"  Min: {df['volume'].min()}")
print(f"  Max: {df['volume'].max()}")
print(f"  Mean: {df['volume'].mean():.2f}")
print(f"  Median: {df['volume'].median():.2f}")
print(f"  Wiersze z volume=0: {(df['volume'] == 0).sum():,}")

# Sprawdź czy są wiersze z volume=0
zero_volume_rows = df[df['volume'] == 0]
if len(zero_volume_rows) > 0:
    print(f"\n⚠️  WIERSZE Z VOLUME=0:")
    print(f"  Liczba: {len(zero_volume_rows):,}")
    print(f"  Procent: {len(zero_volume_rows)/len(df)*100:.4f}%")
    
    # Sprawdź czy wiersze z volume=0 powodują infinity
    zero_volume_indices = zero_volume_rows.index
    infinity_after_zero = []
    
    for idx in zero_volume_indices:
        df_idx = df.index.get_loc(idx)
        if df_idx < len(df) - 1:
            next_idx = df.index[df_idx + 1]
            next_row = df.loc[next_idx]
            if next_row['volume'] > 0:
                infinity_after_zero.append(next_idx)
    
    print(f"  Wiersze z infinity po volume=0: {len(infinity_after_zero)}")

print(f"\n📋 WNIOSEK:")
if len(infinity_rows) > 0:
    print(f"  PROBLEM: volume_change_norm ma infinity z powodu dzielenia przez zero")
    print(f"  ROZWIĄZANIE: Naprawić obliczenia w feature_calculator_ohlc_snapshot")
else:
    print(f"  BRAK PROBLEMÓW")

print("\n" + "=" * 60)
print("DEBUGOWANIE ZAKOŃCZONE") 