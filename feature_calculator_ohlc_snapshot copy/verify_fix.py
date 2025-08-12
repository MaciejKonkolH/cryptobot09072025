import pandas as pd
import numpy as np

print("🔍 WERYFIKACJA POPRAWKI VOLUME_CHANGE_NORM")
print("=" * 60)

# Wczytaj dane po poprawce
df = pd.read_feather("output/ohlc_orderbook_features.feather")

print(f"📊 Wczytano: {len(df):,} wierszy, {len(df.columns)} kolumn")

# Sprawdź volume_change_norm
if 'volume_change_norm' in df.columns:
    print(f"\n🔍 SPRAWDZANIE volume_change_norm:")
    
    # Sprawdź infinity
    inf_count = np.isinf(df['volume_change_norm']).sum()
    print(f"  Wartości infinity: {inf_count}")
    
    # Sprawdź NaN
    nan_count = df['volume_change_norm'].isna().sum()
    print(f"  Wartości NaN: {nan_count}")
    
    # Sprawdź wartości ekstremalne
    large_count = (df['volume_change_norm'] > 1e10).sum()
    small_count = (df['volume_change_norm'] < -1e10).sum()
    print(f"  Wartości >1e10: {large_count}")
    print(f"  Wartości <-1e10: {small_count}")
    
    # Statystyki
    print(f"\n📊 STATYSTYKI volume_change_norm:")
    print(f"  Min: {df['volume_change_norm'].min():.4f}")
    print(f"  Max: {df['volume_change_norm'].max():.4f}")
    print(f"  Mean: {df['volume_change_norm'].mean():.4f}")
    print(f"  Median: {df['volume_change_norm'].median():.4f}")
    
    # Sprawdź czy są wiersze z volume=0
    zero_volume_count = (df['volume'] == 0).sum()
    print(f"\n📊 WIERSZE Z VOLUME=0:")
    print(f"  Liczba: {zero_volume_count:,}")
    print(f"  Procent: {zero_volume_count/len(df)*100:.4f}%")
    
    if zero_volume_count > 0:
        # Sprawdź volume_change_norm dla wierszy z volume=0
        zero_volume_rows = df[df['volume'] == 0]
        print(f"  volume_change_norm dla wierszy z volume=0:")
        print(f"    Min: {zero_volume_rows['volume_change_norm'].min():.4f}")
        print(f"    Max: {zero_volume_rows['volume_change_norm'].max():.4f}")
        print(f"    Mean: {zero_volume_rows['volume_change_norm'].mean():.4f}")
    
    # Wniosek
    print(f"\n📋 WNIOSEK:")
    if inf_count == 0 and nan_count == 0 and large_count == 0 and small_count == 0:
        print(f"  ✅ POPRAWKA DZIAŁA! Brak problemów z volume_change_norm")
    else:
        print(f"  ⚠️  PROBLEMY NADAL ISTNIEJĄ:")
        print(f"    - Infinity: {inf_count}")
        print(f"    - NaN: {nan_count}")
        print(f"    - Ekstremalne: {large_count + small_count}")
else:
    print(f"❌ Kolumna volume_change_norm nie istnieje")

print("\n" + "=" * 60)
print("WERYFIKACJA ZAKOŃCZONA") 