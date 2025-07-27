import pandas as pd
import numpy as np

print("🔍 SPRAWDZANIE WARTOŚCI INFINITY W DANYCH TRENINGOWYCH")
print("=" * 60)

# Wczytaj dane
df = pd.read_feather("../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather")

print(f"📊 Wczytano: {len(df):,} wierszy, {len(df.columns)} kolumn")

# Sprawdź wartości infinity
print(f"\n🔍 SPRAWDZANIE WARTOŚCI INFINITY:")
infinity_cols = []
for col in df.columns:
    if df[col].dtype in ['float64', 'float32']:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            infinity_cols.append(col)
            print(f"  ⚠️  {col}: {inf_count:,} infinity")

if not infinity_cols:
    print("  ✅ Brak wartości infinity")
else:
    print(f"\n📋 KOLUMNY Z INFINITY ({len(infinity_cols)}):")
    for col in infinity_cols:
        inf_count = np.isinf(df[col]).sum()
        percentage = (inf_count / len(df)) * 100
        print(f"  {col}: {inf_count:,} ({percentage:.2f}%)")
        
        # Sprawdź przykładowe wartości
        inf_values = df[df[col] == np.inf][col]
        neg_inf_values = df[df[col] == -np.inf][col]
        print(f"    +inf: {len(inf_values):,}, -inf: {len(neg_inf_values):,}")

# Sprawdź wartości bardzo duże
print(f"\n🔍 SPRAWDZANIE WARTOŚCI BARDZO DUŻYCH (>1e10):")
large_cols = []
for col in df.columns:
    if df[col].dtype in ['float64', 'float32']:
        large_count = (df[col] > 1e10).sum()
        if large_count > 0:
            large_cols.append(col)
            print(f"  ⚠️  {col}: {large_count:,} wartości >1e10")

if not large_cols:
    print("  ✅ Brak bardzo dużych wartości")

# Sprawdź wartości bardzo małe (<-1e10)
print(f"\n🔍 SPRAWDZANIE WARTOŚCI BARDZO MAŁYCH (<-1e10):")
small_cols = []
for col in df.columns:
    if df[col].dtype in ['float64', 'float32']:
        small_count = (df[col] < -1e10).sum()
        if small_count > 0:
            small_cols.append(col)
            print(f"  ⚠️  {col}: {small_count:,} wartości <-1e10")

if not small_cols:
    print("  ✅ Brak bardzo małych wartości")

print("\n" + "=" * 60)
print("ANALIZA ZAKOŃCZONA") 