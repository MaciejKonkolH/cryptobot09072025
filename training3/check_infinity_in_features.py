import pandas as pd
import numpy as np
import sys
import os

# Dodaj ścieżkę do modułu
sys.path.append(os.path.dirname(__file__))
import config as cfg

print("🔍 SPRAWDZANIE WARTOŚCI INFINITY W CECHACH TRENINGOWYCH")
print("=" * 60)

# Wczytaj dane
df = pd.read_feather("../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather")

print(f"📊 Wczytano: {len(df):,} wierszy, {len(df.columns)} kolumn")

# Sprawdź które cechy z cfg.FEATURES są dostępne
available_features = []
missing_features = []

for feature in cfg.FEATURES:
    if feature in df.columns:
        available_features.append(feature)
    else:
        missing_features.append(feature)

print(f"\n📋 CECHY Z cfg.FEATURES:")
print(f"  ✅ Dostępne: {len(available_features)}")
print(f"  ❌ Brakuje: {len(missing_features)}")

if missing_features:
    print(f"  Brakujące cechy: {missing_features[:5]}")
    if len(missing_features) > 5:
        print(f"  ... i {len(missing_features) - 5} więcej")

# Sprawdź infinity tylko w dostępnych cechach
print(f"\n🔍 SPRAWDZANIE INFINITY W DOSTĘPNYCH CECHACH:")
infinity_features = []

for feature in available_features:
    if df[feature].dtype in ['float64', 'float32']:
        inf_count = np.isinf(df[feature]).sum()
        if inf_count > 0:
            infinity_features.append(feature)
            percentage = (inf_count / len(df)) * 100
            print(f"  ⚠️  {feature}: {inf_count:,} infinity ({percentage:.4f}%)")

if not infinity_features:
    print("  ✅ Brak wartości infinity w cechach")
else:
    print(f"\n📋 CECHY Z INFINITY ({len(infinity_features)}):")
    for feature in infinity_features:
        inf_count = np.isinf(df[feature]).sum()
        percentage = (inf_count / len(df)) * 100
        print(f"  {feature}: {inf_count:,} ({percentage:.4f}%)")

# Sprawdź wartości ekstremalne
print(f"\n🔍 SPRAWDZANIE WARTOŚCI EKSTREMALNYCH:")
extreme_features = []

for feature in available_features:
    if df[feature].dtype in ['float64', 'float32']:
        large_count = (df[feature] > 1e10).sum()
        small_count = (df[feature] < -1e10).sum()
        if large_count > 0 or small_count > 0:
            extreme_features.append(feature)
            print(f"  ⚠️  {feature}: {large_count:,} >1e10, {small_count:,} <-1e10")

if not extreme_features:
    print("  ✅ Brak wartości ekstremalnych")

# Sprawdź NaN w cechach
print(f"\n🔍 SPRAWDZANIE NaN W CECHACH:")
nan_features = []

for feature in available_features:
    nan_count = df[feature].isna().sum()
    if nan_count > 0:
        nan_features.append(feature)
        percentage = (nan_count / len(df)) * 100
        print(f"  ⚠️  {feature}: {nan_count:,} NaN ({percentage:.4f}%)")

if not nan_features:
    print("  ✅ Brak NaN w cechach")

print(f"\n📋 PODSUMOWANIE:")
print(f"  Cechy z infinity: {len(infinity_features)}")
print(f"  Cechy z ekstremalnymi: {len(extreme_features)}")
print(f"  Cechy z NaN: {len(nan_features)}")

if infinity_features or extreme_features or nan_features:
    print("  ⚠️  PROBLEMY ZNALEZIONE - trening może się nie powieść")
else:
    print("  ✅ WSZYSTKIE CECHY CZYSTE - trening powinien działać")

print("\n" + "=" * 60)
print("ANALIZA ZAKOŃCZONA") 