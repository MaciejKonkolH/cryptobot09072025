import pandas as pd

print("🔍 ANALIZA KOLUMN NA ETAPIE MERGE")
print("=" * 60)

# Wczytaj dane po merge
print("\n1. KOLUMNY W merged_ohlc_orderbook.feather:")
try:
    df_merged = pd.read_feather("merged_ohlc_orderbook.feather")
    
    print(f"📊 Łącznie kolumn: {len(df_merged.columns)}")
    print(f"📊 Łącznie wierszy: {len(df_merged):,}")
    
    # Kategorie kolumn
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    metadata_cols = ['fill_method', 'gap_duration_minutes', 'price_change_percent']
    snapshot1_cols = [col for col in df_merged.columns if col.startswith('snapshot1_')]
    snapshot2_cols = [col for col in df_merged.columns if col.startswith('snapshot2_')]
    other_cols = [col for col in df_merged.columns if col not in ohlc_cols + metadata_cols + snapshot1_cols + snapshot2_cols]
    
    print(f"\n📈 KOLUMNY OHLC ({len(ohlc_cols)}):")
    for col in ohlc_cols:
        if col in df_merged.columns:
            missing = df_merged[col].isna().sum()
            print(f"  ✅ {col}: {missing:,} NaN")
        else:
            print(f"  ❌ {col}: brak")
    
    print(f"\n🔧 KOLUMNY METADANYCH ({len(metadata_cols)}):")
    for col in metadata_cols:
        if col in df_merged.columns:
            missing = df_merged[col].isna().sum()
            percentage = (missing / len(df_merged)) * 100
            print(f"  📋 {col}: {missing:,} NaN ({percentage:.2f}%)")
        else:
            print(f"  ❌ {col}: brak")
    
    print(f"\n📊 KOLUMNY SNAPSHOT1 ({len(snapshot1_cols)}):")
    print(f"  Pierwsze 5: {snapshot1_cols[:5]}")
    if len(snapshot1_cols) > 5:
        print(f"  ... i {len(snapshot1_cols) - 5} więcej")
    
    print(f"\n📊 KOLUMNY SNAPSHOT2 ({len(snapshot2_cols)}):")
    print(f"  Pierwsze 5: {snapshot2_cols[:5]}")
    if len(snapshot2_cols) > 5:
        print(f"  ... i {len(snapshot2_cols) - 5} więcej")
    
    print(f"\n🔍 INNE KOLUMNY ({len(other_cols)}):")
    for col in other_cols:
        missing = df_merged[col].isna().sum()
        print(f"  ❓ {col}: {missing:,} NaN")
    
    # Sprawdź które kolumny mają dużo NaN
    print(f"\n⚠️  KOLUMNY Z DUŻO NaN (>50%):")
    for col in df_merged.columns:
        missing = df_merged[col].isna().sum()
        percentage = (missing / len(df_merged)) * 100
        if percentage > 50:
            print(f"  ⚠️  {col}: {missing:,} NaN ({percentage:.2f}%)")
    
except Exception as e:
    print(f"❌ Błąd: {e}")

print("\n" + "=" * 60)
print("ANALIZA ZAKOŃCZONA") 