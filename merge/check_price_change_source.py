import pandas as pd

print("🔍 ANALIZA ŹRÓDŁA price_change_percent")
print("=" * 50)

# 1. Sprawdź dane po merge
print("\n1. DANE PO MERGE (merged_ohlc_orderbook.feather):")
try:
    df_merged = pd.read_feather("merged_ohlc_orderbook.feather")
    if 'price_change_percent' in df_merged.columns:
        missing_count = df_merged['price_change_percent'].isna().sum()
        total_count = len(df_merged)
        print(f"  ✅ Kolumna istnieje: {missing_count:,}/{total_count:,} NaN ({missing_count/total_count*100:.2f}%)")
        
        # Sprawdź wartości
        non_null_values = df_merged['price_change_percent'].dropna()
        if len(non_null_values) > 0:
            print(f"  📊 Wartości nie-null: {len(non_null_values):,}")
            print(f"  📊 Min: {non_null_values.min():.4f}")
            print(f"  📊 Max: {non_null_values.max():.4f}")
            print(f"  📊 Mean: {non_null_values.mean():.4f}")
    else:
        print("  ❌ Kolumna nie istnieje")
except Exception as e:
    print(f"  ❌ Błąd: {e}")

# 2. Sprawdź dane orderbook
print("\n2. DANE ORDERBOOK (../download/orderbook_filled.feather):")
try:
    df_orderbook = pd.read_feather("../download/orderbook_filled.feather")
    if 'price_change_percent' in df_orderbook.columns:
        missing_count = df_orderbook['price_change_percent'].isna().sum()
        total_count = len(df_orderbook)
        print(f"  ✅ Kolumna istnieje: {missing_count:,}/{total_count:,} NaN ({missing_count/total_count*100:.2f}%)")
    else:
        print("  ❌ Kolumna nie istnieje")
        
    # Sprawdź inne kolumny metadanych
    meta_cols = ['fill_method', 'gap_duration_minutes']
    for col in meta_cols:
        if col in df_orderbook.columns:
            missing_count = df_orderbook[col].isna().sum()
            total_count = len(df_orderbook)
            print(f"  📋 {col}: {missing_count:,}/{total_count:,} NaN ({missing_count/total_count*100:.2f}%)")
        else:
            print(f"  ❌ {col}: kolumna nie istnieje")
            
except Exception as e:
    print(f"  ❌ Błąd: {e}")

# 3. Sprawdź dane OHLC
print("\n3. DANE OHLC (../download/ohlc_merged.feather):")
try:
    df_ohlc = pd.read_feather("../download/ohlc_merged.feather")
    if 'price_change_percent' in df_ohlc.columns:
        missing_count = df_ohlc['price_change_percent'].isna().sum()
        total_count = len(df_ohlc)
        print(f"  ✅ Kolumna istnieje: {missing_count:,}/{total_count:,} NaN ({missing_count/total_count*100:.2f}%)")
    else:
        print("  ❌ Kolumna nie istnieje")
        
    # Sprawdź kolumny OHLC
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlc_cols:
        if col in df_ohlc.columns:
            missing_count = df_ohlc[col].isna().sum()
            total_count = len(df_ohlc)
            print(f"  📈 {col}: {missing_count:,}/{total_count:,} NaN ({missing_count/total_count*100:.2f}%)")
        else:
            print(f"  ❌ {col}: kolumna nie istnieje")
            
except Exception as e:
    print(f"  ❌ Błąd: {e}")

# 4. Sprawdź dane po etykietowaniu
print("\n4. DANE PO ETYKIETOWANIU (../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather):")
try:
    df_labeled = pd.read_feather("../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather")
    if 'price_change_percent' in df_labeled.columns:
        missing_count = df_labeled['price_change_percent'].isna().sum()
        total_count = len(df_labeled)
        print(f"  ✅ Kolumna istnieje: {missing_count:,}/{total_count:,} NaN ({missing_count/total_count*100:.2f}%)")
    else:
        print("  ❌ Kolumna nie istnieje")
        
except Exception as e:
    print(f"  ❌ Błąd: {e}")

print("\n" + "=" * 50)
print("ANALIZA ZAKOŃCZONA") 