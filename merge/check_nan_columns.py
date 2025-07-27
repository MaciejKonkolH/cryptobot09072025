import pandas as pd

# Wczytaj dane
df = pd.read_feather("../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather")

# Sprawdź które kolumny mają NaN
missing_by_column = df.isnull().sum()
columns_with_missing = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

print("KOLUMNY Z BRAKUJĄCYMI DANYMI:")
for col, count in columns_with_missing.items():
    percentage = (count / len(df)) * 100
    print(f"  {col}: {count:,} ({percentage:.2f}%)")

# Sprawdź czy któreś z kolumn FEATURES z training3 mają NaN
training3_features = [
    'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'adx_14',
    'ma_60', 'ma_240', 'ma_1440', 'price_to_ma_60', 'price_to_ma_240', 
    'price_to_ma_1440', 'ma_60_to_ma_240',
    'volume_change_norm', 'price_change_percent', 'price_momentum',
    'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'imbalance_s1', 'imbalance_s2', 'pressure_change',
    'tp_1pct_depth_s1', 'tp_2pct_depth_s1', 'sl_1pct_depth_s1', 'tp_sl_ratio_1pct', 'tp_sl_ratio_2pct',
    'total_depth_change', 'total_notional_change',
    'depth_price_corr', 'pressure_volume_corr',
    'snapshot1_depth_-5', 'snapshot1_depth_-4', 'snapshot1_depth_-3', 'snapshot1_depth_-2', 'snapshot1_depth_-1',
    'snapshot1_depth_1', 'snapshot1_depth_2', 'snapshot1_depth_3', 'snapshot1_depth_4', 'snapshot1_depth_5',
    'snapshot1_notional_-5', 'snapshot1_notional_-4', 'snapshot1_notional_-3', 'snapshot1_notional_-2', 'snapshot1_notional_-1',
    'snapshot1_notional_1', 'snapshot1_notional_2', 'snapshot1_notional_3', 'snapshot1_notional_4', 'snapshot1_notional_5',
    'snapshot2_depth_-5', 'snapshot2_depth_-4', 'snapshot2_depth_-3', 'snapshot2_depth_-2', 'snapshot2_depth_-1',
    'snapshot2_depth_1', 'snapshot2_depth_2', 'snapshot2_depth_3', 'snapshot2_depth_4', 'snapshot2_depth_5',
    'snapshot2_notional_-5', 'snapshot2_notional_-4', 'snapshot2_notional_-3', 'snapshot2_notional_-2', 'snapshot2_notional_-1',
    'snapshot2_notional_1', 'snapshot2_notional_2', 'snapshot2_notional_3', 'snapshot2_notional_4', 'snapshot2_notional_5',
    'volume', 'taker_buy_volume', 'taker_buy_quote_volume', 'quote_volume',
    'spread'
]

print("\nSPRAWDZENIE KOLUMN FEATURES Z TRAINING3:")
for feature in training3_features:
    if feature in df.columns:
        missing_count = df[feature].isna().sum()
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            print(f"  ❌ {feature}: {missing_count:,} ({percentage:.2f}%)")
        else:
            print(f"  ✅ {feature}: brak NaN")
    else:
        print(f"  ❌ {feature}: kolumna nie istnieje") 