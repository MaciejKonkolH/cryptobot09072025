"""
Skrypt do sprawdzenia jakie cechy zostały rzeczywiście obliczone.
"""
import pandas as pd
import numpy as np

def main():
    print("SPRAWDZANIE OBLICZONYCH CECH")
    print("="*50)
    
    # Wczytaj dane
    df = pd.read_feather('output/ohlc_orderbook_features.feather')
    print(f"Wczytano {len(df):,} wierszy, {len(df.columns)} kolumn")
    
    # Kolumny oryginalne (nie obliczone)
    original_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                    'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
    
    # Kolumny orderbook (nie obliczone)
    ob_cols = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    
    # Kolumny obliczone przez nasz skrypt
    calculated_cols = [col for col in df.columns if col not in original_cols and 
                      not col.startswith(('snapshot1_', 'snapshot2_'))]
    
    print(f"\nKolumny oryginalne: {len(original_cols)}")
    print(f"Kolumny orderbook: {len(ob_cols)}")
    print(f"Kolumny obliczone: {len(calculated_cols)}")
    
    print(f"\nKOLUMNY OBLICZONE PRZEZ NASZ SKRYPT:")
    print("-" * 40)
    for i, col in enumerate(calculated_cols, 1):
        unique_vals = df[col].nunique()
        null_vals = df[col].isnull().sum()
        print(f"{i:2d}. {col}")
        print(f"    Unikalne: {unique_vals:,} z {len(df):,} ({unique_vals/len(df)*100:.2f}%)")
        print(f"    Null: {null_vals:,} ({null_vals/len(df)*100:.2f}%)")
        
        # Sprawdź czy kolumna ma różne wartości
        if unique_vals <= 5:
            print(f"    Wartości: {sorted(df[col].dropna().unique())}")
        elif unique_vals <= 20:
            print(f"    Przykłady: {sorted(df[col].dropna().unique()[:10])}")
        else:
            print(f"    Min: {df[col].min():.6f}, Max: {df[col].max():.6f}")
        print()
    
    # Sprawdź czy orderbook rzeczywiście się różni
    print("SPRAWDZENIE ORDERBOOK:")
    print("-" * 30)
    s1_cols = [col for col in df.columns if col.startswith('snapshot1_') and 'depth_' in col]
    s2_cols = [col for col in df.columns if col.startswith('snapshot2_') and 'depth_' in col]
    
    for i in range(min(3, len(s1_cols))):
        col1 = s1_cols[i]
        col2 = s2_cols[i]
        diff_count = (df[col1] != df[col2]).sum()
        print(f"{col1} vs {col2}: {diff_count:,} różnic ({diff_count/len(df)*100:.2f}%)")
    
    # Sprawdź czy nasze cechy są sensowne
    print("\nSPRAWDZENIE NASZYCH CECH:")
    print("-" * 30)
    test_features = ['bb_width', 'rsi_14', 'macd_hist', 'buy_sell_ratio_s1', 'pressure_change']
    
    for feature in test_features:
        if feature in df.columns:
            print(f"{feature}:")
            print(f"  Min: {df[feature].min():.6f}")
            print(f"  Max: {df[feature].max():.6f}")
            print(f"  Mean: {df[feature].mean():.6f}")
            print(f"  Std: {df[feature].std():.6f}")
            print(f"  Null: {df[feature].isnull().sum():,}")
            print()

if __name__ == "__main__":
    main() 