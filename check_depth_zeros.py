"""
Skrypt do sprawdzenia czy poziomy order book mają wartości zero
"""
import pandas as pd
import numpy as np

def analyze_depth_zeros():
    """Analizuje dane order book pod kątem wartości zero."""
    
    # Wczytaj dane
    print("Wczytywanie danych...")
    df = pd.read_feather('../download/orderbook_ohlc_merged.feather')
    print(f"Liczba wierszy: {len(df):,}")
    
    # Znajdź kolumny depth
    depth_cols = [col for col in df.columns if 'depth' in col]
    print(f"Kolumny depth: {depth_cols}")
    
    print("\n=== PRÓBKA DANYCH ===")
    print(df[depth_cols].head(10))
    
    print("\n=== SPRAWDZENIE ZER ===")
    for col in depth_cols:
        zeros = (df[col] == 0).sum()
        total = len(df)
        percentage = (zeros / total) * 100
        print(f"{col}: {zeros:,} zer ({percentage:.2f}%)")
    
    print("\n=== SPRAWDZENIE NAN ===")
    for col in depth_cols:
        nans = df[col].isna().sum()
        total = len(df)
        percentage = (nans / total) * 100
        print(f"{col}: {nans:,} NaN ({percentage:.2f}%)")
    
    print("\n=== SPRAWDZENIE INF ===")
    for col in depth_cols:
        infs = np.isinf(df[col]).sum()
        total = len(df)
        percentage = (infs / total) * 100
        print(f"{col}: {infs:,} inf ({percentage:.2f}%)")
    
    print("\n=== SPRAWDZENIE WARTOŚCI UJEMNYCH ===")
    for col in depth_cols:
        negatives = (df[col] < 0).sum()
        total = len(df)
        percentage = (negatives / total) * 100
        print(f"{col}: {negatives:,} ujemnych ({percentage:.2f}%)")
    
    print("\n=== STATYSTYKI OGÓLNE ===")
    print(f"Wiersze z kompletnymi danymi order book: {(df['data_quality'] == 'complete').sum():,}")
    print(f"Wiersze z brakującymi danymi order book: {(df['data_quality'] == 'missing').sum():,}")

if __name__ == "__main__":
    analyze_depth_zeros() 