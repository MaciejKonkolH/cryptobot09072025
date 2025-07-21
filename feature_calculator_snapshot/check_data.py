import pandas as pd

# Sprawdź duży plik
df = pd.read_feather('output/BTCUSDT-1m-futures_features.feather')
print(f"Duży plik: {len(df):,} wierszy")

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Okres: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"Dni: {(df['timestamp'].max() - df['timestamp'].min()).days} dni")
else:
    print("Kolumny:", list(df.columns)[:10])
    
print(f"Kolumny: {len(df.columns)}")
print(f"Rozmiar: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB") 