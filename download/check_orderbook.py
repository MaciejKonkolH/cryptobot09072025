import pandas as pd
from datetime import datetime, timedelta

# Wczytaj przykładowy plik order book
df = pd.read_csv('orderbook_raw/BTCUSDT-bookDepth-2023-01-01.csv')

print(f"=== ANALIZA ROZKŁADU ORDER BOOK ===")
print(f"Liczba wierszy w CSV: {len(df)}")
print(f"Liczba unikalnych timestampów (snapshots): {df['timestamp'].nunique()}")
print(f"Poziomy głębokości na snapshot: {len(df) / df['timestamp'].nunique():.0f}")

# Konwertuj timestamp na datetime
df['datetime'] = pd.to_datetime(df['timestamp'])

# Sprawdź zakres czasowy
print(f"\n=== ZAKRES CZASOWY ===")
print(f"Początek: {df['datetime'].min()}")
print(f"Koniec: {df['datetime'].max()}")
print(f"Czas trwania: {df['datetime'].max() - df['datetime'].min()}")

# Sprawdź rozkład w minutach
print(f"\n=== ROZKŁAD W MINUTACH ===")
df['minute'] = df['datetime'].dt.floor('min')
minute_counts = df.groupby('minute')['timestamp'].nunique()  # Liczba snapshots na minutę
print(f"Liczba minut z danymi: {len(minute_counts)}")
print(f"Średnia snapshots na minutę: {minute_counts.mean():.1f}")
print(f"Min snapshots na minutę: {minute_counts.min()}")
print(f"Max snapshots na minutę: {minute_counts.max()}")

# Sprawdź przykładowe minuty
print(f"\n=== PRZYKŁADOWE MINUTY ===")
print("Minuty z największą liczbą snapshots:")
print(minute_counts.nlargest(10))

print(f"\nMinuty z najmniejszą liczbą snapshots:")
print(minute_counts.nsmallest(10))

# Sprawdź rozkład percentage
print(f"\n=== ROZKŁAD POZIOMÓW GŁĘBOKOŚCI ===")
percentage_counts = df['percentage'].value_counts().sort_index()
print(percentage_counts)

# Sprawdź czy każdy timestamp ma wszystkie poziomy
print(f"\n=== KOMPLETNOŚĆ DANYCH ===")
timestamp_levels = df.groupby('timestamp')['percentage'].nunique()
print(f"Średnia poziomów na snapshot: {timestamp_levels.mean():.1f}")
print(f"Min poziomów na snapshot: {timestamp_levels.min()}")
print(f"Max poziomów na snapshot: {timestamp_levels.max()}")

# Sprawdź przykładowe snapshots
print(f"\n=== PRZYKŁADOWE SNAPSHOTS ===")
sample_timestamps = df['timestamp'].unique()[:5]
for ts in sample_timestamps:
    ts_data = df[df['timestamp'] == ts]
    print(f"\nSnapshot timestamp: {ts}")
    print(f"Liczba poziomów: {len(ts_data)}")
    print(f"Poziomy: {sorted(ts_data['percentage'].tolist())}")

# Oblicz całkowite statystyki
print(f"\n=== STATYSTYKI CAŁKOWITE ===")
total_minutes = 24 * 60  # 24 godziny
snapshots_per_minute = len(df) / df['timestamp'].nunique() / total_minutes
print(f"Snapshots na minutę (obliczone): {snapshots_per_minute:.1f}")
print(f"Wiersze na minutę: {len(df) / total_minutes:.1f}") 