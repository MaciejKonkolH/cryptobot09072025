import json
import pandas as pd

# Sprawdź JSON z download
with open('../download/orderbook_ohlc_merged.json') as f:
    data = json.load(f)

print(f"JSON zawiera {len(data)} wierszy")

if len(data) > 0:
    first = data[0]
    last = data[-1]
    print(f"Pierwszy wiersz: {first['timestamp']}")
    print(f"Ostatni wiersz: {last['timestamp']}")
    
    # Sprawdź próbkę dat
    sample_dates = [item['timestamp'] for item in data[::10000]]
    print(f"Próbka dat co 10000 wierszy: {sample_dates[:10]}")

# Sprawdź feather
df = pd.read_feather('../download/orderbook_ohlc_merged.feather')
print(f"\nFeather zawiera {len(df)} wierszy")
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Zakres dat: {df['timestamp'].min()} - {df['timestamp'].max()}")
else:
    print("Kolumny w feather:", list(df.columns)[:5]) 