import pandas as pd
import os

# Ścieżka do pliku, który sprawdzamy
file_path = os.path.join('validation_and_labeling', 'raw_validated', 'BTCUSDT-1m-futures.feather')

try:
    df = pd.read_feather(file_path)
    print(f"Kolumny w pliku {file_path}:")
    print(df.columns.tolist())
except Exception as e:
    print(f"Nie udało się wczytać pliku: {e}") 