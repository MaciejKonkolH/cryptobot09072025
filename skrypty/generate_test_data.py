import os
import sys
import pandas as pd
import numpy as np

# --- Konfiguracja ---
# Upewnij się, że ta ścieżka i nazwa pliku są poprawne
FEATURES_CSV_PATH = '../ft_bot_clean/user_data/logs/features/features_BTC_USDT_USDT_20250628_231551.csv'
OUTPUT_NUMPY_FILE = 'test_sequence.npy'

# Parametry do wycięcia sekwencji
ROW_INDEX_TO_TEST = 1000
SEQUENCE_LENGTH = 120
FEATURE_COLUMNS = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200', 
    'volume_to_ma1440', 'volume_to_ma43200'
]

def main():
    """Główna funkcja skryptu."""
    print(f"--- Generator danych testowych ---")
    
    # Budowanie pełnej ścieżki
    base_dir = os.path.dirname(__file__)
    features_full_path = os.path.abspath(os.path.join(base_dir, FEATURES_CSV_PATH))
    output_full_path = os.path.join(base_dir, OUTPUT_NUMPY_FILE)
    
    # Sprawdzenie, czy plik CSV istnieje
    if not os.path.exists(features_full_path):
        print(f"❌ BŁĄD: Plik z cechami nie został znaleziony w: {features_full_path}")
        sys.exit(1)
        
    print(f"✅ Znaleziono plik z cechami: {features_full_path}")
    
    # Wczytanie danych
    features_df = pd.read_csv(features_full_path)
    
    # Wycięcie sekwencji
    start_index = ROW_INDEX_TO_TEST - SEQUENCE_LENGTH
    end_index = ROW_INDEX_TO_TEST
    
    if start_index < 0 or end_index > len(features_df):
        print(f"❌ BŁĄD: Nie można wyciąć sekwencji. Indeksy poza zakresem.")
        sys.exit(1)
        
    raw_sequence_data = features_df.iloc[start_index:end_index][FEATURE_COLUMNS].values
    
    # Sprawdzenie kształtu
    if raw_sequence_data.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
        print(f"❌ BŁĄD: Ostateczny kształt danych jest niepoprawny: {raw_sequence_data.shape}")
        sys.exit(1)
        
    # Zapis do pliku .npy
    try:
        np.save(output_full_path, raw_sequence_data)
        print(f"✅ Dane testowe zostały pomyślnie zapisane do pliku: {output_full_path}")
        print(f"   - Kształt zapisanych danych: {raw_sequence_data.shape}")
    except Exception as e:
        print(f"❌ BŁĄD: Nie udało się zapisać pliku .npy: {e}")

if __name__ == "__main__":
    main() 