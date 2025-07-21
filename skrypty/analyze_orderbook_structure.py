import zipfile
import pandas as pd
import os

def analyze_orderbook_structure():
    """Analizuje strukturę plików order book ZIP"""
    
    # Sprawdź czy mamy jakiś plik do analizy
    test_file = "orderbook_raw/BTCUSDT-bookDepth-2025-07-16.zip"
    
    if not os.path.exists(test_file):
        print("❌ Brak pliku testowego. Najpierw pobierz jakiś plik order book.")
        return
    
    print(f"🔍 Analizuję strukturę pliku: {test_file}")
    
    try:
        with zipfile.ZipFile(test_file) as zip_file:
            csv_filename = zip_file.namelist()[0]
            print(f"📁 Plik w ZIP: {csv_filename}")
            
            # Wczytaj CSV
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                
                print(f"\n📊 Struktura danych:")
                print(f"  Kolumny: {list(df.columns)}")
                print(f"  Typy danych:")
                for col in df.columns:
                    print(f"    {col}: {df[col].dtype}")
                
                print(f"\n📈 Przykładowe dane (pierwsze 5 wierszy):")
                print(df.head())
                
                print(f"\n📊 Statystyki:")
                print(f"  Liczba wierszy: {len(df)}")
                print(f"  Liczba unikalnych timestampów: {df['timestamp'].nunique()}")
                
                # Sprawdź zakres czasowy
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"  Zakres czasowy: {df['timestamp'].min()} do {df['timestamp'].max()}")
                
                # Sprawdź wartości percentage
                print(f"  Wartości percentage: {sorted(df['percentage'].unique())}")
                
                # Sprawdź częstotliwość próbkowania
                time_diff = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
                print(f"  Przykładowa różnica czasowa: {time_diff}")
                
                # Sprawdź ile wierszy na timestamp
                timestamp_counts = df.groupby('timestamp').size()
                print(f"  Średnia liczba wierszy na timestamp: {timestamp_counts.mean():.1f}")
                print(f"  Min wierszy na timestamp: {timestamp_counts.min()}")
                print(f"  Max wierszy na timestamp: {timestamp_counts.max()}")
                
                # Sprawdź strukturę dla jednego timestamp
                sample_timestamp = df['timestamp'].iloc[0]
                sample_data = df[df['timestamp'] == sample_timestamp]
                print(f"\n📋 Przykład danych dla timestamp {sample_timestamp}:")
                print(sample_data)
                
    except Exception as e:
        print(f"❌ Błąd podczas analizy: {e}")

if __name__ == "__main__":
    analyze_orderbook_structure() 