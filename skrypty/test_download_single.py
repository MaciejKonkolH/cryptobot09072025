import requests
import zipfile
import io
import pandas as pd

def test_download_single():
    """Testuje pobieranie pojedynczego pliku order book"""
    
    # URL pliku, który wiemy że istnieje
    url = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2025-07-16.zip"
    
    print(f"🔍 Pobieram plik: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print(f"✅ Pobrano {len(response.content)} bajtów")
        
        # Rozpakuj ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = zip_file.namelist()[0]
            print(f"📁 Plik w ZIP: {csv_filename}")
            
            # Wczytaj CSV
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                
                print(f"📊 Dane:")
                print(f"  Wiersze: {len(df)}")
                print(f"  Kolumny: {list(df.columns)}")
                print(f"  Pierwsze 3 wiersze:")
                print(df.head(3))
                
                # Sprawdź zakres czasowy
                if 'timestamp' in df.columns:
                    # Sprawdź format timestamp
                    print(f"\n⏰ Zakres czasowy:")
                    print(f"  Od: {df['timestamp'].min()}")
                    print(f"  Do: {df['timestamp'].max()}")
                    print(f"  Liczba unikalnych timestampów: {df['timestamp'].nunique()}")
                    
                    # Sprawdź częstotliwość próbkowania
                    if df['timestamp'].nunique() > 1:
                        time_diff = pd.to_datetime(df['timestamp'].iloc[1]) - pd.to_datetime(df['timestamp'].iloc[0])
                        print(f"  Przykładowa różnica czasowa: {time_diff}")
                
    except Exception as e:
        print(f"❌ Błąd: {e}")

if __name__ == "__main__":
    test_download_single() 