import requests
import zipfile
import io
import pandas as pd
from datetime import datetime

def check_contract_data():
    """Sprawdza dane z kontraktów z datą wygaśnięcia"""
    
    # URL z kontraktem z datą wygaśnięcia
    url = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT_240628/BTCUSDT_240628-bookDepth-2024-04-18.zip"
    
    print(f"🔍 Sprawdzam: {url}")
    
    try:
        # Pobierz plik
        response = requests.get(url)
        response.raise_for_status()
        
        print(f"✅ Plik dostępny! Rozmiar: {len(response.content):,} bajtów")
        
        # Rozpakuj ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = zip_file.namelist()[0]
            print(f"📁 Plik CSV: {csv_filename}")
            
            # Wczytaj dane
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                
                print(f"📊 Dane z kontraktu:")
                print(f"   Liczba wierszy: {len(df):,}")
                print(f"   Kolumny: {list(df.columns)}")
                print(f"   Zakres czasowy: {df['timestamp'].min()} - {df['timestamp'].max()}")
                print(f"   Przykładowe dane:")
                print(df.head())
                
                return True
                
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return False

def check_available_contracts():
    """Sprawdza dostępne kontrakty dla 2024-04-18"""
    
    base_url = "https://data.binance.vision/data/futures/um/daily/bookDepth"
    date = "2024-04-18"
    
    # Lista możliwych kontraktów
    contracts = [
        "BTCUSDT_240628",  # 28 czerwca 2024
        "BTCUSDT_240927",  # 27 września 2024
        "BTCUSDT_241227",  # 27 grudnia 2024
        "BTCUSDT_250328",  # 28 marca 2025
    ]
    
    print(f"🔍 Sprawdzam dostępne kontrakty dla {date}:")
    
    for contract in contracts:
        url = f"{base_url}/{contract}/{contract}-bookDepth-{date}.zip"
        try:
            response = requests.head(url)
            if response.status_code == 200:
                print(f"✅ {contract}: DOSTĘPNY")
            else:
                print(f"❌ {contract}: {response.status_code}")
        except Exception as e:
            print(f"❌ {contract}: BŁĄD - {e}")

if __name__ == "__main__":
    print("=== SPRAWDZANIE DANYCH Z KONTRAKTÓW ===")
    check_contract_data()
    print("\n=== SPRAWDZANIE DOSTĘPNYCH KONTRAKTÓW ===")
    check_available_contracts() 