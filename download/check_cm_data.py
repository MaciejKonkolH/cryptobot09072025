import requests
import os
from datetime import datetime

def check_cm_data_availability():
    """Sprawdza dostępność danych Coin-Margined Futures dla brakujących dat"""
    
    # Daty z lukami
    missing_dates = ['2024-04-18', '2023-02-08', '2023-02-09']
    
    base_url = "https://data.binance.vision/data/futures/cm/daily/bookDepth"
    
    print("=== SPRAWDZANIE DANYCH COIN-MARGINED FUTURES ===")
    
    for date in missing_dates:
        # URL dla Coin-Margined Futures
        cm_url = f"{base_url}/BTCUSD/BTCUSD-bookDepth-{date}.zip"
        
        print(f"\n🔍 Sprawdzam {date}: {cm_url}")
        
        try:
            response = requests.head(cm_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ DOSTĘPNE! Status: {response.status_code}")
                print(f"   Rozmiar: {response.headers.get('content-length', 'N/A')} bajtów")
            else:
                print(f"❌ NIEDOSTĘPNE! Status: {response.status_code}")
        except Exception as e:
            print(f"❌ BŁĄD: {e}")
    
    print(f"\n=== INNE MOŻLIWOŚCI ===")
    print("1. Sprawdź dane spot (spot/daily/bookDepth)")
    print("2. Sprawdź dane futures z innym symbolem")
    print("3. Sprawdź dane z innym przedziałem czasowym")

if __name__ == "__main__":
    check_cm_data_availability() 