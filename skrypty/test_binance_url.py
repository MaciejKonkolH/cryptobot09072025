import requests

def test_binance_url():
    """Testuje bezpośrednio URL z Binance Vision"""
    
    # URL z wyników wyszukiwania
    url = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2025-07-16.zip"
    
    print(f"🔍 Testuję URL: {url}")
    
    try:
        response = requests.head(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Plik istnieje!")
            print(f"Rozmiar: {response.headers.get('content-length', 'nieznany')} bajtów")
        else:
            print("❌ Plik nie istnieje")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")
    
    # Sprawdźmy też katalog
    dir_url = "https://data.binance.vision/?prefix=data/futures/um/daily/bookDepth/BTCUSDT/"
    print(f"\n🔍 Testuję katalog: {dir_url}")
    
    try:
        response = requests.get(dir_url)
        print(f"Status: {response.status_code}")
        print(f"Rozmiar odpowiedzi: {len(response.text)} znaków")
        
        if response.status_code == 200:
            print("✅ Katalog dostępny")
            # Pokaż pierwsze 500 znaków
            print("Pierwsze 500 znaków:")
            print(response.text[:500])
        else:
            print("❌ Katalog niedostępny")
            
    except Exception as e:
        print(f"❌ Błąd: {e}")

if __name__ == "__main__":
    test_binance_url() 