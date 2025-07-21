import requests
import re
from datetime import datetime, timedelta

def check_available_dates(symbol, market='futures'):
    """Sprawdza dostępne daty dla order book danych"""
    base_url = f"https://data.binance.vision/?prefix=data/{market}/um/daily/bookDepth/{symbol}/"
    
    print(f"🔍 Sprawdzam dostępne daty dla {symbol} ({market})")
    print(f"URL: {base_url}")
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        
        # Szukamy plików w formacie BTCUSDT-bookDepth-YYYY-MM-DD.zip
        pattern = f"{symbol}-bookDepth-([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})"
        matches = re.findall(pattern, response.text)
        
        if matches:
            dates = sorted(matches)
            print(f"✅ Znaleziono {len(dates)} dostępnych dat:")
            
            # Pokaż pierwsze i ostatnie 5 dat
            if len(dates) <= 10:
                for date in dates:
                    print(f"  - {date}")
            else:
                print("  Pierwsze 5 dat:")
                for date in dates[:5]:
                    print(f"    - {date}")
                print("  Ostatnie 5 dat:")
                for date in dates[-5:]:
                    print(f"    - {date}")
                print(f"  ... i {len(dates) - 10} innych dat")
            
            # Sprawdź zakres dat
            first_date = datetime.strptime(dates[0], "%Y-%m-%d")
            last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
            today = datetime.now()
            
            print(f"\n📅 Zakres dostępnych danych:")
            print(f"  Najwcześniejsza data: {dates[0]}")
            print(f"  Najpóźniejsza data: {dates[-1]}")
            print(f"  Dzisiaj: {today.strftime('%Y-%m-%d')}")
            print(f"  Różnica od dzisiaj: {(today - last_date).days} dni")
            
            return dates
        else:
            print("❌ Nie znaleziono żadnych dostępnych dat")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd podczas sprawdzania: {e}")
        return []

if __name__ == "__main__":
    # Sprawdź dla BTCUSDT
    available_dates = check_available_dates("BTCUSDT")
    
    if available_dates:
        print(f"\n💡 Sugerowane daty do pobrania:")
        # Pokaż ostatnie 10 dostępnych dat
        recent_dates = available_dates[-10:]
        for date in recent_dates:
            print(f"  {date}")
        
        print(f"\n🎯 Przykład użycia:")
        if len(recent_dates) >= 2:
            start_date = recent_dates[0]
            end_date = recent_dates[-1]
            print(f"  python simple_orderbook_downloader.py BTCUSDT {start_date} {end_date}") 