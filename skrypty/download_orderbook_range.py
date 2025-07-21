import os
import requests
from datetime import datetime, timedelta
import sys

ORDERBOOK_DIR = "orderbook_raw"


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def download_orderbook(symbol, date_str, market="futures", out_dir=ORDERBOOK_DIR):
    url = f"https://data.binance.vision/data/{market}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
    local_path = os.path.join(out_dir, f"{symbol}-bookDepth-{date_str}.zip")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ {date_str}: zapisano {local_path}")
            return True
        else:
            print(f"❌ {date_str}: brak pliku ({resp.status_code})")
            return False
    except Exception as e:
        print(f"❌ {date_str}: błąd pobierania: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Użycie: python download_orderbook_range.py SYMBOL DATA_START DATA_END")
        print("Przykład: python download_orderbook_range.py BTCUSDT 2024-06-01 2024-06-10")
        sys.exit(1)
    
    symbol = sys.argv[1]
    date_start = sys.argv[2]
    date_end = sys.argv[3]
    
    try:
        start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    except Exception as e:
        print(f"Błąd parsowania daty: {e}")
        sys.exit(1)
    
    os.makedirs(ORDERBOOK_DIR, exist_ok=True)
    
    total = (end_dt - start_dt).days + 1
    print(f"\n🚀 Pobieranie order book dla {symbol} ({total} dni)")
    print(f"Docelowy katalog: {ORDERBOOK_DIR}\n")
    
    downloaded = 0
    for i, dt in enumerate(daterange(start_dt, end_dt), 1):
        date_str = dt.strftime("%Y-%m-%d")
        if download_orderbook(symbol, date_str):
            downloaded += 1
        print(f"Postęp: {i}/{total} dni, pobrano {downloaded} plików\n{'-'*40}")
    
    print(f"\n🎉 Zakończono. Pobrano {downloaded}/{total} plików do {ORDERBOOK_DIR}")

if __name__ == "__main__":
    main() 