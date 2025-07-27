import os
import requests
import zipfile
import pandas as pd
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path
import io
import argparse
import logging

# Konfiguracja katalogów
ORDERBOOK_DIR = "orderbook_raw"
OHLC_DIR = "ohlc_raw"
METADATA_FILE = "download_metadata.json"

def daterange(start_date, end_date):
    """Generator dat od start_date do end_date (włącznie)"""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def check_file_exists_on_server(url):
    """Sprawdza czy plik istnieje na serwerze"""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def download_and_extract_file(url, local_zip_path, local_csv_path):
    """Pobiera plik ZIP i rozpakowuje go do CSV"""
    try:
        # Pobierz ZIP
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            # Zapisz ZIP
            with open(local_zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Rozpakuj ZIP
            with zipfile.ZipFile(local_zip_path) as zip_file:
                csv_filename = zip_file.namelist()[0]
                zip_file.extract(csv_filename, os.path.dirname(local_csv_path))
                # Zmień nazwę na standardową
                old_path = os.path.join(os.path.dirname(local_csv_path), csv_filename)
                os.rename(old_path, local_csv_path)
            
            # Usuń ZIP
            os.remove(local_zip_path)
            
            file_size = os.path.getsize(local_csv_path)
            print(f"✅ Pobrano i rozpakowano -> {local_csv_path} ({file_size:,} bajtów)")
            return True
        else:
            print(f"❌ Błąd pobierania ({resp.status_code})")
            return False
    except Exception as e:
        print(f"❌ Błąd pobierania: {e}")
        return False

def download_orderbook_data(symbol, date_str, market="futures"):
    """Pobiera dane order book dla podanej daty"""
    url = f"https://data.binance.vision/data/{market}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
    zip_path = os.path.join(ORDERBOOK_DIR, f"{symbol}-bookDepth-{date_str}.zip")
    csv_path = os.path.join(ORDERBOOK_DIR, f"{symbol}-bookDepth-{date_str}.csv")
    
    # Sprawdź czy CSV już istnieje i jest kompletny
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        if file_size > 1000:  # Sprawdź czy plik ma sensowny rozmiar
            print(f"✅ {date_str}: Order book CSV już istnieje ({file_size:,} bajtów)")
            return True
        else:
            print(f"⚠️ {date_str}: Order book CSV istnieje ale jest za mały ({file_size} bajtów) - usuwam")
            os.remove(csv_path)
    
    # Sprawdź czy plik istnieje na serwerze
    print(f"🔍 {date_str}: Sprawdzam order book na serwerze...")
    if not check_file_exists_on_server(url):
        print(f"❌ {date_str}: Order book niedostępny na serwerze")
        return False
    
    # Pobierz plik
    return download_and_extract_file(url, zip_path, csv_path)

def download_ohlc_data(symbol, date_str, interval="1m", market="futures"):
    """Pobiera dane OHLC dla podanej daty"""
    url = f"https://data.binance.vision/data/{market}/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
    zip_path = os.path.join(OHLC_DIR, f"{symbol}-{interval}-{date_str}.zip")
    csv_path = os.path.join(OHLC_DIR, f"{symbol}-{interval}-{date_str}.csv")
    
    # Sprawdź czy CSV już istnieje i jest kompletny
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        if file_size > 1000:  # Sprawdź czy plik ma sensowny rozmiar
            print(f"✅ {date_str}: OHLC CSV już istnieje ({file_size:,} bajtów)")
            return True
        else:
            print(f"⚠️ {date_str}: OHLC CSV istnieje ale jest za mały ({file_size} bajtów) - usuwam")
            os.remove(csv_path)
    
    # Sprawdź czy plik istnieje na serwerze
    print(f"🔍 {date_str}: Sprawdzam OHLC na serwerze...")
    if not check_file_exists_on_server(url):
        print(f"❌ {date_str}: OHLC niedostępny na serwerze")
        return False
    
    # Pobierz plik
    return download_and_extract_file(url, zip_path, csv_path)

def copy_neighboring_file(source_date, target_date, file_type, symbol, interval="1m"):
    """Kopiuje plik z sąsiedniego dnia i dostosowuje timestampy"""
    if file_type == "orderbook":
        source_file = os.path.join(ORDERBOOK_DIR, f"{symbol}-bookDepth-{source_date}.csv")
        target_file = os.path.join(ORDERBOOK_DIR, f"{symbol}-bookDepth-{target_date}.csv")
    else:  # ohlc
        source_file = os.path.join(OHLC_DIR, f"{symbol}-{interval}-{source_date}.csv")
        target_file = os.path.join(OHLC_DIR, f"{symbol}-{interval}-{target_date}.csv")
    
    if not os.path.exists(source_file):
        print(f"❌ Plik źródłowy nie istnieje: {source_file}")
        return False
    
    try:
        # Wczytaj dane źródłowe
        df = pd.read_csv(source_file)
        print(f"📊 Wczytano {len(df):,} wierszy z {source_date}")
        
        # Skopiuj dane
        new_df = df.copy()
        
        # Dostosuj timestampy
        source_start = pd.to_datetime(source_date)
        target_start = pd.to_datetime(target_date)
        time_shift = target_start - source_start
        
        # Dostosuj timestampy (różne formaty dla OHLC i Orderbook)
        if file_type == "orderbook":
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp']) + time_shift
            new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:  # ohlc
            # OHLC ma timestamp w pierwszej kolumnie (open_time)
            new_df.iloc[:, 0] = pd.to_numeric(new_df.iloc[:, 0]) + int(time_shift.total_seconds() * 1000)
        
        # Zapisz nowy plik
        new_df.to_csv(target_file, index=False)
        
        file_size = os.path.getsize(target_file)
        print(f"✅ Utworzono: {target_file} ({len(new_df):,} wierszy, {file_size:,} bajtów)")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas kopiowania: {e}")
        return False

def find_neighboring_dates(missing_date, date_range):
    """Znajduje sąsiednie daty dla uzupełnienia brakującego dnia"""
    missing_dt = datetime.strptime(missing_date, '%Y-%m-%d')
    
    # Sprawdź dzień przed
    day_before = (missing_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    day_before_exists = day_before in date_range
    
    # Sprawdź dzień po
    day_after = (missing_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    day_after_exists = day_after in date_range
    
    if day_before_exists and day_after_exists:
        # Preferuj dzień przed
        return day_before
    elif day_before_exists:
        return day_before
    elif day_after_exists:
        return day_after
    else:
        return None

def fill_missing_files(missing_dates, date_range, symbol, interval="1m"):
    """Uzupełnia brakujące pliki przez kopiowanie sąsiednich dni"""
    print(f"\n🔧 Uzupełniam {len(missing_dates)} brakujących plików...")
    
    filled_dates = []  # Lista dat które zostały uzupełnione
    
    for missing_date in missing_dates:
        print(f"\n📅 Przetwarzam: {missing_date}")
        
        # Znajdź sąsiedni dzień
        neighbor_date = find_neighboring_dates(missing_date, date_range)
        
        if neighbor_date is None:
            print(f"❌ Brak sąsiednich dni dla {missing_date}")
            continue
        
        print(f"📋 Używam danych z: {neighbor_date}")
        
        # Uzupełnij order book
        if not copy_neighboring_file(neighbor_date, missing_date, "orderbook", symbol):
            print(f"❌ Nie udało się uzupełnić order book dla {missing_date}")
            continue
        
        # Uzupełnij OHLC
        if not copy_neighboring_file(neighbor_date, missing_date, "ohlc", symbol, interval):
            print(f"❌ Nie udało się uzupełnić OHLC dla {missing_date}")
            continue
        
        print(f"✅ Uzupełniono dane dla {missing_date}")
        filled_dates.append(missing_date)
    
    return filled_dates

def check_for_long_gaps(missing_dates):
    """Sprawdza czy są przerwy dłuższe niż 2 dni"""
    if not missing_dates:
        return False
    
    # Sortuj daty
    sorted_dates = sorted(missing_dates)
    
    # Sprawdź przerwy między kolejnymi brakującymi datami
    for i in range(len(sorted_dates) - 1):
        date1 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
        date2 = datetime.strptime(sorted_dates[i + 1], '%Y-%m-%d')
        gap_days = (date2 - date1).days
        
        if gap_days > 2:
            print(f"❌ Znaleziono przerwę dłuższą niż 2 dni: {sorted_dates[i]} - {sorted_dates[i+1]} ({gap_days} dni)")
            return True
    
    return False

def main():
    """Główna funkcja pobierania danych"""
    parser = argparse.ArgumentParser(description='Pobierz dane OHLC i Orderbook z Binance')
    parser.add_argument('symbol', help='Symbol kryptowaluty (np. BTCUSDT)')
    parser.add_argument('start_date', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('end_date', help='Data końcowa (YYYY-MM-DD)')
    parser.add_argument('--interval', default='1m', help='Interwał OHLC (domyślnie: 1m)')
    parser.add_argument('--market', default='futures', help='Typ rynku (domyślnie: futures)')
    
    args = parser.parse_args()
    
    # Konwertuj stringi dat na datetime
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ Błąd formatu daty: {e}")
        print("Użyj formatu: YYYY-MM-DD (np. 2023-01-01)")
        return
    
    symbol = args.symbol
    interval = args.interval
    market = args.market
    
    print(f"🚀 Rozpoczynanie pobierania danych dla {symbol}")
    print(f"📅 Zakres: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Interwał: {interval}")
    print(f"🏪 Rynek: {market}")
    
    # Utwórz katalogi jeśli nie istnieją
    os.makedirs(ORDERBOOK_DIR, exist_ok=True)
    os.makedirs(OHLC_DIR, exist_ok=True)
    
    # KROK 1: Pobierz wszystkie daty z zakresu
    all_dates = [date.strftime('%Y-%m-%d') for date in daterange(start_date, end_date)]
    print(f"\n📋 Zakres zawiera {len(all_dates)} dni")
    
    # KROK 2: Pobierz dane
    print(f"\n📥 Pobieram dane...")
    successful_orderbook = []
    successful_ohlc = []
    missing_orderbook = []
    missing_ohlc = []
    
    for date_str in all_dates:
        print(f"\n📅 Przetwarzam: {date_str}")
        
        # Pobierz order book
        if download_orderbook_data(symbol, date_str, market):
            successful_orderbook.append(date_str)
        else:
            missing_orderbook.append(date_str)
        
        # Pobierz OHLC
        if download_ohlc_data(symbol, date_str, interval, market):
            successful_ohlc.append(date_str)
        else:
            missing_ohlc.append(date_str)
    
    # KROK 3: Sprawdź przerwy
    print(f"\n🔍 Analiza brakujących plików:")
    print(f"   Order book: {len(successful_orderbook)} pobranych, {len(missing_orderbook)} brakujących")
    print(f"   OHLC: {len(successful_ohlc)} pobranych, {len(missing_ohlc)} brakujących")
    
    if missing_orderbook:
        print(f"   Brakujące order book: {missing_orderbook}")
    if missing_ohlc:
        print(f"   Brakujące OHLC: {missing_ohlc}")
    
    # KROK 4: Sprawdź czy są przerwy dłuższe niż 2 dni
    if check_for_long_gaps(missing_orderbook) or check_for_long_gaps(missing_ohlc):
        print(f"\n❌ Znaleziono przerwy dłuższe niż 2 dni. Kończę działanie.")
        return
    
    # KROK 5: Uzupełnij brakujące pliki
    filled_orderbook = fill_missing_files(missing_orderbook, all_dates, symbol, interval)
    filled_ohlc = fill_missing_files(missing_ohlc, all_dates, symbol, interval)
    
    # KROK 6: Podsumowanie
    print(f"\n🎉 Proces zakończony!")
    print(f"📊 Pobrano order book: {len(successful_orderbook)}")
    print(f"📊 Pobrano OHLC: {len(successful_ohlc)}")
    print(f"🔧 Uzupełniono order book: {len(filled_orderbook)}")
    print(f"🔧 Uzupełniono OHLC: {len(filled_ohlc)}")
    
    # Wyświetl szczegółowe informacje o uzupełnionych datach
    if filled_orderbook:
        print(f"\n📋 Uzupełnione daty order book: {', '.join(filled_orderbook)}")
    if filled_ohlc:
        print(f"📋 Uzupełnione daty OHLC: {', '.join(filled_ohlc)}")
    
    # Sprawdź czy wszystkie daty zostały uzupełnione
    if len(filled_orderbook) < len(missing_orderbook):
        print(f"⚠️ Nie udało się uzupełnić wszystkich order book: {len(missing_orderbook) - len(filled_orderbook)} pozostało")
    if len(filled_ohlc) < len(missing_ohlc):
        print(f"⚠️ Nie udało się uzupełnić wszystkich OHLC: {len(missing_ohlc) - len(filled_ohlc)} pozostało")
    
    if len(filled_orderbook) == len(missing_orderbook) and len(filled_ohlc) == len(missing_ohlc):
        print(f"✅ Wszystkie brakujące pliki zostały pomyślnie uzupełnione!")
    
    # Zapisz metadata
    metadata = {
        'symbol': symbol,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'interval': interval,
        'market': market,
        'created_at': datetime.now().isoformat(),
        'successful_orderbook': successful_orderbook,
        'successful_ohlc': successful_ohlc,
        'missing_orderbook': missing_orderbook,
        'missing_ohlc': missing_ohlc
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata zapisana: {METADATA_FILE}")

if __name__ == "__main__":
    main() 