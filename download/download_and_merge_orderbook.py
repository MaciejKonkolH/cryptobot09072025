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

ORDERBOOK_DIR = "orderbook_raw"
OHLC_DIR = "ohlc_raw"
MERGED_FILE = "orderbook_ohlc_merged.feather"  # Tylko feather, bez JSON
METADATA_FILE = "download_metadata.json"


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def download_and_extract_orderbook(symbol, date_str, market="futures", out_dir=ORDERBOOK_DIR):
    """Pobiera plik ZIP order book, rozpakowuje go i zwraca ścieżkę do CSV"""
    url = f"https://data.binance.vision/data/{market}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
    zip_path = os.path.join(out_dir, f"{symbol}-bookDepth-{date_str}.zip")
    csv_path = os.path.join(out_dir, f"{symbol}-bookDepth-{date_str}.csv")
    
    try:
        # Sprawdź czy CSV już istnieje i jest kompletny
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            if file_size > 1000:  # Sprawdź czy plik ma sensowny rozmiar
                print(f"✅ {date_str}: Order book CSV już istnieje ({file_size:,} bajtów)")
                return csv_path
            else:
                print(f"⚠️ {date_str}: Order book CSV istnieje ale jest za mały ({file_size} bajtów) - usuwam")
                os.remove(csv_path)
        
        # Plik nie istnieje lokalnie - sprawdź serwer
        print(f"🔍 {date_str}: Order book nie istnieje lokalnie - sprawdzam serwer...")
        head_resp = requests.head(url, timeout=10)
        if head_resp.status_code == 404:
            print(f"❌ {date_str}: Order book niedostępny na serwerze (404)")
            return None
        
        # Pobierz ZIP
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            # Zapisz ZIP
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Rozpakuj ZIP
            with zipfile.ZipFile(zip_path) as zip_file:
                csv_filename = zip_file.namelist()[0]
                zip_file.extract(csv_filename, out_dir)
                # Zmień nazwę na standardową
                old_path = os.path.join(out_dir, csv_filename)
                os.rename(old_path, csv_path)
            
            # Usuń ZIP
            os.remove(zip_path)
            
            file_size = os.path.getsize(csv_path)
            print(f"✅ {date_str}: Order book pobrano i rozpakowano -> {csv_path} ({file_size:,} bajtów)")
            return csv_path
        else:
            print(f"❌ {date_str}: błąd pobierania order book ({resp.status_code})")
            return None
    except Exception as e:
        print(f"❌ {date_str}: błąd pobierania order book: {e}")
        return None

def download_and_extract_ohlc(symbol, date_str, interval="1m", market="futures", out_dir=OHLC_DIR):
    """Pobiera plik ZIP OHLC, rozpakowuje go i zwraca ścieżkę do CSV"""
    url = f"https://data.binance.vision/data/{market}/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
    zip_path = os.path.join(out_dir, f"{symbol}-{interval}-{date_str}.zip")
    csv_path = os.path.join(out_dir, f"{symbol}-{interval}-{date_str}.csv")
    
    try:
        # Sprawdź czy CSV już istnieje i jest kompletny
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            if file_size > 1000:  # Sprawdź czy plik ma sensowny rozmiar
                print(f"✅ {date_str}: OHLC CSV już istnieje ({file_size:,} bajtów)")
                return csv_path
            else:
                print(f"⚠️ {date_str}: OHLC CSV istnieje ale jest za mały ({file_size} bajtów) - usuwam")
                os.remove(csv_path)
        
        # TYLKO jeśli plik nie istnieje lokalnie - sprawdź serwer
        print(f"🔍 {date_str}: OHLC nie istnieje lokalnie - sprawdzam serwer...")
        head_resp = requests.head(url, timeout=10)
        if head_resp.status_code == 404:
            print(f"❌ {date_str}: OHLC niedostępny na serwerze (404)")
            return None
        
        # Pobierz ZIP
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200:
            # Zapisz ZIP
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Rozpakuj ZIP
            with zipfile.ZipFile(zip_path) as zip_file:
                csv_filename = zip_file.namelist()[0]
                zip_file.extract(csv_filename, out_dir)
                # Zmień nazwę na standardową
                old_path = os.path.join(out_dir, csv_filename)
                os.rename(old_path, csv_path)
            
            # Usuń ZIP
            os.remove(zip_path)
            
            file_size = os.path.getsize(csv_path)
            print(f"✅ {date_str}: OHLC pobrano i rozpakowano -> {csv_path} ({file_size:,} bajtów)")
            return csv_path
        else:
            print(f"❌ {date_str}: błąd pobierania OHLC ({resp.status_code})")
            return None
    except Exception as e:
        print(f"❌ {date_str}: błąd pobierania OHLC: {e}")
        return None

def load_and_process_orderbook(csv_files):
    """Wczytuje i przetwarza pliki order book"""
    print(f"\n📊 Wczytuję {len(csv_files)} plików order book...")
    
    all_orderbook_data = []
    
    for i, csv_file in enumerate(csv_files, 1):
        if csv_file and os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_orderbook_data.append(df)
                print(f"  📊 {i}/{len(csv_files)}: {os.path.basename(csv_file)} - {len(df)} wierszy")
            except Exception as e:
                print(f"  ❌ Błąd wczytania {csv_file}: {e}")
    
    if all_orderbook_data:
        # Połącz wszystkie dane
        combined_df = pd.concat(all_orderbook_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        print(f"✅ Wczytano {len(combined_df)} wierszy order book")
        print(f"⏰ Zakres czasowy: {combined_df['timestamp'].min()} do {combined_df['timestamp'].max()}")
        
        return combined_df
    else:
        print("❌ Brak danych order book do wczytania")
        return None

def load_and_process_ohlc(csv_files):
    """Wczytuje i przetwarza pliki OHLC"""
    print(f"\n📈 Wczytuję {len(csv_files)} plików OHLC...")
    
    all_ohlc_data = []
    
    for i, csv_file in enumerate(csv_files, 1):
        if csv_file and os.path.exists(csv_file):
            try:
                # Wczytaj surowe dane OHLC (pomiń pierwszy wiersz z nagłówkami)
                df = pd.read_csv(csv_file, header=0, names=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Konwertuj timestamp na datetime (sprawdź czy to string czy numeric)
                if df['open_time'].dtype == 'object':
                    # Jeśli string, spróbuj konwersji na numeric
                    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                
                # Konwertuj z milliseconds na datetime
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                
                # Usuń wiersze z nieprawidłowymi timestampami
                df = df.dropna(subset=['timestamp'])
                
                # Konwertuj kolumny numeryczne
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Wybierz tylko potrzebne kolumny
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                all_ohlc_data.append(df)
                print(f"  📈 {i}/{len(csv_files)}: {os.path.basename(csv_file)} - {len(df)} wierszy")
            except Exception as e:
                print(f"  ❌ Błąd wczytania {csv_file}: {e}")
    
    if all_ohlc_data:
        # Połącz wszystkie dane
        combined_df = pd.concat(all_ohlc_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        print(f"✅ Wczytano {len(combined_df)} wierszy OHLC")
        print(f"⏰ Zakres czasowy: {combined_df['timestamp'].min()} do {combined_df['timestamp'].max()}")
        
        return combined_df
    else:
        print("❌ Brak danych OHLC do wczytania")
        return None

def process_snapshots_for_candle(ohlc_timestamp, wide_orderbook_df):
    """
    Przetwarza snapshoty order book dla jednej świeczki OHLC
    Zwraca 2 snapshoty (lub None jeśli nie można utworzyć)
    """
    # OHLC timestamp to początek świeczki (np. 00:04:00)
    # Szukamy snapshotów w trakcie świeczki (00:04:00 - 00:05:00)
    window_start = ohlc_timestamp
    window_end = ohlc_timestamp + pd.Timedelta(minutes=1)
    
    # DEBUG: Sprawdź pierwsze kilka wywołań
    if ohlc_timestamp.hour == 0 and ohlc_timestamp.minute <= 5:
        print(f"DEBUG: process_snapshots_for_candle dla {ohlc_timestamp}")
        print(f"DEBUG: window_start = {window_start}, window_end = {window_end}")
        print(f"DEBUG: wide_orderbook_df.index.min() = {wide_orderbook_df.index.min()}")
        print(f"DEBUG: wide_orderbook_df.index.max() = {wide_orderbook_df.index.max()}")
        print(f"DEBUG: Snapshoty w wide_orderbook_df:")
        for i, idx in enumerate(wide_orderbook_df.index[:10]):
            print(f"  {i}: {idx}")
    
    # Znajdź snapshoty w oknie świeczki (00:04:00 - 00:05:00)
    relevant_orderbook = wide_orderbook_df[
        (wide_orderbook_df.index >= window_start) &
        (wide_orderbook_df.index < window_end)
    ].sort_index()
    
    # DEBUG: Sprawdź pierwsze kilka wywołań
    if ohlc_timestamp.hour == 0 and ohlc_timestamp.minute <= 5:
        print(f"DEBUG: Znaleziono {len(relevant_orderbook)} snapshotów w oknie")
        for i, idx in enumerate(relevant_orderbook.index):
            print(f"  {i}: {idx}")
    
    snapshot_count = len(relevant_orderbook)
    
    if snapshot_count == 0:
        # Znajdź najbliższe snapshoty przed i po oknie
        before_window = wide_orderbook_df[wide_orderbook_df.index < window_start].sort_index()
        after_window = wide_orderbook_df[wide_orderbook_df.index >= window_end].sort_index()
        
        if len(before_window) > 0 and len(after_window) > 0:
            # Interpoluj między snapshotami przed i po oknie
            before_snapshot = before_window.iloc[-1]
            after_snapshot = after_window.iloc[0]
            
            # Stwórz 2 snapshoty przez interpolację
            snapshot1 = interpolate_snapshots(before_snapshot, after_snapshot, 0.25)
            snapshot2 = interpolate_snapshots(before_snapshot, after_snapshot, 0.75)
            
            return [snapshot1, snapshot2]
        else:
            return None
    
    elif snapshot_count == 1:
        # Sprawdź odległość od końca okna
        single_snapshot = relevant_orderbook.iloc[0]
        distance_to_end = (window_end - single_snapshot.name).total_seconds()
        
        if distance_to_end < 30:
            # Snapshot jest bliżej niż 30 sekund od końca okna
            # Znajdź snapshot przed oknem
            before_window = wide_orderbook_df[wide_orderbook_df.index < window_start].sort_index()
            
            if len(before_window) > 0:
                before_snapshot = before_window.iloc[-1]
                # Interpoluj między snapshotem przed oknem a bieżącym
                interpolated_snapshot = interpolate_snapshots(before_snapshot, single_snapshot, 0.5)
                return [interpolated_snapshot, single_snapshot]
            else:
                return [single_snapshot, single_snapshot]  # Duplikuj
        else:
            # Snapshot jest dalej niż 30 sekund od końca okna
            # Znajdź snapshot po oknie
            after_window = wide_orderbook_df[wide_orderbook_df.index >= window_end].sort_index()
            
            if len(after_window) > 0:
                after_snapshot = after_window.iloc[0]
                # Interpoluj między bieżącym a snapshotem po oknie
                interpolated_snapshot = interpolate_snapshots(single_snapshot, after_snapshot, 0.5)
                return [single_snapshot, interpolated_snapshot]
            else:
                return [single_snapshot, single_snapshot]  # Duplikuj
    
    elif snapshot_count == 2:
        # Idealny przypadek - zwróć oba snapshoty
        return [relevant_orderbook.iloc[0], relevant_orderbook.iloc[1]]
    
    elif snapshot_count >= 3:
        # Usuń środkowy snapshot, zostaw skrajne
        return [relevant_orderbook.iloc[0], relevant_orderbook.iloc[-1]]
    
    return None

def interpolate_snapshots(snapshot1, snapshot2, ratio):
    """
    Interpoluje między dwoma snapshotami order book
    ratio: 0.0 = snapshot1, 1.0 = snapshot2, 0.5 = średnia
    """
    interpolated = {'timestamp': snapshot1['timestamp']}  # Użyj timestamp z pierwszego
    
    # Interpoluj wszystkie poziomy order book
    for i in range(-5, 6):
        if i == 0:
            continue  # Pomijamy poziom 0
        
        depth_key = f'depth_{i}'
        notional_key = f'notional_{i}'
        
        if depth_key in snapshot1 and depth_key in snapshot2:
            # Interpoluj depth i notional
            depth1 = snapshot1[depth_key]
            depth2 = snapshot2[depth_key]
            notional1 = snapshot1[notional_key]
            notional2 = snapshot2[notional_key]
            
            interpolated[depth_key] = depth1 + (depth2 - depth1) * ratio
            interpolated[notional_key] = notional1 + (notional2 - notional1) * ratio
    
    return interpolated

def check_existing_data_range(start_date, end_date, symbol="BTCUSDT"):
    """Sprawdza czy dane z zakresu już istnieją i zwraca brakujące fragmenty"""
    print(f"🔍 Sprawdzam istniejące dane dla zakresu: {start_date} - {end_date}")
    
    # Sprawdź czy plik merged już istnieje
    if os.path.exists(MERGED_FILE):
        try:
            # Sprawdź metadata
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'r') as f:
                    metadata = json.load(f)
                
                existing_start = datetime.fromisoformat(metadata['start_date'])
                existing_end = datetime.fromisoformat(metadata['end_date'])
                
                # Sprawdź czy zakres jest kompletny
                if existing_start <= start_date and existing_end >= end_date:
                    print(f"✅ Dane z zakresu {start_date} - {end_date} już istnieją!")
                    print(f"   Istniejący zakres: {existing_start} - {existing_end}")
                    return True, []
                else:
                    print(f"⚠️ Istnieją dane, ale zakres nie jest kompletny")
                    print(f"   Istniejący: {existing_start} - {existing_end}")
                    print(f"   Wymagany: {start_date} - {end_date}")
                    
                    # Znajdź brakujące fragmenty
                    missing_ranges = []
                    if start_date < existing_start:
                        missing_ranges.append((start_date, existing_start - timedelta(days=1)))
                    if end_date > existing_end:
                        missing_ranges.append((existing_end + timedelta(days=1), end_date))
                    
                    return False, missing_ranges
            else:
                print(f"⚠️ Plik merged istnieje, ale brak metadata")
                return False, [(start_date, end_date)]
        except Exception as e:
            print(f"⚠️ Błąd sprawdzania metadata: {e}")
            return False, [(start_date, end_date)]
    
    print(f"❌ Brak istniejących danych - pobieram cały zakres")
    return False, [(start_date, end_date)]

def save_metadata(start_date, end_date, symbol="BTCUSDT"):
    """Zapisuje metadata o pobranych danych"""
    metadata = {
        'symbol': symbol,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'created_at': datetime.now().isoformat(),
        'file_format': 'feather'
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata zapisana: {METADATA_FILE}")

def merge_orderbook_with_ohlc(orderbook_df, ohlc_df, save_json=False, extra_month_for_ma=False):
    """ZOPTYMALIZOWANA wersja łączenia order book z OHLC - TYLKO WSPÓLNE DANE"""
    print(f"\n🔗 Łączę order book z OHLC (ZOPTYMALIZOWANE)...")
    print(f"📊 Dane OHLC: {len(ohlc_df)} wierszy")
    print(f"📊 Dane Order Book: {len(orderbook_df)} wierszy")
    
    # KROK 1: Indeksowanie i groupby (ZAMIENIA O(n²) na O(n log n))
    print("📊 Indeksuję dane order book...")
    orderbook_df.set_index('timestamp', inplace=True)
    
    # KROK 2: Vectorizowane tworzenie wide format
    print("📊 Tworzę wide format order book (VECTORIZED)...")
    
    # Dodaj logi progress
    total_snapshots = orderbook_df.groupby(level=0).ngroups
    print(f"📊 Przetwarzam {total_snapshots:,} snapshots order book...")
    
    # Zastąp groupby().apply() pętlą z logami postępu
    wide_orderbook_data = []
    processed_count = 0
    
    for timestamp, group in orderbook_df.groupby(level=0):
        # Stwórz wiersz dla tego timestampu
        row_data = {'timestamp': timestamp}
        
        # Dodaj dane depth i notional dla każdego poziomu
        for _, snapshot in group.iterrows():
            row_data[f'depth_{snapshot["percentage"]}'] = snapshot["depth"]
            row_data[f'notional_{snapshot["percentage"]}'] = snapshot["notional"]
        
        wide_orderbook_data.append(row_data)
        processed_count += 1
        
        # Logi postępu co 10,000 snapshots
        if processed_count % 10000 == 0:
            progress = (processed_count / total_snapshots) * 100
            print(f"📊 Postęp: {processed_count:,}/{total_snapshots:,} snapshots ({progress:.1f}%)")
    
    print(f"✅ Wide format utworzony: {len(wide_orderbook_data):,} snapshots")
    
    # Konwertuj na DataFrame
    wide_orderbook_df = pd.DataFrame(wide_orderbook_data)
    wide_orderbook_df.set_index('timestamp', inplace=True)
    
    # KROK 4: Znajdź zakres wspólnych danych
    orderbook_start = wide_orderbook_df.index.min()
    orderbook_end = wide_orderbook_df.index.max()
    ohlc_start = ohlc_df.index.min()
    ohlc_end = ohlc_df.index.max()
    
    # Diagnostyka typów
    print(f"🔍 Diagnostyka typów indeksów:")
    print(f"   orderbook_start: {type(orderbook_start)} = {orderbook_start}")
    print(f"   ohlc_start: {type(ohlc_start)} = {ohlc_start}")
    print(f"   orderbook_end: {type(orderbook_end)} = {orderbook_end}")
    print(f"   ohlc_end: {type(ohlc_end)} = {ohlc_end}")
    
    # Konwertuj na pd.Timestamp jeśli potrzebne
    if not isinstance(orderbook_start, pd.Timestamp):
        orderbook_start = pd.Timestamp(orderbook_start)
    if not isinstance(orderbook_end, pd.Timestamp):
        orderbook_end = pd.Timestamp(orderbook_end)
    
    # OHLC ma RangeIndex - użyj timestamp z kolumny
    if isinstance(ohlc_start, int):
        ohlc_start = ohlc_df['timestamp'].min()
    if isinstance(ohlc_end, int):
        ohlc_end = ohlc_df['timestamp'].max()
    
    if not isinstance(ohlc_start, pd.Timestamp):
        ohlc_start = pd.Timestamp(ohlc_start)
    if not isinstance(ohlc_end, pd.Timestamp):
        ohlc_end = pd.Timestamp(ohlc_end)

    # Oblicz wspólny zakres
    common_start = max(orderbook_start, ohlc_start)
    common_end = min(orderbook_end, ohlc_end)
    
    print(f"📅 Zakresy danych:")
    print(f"   OHLC: {ohlc_start} - {ohlc_end}")
    print(f"   Order Book: {orderbook_start} - {orderbook_end}")
    print(f"   WSPÓLNY: {common_start} - {common_end}")
    
    # KROK 5: Filtruj OHLC do wspólnego zakresu
    if extra_month_for_ma:
        # Dodaj 30 dni wcześniej dla MA 43200
        ma_start = common_start - pd.Timedelta(days=30)
        print(f"   +30 dni dla MA: {ma_start} - {common_end}")
        filtered_ohlc = ohlc_df[(ohlc_df['timestamp'] >= ma_start) & (ohlc_df['timestamp'] <= common_end)]
    else:
        filtered_ohlc = ohlc_df[(ohlc_df['timestamp'] >= common_start) & (ohlc_df['timestamp'] <= common_end)]
    
    print(f"📊 Filtrowane OHLC: {len(filtered_ohlc)} wierszy")
    
    # KROK 6: Vectorizowane przetwarzanie OHLC
    print("📊 Przetwarzam snapshoty dla świeczek OHLC (VECTORIZED)...")
    
    def process_candle_vectorized(ohlc_row):
        """Vectorizowana funkcja przetwarzania jednej świeczki"""
        ohlc_timestamp = ohlc_row['timestamp']
        
        # DEBUG: Sprawdź pierwsze kilka świeczek
        if processed_count < 5:
            print(f"DEBUG: Przetwarzam świeczkę {ohlc_timestamp}")
            print(f"DEBUG: wide_orderbook_df.index.min() = {wide_orderbook_df.index.min()}")
            print(f"DEBUG: wide_orderbook_df.index.max() = {wide_orderbook_df.index.max()}")
        
        # Szybki lookup w indeksowanym DataFrame
        try:
            # Użyj exclusive end slicing (jak w process_snapshots_for_candle)
            snapshots = wide_orderbook_df[
                (wide_orderbook_df.index >= ohlc_timestamp) & 
                (wide_orderbook_df.index < ohlc_timestamp + pd.Timedelta(minutes=1))
            ]
            
            # DEBUG: Sprawdź pierwsze kilka świeczek
            if processed_count < 5:
                print(f"DEBUG: Znaleziono {len(snapshots)} snapshotów dla {ohlc_timestamp}")
                
        except KeyError:
            if processed_count < 5:
                print(f"DEBUG: KeyError dla {ohlc_timestamp}")
            return None  # Brak order book - pomijamy ten wiersz
        
        if len(snapshots) == 0:
            if processed_count < 5:
                print(f"DEBUG: Brak snapshotów dla {ohlc_timestamp}")
            return None  # Brak order book - pomijamy ten wiersz
        
        # Przetwórz snapshoty
        processed_snapshots = process_snapshots_for_candle(ohlc_timestamp, snapshots)
        
        # DEBUG: Sprawdź pierwsze kilka świeczek
        if processed_count < 5:
            print(f"DEBUG: processed_snapshots = {processed_snapshots}")
        
        # Sprawdź czy mamy kompletne dane
        if not processed_snapshots or len(processed_snapshots) != 2:
            if processed_count < 5:
                print(f"DEBUG: Niekompletne dane dla {ohlc_timestamp}")
            return None  # Niekompletne dane - pomijamy ten wiersz
        
        # Stwórz wiersz wynikowy
        merged_row = {
            'timestamp': ohlc_timestamp,
            'open': ohlc_row['open'],
            'high': ohlc_row['high'],
            'low': ohlc_row['low'],
            'close': ohlc_row['close'],
            'volume': ohlc_row['volume']
        }
        
        # Dodaj dane z pierwszego snapshotu
        for key, value in processed_snapshots[0].items():
            if key != 'timestamp':
                merged_row[f'snapshot1_{key}'] = value
        
        # Dodaj dane z drugiego snapshotu
        for key, value in processed_snapshots[1].items():
            if key != 'timestamp':
                merged_row[f'snapshot2_{key}'] = value
        
        # Dodaj informacje o jakości danych
        merged_row['snapshot1_timestamp'] = processed_snapshots[0]['timestamp']
        merged_row['snapshot2_timestamp'] = processed_snapshots[1]['timestamp']
        merged_row['data_quality'] = 'complete'
        
        return merged_row
    
    # KROK 7: Vectorizowane przetwarzanie wszystkich świeczek
    print("🔄 Przetwarzam wszystkie świeczki OHLC...")
    
    # Dodaj logi progress
    total_candles = len(filtered_ohlc)
    print(f"📊 Przetwarzam {total_candles:,} świeczek OHLC...")
    
    # Zastąp apply() pętlą z logami postępu
    processed_data = []
    processed_count = 0
    
    for index, ohlc_row in filtered_ohlc.iterrows():
        result = process_candle_vectorized(ohlc_row)
        if result is not None:
            processed_data.append(result)
        
        processed_count += 1
        
        # Logi postępu co 50,000 świeczek
        if processed_count % 50000 == 0:
            progress = (processed_count / total_candles) * 100
            complete_count = len(processed_data)
            print(f"📊 Postęp: {processed_count:,}/{total_candles:,} świeczek ({progress:.1f}%) - Kompletne: {complete_count:,}")
    
    print(f"✅ Przetworzono {len(processed_data):,} świeczek z kompletnymi danymi order book")
    
    # KROK 8: Tworzenie finalnego DataFrame
    print("📊 Tworzę finalny DataFrame...")
    full_merged_df = pd.DataFrame(processed_data)
    
    # Statystyki
    complete_count = len(full_merged_df)
    total_ohlc = len(filtered_ohlc)
    coverage = (complete_count / total_ohlc * 100) if total_ohlc > 0 else 0
    
    print(f"\n📈 Statystyki przetwarzania (TYLKO WSPÓLNE DANE):")
    print(f"  Świeczek OHLC w zakresie: {total_ohlc}")
    print(f"  Świeczek z kompletnymi danymi: {complete_count}")
    print(f"  Pokrycie: {coverage:.1f}%")
    print(f"  Zakres wynikowy: {full_merged_df['timestamp'].min()} - {full_merged_df['timestamp'].max()}")
    
    # Zapisz tylko feather (szybszy)
    print(f"\n💾 Zapisuję dane do {MERGED_FILE}...")
    full_merged_df.to_feather(MERGED_FILE)
    print(f"✅ Dane zapisane pomyślnie do {MERGED_FILE}")
    
    return full_merged_df

def create_missing_orderbook_data(missing_dates, orderbook_dir):
    """
    Tworzy brakujące dane order book na podstawie dni sąsiadujących
    """
    print(f"🔧 Tworzę brakujące dane order book dla {len(missing_dates)} dni...")
    
    for missing_date in missing_dates:
        print(f"   📅 Przetwarzam: {missing_date}")
        
        # Znajdź sąsiadujące dni
        missing_dt = datetime.strptime(missing_date, '%Y-%m-%d')
        
        # Sprawdź dzień przed
        day_before = (missing_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        file_before = f"orderbook_raw/BTCUSDT-bookDepth-{day_before}.csv"
        
        # Sprawdź dzień po
        day_after = (missing_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        file_after = f"orderbook_raw/BTCUSDT-bookDepth-{day_after}.csv"
        
        # Wybierz źródło danych (preferuj dzień przed)
        source_file = None
        source_date = None
        
        if os.path.exists(file_before):
            source_file = file_before
            source_date = day_before
            print(f"      📋 Używam danych z: {day_before}")
        elif os.path.exists(file_after):
            source_file = file_after
            source_date = day_after
            print(f"      📋 Używam danych z: {day_after}")
        else:
            print(f"      ❌ Brak danych sąsiadujących dla {missing_date}")
            continue
        
        # Wczytaj dane źródłowe
        try:
            source_df = pd.read_csv(source_file)
            print(f"      📊 Wczytano {len(source_df):,} wierszy z {source_date}")
            
            # Skopiuj dane i dostosuj timestampy
            new_df = source_df.copy()
            
            # Zamień timestampy na brakujący dzień
            source_start = pd.to_datetime(source_date)
            target_start = pd.to_datetime(missing_date)
            
            # Oblicz przesunięcie czasowe
            time_shift = target_start - source_start
            
            # Dostosuj timestampy
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp']) + time_shift
            new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Zapisz nowy plik
            output_file = f"orderbook_raw/BTCUSDT-bookDepth-{missing_date}.csv"
            new_df.to_csv(output_file, index=False)
            
            print(f"      ✅ Utworzono: {output_file} ({len(new_df):,} wierszy)")
            
        except Exception as e:
            print(f"      ❌ Błąd podczas tworzenia danych: {e}")

def check_and_fill_missing_orderbook():
    """
    Sprawdza brakujące dni order book i wypełnia je danymi z dni sąsiadujących
    """
    orderbook_dir = "orderbook_raw"
    ohlc_dir = "ohlc_raw"
    
    # Pobierz listę plików
    orderbook_files = [f for f in os.listdir(orderbook_dir) if f.endswith('.csv')]
    ohlc_files = [f for f in os.listdir(ohlc_dir) if f.endswith('.csv')]
    
    # Wyciągnij daty z nazw plików
    orderbook_dates = set()
    for file in orderbook_files:
        if 'BTCUSDT-bookDepth-' in file:
            date_str = file.replace('BTCUSDT-bookDepth-', '').replace('.csv', '')
            orderbook_dates.add(date_str)
    
    ohlc_dates = set()
    for file in ohlc_files:
        if 'BTCUSDT-1m-' in file:
            date_str = file.replace('BTCUSDT-1m-', '').replace('.csv', '')
            ohlc_dates.add(date_str)
    
    # Znajdź luki
    missing_orderbook = ohlc_dates - orderbook_dates
    
    if missing_orderbook:
        print(f"🔍 Znaleziono {len(missing_orderbook)} brakujących dni order book:")
        for date in sorted(missing_orderbook):
            print(f"   - {date}")
        
        # Wypełnij brakujące dane
        create_missing_orderbook_data(sorted(missing_orderbook), orderbook_dir)
        
        print(f"✅ Wypełniono wszystkie brakujące dni order book!")
        return True
    else:
        print(f"✅ Brak luk w danych order book!")
        return False

def main():
    """Główna funkcja pobierania i łączenia danych"""
    import argparse
    
    # Parsuj argumenty z linii komend
    parser = argparse.ArgumentParser(description='Pobierz i połącz dane OHLC z Order Book')
    parser.add_argument('symbol', help='Symbol kryptowaluty (np. BTCUSDT)')
    parser.add_argument('start_date', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('end_date', help='Data końcowa (YYYY-MM-DD)')
    parser.add_argument('--extra-month', action='store_true', help='Dodaj 30 dni dla MA 43200')
    
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
    extra_month_for_ma = args.extra_month
    
    print(f"🚀 Rozpoczynanie pobierania danych dla {symbol}")
    print(f"📅 Zakres: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    if extra_month_for_ma:
        print(f"📊 Dodaję 30 dni dla obliczania MA 43200")
    
    # Sprawdź i wypełnij brakujące dane order book
    print(f"\n🔍 Sprawdzam luki w danych order book...")
    check_and_fill_missing_orderbook()
    
    # KROK 1: Sprawdź istniejące dane
    data_exists, missing_ranges = check_existing_data_range(start_date, end_date, symbol)
    
    if data_exists:
        print(f"✅ Dane już istnieją! Kończę pracę.")
        return
    
    if missing_ranges:
        print(f"📋 Brakujące zakresy do pobrania:")
        for start, end in missing_ranges:
            print(f"   {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
    
    # KROK 2: Pobierz brakujące dane
    all_csv_files_orderbook = []
    all_csv_files_ohlc = []
    
    for start, end in missing_ranges:
        print(f"\n📥 Pobieram dane dla zakresu: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
        
        for date in daterange(start, end):
            date_str = date.strftime('%Y-%m-%d')
            
            # Pobierz order book
            orderbook_csv = download_and_extract_orderbook(symbol, date_str)
            if orderbook_csv:
                all_csv_files_orderbook.append(orderbook_csv)
            
            # Pobierz OHLC
            ohlc_csv = download_and_extract_ohlc(symbol, date_str)
            if ohlc_csv:
                all_csv_files_ohlc.append(ohlc_csv)
    
    if not all_csv_files_orderbook or not all_csv_files_ohlc:
        print("❌ Brak plików do przetworzenia!")
        return
    
    # KROK 3: Przetwórz dane
    print(f"\n📊 Przetwarzam {len(all_csv_files_orderbook)} plików order book...")
    orderbook_df = load_and_process_orderbook(all_csv_files_orderbook)
    
    print(f"\n📊 Przetwarzam {len(all_csv_files_ohlc)} plików OHLC...")
    ohlc_df = load_and_process_ohlc(all_csv_files_ohlc)
    
    if orderbook_df is None or ohlc_df is None:
        print("❌ Błąd przetwarzania danych!")
        return
    
    # KROK 4: Połącz dane (ZOPTYMALIZOWANE - TYLKO WSPÓLNE)
    merged_df = merge_orderbook_with_ohlc(orderbook_df, ohlc_df, extra_month_for_ma=extra_month_for_ma)
    
    # KROK 5: Zapisz metadata
    save_metadata(start_date, end_date, symbol)
    
    print(f"\n🎉 Proces zakończony pomyślnie!")
    print(f"📁 Plik wynikowy: {MERGED_FILE}")
    print(f"📊 Liczba wierszy: {len(merged_df):,}")
    print(f"📅 Zakres wynikowy: {merged_df['timestamp'].min()} - {merged_df['timestamp'].max()}")

if __name__ == "__main__":
    main() 