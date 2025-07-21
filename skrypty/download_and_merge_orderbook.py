import os
import requests
import zipfile
import pandas as pd
import json
from datetime import datetime, timedelta
import sys

ORDERBOOK_DIR = "orderbook_raw"
OHLC_DIR = "ohlc_raw"
MERGED_FILE = "orderbook_ohlc_merged.feather"
JSON_FILE = "orderbook_ohlc_merged.json"


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def download_and_extract_orderbook(symbol, date_str, market="futures", out_dir=ORDERBOOK_DIR):
    """Pobiera plik ZIP order book, rozpakowuje go i zwraca ścieżkę do CSV"""
    url = f"https://data.binance.vision/data/{market}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
    zip_path = os.path.join(out_dir, f"{symbol}-bookDepth-{date_str}.zip")
    csv_path = os.path.join(out_dir, f"{symbol}-bookDepth-{date_str}.csv")
    
    try:
        # Sprawdź czy CSV już istnieje
        if os.path.exists(csv_path):
            print(f"✅ {date_str}: Order book CSV już istnieje")
            return csv_path
        
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
            
            print(f"✅ {date_str}: Order book pobrano i rozpakowano -> {csv_path}")
            return csv_path
        else:
            print(f"❌ {date_str}: brak order book pliku ({resp.status_code})")
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
        # Sprawdź czy CSV już istnieje
        if os.path.exists(csv_path):
            print(f"✅ {date_str}: OHLC CSV już istnieje")
            return csv_path
        
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
            
            print(f"✅ {date_str}: OHLC pobrano i rozpakowano -> {csv_path}")
            return csv_path
        else:
            print(f"❌ {date_str}: brak OHLC pliku ({resp.status_code})")
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
    window_start = ohlc_timestamp - pd.Timedelta(seconds=60)
    window_end = ohlc_timestamp
    
    # Znajdź snapshoty w oknie -60 do 0 sekund
    relevant_orderbook = wide_orderbook_df[
        (wide_orderbook_df['timestamp'] >= window_start) &
        (wide_orderbook_df['timestamp'] < window_end)
    ].sort_values('timestamp')
    
    snapshot_count = len(relevant_orderbook)
    
    if snapshot_count == 0:
        # Znajdź najbliższe snapshoty przed i po oknie
        before_window = wide_orderbook_df[wide_orderbook_df['timestamp'] < window_start].sort_values('timestamp')
        after_window = wide_orderbook_df[wide_orderbook_df['timestamp'] >= window_end].sort_values('timestamp')
        
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
        distance_to_end = (window_end - single_snapshot['timestamp']).total_seconds()
        
        if distance_to_end < 30:
            # Snapshot jest bliżej niż 30 sekund od końca okna
            # Znajdź snapshot przed oknem
            before_window = wide_orderbook_df[wide_orderbook_df['timestamp'] < window_start].sort_values('timestamp')
            
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
            after_window = wide_orderbook_df[wide_orderbook_df['timestamp'] >= window_end].sort_values('timestamp')
            
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

def save_to_json(merged_df, output_file=JSON_FILE):
    """Zapisuje dane do pliku JSON z formatowaniem"""
    print(f"\n💾 Zapisuję dane do {output_file}...")
    
    # Konwertuj DataFrame na listę słowników
    json_data = []
    
    for _, row in merged_df.iterrows():
        # Stwórz strukturę dla jednego wiersza
        candle_data = {
            "timestamp": row['timestamp'].isoformat(),
            "ohlc": {
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            },
            "data_quality": row['data_quality']
        }
        
        # Dodaj dane order book jeśli są dostępne
        if row['data_quality'] == 'complete':
            candle_data["orderbook"] = {
                "snapshot1": {
                    "timestamp": row['snapshot1_timestamp'].isoformat(),
                    "levels": {}
                },
                "snapshot2": {
                    "timestamp": row['snapshot2_timestamp'].isoformat(),
                    "levels": {}
                }
            }
            
            # Dodaj poziomy order book dla obu snapshotów
            for i in range(-5, 6):
                if i == 0:
                    continue
                
                level_key = str(i)
                candle_data["orderbook"]["snapshot1"]["levels"][level_key] = {
                    "depth": float(row[f'snapshot1_depth_{i}']),
                    "notional": float(row[f'snapshot1_notional_{i}'])
                }
                candle_data["orderbook"]["snapshot2"]["levels"][level_key] = {
                    "depth": float(row[f'snapshot2_depth_{i}']),
                    "notional": float(row[f'snapshot2_notional_{i}'])
                }
        
        json_data.append(candle_data)
    
    # Zapisz do pliku JSON z formatowaniem
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Zapisano {len(json_data)} świeczek do {output_file}")

def merge_orderbook_with_ohlc(orderbook_df, ohlc_df, save_json=False):
    """Łączy dane order book z OHLC i analizuje rozkład snapshotów"""
    print(f"\n🔗 Łączę order book z OHLC...")
    
    # Konwertuj order book na wide format (jeden wiersz na timestamp)
    print("Przekształcam order book na wide format...")
    
    wide_orderbook = []
    for timestamp in orderbook_df['timestamp'].unique():
        timestamp_data = orderbook_df[orderbook_df['timestamp'] == timestamp]
        
        # Stwórz jeden wiersz z wszystkimi poziomami
        row = {'timestamp': timestamp}
        for _, level_data in timestamp_data.iterrows():
            percentage = level_data['percentage']
            row[f'depth_{percentage}'] = level_data['depth']
            row[f'notional_{percentage}'] = level_data['notional']
        
        wide_orderbook.append(row)
    
    wide_orderbook_df = pd.DataFrame(wide_orderbook)
    print(f"Przekształcono na {len(wide_orderbook_df)} unikalnych timestampów order book")
    
    # Przetwórz każdą świeczkę OHLC
    print("\n📊 Przetwarzam snapshoty order book dla każdej świeczki OHLC...")
    
    processed_data = []
    stats = {'0_snapshots': 0, '1_snapshot': 0, '2_snapshots': 0, '3_plus_snapshots': 0}
    
    for i, ohlc_row in enumerate(ohlc_df.iterrows()):
        ohlc_timestamp = ohlc_row[1]['timestamp']
        
        # Przetwórz snapshoty dla tej świeczki
        processed_snapshots = process_snapshots_for_candle(ohlc_timestamp, wide_orderbook_df)
        
        # Stwórz wiersz z danymi OHLC
        merged_row = {
            'timestamp': ohlc_timestamp,
            'open': ohlc_row[1]['open'],
            'high': ohlc_row[1]['high'],
            'low': ohlc_row[1]['low'],
            'close': ohlc_row[1]['close'],
            'volume': ohlc_row[1]['volume']
        }
        
        # Dodaj dane order book
        if processed_snapshots and len(processed_snapshots) == 2:
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
        else:
            # Brak danych order book
            merged_row['data_quality'] = 'missing'
        
        processed_data.append(merged_row)
        
        # Aktualizuj statystyki
        if processed_snapshots is None:
            stats['0_snapshots'] += 1
        elif len(processed_snapshots) == 1:
            stats['1_snapshot'] += 1
        elif len(processed_snapshots) == 2:
            stats['2_snapshots'] += 1
        else:
            stats['3_plus_snapshots'] += 1
    
    # Wyświetl statystyki
    print(f"\n📈 Statystyki przetwarzania:")
    print(f"  Świeczek z 0 snapshotami: {stats['0_snapshots']}")
    print(f"  Świeczek z 1 snapshotem: {stats['1_snapshot']}")
    print(f"  Świeczek z 2 snapshotami: {stats['2_snapshots']}")
    print(f"  Świeczek z 3+ snapshotami: {stats['3_plus_snapshots']}")
    
    # Zapisz połączone dane
    merged_df = pd.DataFrame(processed_data)
    merged_df.to_feather(MERGED_FILE)
    
    print(f"\n✅ Połączono dane w {MERGED_FILE}")
    print(f"📊 Łącznie {len(merged_df)} wierszy")
    
    # Zapisz do JSON jeśli włączone
    if save_json:
        save_to_json(merged_df)
    
    return merged_df

def main():
    if len(sys.argv) < 4:
        print("Użycie: python download_and_merge_orderbook.py SYMBOL DATA_START DATA_END [--json]")
        print("Przykład: python download_and_merge_orderbook.py BTCUSDT 2024-06-01 2024-06-10 --json")
        sys.exit(1)
    
    symbol = sys.argv[1]
    date_start = sys.argv[2]
    date_end = sys.argv[3]
    
    # Sprawdź flagę --json
    save_json = '--json' in sys.argv
    
    try:
        start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    except Exception as e:
        print(f"Błąd parsowania daty: {e}")
        sys.exit(1)
    
    os.makedirs(ORDERBOOK_DIR, exist_ok=True)
    os.makedirs(OHLC_DIR, exist_ok=True)
    
    total = (end_dt - start_dt).days + 1
    print(f"\n🚀 Pobieranie order book i OHLC dla {symbol} ({total} dni)")
    print(f"Order book katalog: {ORDERBOOK_DIR}")
    print(f"OHLC katalog: {OHLC_DIR}")
    print(f"Plik wynikowy: {MERGED_FILE}")
    if save_json:
        print(f"Plik JSON: {JSON_FILE}")
    print()
    
    orderbook_files = []
    ohlc_files = []
    downloaded = 0
    
    for i, dt in enumerate(daterange(start_dt, end_dt), 1):
        date_str = dt.strftime("%Y-%m-%d")
        
        # Pobierz order book
        orderbook_path = download_and_extract_orderbook(symbol, date_str)
        if orderbook_path:
            orderbook_files.append(orderbook_path)
        
        # Pobierz OHLC
        ohlc_path = download_and_extract_ohlc(symbol, date_str)
        if ohlc_path:
            ohlc_files.append(ohlc_path)
        
        if orderbook_path or ohlc_path:
            downloaded += 1
            
        print(f"Postęp: {i}/{total} dni, pobrano {downloaded} plików\n{'-'*40}")
    
    print(f"\n📥 Pobrano {downloaded}/{total} plików")
    
    if orderbook_files and ohlc_files:
        # Wczytaj i przetwórz dane
        orderbook_df = load_and_process_orderbook(orderbook_files)
        ohlc_df = load_and_process_ohlc(ohlc_files)
        
        if orderbook_df is not None and ohlc_df is not None:
            # Połącz dane
            merge_orderbook_with_ohlc(orderbook_df, ohlc_df, save_json=save_json)
        else:
            print("❌ Nie można wczytać danych order book lub OHLC")
    else:
        print("❌ Nie pobrano żadnych plików order book lub OHLC")

if __name__ == "__main__":
    main() 