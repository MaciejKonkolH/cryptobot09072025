import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path

# Konfiguracja
ORDERBOOK_DIR = "orderbook_raw"
OUTPUT_FILE = "orderbook_merged.feather"

def load_all_orderbook_files(symbol):
    """Wczytuje wszystkie pliki order book i łączy je w jeden DataFrame"""
    print(f"📊 Wczytuję pliki order book dla {symbol}...")
    
    all_data = []
    orderbook_pattern = f"{symbol}-bookDepth-"
    
    # Znajdź wszystkie pliki order book
    if not os.path.exists(ORDERBOOK_DIR):
        print(f"❌ Katalog {ORDERBOOK_DIR} nie istnieje!")
        return None
    
    files = [f for f in os.listdir(ORDERBOOK_DIR) if f.endswith('.csv') and orderbook_pattern in f]
    files.sort()
    
    print(f"📁 Znaleziono {len(files)} plików order book")
    
    # Wczytaj każdy plik
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(ORDERBOOK_DIR, filename)
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Przekształć z long do wide format
            wide_df = transform_long_to_wide(df)
            all_data.append(wide_df)
            
            print(f"  📊 {i}/{len(files)}: {filename} - {len(df):,} wierszy -> {len(wide_df):,} snapshotów")
        except Exception as e:
            print(f"  ❌ Błąd wczytania {filename}: {e}")
    
    if not all_data:
        print("❌ Brak danych do wczytania!")
        return None
    
    # Połącz wszystkie dane
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp')
    
    print(f"✅ Wczytano {len(combined_df):,} snapshotów order book")
    print(f"⏰ Zakres czasowy: {combined_df['timestamp'].min()} do {combined_df['timestamp'].max()}")
    
    return combined_df

def transform_long_to_wide(df):
    """Przekształca dane order book z formatu long do wide"""
    # Grupuj po timestamp
    grouped = df.groupby('timestamp')
    
    wide_data = []
    for timestamp, group in grouped:
        # Stwórz wiersz z timestamp
        row = {'timestamp': timestamp}
        
        # Dodaj wszystkie poziomy
        for _, level_data in group.iterrows():
            percentage = level_data['percentage']
            depth = level_data['depth']
            notional = level_data['notional']
            
            # Utwórz klucze kolumn
            depth_key = f'depth_{percentage}'
            notional_key = f'notional_{percentage}'
            
            row[depth_key] = depth
            row[notional_key] = notional
        
        wide_data.append(row)
    
    return pd.DataFrame(wide_data)

def round_to_minute(timestamp):
    """Zaokrągla timestamp do pełnej minuty"""
    return timestamp.replace(second=0, microsecond=0)

def calculate_interpolated_timestamp(snapshot1_time, snapshot2_time, minute_start, minute_end):
    """Oblicza timestamp dla interpolowanego snapshotu z ograniczeniem do zakresu minuty"""
    # Konwertuj do datetime dla operacji arytmetycznych
    time1 = pd.to_datetime(snapshot1_time)
    time2 = pd.to_datetime(snapshot2_time)
    
    avg_time = time1 + (time2 - time1) / 2
    
    if avg_time < minute_start:
        return minute_start  # początek minuty
    elif avg_time > minute_end:
        return minute_end    # koniec minuty
    else:
        return avg_time      # średnia w zakresie

def interpolate_snapshots(snapshot1, snapshot2, ratio):
    """Interpoluje między dwoma snapshotami order book"""
    interpolated = {}
    
    # Interpoluj wszystkie poziomy order book
    for col in snapshot1.index:
        if col.startswith('depth_') or col.startswith('notional_'):
            if col in snapshot1 and col in snapshot2:
                # Interpoluj depth i notional
                val1 = snapshot1[col]
                val2 = snapshot2[col]
                interpolated[col] = val1 + (val2 - val1) * ratio
    
    return interpolated

def create_wide_format_row(minute_timestamp, minute_data):
    """Tworzy wiersz w formacie wide z dwoma snapshotami dla wszystkich poziomów"""
    if len(minute_data) == 0:
        return None
    
    # Sortuj dane po timestamp
    minute_data = minute_data.sort_values('timestamp')
    
    # Wybierz 2 snapshoty
    if len(minute_data) == 1:
        # Jeden snapshot - duplikuj
        snapshot1 = minute_data.iloc[0]
        snapshot2 = minute_data.iloc[0]
    elif len(minute_data) == 2:
        # Dwa snapshoty - użyj obu
        snapshot1 = minute_data.iloc[0]
        snapshot2 = minute_data.iloc[1]
    else:
        # Więcej snapshotów - użyj skrajnych
        snapshot1 = minute_data.iloc[0]
        snapshot2 = minute_data.iloc[-1]
    
    # Stwórz wiersz
    row = {
        'timestamp': minute_timestamp,
        'snapshot1_timestamp': snapshot1['timestamp'],
        'snapshot2_timestamp': snapshot2['timestamp']
    }
    
    # Dodaj wszystkie poziomy dla pierwszego snapshotu
    for col in snapshot1.index:
        if col.startswith('depth_') or col.startswith('notional_'):
            row[f'snapshot1_{col}'] = snapshot1[col]
    
    # Dodaj wszystkie poziomy dla drugiego snapshotu
    for col in snapshot2.index:
        if col.startswith('depth_') or col.startswith('notional_'):
            row[f'snapshot2_{col}'] = snapshot2[col]
    
    return row

def process_minute_with_interpolation(minute_data, minute_start, minute_end, all_orderbook_df):
    """Przetwarza minutę z interpolacją jeśli potrzeba"""
    snapshot_count = len(minute_data)
    
    if snapshot_count == 0:
        # Brak snapshotów w minucie - interpoluj z sąsiednich minut
        before_minute = all_orderbook_df[all_orderbook_df['timestamp'] < minute_start].sort_values('timestamp')
        after_minute = all_orderbook_df[all_orderbook_df['timestamp'] >= minute_end].sort_values('timestamp')
        
        if len(before_minute) > 0 and len(after_minute) > 0:
            before_snapshot = before_minute.iloc[-1]
            after_snapshot = after_minute.iloc[0]
            
            # Stwórz dwa interpolowane snapshoty
            interpolated_data = []
            
            # Snapshot 1 (początek minuty)
            snapshot1 = interpolate_snapshots(before_snapshot, after_snapshot, 0.25)
            snapshot1['timestamp'] = minute_start
            interpolated_data.append(snapshot1)
            
            # Snapshot 2 (koniec minuty)
            snapshot2 = interpolate_snapshots(before_snapshot, after_snapshot, 0.75)
            snapshot2['timestamp'] = minute_end
            interpolated_data.append(snapshot2)
            
            return pd.DataFrame(interpolated_data)
        else:
            return pd.DataFrame()
    
    elif snapshot_count == 1:
        # Jeden snapshot - sprawdź czy bliżej początku czy końca minuty
        single_snapshot = minute_data.iloc[0]
        distance_to_start = (single_snapshot['timestamp'] - minute_start).total_seconds()
        distance_to_end = (minute_end - single_snapshot['timestamp']).total_seconds()
        
        if distance_to_start <= distance_to_end:
            # Snapshot bliżej początku - interpoluj z przyszłym
            after_minute = all_orderbook_df[all_orderbook_df['timestamp'] >= minute_end].sort_values('timestamp')
            
            if len(after_minute) > 0:
                after_snapshot = after_minute.iloc[0]
                interpolated_timestamp = calculate_interpolated_timestamp(
                    single_snapshot['timestamp'], after_snapshot['timestamp'], 
                    minute_start, minute_end
                )
                
                # Stwórz DataFrame z dwoma snapshotami
                interpolated_data = []
                interpolated_data.append(single_snapshot.to_dict())
                
                snapshot2 = interpolate_snapshots(single_snapshot, after_snapshot, 0.5)
                snapshot2['timestamp'] = interpolated_timestamp
                interpolated_data.append(snapshot2)
                
                return pd.DataFrame(interpolated_data)
            else:
                # Duplikuj snapshot
                return pd.DataFrame([single_snapshot.to_dict(), single_snapshot.to_dict()])
        else:
            # Snapshot bliżej końca - interpoluj z przeszłym
            before_minute = all_orderbook_df[all_orderbook_df['timestamp'] < minute_start].sort_values('timestamp')
            
            if len(before_minute) > 0:
                before_snapshot = before_minute.iloc[-1]
                interpolated_timestamp = calculate_interpolated_timestamp(
                    before_snapshot['timestamp'], single_snapshot['timestamp'], 
                    minute_start, minute_end
                )
                
                # Stwórz DataFrame z dwoma snapshotami
                interpolated_data = []
                
                snapshot1 = interpolate_snapshots(before_snapshot, single_snapshot, 0.5)
                snapshot1['timestamp'] = interpolated_timestamp
                interpolated_data.append(snapshot1)
                interpolated_data.append(single_snapshot.to_dict())
                
                return pd.DataFrame(interpolated_data)
            else:
                # Duplikuj snapshot
                return pd.DataFrame([single_snapshot.to_dict(), single_snapshot.to_dict()])
    
    elif snapshot_count == 2:
        # Idealny przypadek - zwróć oba snapshoty
        return minute_data
    
    elif snapshot_count >= 3:
        # Usuń środkowe snapshoty, zostaw skrajne
        return minute_data.iloc[[0, -1]]
    
    return pd.DataFrame()

def merge_orderbook_to_feather(symbol):
    """Główna funkcja łączenia order book w format feather"""
    print(f"🚀 Rozpoczynam łączenie order book dla {symbol}")
    
    # KROK 1: Wczytaj wszystkie dane
    orderbook_df = load_all_orderbook_files(symbol)
    if orderbook_df is None:
        return None
    
    # KROK 2: Grupuj po minutach
    print(f"\n📊 Grupuję dane po minutach...")
    orderbook_df['minute'] = orderbook_df['timestamp'].apply(round_to_minute)
    
    # KROK 3: Przetwórz każdą minutę
    print(f"🔄 Przetwarzam snapshoty dla każdej minuty...")
    
    wide_format_data = []
    total_minutes = orderbook_df['minute'].nunique()
    processed_count = 0
    
    for minute_timestamp, minute_data in orderbook_df.groupby('minute'):
        minute_start = minute_timestamp
        minute_end = minute_timestamp + timedelta(minutes=1)
        
        # Przetwórz snapshoty dla tej minuty (z interpolacją jeśli potrzeba)
        processed_minute_data = process_minute_with_interpolation(
            minute_data, minute_start, minute_end, orderbook_df
        )
        
        if len(processed_minute_data) > 0:
            # Stwórz wiersz w formacie wide
            row = create_wide_format_row(minute_timestamp, processed_minute_data)
            if row:
                wide_format_data.append(row)
        
        processed_count += 1
        if processed_count % 1000 == 0:
            progress = (processed_count / total_minutes) * 100
            print(f"📊 Postęp: {processed_count:,}/{total_minutes:,} minut ({progress:.1f}%)")
    
    # KROK 4: Stwórz finalny DataFrame
    print(f"\n📊 Tworzę finalny DataFrame...")
    result_df = pd.DataFrame(wide_format_data)
    
    if len(result_df) == 0:
        print("❌ Brak danych do zapisania!")
        return None
    
    # KROK 5: Zapisz do feather
    print(f"💾 Zapisuję do {OUTPUT_FILE}...")
    result_df.to_feather(OUTPUT_FILE)
    
    # KROK 6: Podsumowanie
    print(f"\n🎉 Proces zakończony!")
    print(f"📊 Wierszy w wyniku: {len(result_df):,}")
    print(f"📅 Zakres czasowy: {result_df['timestamp'].min()} - {result_df['timestamp'].max()}")
    print(f"📋 Kolumny: {len(result_df.columns)}")
    print(f"📁 Plik: {OUTPUT_FILE}")
    
    # KROK 7: Wyświetl informacje o kolumnach
    print(f"\n📋 Struktura kolumn:")
    snapshot1_cols = [col for col in result_df.columns if col.startswith('snapshot1_')]
    snapshot2_cols = [col for col in result_df.columns if col.startswith('snapshot2_')]
    other_cols = [col for col in result_df.columns if not col.startswith('snapshot')]
    
    print(f"   Kolumny główne: {other_cols}")
    print(f"   Snapshot 1 kolumny: {len(snapshot1_cols)} (poziomy -5 do +5)")
    print(f"   Snapshot 2 kolumny: {len(snapshot2_cols)} (poziomy -5 do +5)")
    
    return result_df

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Połącz dane order book w format feather')
    parser.add_argument('symbol', help='Symbol kryptowaluty (np. BTCUSDT)')
    
    args = parser.parse_args()
    symbol = args.symbol
    
    # Uruchom proces łączenia
    result = merge_orderbook_to_feather(symbol)
    
    if result is not None:
        print(f"\n✅ Dane zostały pomyślnie połączone i zapisane!")
    else:
        print(f"\n❌ Błąd podczas łączenia danych!")

if __name__ == "__main__":
    main() 