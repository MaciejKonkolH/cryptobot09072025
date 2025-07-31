#!/usr/bin/env python3
"""
Skrypt łączenia danych orderbook w format feather
Dostosowany do nowego modułu orderbook - przetwarza wszystkie pary z konfiguracji
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging
import sys
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import konfiguracji
from config import PAIRS, FILE_CONFIG, LOGGING_CONFIG

def setup_logging():
    """Konfiguruje system logowania"""
    log_file = Path("merge_orderbook.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_single_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Wczytuje pojedynczy plik CSV"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Błąd wczytania {file_path.name}: {e}")
        return None

def load_all_orderbook_files(symbol: str, logger) -> Optional[pd.DataFrame]:
    """Wczytuje wszystkie pliki order book dla jednej pary z optymalizacją"""
    logger.info(f"Wczytuję pliki order book dla {symbol}...")
    
    orderbook_dir = Path(FILE_CONFIG['output_dir'])
    
    # Znajdź wszystkie pliki order book
    if not orderbook_dir.exists():
        logger.error(f"Katalog {orderbook_dir} nie istnieje!")
        return None
    
    files = list(orderbook_dir.glob(f"{symbol}-bookDepth-*.csv"))
    files.sort()
    
    logger.info(f"Znaleziono {len(files)} plików order book dla {symbol}")
    
    # KROK 1: Równoległe wczytywanie wszystkich plików
    logger.info(f"Rozpoczynam równoległe wczytywanie {len(files)} plików...")
    
    all_dfs = []
    max_workers = min(8, len(files))  # Maksymalnie 8 wątków
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submituj wszystkie zadania
        future_to_file = {executor.submit(load_single_file, file_path): file_path for file_path in files}
        
        # Zbierz wyniki z progress bar
        completed = 0
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed += 1
            
            try:
                df = future.result()
                if df is not None:
                    all_dfs.append(df)
                
                if completed % 50 == 0 or completed == len(files):
                    progress = (completed / len(files)) * 100
                    logger.info(f"Wczytano {completed}/{len(files)} plików ({progress:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Błąd przetwarzania {file_path.name}: {e}")
    
    if not all_dfs:
        logger.warning(f"Brak danych do wczytania dla {symbol}!")
        return None
    
    # KROK 2: Połącz wszystkie dane w jeden DataFrame
    logger.info(f"Łączę {len(all_dfs)} plików w jeden DataFrame...")
    combined_long_df = pd.concat(all_dfs, ignore_index=True)
    combined_long_df = combined_long_df.sort_values('timestamp')
    
    logger.info(f"Wczytano {len(combined_long_df):,} wierszy w formacie long")
    logger.info(f"Zakres czasowy {symbol}: {combined_long_df['timestamp'].min()} do {combined_long_df['timestamp'].max()}")
    
    # KROK 3: Jedna transformacja long→wide na końcu
    logger.info(f"Przekształcam z long do wide format...")
    wide_df = transform_long_to_wide(combined_long_df)
    
    logger.info(f"Przekształcono do {len(wide_df):,} snapshotów w formacie wide")
    
    return wide_df

def transform_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Przekształca dane order book z formatu long do wide - OPTYMALIZOWANA WERSJA"""
    # Użyj pivot_table zamiast groupby - znacznie szybsze dla dużych danych
    logger = logging.getLogger(__name__)
    logger.info(f"Rozpoczynam transformację {len(df):,} wierszy...")
    
    # Przygotuj dane do pivot
    df_pivot = df.copy()
    
    # Stwórz kolumny depth i notional osobno
    depth_pivot = df_pivot.pivot_table(
        index='timestamp',
        columns='percentage',
        values='depth',
        aggfunc='first'
    ).add_prefix('depth_')
    
    notional_pivot = df_pivot.pivot_table(
        index='timestamp',
        columns='percentage',
        values='notional',
        aggfunc='first'
    ).add_prefix('notional_')
    
    # Połącz kolumny
    result = pd.concat([depth_pivot, notional_pivot], axis=1)
    result = result.reset_index()
    
    logger.info(f"Transformacja zakończona: {len(result):,} snapshotów, {len(result.columns)} kolumn")
    
    return result

def round_to_minute(timestamp):
    """Zaokrągla timestamp do pełnej minuty"""
    return timestamp.replace(second=0, microsecond=0)

def calculate_interpolated_timestamp(snapshot1_time, snapshot2_time, minute_start, minute_end):
    """Oblicza timestamp dla interpolowanego snapshotu z ograniczeniem do zakresu minuty"""
    time1 = pd.to_datetime(snapshot1_time)
    time2 = pd.to_datetime(snapshot2_time)
    
    avg_time = time1 + (time2 - time1) / 2
    
    if avg_time < minute_start:
        return minute_start
    elif avg_time > minute_end:
        return minute_end
    else:
        return avg_time

def interpolate_snapshots(snapshot1, snapshot2, ratio):
    """Interpoluje między dwoma snapshotami order book"""
    interpolated = {}
    
    # Interpoluj wszystkie poziomy order book
    for col in snapshot1.index:
        if col.startswith('depth_') or col.startswith('notional_'):
            if col in snapshot1 and col in snapshot2:
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
        snapshot1 = minute_data.iloc[0]
        snapshot2 = minute_data.iloc[0]
    elif len(minute_data) == 2:
        snapshot1 = minute_data.iloc[0]
        snapshot2 = minute_data.iloc[1]
    else:
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
                
                interpolated_data = []
                interpolated_data.append(single_snapshot.to_dict())
                
                snapshot2 = interpolate_snapshots(single_snapshot, after_snapshot, 0.5)
                snapshot2['timestamp'] = interpolated_timestamp
                interpolated_data.append(snapshot2)
                
                return pd.DataFrame(interpolated_data)
            else:
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
                
                interpolated_data = []
                
                snapshot1 = interpolate_snapshots(before_snapshot, single_snapshot, 0.5)
                snapshot1['timestamp'] = interpolated_timestamp
                interpolated_data.append(snapshot1)
                interpolated_data.append(single_snapshot.to_dict())
                
                return pd.DataFrame(interpolated_data)
            else:
                return pd.DataFrame([single_snapshot.to_dict(), single_snapshot.to_dict()])
    
    elif snapshot_count == 2:
        return minute_data
    
    elif snapshot_count >= 3:
        return minute_data.iloc[[0, -1]]
    
    return pd.DataFrame()

def merge_orderbook_to_feather(symbol: str, logger) -> Optional[pd.DataFrame]:
    """Łączy order book dla jednej pary w format feather"""
    logger.info(f"Rozpoczynam łączenie order book dla {symbol}")
    
    # KROK 1: Wczytaj wszystkie dane
    orderbook_df = load_all_orderbook_files(symbol, logger)
    if orderbook_df is None:
        return None
    
    # KROK 2: Grupuj po minutach
    logger.info(f"Grupuję dane po minutach...")
    orderbook_df['minute'] = orderbook_df['timestamp'].apply(round_to_minute)
    
    # KROK 2.5: OPTYMALIZACJA - Indeksuj dane dla szybkiego wyszukiwania
    logger.info(f"Indeksuję dane dla szybkiego wyszukiwania...")
    orderbook_df_sorted = orderbook_df.sort_values('timestamp').reset_index(drop=True)
    
    def find_nearest_snapshot(target_time, direction='before'):
        """Szybkie wyszukiwanie najbliższego snapshotu używając pandas"""
        if len(orderbook_df_sorted) == 0:
            return None
            
        if direction == 'before':
            # Znajdź ostatni snapshot przed target_time
            mask = orderbook_df_sorted['timestamp'] < target_time
            if mask.any():
                return orderbook_df_sorted[mask].iloc[-1]
        else:  # direction == 'after'
            # Znajdź pierwszy snapshot po target_time
            mask = orderbook_df_sorted['timestamp'] >= target_time
            if mask.any():
                return orderbook_df_sorted[mask].iloc[0]
        
        return None
    
    # KROK 3: Przetwórz każdą minutę
    logger.info(f"Przetwarzam minuty...")
    wide_format_data = []
    processed_count = 0
    total_minutes = len(orderbook_df.groupby('minute'))
    
    for minute_timestamp, minute_data in orderbook_df.groupby('minute'):
        minute_start = minute_timestamp
        minute_end = minute_timestamp + timedelta(minutes=1)
        
        # Przetwórz snapshoty dla tej minuty z zoptymalizowanym wyszukiwaniem
        processed_minute_data = process_minute_with_interpolation_optimized(
            minute_data, minute_start, minute_end, find_nearest_snapshot
        )
        
        if len(processed_minute_data) > 0:
            row = create_wide_format_row(minute_timestamp, processed_minute_data)
            if row:
                wide_format_data.append(row)
        
        processed_count += 1
        if processed_count % 1000 == 0:
            progress = (processed_count / total_minutes) * 100
            logger.info(f"Postęp {symbol}: {processed_count:,}/{total_minutes:,} minut ({progress:.1f}%)")
    
    # KROK 4: Stwórz finalny DataFrame
    logger.info(f"Tworzę finalny DataFrame dla {symbol}...")
    result_df = pd.DataFrame(wide_format_data)
    
    if len(result_df) == 0:
        logger.warning(f"Brak danych do zapisania dla {symbol}!")
        return None
    
    # KROK 5: Zapisz do feather
    output_dir = Path("merged_raw")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"orderbook_merged_{symbol}.feather"
    logger.info(f"Zapisuję {symbol} do {output_file}...")
    result_df.to_feather(output_file)
    
    # KROK 6: Podsumowanie
    logger.info(f"Proces zakończony dla {symbol}!")
    logger.info(f"  Wierszy: {len(result_df):,}")
    logger.info(f"  Zakres: {result_df['timestamp'].min()} - {result_df['timestamp'].max()}")
    logger.info(f"  Kolumny: {len(result_df.columns)}")
    logger.info(f"  Plik: {output_file}")
    
    return result_df

def process_minute_with_interpolation_optimized(minute_data, minute_start, minute_end, find_nearest_snapshot):
    """Przetwarza minutę z interpolacją - ZOPTYMALIZOWANA WERSJA"""
    snapshot_count = len(minute_data)
    
    if snapshot_count == 0:
        # Brak snapshotów w minucie - interpoluj z sąsiednich minut
        before_snapshot = find_nearest_snapshot(minute_start, 'before')
        after_snapshot = find_nearest_snapshot(minute_end, 'after')
        
        if before_snapshot is not None and after_snapshot is not None:
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
            after_snapshot = find_nearest_snapshot(minute_end, 'after')
            
            if after_snapshot is not None:
                interpolated_timestamp = calculate_interpolated_timestamp(
                    single_snapshot['timestamp'], after_snapshot['timestamp'], 
                    minute_start, minute_end
                )
                
                interpolated_data = []
                interpolated_data.append(single_snapshot.to_dict())
                
                snapshot2 = interpolate_snapshots(single_snapshot, after_snapshot, 0.5)
                snapshot2['timestamp'] = interpolated_timestamp
                interpolated_data.append(snapshot2)
                
                return pd.DataFrame(interpolated_data)
            else:
                return pd.DataFrame([single_snapshot.to_dict(), single_snapshot.to_dict()])
        else:
            # Snapshot bliżej końca - interpoluj z przeszłym
            before_snapshot = find_nearest_snapshot(minute_start, 'before')
            
            if before_snapshot is not None:
                interpolated_timestamp = calculate_interpolated_timestamp(
                    before_snapshot['timestamp'], single_snapshot['timestamp'], 
                    minute_start, minute_end
                )
                
                interpolated_data = []
                
                snapshot1 = interpolate_snapshots(before_snapshot, single_snapshot, 0.5)
                snapshot1['timestamp'] = interpolated_timestamp
                interpolated_data.append(snapshot1)
                interpolated_data.append(single_snapshot.to_dict())
                
                return pd.DataFrame(interpolated_data)
            else:
                return pd.DataFrame([single_snapshot.to_dict(), single_snapshot.to_dict()])
    
    elif snapshot_count == 2:
        return minute_data
    
    elif snapshot_count >= 3:
        return minute_data.iloc[[0, -1]]
    
    return pd.DataFrame()

def merge_all_pairs_to_feather():
    """Główna funkcja łączenia order book dla wszystkich par"""
    logger = setup_logging()
    
    logger.info("Rozpoczynam łączenie order book dla wszystkich par")
    logger.info(f"Pary: {', '.join(PAIRS)}")
    
    start_time = datetime.now()
    successful_pairs = []
    failed_pairs = []
    
    for i, symbol in enumerate(PAIRS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Przetwarzam parę {i}/{len(PAIRS)}: {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            result = merge_orderbook_to_feather(symbol, logger)
            if result is not None:
                successful_pairs.append(symbol)
                logger.info(f"[OK] {symbol} - pomyślnie przetworzono")
            else:
                failed_pairs.append(symbol)
                logger.warning(f"[ERROR] {symbol} - błąd przetwarzania")
        except Exception as e:
            failed_pairs.append(symbol)
            logger.error(f"[ERROR] {symbol} - błąd krytyczny: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Podsumowanie
    logger.info(f"\n{'='*60}")
    logger.info(f"ŁĄCZENIE ORDERBOOK ZAKOŃCZONE!")
    logger.info(f"{'='*60}")
    logger.info(f"Udało się: {len(successful_pairs)}/{len(PAIRS)} par")
    logger.info(f"Czas: {duration}")
    
    if successful_pairs:
        logger.info(f"Pomyślnie przetworzone pary: {', '.join(successful_pairs)}")
    
    if failed_pairs:
        logger.warning(f"Nieudane pary: {', '.join(failed_pairs)}")
    
    # Zapisz metadane
    metadata = {
        'merge_date': datetime.now().isoformat(),
        'pairs': PAIRS,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'duration_seconds': duration.total_seconds(),
        'success_rate': f"{len(successful_pairs)/len(PAIRS)*100:.1f}%"
    }
    
    metadata_file = Path("merge_orderbook_metadata.json")
    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadane zapisane: {metadata_file}")
    except Exception as e:
        logger.error(f"Błąd zapisywania metadanych: {e}")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Połącz dane order book wszystkich par w format feather')
    parser.add_argument('--symbol', help='Przetwórz tylko jedną parę (opcjonalnie)')
    
    args = parser.parse_args()
    
    if args.symbol:
        # Przetwórz tylko jedną parę
        logger = setup_logging()
        result = merge_orderbook_to_feather(args.symbol, logger)
        if result is not None:
            print(f"\n[OK] Para {args.symbol} została pomyślnie przetworzona!")
        else:
            print(f"\n[ERROR] Błąd podczas przetwarzania {args.symbol}!")
    else:
        # Przetwórz wszystkie pary
        merge_all_pairs_to_feather()

if __name__ == "__main__":
    main() 