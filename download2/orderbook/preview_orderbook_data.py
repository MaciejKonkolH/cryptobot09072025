#!/usr/bin/env python3
"""
Skrypt do podglądu danych orderbook
Pobiera przykładowe wiersze z plików feather dla określonej daty
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
from pathlib import Path
import logging
import sys
from typing import List, Dict, Optional

# Import konfiguracji
from config import PAIRS, LOGGING_CONFIG

def setup_logging():
    """Konfiguruje system logowania"""
    log_file = Path("preview.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_orderbook_data_for_date(symbol: str, target_date: datetime, logger, rows_limit: int = 20) -> Optional[pd.DataFrame]:
    """Wczytuje dane order book dla jednej pary z określonej daty"""
    feather_file = Path("orderbook_completed") / f"orderbook_filled_{symbol}.feather"
    
    logger.info(f"Wczytuję dane z {feather_file} dla daty {target_date.date()}...")
    
    if not os.path.exists(feather_file):
        logger.error(f"Plik {feather_file} nie istnieje!")
        return None
    
    try:
        # Wczytaj dane
        df = pd.read_feather(feather_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filtruj dane dla określonej daty
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        day_data = df[(df['timestamp'] >= start_of_day) & (df['timestamp'] < end_of_day)]
        
        if len(day_data) == 0:
            logger.warning(f"Brak danych dla {symbol} w dniu {target_date.date()}")
            return None
        
        # Sortuj i ogranicz liczbę wierszy
        day_data = day_data.sort_values('timestamp').head(rows_limit)
        
        logger.info(f"Znaleziono {len(day_data)} wierszy dla {symbol} w dniu {target_date.date()}")
        
        return day_data
    except Exception as e:
        logger.error(f"Błąd wczytania {feather_file}: {e}")
        return None

def preview_orderbook_data(symbol: str, target_date: datetime, logger, rows_limit: int = 20):
    """Wyświetla podgląd danych orderbook dla jednej pary"""
    logger.info(f"Podgląd danych dla {symbol} z dnia {target_date.date()}")
    
    # Wczytaj dane
    df = load_orderbook_data_for_date(symbol, target_date, logger, rows_limit)
    if df is None:
        return
    
    # Wyświetl podstawowe informacje
    print(f"\n{'='*80}")
    print(f"PODGLĄD DANYCH ORDERBOOK: {symbol}")
    print(f"Data: {target_date.date()}")
    print(f"Liczba wierszy: {len(df)}")
    print(f"Zakres czasowy: {df['timestamp'].min()} do {df['timestamp'].max()}")
    print(f"{'='*80}")
    
    # Sprawdź kolumny
    orderbook_cols = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    timestamp_cols = ['timestamp', 'snapshot1_timestamp', 'snapshot2_timestamp']
    metadata_cols = [col for col in df.columns if col in ['fill_method', 'gap_duration_minutes', 'price_change_percent']]
    
    print(f"\n📊 STRUKTURA DANYCH:")
    print(f"  Kolumny timestamp: {len(timestamp_cols)}")
    print(f"  Kolumny orderbook: {len(orderbook_cols)}")
    print(f"  Kolumny metadane: {len(metadata_cols)}")
    print(f"  Wszystkie kolumny: {len(df.columns)}")
    
    # Wyświetl pierwsze kilka wierszy
    print(f"\n📋 PRZYKŁADOWE DANE (pierwsze {min(5, len(df))} wierszy):")
    print("-" * 80)
    
    # Wybierz kolumny do wyświetlenia
    display_cols = timestamp_cols + metadata_cols + orderbook_cols[:6]  # Pierwsze 6 kolumn orderbook
    
    # Wyświetl dane w czytelnym formacie
    for i, (idx, row) in enumerate(df.head(5).iterrows()):
        print(f"\nWiersz {i+1}:")
        print(f"  Timestamp: {row['timestamp']}")
        print(f"  Snapshot1: {row['snapshot1_timestamp']}")
        print(f"  Snapshot2: {row['snapshot2_timestamp']}")
        
        if 'fill_method' in row:
            print(f"  Metoda wypełniania: {row['fill_method']}")
        
        # Wyświetl pierwsze kilka wartości orderbook
        print(f"  Orderbook (pierwsze 3 poziomy):")
        for col in orderbook_cols[:6]:
            if col in row:
                if 'timestamp' in col:
                    print(f"    {col}: {row[col]}")
                else:
                    print(f"    {col}: {row[col]:.6f}")
    
    # Statystyki dla całego dnia
    print(f"\n📈 STATYSTYKI DLA DNIA {target_date.date()}:")
    print("-" * 40)
    
    # Sprawdź metody wypełniania
    if 'fill_method' in df.columns:
        fill_stats = df['fill_method'].value_counts()
        print(f"Metody wypełniania:")
        for method, count in fill_stats.items():
            print(f"  {method}: {count} wierszy")
    
    # Sprawdź zakres wartości orderbook
    if len(orderbook_cols) > 0:
        print(f"\nZakres wartości orderbook:")
        for col in orderbook_cols[:3]:  # Pierwsze 3 kolumny
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                print(f"  {col}: min={min_val:.6f}, max={max_val:.6f}, średnia={mean_val:.6f}")
    
    # Sprawdź czy dane są ciągłe
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff()
    gaps = time_diffs[time_diffs > timedelta(minutes=1)]
    
    if len(gaps) == 0:
        print(f"\n✅ Dane są ciągłe - brak luk czasowych")
    else:
        print(f"\n⚠️ Znaleziono {len(gaps)} luk czasowych")
        for i, gap in enumerate(gaps.head(3)):
            print(f"  Luka {i+1}: {gap}")

def preview_all_pairs_for_date(target_date: datetime, logger, rows_limit: int = 20):
    """Podgląd danych dla wszystkich par z określonej daty"""
    logger.info(f"Podgląd danych dla wszystkich par z dnia {target_date.date()}")
    
    successful_pairs = []
    failed_pairs = []
    
    for i, symbol in enumerate(PAIRS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sprawdzam parę {i}/{len(PAIRS)}: {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            preview_orderbook_data(symbol, target_date, logger, rows_limit)
            successful_pairs.append(symbol)
        except Exception as e:
            failed_pairs.append(symbol)
            logger.error(f"[ERROR] {symbol} - błąd: {e}")
    
    # Podsumowanie
    logger.info(f"\n{'='*60}")
    logger.info(f"PODGLĄD ZAKOŃCZONY!")
    logger.info(f"{'='*60}")
    logger.info(f"Udało się: {len(successful_pairs)}/{len(PAIRS)} par")
    
    if successful_pairs:
        logger.info(f"Pomyślnie sprawdzone pary: {', '.join(successful_pairs)}")
    
    if failed_pairs:
        logger.warning(f"Nieudane pary: {', '.join(failed_pairs)}")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Podgląd danych orderbook z określonej daty')
    parser.add_argument('--date', required=True, help='Data w formacie YYYY-MM-DD (np. 2024-01-15)')
    parser.add_argument('--symbol', help='Sprawdź tylko jedną parę (opcjonalnie)')
    parser.add_argument('--rows', type=int, default=20, help='Liczba wierszy do wyświetlenia (domyślnie: 20)')
    
    args = parser.parse_args()
    
    # Parsuj datę
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"Błąd: Nieprawidłowy format daty. Użyj YYYY-MM-DD (np. 2024-01-15)")
        return
    
    logger = setup_logging()
    
    if args.symbol:
        # Podgląd tylko jednej pary
        preview_orderbook_data(args.symbol, target_date, logger, args.rows)
    else:
        # Podgląd wszystkich par
        preview_all_pairs_for_date(target_date, logger, args.rows)

if __name__ == "__main__":
    main() 