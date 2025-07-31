#!/usr/bin/env python3
"""
Skrypt inteligentnego wypełniania luk w danych orderbook
Dostosowany do nowego modułu orderbook - obsługuje wszystkie pary z konfiguracji
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
from config import PAIRS, FILE_CONFIG, LOGGING_CONFIG

def setup_logging():
    """Konfiguruje system logowania"""
    log_file = Path("fill_gaps.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_orderbook_data(symbol: str, logger) -> Optional[pd.DataFrame]:
    """Wczytuje dane order book dla jednej pary"""
    feather_file = Path("merged_raw") / f"orderbook_merged_{symbol}.feather"
    
    logger.info(f"Wczytuję dane z {feather_file}...")
    
    if not os.path.exists(feather_file):
        logger.error(f"Plik {feather_file} nie istnieje!")
        return None
    
    try:
        df = pd.read_feather(feather_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Wczytano {len(df):,} wierszy")
        logger.info(f"Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
        
        return df
    except Exception as e:
        logger.error(f"Błąd wczytania {feather_file}: {e}")
        return None

def calculate_price_change(df, gap_start_idx, gap_end_idx):
    """Oblicza zmianę ceny wokół luki"""
    # Znajdź indeksy przed i po luce
    before_idx = gap_start_idx - 1
    after_idx = gap_end_idx
    
    if before_idx < 0 or after_idx >= len(df):
        return 0.0
    
    # Oblicz średnią cenę z order book (użyj poziomu 0.1% jako proxy)
    def get_price_from_orderbook(row):
        depth_cols = [col for col in row.index if col.startswith('snapshot1_depth_')]
        notional_cols = [col for col in row.index if col.startswith('snapshot1_notional_')]
        
        if len(depth_cols) > 0 and len(notional_cols) > 0:
            # Użyj pierwszego poziomu jako proxy ceny
            depth = row[depth_cols[0]]
            notional = row[notional_cols[0]]
            if depth > 0:
                return notional / depth
        return 0.0
    
    price_before = get_price_from_orderbook(df.iloc[before_idx])
    price_after = get_price_from_orderbook(df.iloc[after_idx])
    
    if price_before > 0:
        return abs(price_after - price_before) / price_before * 100
    return 0.0

def interpolate_orderbook(snapshot1, snapshot2, ratio):
    """Interpoluje między dwoma snapshotami order book"""
    interpolated = {}
    
    # Interpoluj wszystkie kolumny order book
    for col in snapshot1.index:
        if col.startswith('snapshot1_depth_') or col.startswith('snapshot1_notional_') or \
           col.startswith('snapshot2_depth_') or col.startswith('snapshot2_notional_'):
            if col in snapshot1 and col in snapshot2:
                val1 = snapshot1[col]
                val2 = snapshot2[col]
                interpolated[col] = val1 + (val2 - val1) * ratio
    
    return interpolated

def rolling_average_orderbook(df, center_idx, window_minutes=30):
    """Oblicza rolling average order book"""
    # Znajdź indeksy w oknie czasowym
    center_time = df.iloc[center_idx]['timestamp']
    start_time = center_time - timedelta(minutes=window_minutes//2)
    end_time = center_time + timedelta(minutes=window_minutes//2)
    
    window_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    if len(window_data) == 0:
        return None
    
    # Oblicz średnią dla każdej kolumny order book
    avg_snapshot = {}
    for col in df.columns:
        if col.startswith('snapshot1_depth_') or col.startswith('snapshot1_notional_') or \
           col.startswith('snapshot2_depth_') or col.startswith('snapshot2_notional_'):
            avg_snapshot[col] = window_data[col].mean()
    
    return avg_snapshot

def fill_gaps_intelligently(df, symbol: str, logger, max_small_gap_minutes=5, max_medium_gap_minutes=60, 
                          price_change_threshold=2.0):
    """Inteligentnie wypełnia luki w danych order book dla jednej pary"""
    logger.info(f"Rozpoczynam inteligentne wypełnianie luk dla {symbol}...")
    
    # Znajdź wszystkie luki
    df['time_diff'] = df['timestamp'].diff()
    gaps = df[df['time_diff'] > timedelta(minutes=1)].copy()
    
    logger.info(f"Znaleziono {len(gaps)} luk do wypełnienia")
    
    if len(gaps) == 0:
        logger.info(f"Brak luk do wypełnienia dla {symbol}")
        return df
    
    # Przygotuj nowy DataFrame
    filled_df = df.copy()
    filled_rows = []
    
    total_gaps = len(gaps)
    processed_gaps = 0
    
    for gap_idx, gap_row in gaps.iterrows():
        processed_gaps += 1
        
        if processed_gaps % 50 == 0:
            progress = (processed_gaps / total_gaps) * 100
            logger.info(f"Postęp {symbol}: {processed_gaps:,}/{total_gaps:,} luk ({progress:.1f}%)")
        
        # Oblicz parametry luki
        gap_start_time = gap_row['timestamp'] - gap_row['time_diff']
        gap_end_time = gap_row['timestamp']
        gap_duration_minutes = gap_row['time_diff'].total_seconds() / 60
        
        # Znajdź indeksy
        gap_start_idx = filled_df[filled_df['timestamp'] == gap_start_time].index[0]
        gap_end_idx = gap_start_idx + 1
        
        # Oblicz zmianę ceny
        price_change = calculate_price_change(filled_df, gap_start_idx, gap_end_idx)
        
        # Wybierz metodę wypełniania
        if gap_duration_minutes <= max_small_gap_minutes:
            method = "interpolacja"
        elif gap_duration_minutes <= max_medium_gap_minutes and price_change < price_change_threshold:
            method = "rolling_average"
        else:
            method = "forward_fill"
        
        # Wypełnij luki
        current_time = gap_start_time + timedelta(minutes=1)
        while current_time < gap_end_time:
            if method == "interpolacja":
                # Interpolacja liniowa
                total_gap_minutes = gap_duration_minutes
                minutes_from_start = (current_time - gap_start_time).total_seconds() / 60
                ratio = minutes_from_start / total_gap_minutes
                
                snapshot1 = filled_df.iloc[gap_start_idx]
                snapshot2 = filled_df.iloc[gap_end_idx]
                
                interpolated = interpolate_orderbook(snapshot1, snapshot2, ratio)
                
            elif method == "rolling_average":
                # Rolling average
                center_idx = gap_start_idx + (gap_end_idx - gap_start_idx) // 2
                interpolated = rolling_average_orderbook(filled_df, center_idx)
                
            else:  # forward_fill
                # Użyj ostatniego znanego snapshotu
                interpolated = filled_df.iloc[gap_start_idx].to_dict()
            
            if interpolated:
                # Stwórz nowy wiersz
                new_row = {
                    'timestamp': current_time,
                    'snapshot1_timestamp': current_time,
                    'snapshot2_timestamp': current_time,
                    'fill_method': method,
                    'gap_duration_minutes': gap_duration_minutes,
                    'price_change_percent': price_change
                }
                
                # Dodaj dane order book
                for col, value in interpolated.items():
                    if col.startswith('snapshot1_') or col.startswith('snapshot2_'):
                        new_row[col] = value
                
                filled_rows.append(new_row)
            
            current_time += timedelta(minutes=1)
    
    # Dodaj wypełnione wiersze do DataFrame
    if filled_rows:
        filled_df_new = pd.concat([filled_df, pd.DataFrame(filled_rows)], ignore_index=True)
        filled_df_new = filled_df_new.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Wypełniono {len(filled_rows):,} brakujących minut dla {symbol}")
        logger.info(f"Nowy rozmiar {symbol}: {len(filled_df_new):,} wierszy")
        
        return filled_df_new
    else:
        logger.info(f"Brak luk do wypełnienia dla {symbol}")
        return filled_df

def fill_orderbook_gaps_for_symbol(symbol: str, logger, max_small_gap_minutes=5, max_medium_gap_minutes=60, 
                                 price_change_threshold=2.0) -> Optional[pd.DataFrame]:
    """Wypełnia luki w danych order book dla jednej pary"""
    logger.info(f"Rozpoczynam wypełnianie luk dla {symbol}")
    
    # KROK 1: Wczytaj dane
    df = load_orderbook_data(symbol, logger)
    if df is None:
        return None
    
    # KROK 2: Wypełnij luki
    filled_df = fill_gaps_intelligently(
        df, symbol, logger,
        max_small_gap_minutes=max_small_gap_minutes,
        max_medium_gap_minutes=max_medium_gap_minutes,
        price_change_threshold=price_change_threshold
    )
    
    # KROK 3: Zapisz wynik
    output_dir = Path("orderbook_completed")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"orderbook_filled_{symbol}.feather"
    logger.info(f"Zapisuję {symbol} do {output_file}...")
    filled_df.to_feather(output_file)
    
    # KROK 4: Podsumowanie
    logger.info(f"Proces zakończony dla {symbol}!")
    logger.info(f"  Oryginalny rozmiar: {len(df):,} wierszy")
    logger.info(f"  Nowy rozmiar: {len(filled_df):,} wierszy")
    logger.info(f"  Dodano: {len(filled_df) - len(df):,} wierszy")
    
    # Statystyki metod wypełniania
    if 'fill_method' in filled_df.columns:
        method_stats = filled_df['fill_method'].value_counts()
        logger.info(f"  Statystyki metod wypełniania:")
        for method, count in method_stats.items():
            logger.info(f"    {method}: {count:,} wierszy")
    
    return filled_df

def fill_all_pairs_gaps(max_small_gap_minutes=5, max_medium_gap_minutes=60, price_change_threshold=2.0):
    """Główna funkcja wypełniania luk dla wszystkich par"""
    logger = setup_logging()
    
    logger.info("Rozpoczynam wypełnianie luk dla wszystkich par")
    logger.info(f"Pary: {', '.join(PAIRS)}")
    logger.info(f"Parametry:")
    logger.info(f"  - Małe luki (interpolacja): ≤ {max_small_gap_minutes} min")
    logger.info(f"  - Średnie luki (rolling): ≤ {max_medium_gap_minutes} min")
    logger.info(f"  - Próg zmiany ceny: {price_change_threshold}%")
    
    start_time = datetime.now()
    successful_pairs = []
    failed_pairs = []
    
    for i, symbol in enumerate(PAIRS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Przetwarzam parę {i}/{len(PAIRS)}: {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            result = fill_orderbook_gaps_for_symbol(
                symbol, logger,
                max_small_gap_minutes=max_small_gap_minutes,
                max_medium_gap_minutes=max_medium_gap_minutes,
                price_change_threshold=price_change_threshold
            )
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
    logger.info(f"WYPEŁNIANIE LUK ZAKOŃCZONE!")
    logger.info(f"{'='*60}")
    logger.info(f"Udało się: {len(successful_pairs)}/{len(PAIRS)} par")
    logger.info(f"Czas: {duration}")
    
    if successful_pairs:
        logger.info(f"Pomyślnie przetworzone pary: {', '.join(successful_pairs)}")
    
    if failed_pairs:
        logger.warning(f"Nieudane pary: {', '.join(failed_pairs)}")
    
    # Zapisz metadane
    metadata = {
        'fill_date': datetime.now().isoformat(),
        'pairs': PAIRS,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'duration_seconds': duration.total_seconds(),
        'success_rate': f"{len(successful_pairs)/len(PAIRS)*100:.1f}%",
        'parameters': {
            'max_small_gap_minutes': max_small_gap_minutes,
            'max_medium_gap_minutes': max_medium_gap_minutes,
            'price_change_threshold': price_change_threshold
        }
    }
    
    metadata_file = Path("fill_gaps_metadata.json")
    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadane zapisane: {metadata_file}")
    except Exception as e:
        logger.error(f"Błąd zapisywania metadanych: {e}")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Inteligentnie wypełnia luki w danych order book')
    parser.add_argument('--symbol', help='Przetwórz tylko jedną parę (opcjonalnie)')
    parser.add_argument('--max-small-gap', type=int, default=5, help='Maksymalna luka dla interpolacji (minuty)')
    parser.add_argument('--max-medium-gap', type=int, default=60, help='Maksymalna luka dla rolling average (minuty)')
    parser.add_argument('--price-threshold', type=float, default=2.0, help='Próg zmiany ceny (%)')
    
    args = parser.parse_args()
    
    if args.symbol:
        # Przetwórz tylko jedną parę
        logger = setup_logging()
        result = fill_orderbook_gaps_for_symbol(
            args.symbol, logger,
            max_small_gap_minutes=args.max_small_gap,
            max_medium_gap_minutes=args.max_medium_gap,
            price_change_threshold=args.price_threshold
        )
        if result is not None:
            print(f"\n[OK] Para {args.symbol} została pomyślnie przetworzona!")
        else:
            print(f"\n[ERROR] Błąd podczas przetwarzania {args.symbol}!")
    else:
        # Przetwórz wszystkie pary
        fill_all_pairs_gaps(
            max_small_gap_minutes=args.max_small_gap,
            max_medium_gap_minutes=args.max_medium_gap,
            price_change_threshold=args.price_threshold
        )

if __name__ == "__main__":
    main() 