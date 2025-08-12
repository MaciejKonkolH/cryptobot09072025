#!/usr/bin/env python3
"""
Skrypt sprawdzający zakres dostępnych dat w plikach features feather.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import logging

def setup_logging():
    """Konfiguruje system logowania."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_features_file_range(file_path: str) -> dict:
    """Sprawdza zakres dat w pliku features feather."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Sprawdzam plik: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Plik nie istnieje: {file_path}")
            return None
        
        # Wczytaj dane
        logger.info("Wczytywanie danych...")
        df = pd.read_feather(file_path)
        
        # Sprawdź czy timestamp jest kolumną czy indeksem
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            timestamps = df['timestamp']
        else:
            # Sprawdź czy indeks to timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index
            else:
                logger.error("Nie znaleziono kolumny timestamp ani DatetimeIndex")
                return None
        
        # Oblicz zakres
        min_date = timestamps.min()
        max_date = timestamps.max()
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Oblicz rozmiar pliku
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Sprawdź czy są luki w danych (co minutę)
        expected_minutes = int((max_date - min_date).total_seconds() / 60)
        actual_minutes = len(timestamps)
        missing_minutes = expected_minutes - actual_minutes
        
        # Sprawdź gęstość danych
        data_density = actual_minutes / (expected_minutes + 1) * 100
        
        result = {
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'min_date': min_date,
            'max_date': max_date,
            'date_range_days': (max_date - min_date).days,
            'expected_minutes': expected_minutes,
            'actual_minutes': actual_minutes,
            'missing_minutes': missing_minutes,
            'data_density_percent': data_density,
            'columns_list': list(df.columns)
        }
        
        logger.info(f"✅ Analiza zakończona pomyślnie")
        return result
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy pliku: {e}")
        return None

def print_range_summary(result: dict):
    """Wyświetla podsumowanie zakresu dat."""
    if not result:
        return
    
    print("\n" + "="*80)
    print("📊 ANALIZA ZAKRESU DAT - PLIK FEATURES")
    print("="*80)
    
    print(f"📁 Plik: {result['file_path']}")
    print(f"📦 Rozmiar: {result['file_size_mb']:.2f} MB")
    print(f"📊 Wiersze: {result['total_rows']:,}")
    print(f"📋 Kolumny: {result['total_columns']}")
    
    print(f"\n📅 ZAKRES DAT:")
    print(f"   Najstarsza data: {result['min_date']}")
    print(f"   Najnowsza data:  {result['max_date']}")
    print(f"   Zakres dni:      {result['date_range_days']:,} dni")
    
    print(f"\n⏱️  SZCZEGÓŁY CZASOWE:")
    print(f"   Oczekiwane minuty: {result['expected_minutes']:,}")
    print(f"   Rzeczywiste minuty: {result['actual_minutes']:,}")
    print(f"   Brakujące minuty:   {result['missing_minutes']:,}")
    print(f"   Gęstość danych:     {result['data_density_percent']:.2f}%")
    
    print(f"\n📋 LISTA KOLUMN ({result['total_columns']}):")
    for i, col in enumerate(result['columns_list'], 1):
        print(f"   {i:2d}. {col}")
    
    print("="*80)

def check_all_features_files(output_dir: str):
    """Sprawdza wszystkie pliki features w katalogu."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.error(f"Katalog nie istnieje: {output_dir}")
        return
    
    # Znajdź wszystkie pliki features_*.feather
    feature_files = list(output_path.glob("features_*.feather"))
    
    if not feature_files:
        logger.warning(f"Nie znaleziono plików features_*.feather w {output_dir}")
        return
    
    logger.info(f"Znaleziono {len(feature_files)} plików features")
    
    results = []
    for file_path in feature_files:
        result = check_features_file_range(str(file_path))
        if result:
            results.append(result)
            print_range_summary(result)
    
    # Podsumowanie wszystkich plików
    if results:
        print("\n" + "="*80)
        print("📊 PODSUMOWANIE WSZYSTKICH PLIKÓW")
        print("="*80)
        
        for result in results:
            symbol = Path(result['file_path']).stem.replace('features_', '')
            print(f"{symbol:10s} | {result['min_date'].strftime('%Y-%m-%d')} | {result['max_date'].strftime('%Y-%m-%d')} | {result['total_rows']:8,} wierszy | {result['file_size_mb']:6.1f} MB")
        
        print("="*80)

def main():
    """Główna funkcja programu."""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Sprawdzanie zakresu dat w plikach features feather')
    parser.add_argument('--file', type=str, help='Ścieżka do konkretnego pliku features')
    parser.add_argument('--dir', type=str, default='output', help='Katalog z plikami features (domyślnie: output)')
    parser.add_argument('--all', action='store_true', help='Sprawdź wszystkie pliki features w katalogu')
    
    args = parser.parse_args()
    
    if args.file:
        # Sprawdź konkretny plik
        result = check_features_file_range(args.file)
        if result:
            print_range_summary(result)
    elif args.all:
        # Sprawdź wszystkie pliki w katalogu
        check_all_features_files(args.dir)
    else:
        # Sprawdź domyślny plik BTCUSDT
        default_file = os.path.join(args.dir, 'features_BTCUSDT.feather')
        logger.info(f"Sprawdzam domyślny plik: {default_file}")
        result = check_features_file_range(default_file)
        if result:
            print_range_summary(result)
        else:
            logger.error("Nie można sprawdzić domyślnego pliku")

if __name__ == "__main__":
    main() 