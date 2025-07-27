#!/usr/bin/env python3
"""
Skrypt do analizy rozkładu etykiet w pliku feather używanym przez moduł trenujący.
Umożliwia filtrowanie po zakresie dat.
"""

import pandas as pd
import argparse
from datetime import datetime
import sys
import os

def load_data(file_path):
    """Wczytuje dane z pliku feather."""
    try:
        print(f"Wczytywanie danych z: {file_path}")
        df = pd.read_feather(file_path)
        print(f"Wczytano: {len(df):,} wierszy, {len(df.columns)} kolumn")
        return df
    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        return None

def analyze_label_distribution(df, start_date=None, end_date=None):
    """Analizuje rozkład etykiet w danych."""
    
    # Konwertuj timestamp na datetime jeśli nie jest
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Filtruj po zakresie dat jeśli podano
    if start_date or end_date:
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
            print(f"Filtrowanie od: {start_dt}")
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            print(f"Filtrowanie do: {end_dt}")
        
        print(f"Po filtrowaniu: {len(df):,} wierszy")
    
    # Znajdź kolumny z etykietami
    label_columns = [col for col in df.columns if col.startswith('label_')]
    
    if not label_columns:
        print("Nie znaleziono kolumn z etykietami (label_*)")
        return
    
    print(f"\nZnaleziono {len(label_columns)} kolumn z etykietami:")
    for col in label_columns:
        print(f"  - {col}")
    
    print("\n" + "="*80)
    print("ANALIZA ROZKŁADU ETYKIET")
    print("="*80)
    
    # Analiza dla każdej kolumny etykiet
    for col in label_columns:
        print(f"\n--- {col} ---")
        
        # Rozkład klas
        value_counts = df[col].value_counts().sort_index()
        total = len(df[col].dropna())
        
        print(f"Liczba próbek: {total:,}")
        print(f"Rozkład klas:")
        
        for class_id, count in value_counts.items():
            percentage = (count / total) * 100
            class_name = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}.get(class_id, f'KLASA_{class_id}')
            print(f"  {class_name} ({class_id}): {count:,} ({percentage:.2f}%)")
        
        # Sprawdź proporcje LONG vs SHORT
        if 0 in value_counts and 1 in value_counts:
            long_count = value_counts[0]
            short_count = value_counts[1]
            ratio = short_count / long_count if long_count > 0 else float('inf')
            print(f"Proporcja SHORT/LONG: {ratio:.3f}")
            
            if ratio > 1.1:
                print(f"  ⚠️  UWAGA: Znacznie więcej pozycji SHORT niż LONG!")
            elif ratio < 0.9:
                print(f"  ⚠️  UWAGA: Znacznie więcej pozycji LONG niż SHORT!")
            else:
                print(f"  ✅ Proporcja w normie")
        
        # Sprawdź czy SHORT > NEUTRAL (bardzo dziwne)
        if 1 in value_counts and 2 in value_counts:
            short_count = value_counts[1]
            neutral_count = value_counts[2]
            if short_count > neutral_count:
                print(f"  🚨 BŁĄD: SHORT ({short_count:,}) > NEUTRAL ({neutral_count:,}) - to nie ma sensu!")
        
        # Analiza czasowa
        print(f"\nAnaliza czasowa:")
        print(f"  Zakres dat: {df.index.min()} - {df.index.max()}")
        print(f"  Okres: {(df.index.max() - df.index.min()).days} dni")
        
        # Sprawdź czy są jakieś wzorce czasowe
        monthly_dist = df.groupby([df.index.year, df.index.month, col]).size().unstack(fill_value=0)
        if not monthly_dist.empty:
            print(f"  Rozkład miesięczny (pierwsze 3 miesiące):")
            for i, (year, month) in enumerate(monthly_dist.index[:3]):
                month_data = monthly_dist.loc[(year, month)]
                print(f"    {year}-{month:02d}: LONG={month_data.get(0, 0):,}, SHORT={month_data.get(1, 0):,}, NEUTRAL={month_data.get(2, 0):,}")

def main():
    parser = argparse.ArgumentParser(description='Analiza rozkładu etykiet w danych treningowych')
    parser.add_argument('--file', '-f', 
                       default='../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather',
                       help='Ścieżka do pliku feather (domyślnie: ../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather)')
    parser.add_argument('--start-date', '-s',
                       help='Data początkowa (format: YYYY-MM-DD lub YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-date', '-e',
                       help='Data końcowa (format: YYYY-MM-DD lub YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--list-columns', '-l', action='store_true',
                       help='Wyświetl wszystkie kolumny w pliku')
    
    args = parser.parse_args()
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(args.file):
        print(f"Błąd: Plik {args.file} nie istnieje!")
        sys.exit(1)
    
    # Wczytaj dane
    df = load_data(args.file)
    if df is None:
        sys.exit(1)
    
    # Wyświetl kolumny jeśli żądane
    if args.list_columns:
        print(f"\nWszystkie kolumny ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        print()
    
    # Analizuj rozkład etykiet
    analyze_label_distribution(df, args.start_date, args.end_date)

if __name__ == "__main__":
    main() 