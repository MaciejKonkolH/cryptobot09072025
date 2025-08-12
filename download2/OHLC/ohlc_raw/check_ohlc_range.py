#!/usr/bin/env python3
"""
Skrypt sprawdzający zakres dat w plikach OHLC CSV
"""

import pandas as pd
import os
import sys

def check_ohlc_range(file_path):
    """Sprawdza zakres dat w pliku OHLC CSV"""
    try:
        print(f"Sprawdzam plik: {file_path}")
        
        # Wczytaj dane
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Oblicz zakres
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        total_rows = len(df)
        
        print(f"  Zakres: {min_date} do {max_date}")
        print(f"  Wiersze: {total_rows:,}")
        print(f"  Dni: {(max_date - min_date).days}")
        print()
        
        return {
            'file': file_path,
            'min_date': min_date,
            'max_date': max_date,
            'rows': total_rows
        }
        
    except Exception as e:
        print(f"  Błąd: {e}")
        return None

def main():
    """Główna funkcja"""
    # Sprawdź konkretny plik BTCUSDT
    result = check_ohlc_range('BTCUSDT_1m.csv')
    
    if result:
        print("="*60)
        print("PODSUMOWANIE BTCUSDT:")
        print(f"  Plik: {result['file']}")
        print(f"  Zakres: {result['min_date']} - {result['max_date']}")
        print(f"  Wiersze: {result['rows']:,}")
        print("="*60)

if __name__ == "__main__":
    main() 