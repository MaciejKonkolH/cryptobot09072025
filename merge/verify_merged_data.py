import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime

def load_merged_data(input_file="merged_ohlc_orderbook.feather"):
    """Wczytuje połączone dane"""
    print(f"📊 Wczytuję połączone dane z {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"❌ Plik {input_file} nie istnieje!")
        return None
    
    df = pd.read_feather(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Wczytano {len(df):,} wierszy")
    print(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
    print(f"📋 Liczba kolumn: {len(df.columns)}")
    
    return df

def analyze_missing_data(df):
    """Analizuje brakujące dane w szczegółach"""
    print(f"\n🔍 SZCZEGÓŁOWA ANALIZA BRAKUJĄCYCH DANYCH")
    print("=" * 80)
    
    # 1. Ogólne statystyki brakujących danych
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    print(f"📊 OGÓLNE STATYSTYKI:")
    print(f"  Łączna liczba komórek: {total_cells:,}")
    print(f"  Brakujące komórki: {missing_cells:,}")
    print(f"  Procent brakujących: {missing_percentage:.2f}%")
    
    # 2. Analiza kolumn z brakującymi danymi
    missing_by_column = df.isnull().sum()
    columns_with_missing = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
    
    print(f"\n📋 KOLUMNY Z BRAKUJĄCYMI DANYMI:")
    if len(columns_with_missing) > 0:
        print(f"  Znaleziono {len(columns_with_missing)} kolumn z brakującymi danymi:")
        print()
        
        for i, (col, count) in enumerate(columns_with_missing.items(), 1):
            percentage = (count / len(df)) * 100
            print(f"  {i:2d}. {col}:")
            print(f"      Brakuje: {count:,} wierszy ({percentage:.2f}%)")
            
            # Dodatkowe informacje dla kolumn orderbook
            if col.startswith(('snapshot1_', 'snapshot2_')):
                print(f"      Typ: Orderbook snapshot")
            elif col in ['open', 'high', 'low', 'close', 'volume']:
                print(f"      Typ: OHLC")
            else:
                print(f"      Typ: Inne")
            print()
    else:
        print("  ✅ Brak kolumn z brakującymi danymi!")
    
    # 3. Analiza wierszy z brakującymi danymi
    missing_by_row = df.isnull().sum(axis=1)
    rows_with_missing = missing_by_row[missing_by_row > 0]
    
    print(f"📊 ANALIZA WIERSZY Z BRAKUJĄCYMI DANYMI:")
    if len(rows_with_missing) > 0:
        print(f"  Wiersze z brakującymi danymi: {len(rows_with_missing):,} ({len(rows_with_missing)/len(df)*100:.2f}%)")
        
        # Statystyki liczby brakujących kolumn na wiersz
        missing_counts = rows_with_missing.value_counts().sort_index()
        print(f"  Rozkład liczby brakujących kolumn na wiersz:")
        for missing_count, row_count in missing_counts.head(10).items():
            print(f"    {missing_count} kolumn brakuje: {row_count:,} wierszy")
        
        if len(missing_counts) > 10:
            print(f"    ... i {len(missing_counts) - 10} więcej kategorii")
    else:
        print("  ✅ Brak wierszy z brakującymi danymi!")
    
    # 4. Analiza czasowa brakujących danych
    print(f"\n⏰ ANALIZA CZASOWA BRAKUJĄCYCH DANYCH:")
    
    # Znajdź wiersze z brakującymi danymi orderbook
    orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    if orderbook_columns:
        first_ob_col = orderbook_columns[0]
        missing_orderbook_rows = df[df[first_ob_col].isna()]
        
        if len(missing_orderbook_rows) > 0:
            print(f"  Wiersze bez danych orderbook: {len(missing_orderbook_rows):,}")
            print(f"  Zakres czasowy brakujących orderbook:")
            print(f"    Od: {missing_orderbook_rows['timestamp'].min()}")
            print(f"    Do: {missing_orderbook_rows['timestamp'].max()}")
            
            # Sprawdź czy są luki czasowe
            missing_timestamps = missing_orderbook_rows['timestamp'].sort_values()
            if len(missing_timestamps) > 1:
                time_diffs = missing_timestamps.diff().dropna()
                max_gap = time_diffs.max()
                print(f"  Maksymalna luka czasowa: {max_gap}")
        else:
            print("  ✅ Wszystkie wiersze mają dane orderbook!")
    
    # 5. Analiza kolumn OHLC
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    print(f"\n📈 ANALIZA KOLUMN OHLC:")
    for col in ohlc_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count:,} brakujących ({percentage:.2f}%)")
            else:
                print(f"  {col}: ✅ Brak brakujących danych")
        else:
            print(f"  {col}: ❌ Kolumna nie istnieje")
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_cells': missing_cells,
        'missing_percentage': missing_percentage,
        'columns_with_missing': len(columns_with_missing),
        'rows_with_missing': len(rows_with_missing)
    }

def check_data_quality(df):
    """Sprawdza jakość danych"""
    print(f"\n🔬 SPRAWDZANIE JAKOŚCI DANYCH")
    print("=" * 80)
    
    issues = []
    
    # 1. Sprawdź wartości ujemne w cenach
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            negative_prices = (df[col] < 0).sum()
            if negative_prices > 0:
                issues.append(f"Ujemne ceny w {col}: {negative_prices:,} wierszy")
    
    # 2. Sprawdź spójność OHLC
    if all(col in df.columns for col in price_columns):
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['close'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] < df['low'])
        ).sum()
        
        if invalid_ohlc > 0:
            issues.append(f"Niespójne dane OHLC: {invalid_ohlc:,} wierszy")
    
    # 3. Sprawdź wartości zerowe w volume
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"Zerowy volume: {zero_volume:,} wierszy")
    
    # 4. Sprawdź orderbook snapshoty
    orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    if orderbook_columns:
        # Sprawdź czy są wartości ujemne w orderbook
        for col in orderbook_columns:
            if df[col].dtype in ['float64', 'int64']:
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    issues.append(f"Ujemne wartości w {col}: {negative_values:,} wierszy")
    
    # 5. Sprawdź duplikaty timestampów
    duplicate_timestamps = df['timestamp'].duplicated().sum()
    if duplicate_timestamps > 0:
        issues.append(f"Duplikaty timestampów: {duplicate_timestamps:,} wierszy")
    
    # Wyświetl wyniki
    if issues:
        print("❌ ZNALEZIONE PROBLEMY:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ Brak problemów z jakością danych!")
    
    return issues

def generate_report(stats, issues, output_file="data_verification_report.txt"):
    """Generuje raport z weryfikacji"""
    print(f"\n📄 GENEROWANIE RAPORTU...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RAPORT WERYFIKACJI DANYCH PO ŁĄCZENIU OHLC Z ORDERBOOK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data generowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PODSUMOWANIE STATYSTYK:\n")
        f.write(f"  Łączna liczba wierszy: {stats['total_rows']:,}\n")
        f.write(f"  Łączna liczba kolumn: {stats['total_columns']:,}\n")
        f.write(f"  Brakujące komórki: {stats['missing_cells']:,}\n")
        f.write(f"  Procent brakujących: {stats['missing_percentage']:.2f}%\n")
        f.write(f"  Kolumny z brakującymi danymi: {stats['columns_with_missing']}\n")
        f.write(f"  Wiersze z brakującymi danymi: {stats['rows_with_missing']:,}\n\n")
        
        f.write("PROBLEMY Z JAKOŚCIĄ:\n")
        if issues:
            for i, issue in enumerate(issues, 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("  Brak problemów z jakością danych\n")
    
    print(f"✅ Raport zapisany: {output_file}")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Weryfikuje połączone dane OHLC z orderbook')
    parser.add_argument('--input', default='merged_ohlc_orderbook.feather', 
                       help='Ścieżka do połączonego pliku danych')
    parser.add_argument('--report', default='data_verification_report.txt',
                       help='Nazwa pliku raportu')
    
    args = parser.parse_args()
    
    print("🔍 ROZPOCZYNAM WERYFIKACJĘ POŁĄCZONYCH DANYCH")
    print("=" * 80)
    
    # Wczytaj dane
    df = load_merged_data(args.input)
    if df is None:
        return
    
    # Analizuj brakujące dane
    stats = analyze_missing_data(df)
    
    # Sprawdź jakość danych
    issues = check_data_quality(df)
    
    # Wygeneruj raport
    generate_report(stats, issues, args.report)
    
    print(f"\n🎉 WERYFIKACJA ZAKOŃCZONA!")
    print(f"📁 Raport: {args.report}")
    
    # Podsumowanie
    if stats['missing_percentage'] > 10:
        print(f"⚠️  UWAGA: Wysoki procent brakujących danych ({stats['missing_percentage']:.2f}%)")
    elif stats['missing_percentage'] > 5:
        print(f"⚠️  Uwaga: Średni procent brakujących danych ({stats['missing_percentage']:.2f}%)")
    else:
        print(f"✅ Niski procent brakujących danych ({stats['missing_percentage']:.2f}%)")
    
    if issues:
        print(f"⚠️  Znaleziono {len(issues)} problemów z jakością danych")
    else:
        print("✅ Brak problemów z jakością danych")

if __name__ == "__main__":
    main() 