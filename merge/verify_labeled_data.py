import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime

def load_labeled_data(input_file="../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather"):
    """Wczytuje dane po etykietowaniu"""
    print(f"📊 Wczytuję dane po etykietowaniu z {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"❌ Plik {input_file} nie istnieje!")
        return None
    
    df = pd.read_feather(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Wczytano {len(df):,} wierszy")
    print(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
    print(f"📋 Liczba kolumn: {len(df.columns)}")
    
    return df

def analyze_labeled_data_structure(df):
    """Analizuje strukturę danych po etykietowaniu"""
    print(f"\n📋 ANALIZA STRUKTURY DANYCH PO ETYKIETOWANIU")
    print("=" * 80)
    
    # 1. Kategorie kolumn
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    label_columns = [col for col in df.columns if col.startswith('label_')]
    feature_columns = [col for col in df.columns if col not in ohlc_columns + orderbook_columns + label_columns + ['timestamp']]
    
    print(f"📊 KATEGORIE KOLUMN:")
    print(f"  OHLC: {len(ohlc_columns)} kolumn")
    print(f"  Orderbook: {len(orderbook_columns)} kolumn")
    print(f"  Etykiety: {len(label_columns)} kolumn")
    print(f"  Cechy: {len(feature_columns)} kolumn")
    print(f"  Inne: {len(df.columns) - len(ohlc_columns) - len(orderbook_columns) - len(label_columns) - len(feature_columns) - 1} kolumn")
    
    # 2. Sprawdź etykiety
    print(f"\n🏷️  ANALIZA ETYKIET:")
    for label_col in label_columns:
        if label_col in df.columns:
            unique_labels = df[label_col].value_counts().sort_index()
            print(f"  {label_col}:")
            print(f"    Unikalne wartości: {unique_labels.to_dict()}")
            print(f"    Rozkład: {unique_labels.to_dict()}")
            print()

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
            
            # Kategoryzacja kolumn
            if col.startswith('label_'):
                print(f"      Typ: Etykieta")
            elif col.startswith(('snapshot1_', 'snapshot2_')):
                print(f"      Typ: Orderbook snapshot")
            elif col in ['open', 'high', 'low', 'close', 'volume']:
                print(f"      Typ: OHLC")
            elif col in ['bb_width', 'rsi_14', 'macd_hist', 'buy_sell_ratio_s1']:
                print(f"      Typ: Cecha techniczna")
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
    
    # 4. Analiza kategorii kolumn
    print(f"\n📊 ANALIZA BRAKUJĄCYCH DANYCH WG KATEGORII:")
    
    # OHLC
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    ohlc_missing = sum(df[col].isna().sum() for col in ohlc_columns if col in df.columns)
    print(f"  OHLC: {ohlc_missing:,} brakujących komórek")
    
    # Orderbook
    orderbook_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    orderbook_missing = sum(df[col].isna().sum() for col in orderbook_columns)
    print(f"  Orderbook: {orderbook_missing:,} brakujących komórek")
    
    # Etykiety
    label_columns = [col for col in df.columns if col.startswith('label_')]
    label_missing = sum(df[col].isna().sum() for col in label_columns)
    print(f"  Etykiety: {label_missing:,} brakujących komórek")
    
    # Cechy techniczne
    feature_columns = ['bb_width', 'rsi_14', 'macd_hist', 'buy_sell_ratio_s1']
    feature_missing = sum(df[col].isna().sum() for col in feature_columns if col in df.columns)
    print(f"  Cechy techniczne: {feature_missing:,} brakujących komórek")
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_cells': missing_cells,
        'missing_percentage': missing_percentage,
        'columns_with_missing': len(columns_with_missing),
        'rows_with_missing': len(rows_with_missing),
        'ohlc_missing': ohlc_missing,
        'orderbook_missing': orderbook_missing,
        'label_missing': label_missing,
        'feature_missing': feature_missing
    }

def check_label_quality(df):
    """Sprawdza jakość etykiet"""
    print(f"\n🏷️  SPRAWDZANIE JAKOŚCI ETYKIET")
    print("=" * 80)
    
    issues = []
    label_columns = [col for col in df.columns if col.startswith('label_')]
    
    for label_col in label_columns:
        if label_col in df.columns:
            # Sprawdź czy są tylko wartości 0, 1, 2
            unique_values = df[label_col].unique()
            invalid_values = [val for val in unique_values if val not in [0, 1, 2]]
            
            if invalid_values:
                issues.append(f"Nieprawidłowe wartości w {label_col}: {invalid_values}")
            
            # Sprawdź rozkład
            value_counts = df[label_col].value_counts().sort_index()
            print(f"  {label_col}: {value_counts.to_dict()}")
            
            # Sprawdź czy nie ma tylko jednej klasy
            if len(value_counts) == 1:
                issues.append(f"Tylko jedna klasa w {label_col}: {value_counts.index[0]}")
    
    if issues:
        print("❌ PROBLEMY Z ETYKIETAMI:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ Brak problemów z etykietami!")
    
    return issues

def check_data_quality(df):
    """Sprawdza ogólną jakość danych"""
    print(f"\n🔬 SPRAWDZANIE OGÓLNEJ JAKOŚCI DANYCH")
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
    
    # 4. Sprawdź duplikaty timestampów
    duplicate_timestamps = df['timestamp'].duplicated().sum()
    if duplicate_timestamps > 0:
        issues.append(f"Duplikaty timestampów: {duplicate_timestamps:,} wierszy")
    
    # 5. Sprawdź czy etykiety są w zakresie [0, 1, 2]
    label_columns = [col for col in df.columns if col.startswith('label_')]
    for label_col in label_columns:
        if label_col in df.columns:
            invalid_labels = df[~df[label_col].isin([0, 1, 2])]
            if len(invalid_labels) > 0:
                issues.append(f"Nieprawidłowe etykiety w {label_col}: {len(invalid_labels):,} wierszy")
    
    # Wyświetl wyniki
    if issues:
        print("❌ ZNALEZIONE PROBLEMY:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ Brak problemów z jakością danych!")
    
    return issues

def generate_report(stats, label_issues, data_issues, output_file="labeled_data_verification_report.txt"):
    """Generuje raport z weryfikacji"""
    print(f"\n📄 GENEROWANIE RAPORTU...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RAPORT WERYFIKACJI DANYCH PO ETYKIETOWANIU (PRZED TRENINGIEM)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data generowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PODSUMOWANIE STATYSTYK:\n")
        f.write(f"  Łączna liczba wierszy: {stats['total_rows']:,}\n")
        f.write(f"  Łączna liczba kolumn: {stats['total_columns']:,}\n")
        f.write(f"  Brakujące komórki: {stats['missing_cells']:,}\n")
        f.write(f"  Procent brakujących: {stats['missing_percentage']:.2f}%\n")
        f.write(f"  Kolumny z brakującymi danymi: {stats['columns_with_missing']}\n")
        f.write(f"  Wiersze z brakującymi danymi: {stats['rows_with_missing']:,}\n\n")
        
        f.write("BRAKUJĄCE DANE WG KATEGORII:\n")
        f.write(f"  OHLC: {stats['ohlc_missing']:,} komórek\n")
        f.write(f"  Orderbook: {stats['orderbook_missing']:,} komórek\n")
        f.write(f"  Etykiety: {stats['label_missing']:,} komórek\n")
        f.write(f"  Cechy techniczne: {stats['feature_missing']:,} komórek\n\n")
        
        f.write("PROBLEMY Z ETYKIETAMI:\n")
        if label_issues:
            for i, issue in enumerate(label_issues, 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("  Brak problemów z etykietami\n")
        f.write("\n")
        
        f.write("PROBLEMY Z JAKOŚCIĄ DANYCH:\n")
        if data_issues:
            for i, issue in enumerate(data_issues, 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("  Brak problemów z jakością danych\n")
    
    print(f"✅ Raport zapisany: {output_file}")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Weryfikuje dane po etykietowaniu (przed treningiem)')
    parser.add_argument('--input', default='../labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather', 
                       help='Ścieżka do pliku z danymi po etykietowaniu')
    parser.add_argument('--report', default='labeled_data_verification_report.txt',
                       help='Nazwa pliku raportu')
    
    args = parser.parse_args()
    
    print("🔍 ROZPOCZYNAM WERYFIKACJĘ DANYCH PO ETYKIETOWANIU")
    print("=" * 80)
    
    # Wczytaj dane
    df = load_labeled_data(args.input)
    if df is None:
        return
    
    # Analizuj strukturę
    analyze_labeled_data_structure(df)
    
    # Analizuj brakujące dane
    stats = analyze_missing_data(df)
    
    # Sprawdź jakość etykiet
    label_issues = check_label_quality(df)
    
    # Sprawdź ogólną jakość danych
    data_issues = check_data_quality(df)
    
    # Wygeneruj raport
    generate_report(stats, label_issues, data_issues, args.report)
    
    print(f"\n🎉 WERYFIKACJA ZAKOŃCZONA!")
    print(f"📁 Raport: {args.report}")
    
    # Podsumowanie
    if stats['missing_percentage'] > 10:
        print(f"⚠️  UWAGA: Wysoki procent brakujących danych ({stats['missing_percentage']:.2f}%)")
    elif stats['missing_percentage'] > 5:
        print(f"⚠️  Uwaga: Średni procent brakujących danych ({stats['missing_percentage']:.2f}%)")
    else:
        print(f"✅ Niski procent brakujących danych ({stats['missing_percentage']:.2f}%)")
    
    if label_issues or data_issues:
        total_issues = len(label_issues) + len(data_issues)
        print(f"⚠️  Znaleziono {total_issues} problemów")
    else:
        print("✅ Brak problemów z danymi")

if __name__ == "__main__":
    main() 