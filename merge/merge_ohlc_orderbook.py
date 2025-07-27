import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import sys

# Dodaj ścieżkę do folderu download
sys.path.append('../download')

def load_ohlc_data(ohlc_file="../download/ohlc_merged.feather"):
    """Wczytuje dane OHLC"""
    print(f"📊 Wczytuję dane OHLC z {ohlc_file}...")
    
    if not os.path.exists(ohlc_file):
        print(f"❌ Plik {ohlc_file} nie istnieje!")
        return None
    
    df = pd.read_feather(ohlc_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Wczytano {len(df):,} wierszy OHLC")
    print(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
    
    return df

def load_orderbook_data(orderbook_file="../download/orderbook_filled.feather"):
    """Wczytuje dane orderbook"""
    print(f"📊 Wczytuję dane orderbook z {orderbook_file}...")
    
    if not os.path.exists(orderbook_file):
        print(f"❌ Plik {orderbook_file} nie istnieje!")
        return None
    
    df = pd.read_feather(orderbook_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Wczytano {len(df):,} wierszy orderbook")
    print(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
    
    return df

def align_timestamps(ohlc_df, orderbook_df):
    """Wyrównuje timestampy między danymi OHLC i orderbook"""
    print(f"\n🔧 Wyrównuję timestampy...")
    
    # Znajdź wspólny zakres czasowy
    ohlc_start = ohlc_df['timestamp'].min()
    ohlc_end = ohlc_df['timestamp'].max()
    ob_start = orderbook_df['timestamp'].min()
    ob_end = orderbook_df['timestamp'].max()
    
    print(f"📅 OHLC zakres: {ohlc_start} - {ohlc_end}")
    print(f"📅 Orderbook zakres: {ob_start} - {ob_end}")
    
    # Znajdź wspólny zakres
    common_start = max(ohlc_start, ob_start)
    common_end = min(ohlc_end, ob_end)
    
    print(f"📅 Wspólny zakres: {common_start} - {common_end}")
    
    # Filtruj dane do wspólnego zakresu
    ohlc_filtered = ohlc_df[
        (ohlc_df['timestamp'] >= common_start) & 
        (ohlc_df['timestamp'] <= common_end)
    ].copy()
    
    orderbook_filtered = orderbook_df[
        (orderbook_df['timestamp'] >= common_start) & 
        (orderbook_df['timestamp'] <= common_end)
    ].copy()
    
    print(f"📊 OHLC po filtrowaniu: {len(ohlc_filtered):,} wierszy")
    print(f"📊 Orderbook po filtrowaniu: {len(orderbook_filtered):,} wierszy")
    
    return ohlc_filtered, orderbook_filtered

def merge_data(ohlc_df, orderbook_df):
    """Łączy dane OHLC z orderbook"""
    print(f"\n🔗 Łączę dane OHLC z orderbook...")
    
    # Usuń kolumny które mogą powodować konflikty
    columns_to_drop = ['time_diff']
    for col in columns_to_drop:
        if col in ohlc_df.columns:
            ohlc_df = ohlc_df.drop(columns=[col])
        if col in orderbook_df.columns:
            orderbook_df = orderbook_df.drop(columns=[col])
    
    # Ustaw timestamp jako indeks dla szybszego merge
    ohlc_df = ohlc_df.set_index('timestamp')
    orderbook_df = orderbook_df.set_index('timestamp')
    
    # Wykonaj left join (zachowaj wszystkie wiersze OHLC)
    merged_df = ohlc_df.join(orderbook_df, how='left')
    
    # Resetuj indeks
    merged_df = merged_df.reset_index()
    
    print(f"✅ Połączono dane: {len(merged_df):,} wierszy")
    
    # Sprawdź pokrycie - znajdź pierwszą kolumnę orderbook
    orderbook_columns = [col for col in merged_df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    if orderbook_columns:
        first_ob_col = orderbook_columns[0]
        missing_orderbook = merged_df[merged_df[first_ob_col].isna()]
        print(f"⚠️  Wiersze bez danych orderbook: {len(missing_orderbook):,} ({len(missing_orderbook)/len(merged_df)*100:.2f}%)")
    else:
        print("⚠️  Nie znaleziono kolumn orderbook!")
    
    return merged_df

def analyze_merged_data(df):
    """Analizuje połączone dane"""
    print(f"\n📊 ANALIZA POŁĄCZONYCH DANYCH:")
    print("-" * 60)
    
    # Podstawowe statystyki
    print(f"📈 Łączna liczba wierszy: {len(df):,}")
    print(f"⏰ Zakres czasowy: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"📋 Liczba kolumn: {len(df.columns)}")
    
    # Sprawdź brakujące dane
    print(f"\n🔍 ANALIZA BRAKUJĄCYCH DANYCH:")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        print("Kolumny z brakującymi danymi:")
        for col, count in missing_data.head(10).items():
            percentage = count / len(df) * 100
            print(f"  {col}: {count:,} ({percentage:.2f}%)")
    else:
        print("✅ Brak brakujących danych!")
    
    # Sprawdź kolumny OHLC
    ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
    print(f"\n📈 KOLUMNY OHLC:")
    for col in ohlc_columns:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
    
    # Sprawdź kolumny orderbook (z prefiksami snapshot1_ i snapshot2_)
    ob_columns = [col for col in df.columns if col.startswith(('snapshot1_', 'snapshot2_'))]
    print(f"\n📊 KOLUMNY ORDERBOOK (pierwsze 10):")
    for col in ob_columns[:10]:
        print(f"  {col}: {df[col].dtype}")
    
    if len(ob_columns) > 10:
        print(f"  ... i {len(ob_columns) - 10} więcej kolumn orderbook")
    
    # Sprawdź kolumny metadanych
    meta_columns = ['fill_method', 'gap_duration_minutes', 'price_change_percent']
    print(f"\n🔧 KOLUMNY METADANYCH:")
    for col in meta_columns:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
    
    # Podsumowanie
    print(f"\n📋 PODSUMOWANIE KOLUMN:")
    print(f"  OHLC: {len(ohlc_columns)} kolumn")
    print(f"  Orderbook: {len(ob_columns)} kolumn")
    print(f"  Metadane: {len(meta_columns)} kolumn")
    print(f"  Inne: {len(df.columns) - len(ohlc_columns) - len(ob_columns) - len(meta_columns)} kolumn")

def save_merged_data(df, output_file="merged_ohlc_orderbook.feather"):
    """Zapisuje połączone dane"""
    print(f"\n💾 Zapisuję połączone dane do {output_file}...")
    
    # Usuń kolumny z oryginalnymi timestampami
    columns_to_drop = ['open_time', 'close_time']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 🧹 USUŃ METADANE PO WYPEŁNIENIU LUK
    metadata_columns = ['fill_method', 'gap_duration_minutes', 'price_change_percent']
    removed_count = 0
    for col in metadata_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_count += 1
            print(f"  🧹 Usunięto metadane: {col}")
    
    if removed_count > 0:
        print(f"  📊 Usunięto {removed_count} kolumn metadanych")
    
    # Zapisz do feather
    df.to_feather(output_file)
    
    # Sprawdź rozmiar pliku
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"✅ Zapisano: {output_file}")
    print(f"📁 Rozmiar pliku: {file_size:.2f} MB")
    print(f"📋 Kolumn w pliku: {len(df.columns)}")
    
    return output_file

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Łączy dane OHLC z orderbook')
    parser.add_argument('--ohlc', default='../download/ohlc_merged.feather', 
                       help='Ścieżka do pliku OHLC')
    parser.add_argument('--orderbook', default='../download/orderbook_filled.feather',
                       help='Ścieżka do pliku orderbook')
    parser.add_argument('--output', default='merged_ohlc_orderbook.feather',
                       help='Nazwa pliku wyjściowego')
    
    args = parser.parse_args()
    
    print("🚀 ROZPOCZYNAM ŁĄCZENIE DANYCH OHLC Z ORDERBOOK")
    print("=" * 60)
    
    # Wczytaj dane
    ohlc_df = load_ohlc_data(args.ohlc)
    if ohlc_df is None:
        return
    
    orderbook_df = load_orderbook_data(args.orderbook)
    if orderbook_df is None:
        return
    
    # Wyrównaj timestampy
    ohlc_aligned, orderbook_aligned = align_timestamps(ohlc_df, orderbook_df)
    
    # Połącz dane
    merged_df = merge_data(ohlc_aligned, orderbook_aligned)
    
    # Przeanalizuj wyniki
    analyze_merged_data(merged_df)
    
    # Zapisz dane
    output_file = save_merged_data(merged_df, args.output)
    
    print(f"\n🎉 ŁĄCZENIE ZAKOŃCZONE POMYŚLNIE!")
    print(f"📁 Plik wynikowy: {output_file}")
    print(f"📊 Wierszy: {len(merged_df):,}")
    print(f"📋 Kolumn: {len(merged_df.columns)}")

if __name__ == "__main__":
    main() 