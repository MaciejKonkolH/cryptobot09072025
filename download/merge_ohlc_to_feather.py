import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Konfiguracja
OHLC_DIR = "ohlc_raw"
OUTPUT_FILE = "ohlc_merged.feather"

def load_all_ohlc_files(symbol):
    """Wczytuje wszystkie pliki OHLC i łączy je w jeden DataFrame"""
    print(f"📊 Wczytuję pliki OHLC dla {symbol}...")
    
    all_data = []
    
    # Znajdź wszystkie pliki OHLC
    if not os.path.exists(OHLC_DIR):
        print(f"❌ Katalog {OHLC_DIR} nie istnieje!")
        return None
    
    files = [f for f in os.listdir(OHLC_DIR) if f.endswith('.csv') and f.startswith(f"{symbol}-1m-")]
    files.sort()
    
    print(f"📁 Znaleziono {len(files)} plików OHLC")
    
    # Wczytaj każdy plik
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(OHLC_DIR, filename)
        try:
            df = pd.read_csv(file_path)
            
            # Sprawdź czy plik ma odpowiednie kolumny
            expected_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in expected_columns):
                print(f"  ⚠️  {filename} - nieprawidłowe kolumny, pomijam")
                continue
            
            # Konwertuj timestamp
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Konwertuj kolumny numeryczne
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            all_data.append(df)
            print(f"  📊 {i}/{len(files)}: {filename} - {len(df):,} wierszy")
        except Exception as e:
            print(f"  ❌ Błąd wczytania {filename}: {e}")
    
    if not all_data:
        print("❌ Brak danych do wczytania!")
        return None
    
    # Połącz wszystkie dane
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp')
    
    # Usuń duplikaty timestampów (zachowaj ostatni)
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    print(f"✅ Wczytano {len(combined_df):,} wierszy OHLC")
    print(f"⏰ Zakres czasowy: {combined_df['timestamp'].min()} do {combined_df['timestamp'].max()}")
    
    return combined_df

def check_ohlc_continuity(df):
    """Sprawdza ciągłość czasową danych OHLC"""
    print(f"\n🔍 Sprawdzam ciągłość danych OHLC...")
    
    # Sprawdź czy dane są posortowane
    if not df['timestamp'].is_monotonic_increasing:
        print("⚠️  Dane nie są posortowane chronologicznie, sortuję...")
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Oblicz różnice między kolejnymi timestampami
    df['time_diff'] = df['timestamp'].diff()
    
    # Znajdź luki (różnica większa niż 1 minuta)
    gaps = df[df['time_diff'] > timedelta(minutes=1)].copy()
    
    if len(gaps) == 0:
        print("✅ Brak luk czasowych - dane są ciągłe!")
        return df, None
    
    print(f"❌ Znaleziono {len(gaps)} luk czasowych:")
    print()
    
    # Grupuj luki po rozmiarze
    gap_sizes = gaps['time_diff'].value_counts().sort_index()
    
    print("📊 Statystyki luk:")
    for gap_size, count in gap_sizes.head(10).items():
        print(f"   {gap_size}: {count} luk")
    
    print()
    print("🔍 Szczegóły pierwszych 20 luk:")
    print("-" * 80)
    
    for i, (_, row) in enumerate(gaps.head(20).iterrows()):
        gap_start = row['timestamp'] - row['time_diff']
        gap_end = row['timestamp']
        gap_duration = row['time_diff']
        
        print(f"Luka {i+1:2d}: {gap_start} -> {gap_end} (brak: {gap_duration})")
    
    if len(gaps) > 20:
        print(f"... i {len(gaps) - 20} więcej luk")
    
    # Analiza luk według miesięcy
    print(f"\n📅 ANALIZA LUK WEDŁUG MIESIĘCY:")
    print("-" * 60)
    
    gaps['month'] = gaps['timestamp'].dt.to_period('M')
    monthly_gaps = gaps.groupby('month').agg({
        'time_diff': ['count', 'sum', 'mean', 'max']
    }).round(2)
    
    monthly_gaps.columns = ['liczba_luk', 'suma_czasu', 'srednia_luka', 'max_luka']
    monthly_gaps = monthly_gaps.reset_index()
    monthly_gaps['month_str'] = monthly_gaps['month'].astype(str)
    monthly_gaps = monthly_gaps.sort_values('liczba_luk', ascending=False)
    
    print(f"{'Miesiąc':<12} {'Luki':<6} {'Suma czasu':<15} {'Średnia':<10} {'Max luka':<12}")
    print("-" * 60)
    
    for _, row in monthly_gaps.head(15).iterrows():
        month = row['month_str']
        count = int(row['liczba_luk'])
        total_time = str(row['suma_czasu']).split()[-1]
        avg_gap = str(row['srednia_luka']).split()[-1]
        max_gap = str(row['max_luka']).split()[-1]
        
        print(f"{month:<12} {count:<6} {total_time:<15} {avg_gap:<10} {max_gap:<12}")
    
    return df, gaps

def check_data_quality(df):
    """Sprawdza jakość danych OHLC"""
    print(f"\n🔍 Sprawdzam jakość danych OHLC...")
    
    # Sprawdź brakujące wartości
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"⚠️  Znaleziono brakujące wartości:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"   {col}: {count:,} brakujących wartości")
    else:
        print("✅ Brak brakujących wartości")
    
    # Sprawdź logikę OHLC
    print(f"\n🔍 Sprawdzam logikę OHLC...")
    
    # High powinno być >= Open, Close, Low
    high_errors = df[(df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])]
    if len(high_errors) > 0:
        print(f"⚠️  Znaleziono {len(high_errors)} błędów w kolumnie 'high'")
    else:
        print("✅ Kolumna 'high' jest poprawna")
    
    # Low powinno być <= Open, Close, High
    low_errors = df[(df['low'] > df['open']) | (df['low'] > df['close']) | (df['low'] > df['high'])]
    if len(low_errors) > 0:
        print(f"⚠️  Znaleziono {len(low_errors)} błędów w kolumnie 'low'")
    else:
        print("✅ Kolumna 'low' jest poprawna")
    
    # Volume powinno być >= 0
    negative_volume = df[df['volume'] < 0]
    if len(negative_volume) > 0:
        print(f"⚠️  Znaleziono {len(negative_volume)} ujemnych wartości volume")
    else:
        print("✅ Kolumna 'volume' jest poprawna")

def merge_ohlc_to_feather(symbol):
    """Główna funkcja łączenia OHLC w format feather"""
    print(f"🚀 Rozpoczynam łączenie OHLC dla {symbol}")
    
    # KROK 1: Wczytaj wszystkie dane
    ohlc_df = load_all_ohlc_files(symbol)
    if ohlc_df is None:
        return None
    
    # KROK 2: Sprawdź ciągłość
    ohlc_df, gaps = check_ohlc_continuity(ohlc_df)
    
    # KROK 3: Sprawdź jakość danych
    check_data_quality(ohlc_df)
    
    # KROK 4: Zapisz do feather
    print(f"\n💾 Zapisuję do {OUTPUT_FILE}...")
    ohlc_df.to_feather(OUTPUT_FILE)
    
    # KROK 5: Podsumowanie
    print(f"\n🎉 Proces zakończony!")
    print(f"📊 Wierszy w wyniku: {len(ohlc_df):,}")
    print(f"📅 Zakres czasowy: {ohlc_df['timestamp'].min()} - {ohlc_df['timestamp'].max()}")
    print(f"📋 Kolumny: {len(ohlc_df.columns)}")
    print(f"📁 Plik: {OUTPUT_FILE}")
    
    # KROK 6: Statystyki luk
    if gaps is not None:
        total_expected_minutes = int((ohlc_df['timestamp'].max() - ohlc_df['timestamp'].min()).total_seconds() / 60) + 1
        missing_minutes = total_expected_minutes - len(ohlc_df)
        
        print(f"\n📊 PODSUMOWANIE CIĄGŁOŚCI:")
        print(f"   Całkowity czas: {ohlc_df['timestamp'].max() - ohlc_df['timestamp'].min()}")
        print(f"   Oczekiwane minuty: {total_expected_minutes:,}")
        print(f"   Rzeczywiste minuty: {len(ohlc_df):,}")
        print(f"   Brakujące minuty: {missing_minutes:,}")
        print(f"   Luki czasowe: {len(gaps):,}")
        print(f"   Pokrycie: {len(ohlc_df)/total_expected_minutes*100:.2f}%")
    
    return ohlc_df

def main():
    parser = argparse.ArgumentParser(description='Łączy pliki OHLC w jeden plik feather i sprawdza ciągłość')
    parser.add_argument('symbol', help='Symbol kryptowaluty (np. BTCUSDT)')
    
    args = parser.parse_args()
    
    result = merge_ohlc_to_feather(args.symbol)

if __name__ == "__main__":
    main() 