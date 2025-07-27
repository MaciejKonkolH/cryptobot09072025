import pandas as pd
import argparse
from datetime import datetime, timedelta

# Konfiguracja
FEATHER_FILE = "orderbook_merged.feather"

def view_feather_data(symbol, target_date, output_csv=None):
    """Wyświetla dane z pliku feather dla konkretnego dnia"""
    print(f"🔍 Podgląd danych z {FEATHER_FILE}")
    print(f"📅 Dzień: {target_date}")
    
    # KROK 1: Wczytaj plik feather
    try:
        print(f"📊 Wczytuję plik feather...")
        df = pd.read_feather(FEATHER_FILE)
        print(f"✅ Wczytano {len(df):,} wierszy")
        print(f"📅 Zakres czasowy: {df['timestamp'].min()} - {df['timestamp'].max()}")
    except Exception as e:
        print(f"❌ Błąd wczytania pliku feather: {e}")
        return None
    
    # KROK 2: Konwertuj target_date na datetime
    try:
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        target_start = target_date
        target_end = target_date + timedelta(days=1)
    except Exception as e:
        print(f"❌ Błąd konwersji daty: {e}")
        return None
    
    # KROK 3: Filtruj dane dla wybranego dnia
    print(f"🔍 Filtruję dane dla {target_date.strftime('%Y-%m-%d')}...")
    filtered_df = df[
        (df['timestamp'] >= target_start) & 
        (df['timestamp'] < target_end)
    ].copy()
    
    print(f"📊 Znaleziono {len(filtered_df)} wierszy dla wybranego dnia")
    
    if len(filtered_df) == 0:
        print(f"❌ Brak danych dla {target_date.strftime('%Y-%m-%d')}")
        return None
    
    # KROK 4: Wyświetl informacje o strukturze
    print(f"\n📋 Struktura danych:")
    print(f"   Kolumny: {len(filtered_df.columns)}")
    print(f"   Wiersze: {len(filtered_df)}")
    
    # KROK 5: Wyświetl pierwsze kilka wierszy
    print(f"\n📊 Pierwsze 5 wierszy:")
    print(filtered_df.head().to_string())
    
    # KROK 6: Wyświetl statystyki
    print(f"\n📈 Statystyki:")
    print(f"   Zakres czasowy: {filtered_df['timestamp'].min()} - {filtered_df['timestamp'].max()}")
    print(f"   Liczba minut: {len(filtered_df)}")
    
    # KROK 7: Sprawdź czy są snapshoty z interpolacji
    if 'snapshot1_timestamp' in filtered_df.columns and 'snapshot2_timestamp' in filtered_df.columns:
        interpolated_count = len(filtered_df[
            (filtered_df['snapshot1_timestamp'] != filtered_df['timestamp']) |
            (filtered_df['snapshot2_timestamp'] != filtered_df['timestamp'])
        ])
        print(f"   Snapshoty z interpolacji: {interpolated_count}")
    
    # KROK 8: Zapisz do CSV jeśli żądane
    if output_csv:
        print(f"\n💾 Zapisuję do {output_csv}...")
        try:
            filtered_df.to_csv(output_csv, index=False)
            print(f"✅ Dane zapisane do {output_csv}")
        except Exception as e:
            print(f"❌ Błąd zapisu do CSV: {e}")
    
    # KROK 9: Wyświetl listę kolumn
    print(f"\n📋 Lista kolumn:")
    for i, col in enumerate(filtered_df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    return filtered_df

def show_sample_data(df, num_rows=3):
    """Wyświetla przykładowe dane z wybranymi kolumnami"""
    print(f"\n🔍 Przykładowe dane (pierwsze {num_rows} wierszy):")
    
    # Wybierz kluczowe kolumny do wyświetlenia
    key_columns = ['timestamp', 'snapshot1_timestamp', 'snapshot2_timestamp']
    
    # Dodaj kilka przykładowych kolumn depth i notional
    for i in range(-2, 3):  # -2, -1, 1, 2 (bez 0)
        if i != 0:
            key_columns.extend([
                f'snapshot1_depth_{i}',
                f'snapshot1_notional_{i}',
                f'snapshot2_depth_{i}',
                f'snapshot2_notional_{i}'
            ])
    
    # Filtruj tylko kolumny które istnieją
    existing_columns = [col for col in key_columns if col in df.columns]
    
    if existing_columns:
        sample_df = df[existing_columns].head(num_rows)
        print(sample_df.to_string())
    else:
        print("❌ Brak odpowiednich kolumn w danych")

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Podgląd danych z pliku feather')
    parser.add_argument('symbol', help='Symbol kryptowaluty (np. BTCUSDT)')
    parser.add_argument('date', help='Data do wyświetlenia (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', help='Plik CSV do zapisu (opcjonalnie)')
    parser.add_argument('--sample', '-s', type=int, default=3, help='Liczba przykładowych wierszy do wyświetlenia (domyślnie: 3)')
    
    args = parser.parse_args()
    
    # Uruchom podgląd danych
    df = view_feather_data(args.symbol, args.date, args.output)
    
    if df is not None:
        # Wyświetl przykładowe dane
        show_sample_data(df, args.sample)
        
        print(f"\n✅ Podgląd zakończony!")
        if args.output:
            print(f"📁 Dane zapisane w: {args.output}")
    else:
        print(f"\n❌ Nie udało się wyświetlić danych!")

if __name__ == "__main__":
    main() 