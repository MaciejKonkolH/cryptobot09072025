import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def check_orderbook_continuity(feather_file="orderbook_merged.feather"):
    """Sprawdza ciągłość czasową w pliku orderbook feather"""
    print(f"🔍 Sprawdzam ciągłość danych w {feather_file}")
    
    # Wczytaj dane
    try:
        df = pd.read_feather(feather_file)
        print(f"✅ Wczytano {len(df):,} wierszy")
    except Exception as e:
        print(f"❌ Błąd wczytania pliku: {e}")
        return
    
    # Sprawdź czy kolumna timestamp istnieje
    if 'timestamp' not in df.columns:
        print("❌ Brak kolumny 'timestamp' w danych!")
        return
    
    # Konwertuj timestamp do datetime jeśli potrzeba
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sortuj po timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"⏰ Zakres czasowy: {df['timestamp'].min()} do {df['timestamp'].max()}")
    print(f"📊 Różnica czasowa: {df['timestamp'].max() - df['timestamp'].min()}")
    
    # Sprawdź czy timestamps są zaokrąglone do minut
    print(f"\n🔍 Sprawdzam czy timestamps są zaokrąglone do minut...")
    non_minute_timestamps = df[df['timestamp'].dt.second != 0]
    if len(non_minute_timestamps) > 0:
        print(f"⚠️  Znaleziono {len(non_minute_timestamps)} timestampów niezaokrąglonych do minut")
        print("   Przykłady:")
        for i, ts in enumerate(non_minute_timestamps['timestamp'].head(5)):
            print(f"   - {ts}")
    else:
        print("✅ Wszystkie timestamps są zaokrąglone do minut")
    
    # Znajdź luki czasowe
    print(f"\n🔍 Szukam luk czasowych...")
    
    # Oblicz różnice między kolejnymi timestampami
    df['time_diff'] = df['timestamp'].diff()
    
    # Znajdź luki większe niż 1 minuta
    gaps = df[df['time_diff'] > timedelta(minutes=1)].copy()
    
    if len(gaps) == 0:
        print("✅ Brak luk czasowych - dane są ciągłe!")
    else:
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
    
    # Sprawdź czy są duplikaty timestampów
    print(f"\n🔍 Sprawdzam duplikaty timestampów...")
    duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
    
    if len(duplicates) > 0:
        print(f"⚠️  Znaleziono {len(duplicates)} duplikatów timestampów")
        print("   Przykłady:")
        for ts in duplicates['timestamp'].unique()[:5]:
            count = len(duplicates[duplicates['timestamp'] == ts])
            print(f"   - {ts}: {count} wystąpień")
    else:
        print("✅ Brak duplikatów timestampów")
    
    # Analiza luk według miesięcy
    if len(gaps) > 0:
        print(f"\n📅 ANALIZA LUK WEDŁUG MIESIĘCY:")
        print("-" * 60)
        
        # Dodaj kolumnę z miesiącem do gaps
        gaps['month'] = gaps['timestamp'].dt.to_period('M')
        
        # Grupuj luki według miesięcy
        monthly_gaps = gaps.groupby('month').agg({
            'time_diff': ['count', 'sum', 'mean', 'max']
        }).round(2)
        
        # Spłaszcz kolumny
        monthly_gaps.columns = ['liczba_luk', 'suma_czasu', 'srednia_luka', 'max_luka']
        monthly_gaps = monthly_gaps.reset_index()
        
        # Konwertuj miesiące na string
        monthly_gaps['month_str'] = monthly_gaps['month'].astype(str)
        
        # Sortuj według liczby luk (malejąco)
        monthly_gaps = monthly_gaps.sort_values('liczba_luk', ascending=False)
        
        print(f"{'Miesiąc':<12} {'Luki':<6} {'Suma czasu':<15} {'Średnia':<10} {'Max luka':<12}")
        print("-" * 60)
        
        for _, row in monthly_gaps.head(20).iterrows():
            month = row['month_str']
            count = int(row['liczba_luk'])
            total_time = str(row['suma_czasu']).split()[-1]  # Pobierz tylko czas
            avg_gap = str(row['srednia_luka']).split()[-1]
            max_gap = str(row['max_luka']).split()[-1]
            
            print(f"{month:<12} {count:<6} {total_time:<15} {avg_gap:<10} {max_gap:<12}")
        
        if len(monthly_gaps) > 20:
            print(f"... i {len(monthly_gaps) - 20} więcej miesięcy")
        
        # Znajdź miesiące z największymi lukami
        print(f"\n🏆 TOP 5 MIESIĘCY Z NAJWIĘKSZYMI LUKAMI:")
        print("-" * 40)
        
        for i, (_, row) in enumerate(monthly_gaps.head(5).iterrows()):
            month = row['month_str']
            count = int(row['liczba_luk'])
            max_gap = str(row['max_luka']).split()[-1]
            total_time = str(row['suma_czasu']).split()[-1]
            
            print(f"{i+1}. {month}: {count} luk, max: {max_gap}, łącznie: {total_time}")
    
    # Podsumowanie
    print(f"\n📊 PODSUMOWANIE:")
    print(f"   Całkowity czas: {df['timestamp'].max() - df['timestamp'].min()}")
    print(f"   Oczekiwane minuty: {int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60) + 1:,}")
    print(f"   Rzeczywiste minuty: {len(df):,}")
    print(f"   Brakujące minuty: {int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60) + 1 - len(df):,}")
    print(f"   Luki czasowe: {len(gaps):,}")
    print(f"   Duplikaty: {len(duplicates):,}")

def main():
    parser = argparse.ArgumentParser(description='Sprawdza ciągłość czasową w pliku orderbook feather')
    parser.add_argument('--file', default='orderbook_merged.feather', help='Ścieżka do pliku feather')
    
    args = parser.parse_args()
    
    check_orderbook_continuity(args.file)

if __name__ == "__main__":
    main() 