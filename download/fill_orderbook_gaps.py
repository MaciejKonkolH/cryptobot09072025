import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os

def load_orderbook_data(feather_file="orderbook_merged.feather"):
    """Wczytuje dane order book"""
    print(f"📊 Wczytuję dane z {feather_file}...")
    
    df = pd.read_feather(feather_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Wczytano {len(df):,} wierszy")
    print(f"⏰ Zakres: {df['timestamp'].min()} do {df['timestamp'].max()}")
    
    return df

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

def fill_gaps_intelligently(df, max_small_gap_minutes=5, max_medium_gap_minutes=60, 
                          price_change_threshold=2.0):
    """Inteligentnie wypełnia luki w danych order book"""
    print(f"\n🔧 Rozpoczynam inteligentne wypełnianie luk...")
    
    # Znajdź wszystkie luki
    df['time_diff'] = df['timestamp'].diff()
    gaps = df[df['time_diff'] > timedelta(minutes=1)].copy()
    
    print(f"📊 Znaleziono {len(gaps)} luk do wypełnienia")
    
    # Przygotuj nowy DataFrame
    filled_df = df.copy()
    filled_rows = []
    
    total_gaps = len(gaps)
    processed_gaps = 0
    
    for gap_idx, gap_row in gaps.iterrows():
        processed_gaps += 1
        
        if processed_gaps % 50 == 0:
            progress = (processed_gaps / total_gaps) * 100
            print(f"📊 Postęp: {processed_gaps:,}/{total_gaps:,} luk ({progress:.1f}%)")
        
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
        
        print(f"\n✅ Wypełniono {len(filled_rows):,} brakujących minut")
        print(f"📊 Nowy rozmiar: {len(filled_df_new):,} wierszy")
        
        return filled_df_new
    else:
        print(f"\n✅ Brak luk do wypełnienia")
        return filled_df

def main():
    parser = argparse.ArgumentParser(description='Inteligentnie wypełnia luki w danych order book')
    parser.add_argument('--input', default='orderbook_merged.feather', help='Plik wejściowy')
    parser.add_argument('--output', default='orderbook_filled.feather', help='Plik wyjściowy')
    parser.add_argument('--max-small-gap', type=int, default=5, help='Maksymalna luka dla interpolacji (minuty)')
    parser.add_argument('--max-medium-gap', type=int, default=60, help='Maksymalna luka dla rolling average (minuty)')
    parser.add_argument('--price-threshold', type=float, default=2.0, help='Próg zmiany ceny (%)')
    
    args = parser.parse_args()
    
    print(f"🚀 Rozpoczynam inteligentne wypełnianie luk")
    print(f"📁 Wejście: {args.input}")
    print(f"📁 Wyjście: {args.output}")
    print(f"⚙️  Parametry:")
    print(f"   - Małe luki (interpolacja): ≤ {args.max_small_gap} min")
    print(f"   - Średnie luki (rolling): ≤ {args.max_medium_gap} min")
    print(f"   - Próg zmiany ceny: {args.price_threshold}%")
    
    # Wczytaj dane
    df = load_orderbook_data(args.input)
    
    # Wypełnij luki
    filled_df = fill_gaps_intelligently(
        df, 
        max_small_gap_minutes=args.max_small_gap,
        max_medium_gap_minutes=args.max_medium_gap,
        price_change_threshold=args.price_threshold
    )
    
    # Zapisz wynik
    print(f"\n💾 Zapisuję wynik do {args.output}...")
    filled_df.to_feather(args.output)
    
    # Podsumowanie
    print(f"\n🎉 ZADANIE ZAKOŃCZONE!")
    print(f"📊 Oryginalny rozmiar: {len(df):,} wierszy")
    print(f"📊 Nowy rozmiar: {len(filled_df):,} wierszy")
    print(f"📊 Dodano: {len(filled_df) - len(df):,} wierszy")
    
    # Statystyki metod wypełniania
    if 'fill_method' in filled_df.columns:
        method_stats = filled_df['fill_method'].value_counts()
        print(f"\n📈 Statystyki metod wypełniania:")
        for method, count in method_stats.items():
            print(f"   {method}: {count:,} wierszy")

if __name__ == "__main__":
    main() 