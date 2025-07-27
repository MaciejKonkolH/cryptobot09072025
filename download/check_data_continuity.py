import os
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Konfiguracja katalogów
ORDERBOOK_DIR = "orderbook_raw"
OHLC_DIR = "ohlc_raw"

def daterange(start_date, end_date):
    """Generator dat od start_date do end_date (włącznie)"""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def get_existing_files(directory, pattern):
    """Pobiera listę istniejących plików z katalogu"""
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and pattern in filename:
            files.append(filename)
    
    return sorted(files)

def extract_date_from_filename(filename, pattern):
    """Wyciąga datę z nazwy pliku"""
    # Usuń prefix i suffix
    date_str = filename.replace(pattern, '').replace('.csv', '')
    return date_str

def check_continuity(dates, max_gap_days=2):
    """Sprawdza ciągłość dat i zwraca luki"""
    if not dates:
        return [], []
    
    # Konwertuj stringi na datetime
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    date_objects.sort()
    
    gaps = []
    missing_dates = []
    
    # Sprawdź luki między kolejnymi datami
    for i in range(len(date_objects) - 1):
        current_date = date_objects[i]
        next_date = date_objects[i + 1]
        
        # Oblicz różnicę w dniach
        gap_days = (next_date - current_date).days - 1
        
        if gap_days > 0:
            # Znajdź brakujące daty w luce
            missing_in_gap = []
            for day in range(1, gap_days + 1):
                missing_date = current_date + timedelta(days=day)
                missing_in_gap.append(missing_date.strftime('%Y-%m-%d'))
            
            gaps.append({
                'start': current_date.strftime('%Y-%m-%d'),
                'end': next_date.strftime('%Y-%m-%d'),
                'gap_days': gap_days,
                'missing_dates': missing_in_gap
            })
            missing_dates.extend(missing_in_gap)
    
    return gaps, missing_dates

def analyze_data_continuity(start_date, end_date, symbol):
    """Analizuje ciągłość danych OHLC i Orderbook"""
    print(f"🔍 Analizuję ciągłość danych dla {symbol}")
    print(f"📅 Zakres: {start_date} - {end_date}")
    
    # KROK 1: Pobierz wszystkie daty z zakresu
    all_dates = [date.strftime('%Y-%m-%d') for date in daterange(start_date, end_date)]
    print(f"📋 Zakres zawiera {len(all_dates)} dni")
    
    # KROK 2: Sprawdź pliki order book
    print(f"\n📊 Sprawdzam pliki order book...")
    orderbook_pattern = f"{symbol}-bookDepth-"
    orderbook_files = get_existing_files(ORDERBOOK_DIR, orderbook_pattern)
    
    orderbook_dates = []
    for filename in orderbook_files:
        date_str = extract_date_from_filename(filename, orderbook_pattern)
        if date_str in all_dates:
            orderbook_dates.append(date_str)
    
    print(f"   Znaleziono {len(orderbook_dates)} plików order book w zakresie")
    
    # KROK 3: Sprawdź pliki OHLC
    print(f"📊 Sprawdzam pliki OHLC...")
    ohlc_pattern = f"{symbol}-1m-"
    ohlc_files = get_existing_files(OHLC_DIR, ohlc_pattern)
    
    ohlc_dates = []
    for filename in ohlc_files:
        date_str = extract_date_from_filename(filename, ohlc_pattern)
        if date_str in all_dates:
            ohlc_dates.append(date_str)
    
    print(f"   Znaleziono {len(ohlc_dates)} plików OHLC w zakresie")
    
    # KROK 4: Sprawdź ciągłość order book
    print(f"\n🔍 Analiza ciągłości order book:")
    orderbook_gaps, orderbook_missing = check_continuity(orderbook_dates)
    
    if orderbook_gaps:
        print(f"   ❌ Znaleziono {len(orderbook_gaps)} luk w order book:")
        for gap in orderbook_gaps:
            print(f"      {gap['start']} - {gap['end']} (luka {gap['gap_days']} dni)")
            print(f"      Brakujące: {', '.join(gap['missing_dates'])}")
    else:
        print(f"   ✅ Order book ma pełną ciągłość!")
    
    # KROK 5: Sprawdź ciągłość OHLC
    print(f"\n🔍 Analiza ciągłości OHLC:")
    ohlc_gaps, ohlc_missing = check_continuity(ohlc_dates)
    
    if ohlc_gaps:
        print(f"   ❌ Znaleziono {len(ohlc_gaps)} luk w OHLC:")
        for gap in ohlc_gaps:
            print(f"      {gap['start']} - {gap['end']} (luka {gap['gap_days']} dni)")
            print(f"      Brakujące: {', '.join(gap['missing_dates'])}")
    else:
        print(f"   ✅ OHLC ma pełną ciągłość!")
    
    # KROK 6: Podsumowanie
    print(f"\n📈 Podsumowanie:")
    print(f"   Order book: {len(orderbook_dates)}/{len(all_dates)} dni ({len(orderbook_dates)/len(all_dates)*100:.1f}%)")
    print(f"   OHLC: {len(ohlc_dates)}/{len(all_dates)} dni ({len(ohlc_dates)/len(all_dates)*100:.1f}%)")
    
    if orderbook_missing:
        print(f"   Brakujące order book: {len(orderbook_missing)} dni")
    if ohlc_missing:
        print(f"   Brakujące OHLC: {len(ohlc_missing)} dni")
    
    # KROK 7: Sprawdź największe luki
    if orderbook_gaps:
        max_orderbook_gap = max(orderbook_gaps, key=lambda x: x['gap_days'])
        print(f"   Największa luka order book: {max_orderbook_gap['gap_days']} dni ({max_orderbook_gap['start']} - {max_orderbook_gap['end']})")
    
    if ohlc_gaps:
        max_ohlc_gap = max(ohlc_gaps, key=lambda x: x['gap_days'])
        print(f"   Największa luka OHLC: {max_ohlc_gap['gap_days']} dni ({max_ohlc_gap['start']} - {max_ohlc_gap['end']})")
    
    return orderbook_missing, ohlc_missing

def copy_neighboring_orderbook(source_date, target_date, symbol):
    """Kopiuje plik order book z sąsiedniego dnia i dostosowuje timestampy"""
    source_file = os.path.join(ORDERBOOK_DIR, f"{symbol}-bookDepth-{source_date}.csv")
    target_file = os.path.join(ORDERBOOK_DIR, f"{symbol}-bookDepth-{target_date}.csv")
    
    if not os.path.exists(source_file):
        print(f"❌ Plik źródłowy nie istnieje: {source_file}")
        return False
    
    try:
        # Wczytaj dane źródłowe
        df = pd.read_csv(source_file)
        print(f"📊 Wczytano {len(df):,} wierszy z {source_date}")
        
        # Skopiuj dane
        new_df = df.copy()
        
        # Dostosuj timestampy
        source_start = pd.to_datetime(source_date)
        target_start = pd.to_datetime(target_date)
        time_shift = target_start - source_start
        
        # Dostosuj timestampy w kolumnie 'timestamp'
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp']) + time_shift
        new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Zapisz nowy plik
        new_df.to_csv(target_file, index=False)
        
        file_size = os.path.getsize(target_file)
        print(f"✅ Utworzono: {target_file} ({len(new_df):,} wierszy, {file_size:,} bajtów)")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas kopiowania: {e}")
        return False

def find_neighboring_orderbook_date(missing_date, existing_dates):
    """Znajduje sąsiedni dzień dla uzupełnienia brakującego order book"""
    missing_dt = datetime.strptime(missing_date, '%Y-%m-%d')
    
    # Sprawdź dzień przed
    day_before = (missing_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    day_before_exists = day_before in existing_dates
    
    # Sprawdź dzień po
    day_after = (missing_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    day_after_exists = day_after in existing_dates
    
    if day_before_exists and day_after_exists:
        # Preferuj dzień przed
        return day_before
    elif day_before_exists:
        return day_before
    elif day_after_exists:
        return day_after
    else:
        return None

def fill_missing_orderbook_files(missing_dates, existing_dates, symbol):
    """Uzupełnia brakujące pliki order book przez kopiowanie sąsiednich dni"""
    print(f"\n🔧 Uzupełniam {len(missing_dates)} brakujących plików order book...")
    
    filled_dates = []
    failed_dates = []
    
    for missing_date in missing_dates:
        print(f"\n📅 Przetwarzam: {missing_date}")
        
        # Znajdź sąsiedni dzień
        neighbor_date = find_neighboring_orderbook_date(missing_date, existing_dates)
        
        if neighbor_date is None:
            print(f"❌ Brak sąsiednich dni dla {missing_date}")
            failed_dates.append(missing_date)
            continue
        
        print(f"📋 Używam danych z: {neighbor_date}")
        
        # Uzupełnij order book
        if copy_neighboring_orderbook(neighbor_date, missing_date, symbol):
            print(f"✅ Uzupełniono order book dla {missing_date}")
            filled_dates.append(missing_date)
        else:
            print(f"❌ Nie udało się uzupełnić order book dla {missing_date}")
            failed_dates.append(missing_date)
    
    return filled_dates, failed_dates

def main():
    """Główna funkcja sprawdzania ciągłości"""
    parser = argparse.ArgumentParser(description='Sprawdź ciągłość pobranych danych')
    parser.add_argument('symbol', help='Symbol kryptowaluty (np. BTCUSDT)')
    parser.add_argument('start_date', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('end_date', help='Data końcowa (YYYY-MM-DD)')
    parser.add_argument('--fill-missing', action='store_true', help='Uzupełnij brakujące pliki order book')
    
    args = parser.parse_args()
    
    # Konwertuj stringi dat na datetime
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ Błąd formatu daty: {e}")
        print("Użyj formatu: YYYY-MM-DD (np. 2023-01-01)")
        return
    
    symbol = args.symbol
    
    # Sprawdź ciągłość
    orderbook_missing, ohlc_missing = analyze_data_continuity(start_date, end_date, symbol)
    
    # Uzupełnij brakujące pliki order book jeśli żądane
    if args.fill_missing and orderbook_missing:
        print(f"\n🔧 Rozpoczynam uzupełnianie brakujących plików order book...")
        
        # Pobierz listę istniejących dat order book
        orderbook_pattern = f"{symbol}-bookDepth-"
        orderbook_files = get_existing_files(ORDERBOOK_DIR, orderbook_pattern)
        existing_orderbook_dates = []
        for filename in orderbook_files:
            date_str = extract_date_from_filename(filename, orderbook_pattern)
            if date_str in [date.strftime('%Y-%m-%d') for date in daterange(start_date, end_date)]:
                existing_orderbook_dates.append(date_str)
        
        # Uzupełnij brakujące pliki
        filled_dates, failed_dates = fill_missing_orderbook_files(
            orderbook_missing, existing_orderbook_dates, symbol
        )
        
        # Podsumowanie uzupełniania
        print(f"\n📊 Podsumowanie uzupełniania order book:")
        print(f"   Uzupełniono: {len(filled_dates)} plików")
        print(f"   Nie udało się: {len(failed_dates)} plików")
        
        if filled_dates:
            print(f"   Uzupełnione daty: {', '.join(filled_dates)}")
        if failed_dates:
            print(f"   Nieudane daty: {', '.join(failed_dates)}")
        
        # Sprawdź ponownie ciągłość po uzupełnieniu
        if filled_dates:
            print(f"\n🔍 Sprawdzam ciągłość po uzupełnieniu...")
            remaining_missing, _ = analyze_data_continuity(start_date, end_date, symbol)
            
            if not remaining_missing:
                print(f"\n🎉 Wszystkie luki w order book zostały uzupełnione!")
            else:
                print(f"\n⚠️ Pozostały luki w order book: {len(remaining_missing)} dni")
    
    # Wynik końcowy
    if not orderbook_missing and not ohlc_missing:
        print(f"\n🎉 Wszystkie dane mają pełną ciągłość!")
    else:
        print(f"\n⚠️ Znaleziono luki w danych!")
        if orderbook_missing:
            print(f"   Order book brakuje: {len(orderbook_missing)} dni")
        if ohlc_missing:
            print(f"   OHLC brakuje: {len(ohlc_missing)} dni")

if __name__ == "__main__":
    main() 