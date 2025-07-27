import os
import pandas as pd
from datetime import datetime, timedelta

def check_all_missing_days():
    """Sprawdza wszystkie brakujące dni order book"""
    
    orderbook_dir = "orderbook_raw"
    ohlc_dir = "ohlc_raw"
    
    # Pobierz listę plików
    orderbook_files = [f for f in os.listdir(orderbook_dir) if f.endswith('.csv')]
    ohlc_files = [f for f in os.listdir(ohlc_dir) if f.endswith('.csv')]
    
    # Wyciągnij daty z nazw plików
    orderbook_dates = set()
    for file in orderbook_files:
        if 'BTCUSDT-bookDepth-' in file:
            date_str = file.replace('BTCUSDT-bookDepth-', '').replace('.csv', '')
            orderbook_dates.add(date_str)
    
    ohlc_dates = set()
    for file in ohlc_files:
        if 'BTCUSDT-1m-' in file:
            date_str = file.replace('BTCUSDT-1m-', '').replace('.csv', '')
            ohlc_dates.add(date_str)
    
    # Znajdź luki
    missing_orderbook = ohlc_dates - orderbook_dates
    
    print(f"=== ANALIZA WSZYSTKICH BRAKUJĄCYCH DNI ===")
    print(f"Liczba dat OHLC: {len(ohlc_dates)}")
    print(f"Liczba dat Order Book: {len(orderbook_dates)}")
    print(f"Liczba brakujących dni Order Book: {len(missing_orderbook)}")
    
    if missing_orderbook:
        print(f"\n❌ BRAKUJĄCE DNI ORDER BOOK:")
        missing_list = sorted(list(missing_orderbook))
        for i, date in enumerate(missing_list, 1):
            print(f"   {i:2d}. {date}")
        
        # Sprawdź czy są pojedyncze dni czy większe luki
        print(f"\n📊 ANALIZA LUK:")
        
        # Konwertuj na datetime
        missing_dates = [datetime.strptime(date, '%Y-%m-%d') for date in missing_list]
        missing_dates.sort()
        
        # Znajdź ciągłe luki
        gaps = []
        current_gap_start = missing_dates[0]
        current_gap_end = missing_dates[0]
        
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i-1]).days == 1:
                # Ciągły gap
                current_gap_end = missing_dates[i]
            else:
                # Nowy gap
                gaps.append((current_gap_start, current_gap_end))
                current_gap_start = missing_dates[i]
                current_gap_end = missing_dates[i]
        
        # Dodaj ostatni gap
        gaps.append((current_gap_start, current_gap_end))
        
        print(f"Liczba luk: {len(gaps)}")
        for i, (start, end) in enumerate(gaps, 1):
            days_in_gap = (end - start).days + 1
            print(f"   Luka {i}: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')} ({days_in_gap} dni)")
        
        # Sprawdź czy są tylko pojedyncze dni
        single_days = [gap for gap in gaps if (gap[1] - gap[0]).days == 0]
        print(f"\n📈 PODSUMOWANIE:")
        print(f"   Pojedyncze dni: {len(single_days)}")
        print(f"   Większe luki: {len(gaps) - len(single_days)}")
        
        if len(single_days) == len(missing_orderbook):
            print(f"✅ WSZYSTKIE BRAKUJĄCE DNI TO POJEDYNCZE DNI!")
            print(f"   Można je łatwo zastąpić danymi z sąsiednich dni.")
        else:
            print(f"⚠️ SĄ WIĘKSZE LUKI - wymagają innego podejścia.")
    
    else:
        print(f"\n✅ BRAK LUK - wszystkie dane są kompletne!")

if __name__ == "__main__":
    check_all_missing_days() 