import os
import pandas as pd
from datetime import datetime, timedelta

def check_missing_orderbook_files():
    """Sprawdza luki w danych order book"""
    
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
    missing_ohlc = orderbook_dates - ohlc_dates
    
    print(f"=== ANALIZA LUK W DANYCH ===")
    print(f"Liczba plików Order Book: {len(orderbook_files)}")
    print(f"Liczba plików OHLC: {len(ohlc_files)}")
    print(f"Liczba dat Order Book: {len(orderbook_dates)}")
    print(f"Liczba dat OHLC: {len(ohlc_dates)}")
    
    if missing_orderbook:
        print(f"\n❌ BRAKUJĄCE ORDER BOOK ({len(missing_orderbook)} dat):")
        for date in sorted(missing_orderbook):
            print(f"   {date}")
    
    if missing_ohlc:
        print(f"\n❌ BRAKUJĄCE OHLC ({len(missing_ohlc)} dat):")
        for date in sorted(missing_ohlc):
            print(f"   {date}")
    
    if not missing_orderbook and not missing_ohlc:
        print(f"\n✅ BRAK LUK - wszystkie dane są kompletne!")
    
    # Sprawdź zakresy
    if orderbook_dates:
        orderbook_dates_list = sorted(list(orderbook_dates))
        print(f"\n📅 Zakres Order Book: {orderbook_dates_list[0]} - {orderbook_dates_list[-1]}")
    
    if ohlc_dates:
        ohlc_dates_list = sorted(list(ohlc_dates))
        print(f"📅 Zakres OHLC: {ohlc_dates_list[0]} - {ohlc_dates_list[-1]}")

if __name__ == "__main__":
    check_missing_orderbook_files() 