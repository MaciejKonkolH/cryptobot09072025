import pandas as pd
import os

def debug_lookup():
    """Debuguje lookup w wide_orderbook_df"""
    
    print("🔍 DEBUGOWANIE LOOKUP")
    
    # Wczytaj order book
    orderbook_file = "orderbook_raw/BTCUSDT-bookDepth-2023-01-01.csv"
    orderbook_df = pd.read_csv(orderbook_file)
    
    # Konwertuj timestamp i ustaw indeks
    orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
    orderbook_df.set_index('timestamp', inplace=True)
    
    print(f"📊 Order book DataFrame:")
    print(f"   Rozmiar: {len(orderbook_df)} wierszy")
    print(f"   Unikalne timestampy: {orderbook_df.index.nunique()}")
    print(f"   Zakres: {orderbook_df.index.min()} - {orderbook_df.index.max()}")
    
    # Test lookup dla pierwszego timestampu
    test_timestamp = orderbook_df.index.min()
    print(f"\n🔍 Test lookup dla: {test_timestamp}")
    
    try:
        # Lookup dla 1 minuty
        result = orderbook_df.loc[test_timestamp:test_timestamp + pd.Timedelta(minutes=1)]
        print(f"   ✅ Lookup działa: {len(result)} wierszy")
        print(f"   Unikalne timestampy w wyniku: {result.index.nunique()}")
        print(f"   Przykładowe dane:")
        print(result.head(10))
        
        # Sprawdź czy są dokładnie 2 snapshoty
        unique_timestamps = result.index.unique()
        print(f"\n🔍 Unikalne timestampy w wyniku:")
        for i, ts in enumerate(unique_timestamps):
            print(f"   {i+1}. {ts}")
        
        if len(unique_timestamps) == 2:
            print(f"   ✅ Znaleziono dokładnie 2 snapshoty!")
        else:
            print(f"   ❌ Znaleziono {len(unique_timestamps)} snapshotów (oczekiwano 2)")
            
    except Exception as e:
        print(f"   ❌ Lookup nie działa: {e}")
    
    # Test lookup dla OHLC timestamp (00:00:00)
    print(f"\n🔍 Test lookup dla OHLC timestamp: 2023-01-01 00:00:00")
    ohlc_timestamp = pd.Timestamp('2023-01-01 00:00:00')
    
    try:
        result = orderbook_df.loc[ohlc_timestamp:ohlc_timestamp + pd.Timedelta(minutes=1)]
        print(f"   ✅ Lookup działa: {len(result)} wierszy")
        print(f"   Unikalne timestampy: {result.index.nunique()}")
    except Exception as e:
        print(f"   ❌ Lookup nie działa: {e}")
    
    # Test lookup dla timestamp po 00:06:05
    print(f"\n🔍 Test lookup dla timestamp po 00:06:05: 2023-01-01 00:07:00")
    later_timestamp = pd.Timestamp('2023-01-01 00:07:00')
    
    try:
        result = orderbook_df.loc[later_timestamp:later_timestamp + pd.Timedelta(minutes=1)]
        print(f"   ✅ Lookup działa: {len(result)} wierszy")
        print(f"   Unikalne timestampy: {result.index.nunique()}")
        if len(result) > 0:
            print(f"   Przykładowe dane:")
            print(result.head(5))
    except Exception as e:
        print(f"   ❌ Lookup nie działa: {e}")

if __name__ == "__main__":
    debug_lookup() 