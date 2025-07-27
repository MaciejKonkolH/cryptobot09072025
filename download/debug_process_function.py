import pandas as pd
import os

def debug_process_function():
    """Debuguje czy funkcja process_snapshots_for_candle jest wywoływana"""
    
    print("🔍 DEBUGOWANIE FUNKCJI PROCESS_SNAPSHOTS_FOR_CANDLE")
    
    # Wczytaj order book
    orderbook_file = "orderbook_raw/BTCUSDT-bookDepth-2023-01-01.csv"
    orderbook_df = pd.read_csv(orderbook_file)
    
    # Konwertuj timestamp i ustaw indeks
    orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
    orderbook_df.set_index('timestamp', inplace=True)
    
    print(f"📊 Order book DataFrame:")
    print(f"   Rozmiar: {len(orderbook_df)} wierszy")
    print(f"   Unikalne timestampy: {orderbook_df.index.nunique()}")
    
    # Test funkcji process_snapshots_for_candle
    from download_and_merge_orderbook import process_snapshots_for_candle
    
    # Test dla timestampu z 3 snapshotami
    test_timestamp = pd.Timestamp('2023-01-01 00:06:00')  # OHLC timestamp
    print(f"\n🔍 Test process_snapshots_for_candle dla: {test_timestamp}")
    
    try:
        result = process_snapshots_for_candle(test_timestamp, orderbook_df)
        print(f"   ✅ Funkcja zwróciła: {type(result)}")
        if result:
            print(f"   Liczba snapshotów: {len(result)}")
            for i, snapshot in enumerate(result):
                print(f"   Snapshot {i+1}: {snapshot}")
        else:
            print(f"   ❌ Funkcja zwróciła None")
    except Exception as e:
        print(f"   ❌ Błąd w funkcji: {e}")
        import traceback
        traceback.print_exc()
    
    # Test dla timestampu z 2 snapshotami
    test_timestamp2 = pd.Timestamp('2023-01-01 00:07:00')
    print(f"\n🔍 Test process_snapshots_for_candle dla: {test_timestamp2}")
    
    try:
        result = process_snapshots_for_candle(test_timestamp2, orderbook_df)
        print(f"   ✅ Funkcja zwróciła: {type(result)}")
        if result:
            print(f"   Liczba snapshotów: {len(result)}")
        else:
            print(f"   ❌ Funkcja zwróciła None")
    except Exception as e:
        print(f"   ❌ Błąd w funkcji: {e}")

if __name__ == "__main__":
    debug_process_function() 