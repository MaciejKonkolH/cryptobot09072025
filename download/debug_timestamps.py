import pandas as pd
import os

def debug_timestamps():
    """Debuguje timestampy w danych OHLC i Order Book"""
    
    print("🔍 DEBUGOWANIE TIMESTAMPÓW")
    
    # Sprawdź pierwszy plik OHLC
    ohlc_files = [f for f in os.listdir("ohlc_raw") if f.startswith("BTCUSDT-1m-2023-01-01")]
    if ohlc_files:
        ohlc_file = f"ohlc_raw/{ohlc_files[0]}"
        print(f"\n📊 Pierwszy plik OHLC: {ohlc_file}")
        
        ohlc_df = pd.read_csv(ohlc_file)
        print(f"   Liczba wierszy: {len(ohlc_df)}")
        print(f"   Pierwsze 5 timestampów:")
        for i, ts in enumerate(ohlc_df['timestamp'].head()):
            print(f"     {i+1}. {ts}")
        print(f"   Ostatnie 5 timestampów:")
        for i, ts in enumerate(ohlc_df['timestamp'].tail()):
            print(f"     {i+1}. {ts}")
    
    # Sprawdź pierwszy plik Order Book
    orderbook_files = [f for f in os.listdir("orderbook_raw") if f.startswith("BTCUSDT-bookDepth-2023-01-01")]
    if orderbook_files:
        orderbook_file = f"orderbook_raw/{orderbook_files[0]}"
        print(f"\n📊 Pierwszy plik Order Book: {orderbook_file}")
        
        orderbook_df = pd.read_csv(orderbook_file)
        print(f"   Liczba wierszy: {len(orderbook_df)}")
        print(f"   Pierwsze 5 timestampów:")
        for i, ts in enumerate(orderbook_df['timestamp'].head()):
            print(f"     {i+1}. {ts}")
        print(f"   Ostatnie 5 timestampów:")
        for i, ts in enumerate(orderbook_df['timestamp'].tail()):
            print(f"     {i+1}. {ts}")
    
    # Sprawdź czy są dane order book przed 00:06:05
    print(f"\n🔍 Sprawdzam dane order book przed 00:06:05...")
    if orderbook_files:
        early_data = orderbook_df[orderbook_df['timestamp'] < '2023-01-01 00:06:05']
        print(f"   Liczba wierszy przed 00:06:05: {len(early_data)}")
        if len(early_data) > 0:
            print(f"   Pierwsze timestampy przed 00:06:05:")
            for ts in early_data['timestamp'].head():
                print(f"     - {ts}")
        else:
            print(f"   ❌ BRAK DANYCH order book przed 00:06:05!")
    
    # Sprawdź czy są dane OHLC po 00:06:05
    print(f"\n🔍 Sprawdzam dane OHLC po 00:06:05...")
    if ohlc_files:
        late_data = ohlc_df[ohlc_df['timestamp'] > '2023-01-01 00:06:05']
        print(f"   Liczba wierszy OHLC po 00:06:05: {len(late_data)}")
        if len(late_data) > 0:
            print(f"   Pierwsze timestampy OHLC po 00:06:05:")
            for ts in late_data['timestamp'].head():
                print(f"     - {ts}")

if __name__ == "__main__":
    debug_timestamps() 