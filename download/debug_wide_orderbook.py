import pandas as pd
import os

def debug_wide_orderbook():
    """Debuguje wide_orderbook_df"""
    
    print("🔍 DEBUGOWANIE WIDE ORDERBOOK")
    
    # Sprawdź czy istnieje plik merged
    if os.path.exists("orderbook_ohlc_merged.feather"):
        print("✅ Plik merged istnieje - wczytuję...")
        df = pd.read_feather("orderbook_ohlc_merged.feather")
        print(f"   Rozmiar: {len(df)} wierszy")
        print(f"   Kolumny: {list(df.columns)}")
        return
    
    print("❌ Plik merged nie istnieje - sprawdzam surowe dane...")
    
    # Wczytaj order book
    orderbook_files = [f for f in os.listdir("orderbook_raw") if f.startswith("BTCUSDT-bookDepth-2023-01-01")]
    if orderbook_files:
        orderbook_file = f"orderbook_raw/{orderbook_files[0]}"
        print(f"\n📊 Wczytuję: {orderbook_file}")
        
        orderbook_df = pd.read_csv(orderbook_file)
        print(f"   Surowe dane: {len(orderbook_df)} wierszy")
        print(f"   Unikalne timestampy: {orderbook_df['timestamp'].nunique()}")
        
        # Sprawdź pierwsze timestampy
        print(f"   Pierwsze 5 timestampów:")
        for ts in orderbook_df['timestamp'].unique()[:5]:
            print(f"     - {ts}")
        
        # Sprawdź czy timestampy są w formacie datetime
        print(f"\n🔍 Sprawdzam format timestampów...")
        try:
            pd.to_datetime(orderbook_df['timestamp'].iloc[0])
            print("   ✅ Timestampy są w formacie datetime")
        except:
            print("   ❌ Timestampy nie są w formacie datetime")
        
        # Sprawdź czy można utworzyć wide format
        print(f"\n🔍 Sprawdzam czy można utworzyć wide format...")
        try:
            # Konwertuj timestamp
            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
            
            # Ustaw indeks
            orderbook_df.set_index('timestamp', inplace=True)
            
            # Sprawdź indeks
            print(f"   Indeks: {type(orderbook_df.index)}")
            print(f"   Zakres indeksu: {orderbook_df.index.min()} - {orderbook_df.index.max()}")
            print(f"   Liczba unikalnych timestampów: {orderbook_df.index.nunique()}")
            
            # Sprawdź czy można zrobić lookup
            test_timestamp = orderbook_df.index.min()
            print(f"\n🔍 Test lookup dla: {test_timestamp}")
            
            try:
                result = orderbook_df.loc[test_timestamp:test_timestamp + pd.Timedelta(minutes=1)]
                print(f"   ✅ Lookup działa: {len(result)} wierszy")
                print(f"   Przykładowe dane:")
                print(result.head())
            except Exception as e:
                print(f"   ❌ Lookup nie działa: {e}")
                
        except Exception as e:
            print(f"   ❌ Błąd tworzenia wide format: {e}")

if __name__ == "__main__":
    debug_wide_orderbook() 