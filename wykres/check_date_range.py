import pandas as pd
from pathlib import Path

def check_date_range(candles_file: Path):
    """
    Sprawdza i wypisuje pierwszą i ostatnią datę w pliku danych historycznych.
    """
    print(f"Sprawdzanie zakresu dat w pliku: {candles_file}")
    
    # Wczytaj dane
    df = pd.read_feather(candles_file)
    
    # Przekształć indeks w kolumnę 'date' (bezpieczne podejście)
    if 'date' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        
    # Poprawka: Jawne określenie jednostki czasu na milisekundy
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    
    # Znajdź pierwszą i ostatnią datę
    first_date = df['date'].min()
    last_date = df['date'].max()
    
    print("\\n--- ZAKRES DAT W PLIKU ---")
    print(f"Pierwsza świeca: {first_date}")
    print(f"Ostatnia świeca:  {last_date}")
    print("--------------------------\\n")
    print("Użyj tych dat, aby ustawić odpowiedni zakres w skrypcie visualize_backtest.py")

if __name__ == '__main__':
    CANDLES_FEATHER_FILE = Path('ft_bot_clean/user_data/strategies/inputs/BTC_USDT_USDT/BTCUSDT-1m-futures.feather')
    
    if not CANDLES_FEATHER_FILE.exists():
        print(f"BŁĄD: Plik danych historycznych nie istnieje: {CANDLES_FEATHER_FILE}")
    else:
        check_date_range(CANDLES_FEATHER_FILE) 