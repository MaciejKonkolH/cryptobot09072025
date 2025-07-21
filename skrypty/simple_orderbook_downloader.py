"""
Prosty Order Book Downloader - Pobieranie surowych danych o głębokości rynku

Skrypt pobiera surowe dane order book z Binance Vision bez dodatkowego przetwarzania.

Użycie:
    python simple_orderbook_downloader.py BTCUSDT 2025-01-01 2025-01-15
    python simple_orderbook_downloader.py ETHUSDT 2024-12-01 2024-12-31 --output-dir ./data
"""

import argparse
import requests
import pandas as pd
import zipfile
import io
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import time

class SimpleOrderBookDownloader:
    """Prosty downloader danych order book z Binance Vision"""
    
    # Bazowy URL dla Binance Vision
    BASE_URL = "https://data.binance.vision/data"
    
    def __init__(self, market_type: str = "futures"):
        """
        Args:
            market_type: 'spot' lub 'futures' (domyślnie futures)
        """
        self.market_type = market_type
        self.session = requests.Session()
        
        # Konfiguruj session dla lepszej wydajności
        self.session.headers.update({
            'User-Agent': 'SimpleOrderBookDownloader/1.0'
        })
        
        print(f"Inicjalizacja Simple OrderBook Downloader (market: {market_type})")
    
    def download_orderbook_data(
        self, 
        symbol: str,
        start_date: str,
        end_date: str,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Pobiera surowe dane order book dla podanego zakresu dat
        
        Args:
            symbol: Para walutowa (np. 'BTCUSDT')
            start_date: Data początkowa (YYYY-MM-DD)
            end_date: Data końcowa (YYYY-MM-DD)
            output_dir: Katalog docelowy
            
        Returns:
            pd.DataFrame: Surowe dane order book
        """
        print(f"Rozpoczynam pobieranie order book dla {symbol}")
        print(f"Zakres dat: {start_date} do {end_date}")
        
        # Określ katalog docelowy
        if output_dir is None:
            output_dir = Path("./orderbook_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pobierz dane dla każdego dnia
        all_dataframes = []
        dates = self._generate_date_range(start_date, end_date)
        
        print(f"Będę pobierać dane dla {len(dates)} dni")
        
        for i, date in enumerate(dates):
            try:
                daily_df = self._download_daily_orderbook(symbol, date)
                if daily_df is not None and len(daily_df) > 0:
                    all_dataframes.append(daily_df)
                
                # Progress reporting
                if (i + 1) % 5 == 0 or i == len(dates) - 1:
                    print(f"Postęp: {i + 1}/{len(dates)} dni, zebrano {len(all_dataframes)} plików")
                    
            except Exception as e:
                print(f"Błąd dla daty {date}: {e}")
                continue
                
            # Krótka pauza żeby nie przeciążać serwera
            time.sleep(0.1)
        
        if not all_dataframes:
            raise Exception(f"Nie pobrano żadnych danych order book dla {symbol}")
        
        # Połącz wszystkie dane
        print("Łączę wszystkie pobrane dane...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Zapisz surowe dane
        output_path = output_dir / f"{symbol}_orderbook_raw.feather"
        combined_df.to_feather(output_path)
        
        # Zapisz też jako CSV
        csv_path = output_dir / f"{symbol}_orderbook_raw.csv"
        combined_df.to_csv(csv_path, index=False)
        
        print(f"Pobrano łącznie {len(combined_df):,} wierszy danych order book")
        print(f"Zapisano: {output_path}")
        print(f"Backup CSV: {csv_path}")
        
        return combined_df
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Generuje listę dat w formacie YYYY-MM-DD"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
            
        return dates
    
    def _download_daily_orderbook(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """
        Pobiera dane order book dla jednego dnia z Binance Vision
        
        Args:
            symbol: Para walutowa
            date: Data w formacie YYYY-MM-DD
            
        Returns:
            pd.DataFrame lub None jeśli nie ma danych
        """
        # Konstruuj URL dla Binance Vision
        if self.market_type == "futures":
            url = f"{self.BASE_URL}/futures/um/daily/bookDepth/{symbol}/{symbol}-book_depth-{date}.zip"
        else:
            url = f"{self.BASE_URL}/spot/daily/bookDepth/{symbol}/{symbol}-book_depth-{date}.zip"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Rozpakuj ZIP i wczytaj CSV
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_filename = f"{symbol}-book_depth-{date}.csv"
                    
                    if csv_filename in zip_file.namelist():
                        csv_data = zip_file.read(csv_filename)
                        
                        # Wczytaj jako DataFrame
                        df = pd.read_csv(
                            io.StringIO(csv_data.decode('utf-8')), 
                            header=None,
                            names=['timestamp', 'percentage', 'depth', 'notional']
                        )
                        
                        print(f"Pobrano {len(df)} wierszy order book dla {date}")
                        return df
                        
            elif response.status_code == 404:
                print(f"Brak danych order book dla {date} (404)")
                return None
            else:
                print(f"HTTP {response.status_code} dla {date}")
                return None
                
        except Exception as e:
            print(f"Błąd pobierania order book {date}: {e}")
            return None

def parse_arguments():
    """Parsuje argumenty z linii komend"""
    parser = argparse.ArgumentParser(
        description="Pobiera surowe dane order book z Binance Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  Podstawowe pobieranie:
    python simple_orderbook_downloader.py BTCUSDT 2025-01-01 2025-01-15

  Z niestandardowym katalogiem:
    python simple_orderbook_downloader.py ETHUSDT 2024-12-01 2024-12-31 --output-dir ./data

  Różne rynki:
    python simple_orderbook_downloader.py BTCUSDT 2025-01-01 2025-01-15 --market spot
"""
    )
    
    # Argumenty obowiązkowe
    parser.add_argument('symbol', help='Para walutowa (np. BTCUSDT)')
    parser.add_argument('start_date', help='Data początkowa (YYYY-MM-DD)')
    parser.add_argument('end_date', help='Data końcowa (YYYY-MM-DD)')
    
    # Argumenty opcjonalne
    parser.add_argument(
        '--market', 
        choices=['spot', 'futures'], 
        default='futures',
        help='Typ rynku Binance (domyślnie futures)'
    )
    parser.add_argument(
        '--output-dir', 
        type=Path,
        default=Path('./orderbook_data'),
        help='Katalog docelowy (domyślnie ./orderbook_data)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Szczegółowe logowanie'
    )
    
    return parser.parse_args()

def main():
    """Główna funkcja skryptu"""
    args = parse_arguments()
    
    # Walidacja argumentów
    symbol = args.symbol.upper()
    
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Błąd formatu daty: {e}")
        print("Użyj formatu YYYY-MM-DD (np. 2025-01-15)")
        sys.exit(1)
    
    if start_date > end_date:
        print("Data początkowa nie może być późniejsza niż końcowa")
        sys.exit(1)
    
    print(f"🚀 Rozpoczynam pobieranie order book dla {symbol}")
    print(f"📊 Rynek: {args.market}")
    print(f"📅 Zakres: {args.start_date} do {args.end_date}")
    print("="*60)
    
    # Utwórz downloader
    downloader = SimpleOrderBookDownloader(market_type=args.market)
    
    try:
        # Pobierz dane order book
        orderbook_df = downloader.download_orderbook_data(
            symbol=symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*60)
        print("🎉 Pobieranie zakończone pomyślnie!")
        print(f"   Symbol: {symbol}")
        print(f"   Zakres: {args.start_date} do {args.end_date}")
        print(f"   Wierszy: {len(orderbook_df):,}")
        print(f"   Kolumny: {list(orderbook_df.columns)}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Pobieranie przerwane przez użytkownika")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 