"""
Binance Order Book Downloader - Pobieranie danych o głębokości rynku

Skrypt pobiera dane order book z Binance Vision i przetwarza je na cechy
kompatybilne z modelem ML.

Użycie:
    python binance_orderbook_downloader.py BTCUSDT 2025-01-01 2025-01-15
    python binance_orderbook_downloader.py ETHUSDT 2024-12-01 2024-12-31 --output-dir ./orderbook_data
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
from typing import Optional, List, Tuple
import time
import numpy as np

class BinanceOrderBookDownloader:
    """Klasa do pobierania danych order book z Binance Vision"""
    
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
            'User-Agent': 'BinanceOrderBookDownloader/1.0'
        })
        
        print(f"Inicjalizacja OrderBook Downloader (market: {market_type})")
    
    def download_orderbook_data(
        self, 
        symbol: str,
        start_date: str,
        end_date: str,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Pobiera dane order book dla podanego zakresu dat
        
        Args:
            symbol: Para walutowa (np. 'BTCUSDT')
            start_date: Data początkowa (YYYY-MM-DD)
            end_date: Data końcowa (YYYY-MM-DD)
            output_dir: Katalog docelowy
            
        Returns:
            pd.DataFrame: Przetworzone dane order book z cechami
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
        
        # Przetwórz na cechy
        print("Przetwarzam dane na cechy...")
        features_df = self._process_orderbook_features(combined_df)
        
        # Zapisz wyniki
        output_path = output_dir / f"{symbol}_orderbook_features.feather"
        features_df.to_feather(output_path)
        
        # Zapisz też surowe dane
        raw_output_path = output_dir / f"{symbol}_orderbook_raw.feather"
        combined_df.to_feather(raw_output_path)
        
        print(f"Pobrano łącznie {len(features_df):,} wierszy danych order book")
        print(f"Zapisano cechy: {output_path}")
        print(f"Zapisano surowe dane: {raw_output_path}")
        
        return features_df
    
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
    
    def _process_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Przetwarza surowe dane order book na cechy ML
        
        Args:
            df: Surowe dane order book z kolumnami [timestamp, percentage, depth, notional]
            
        Returns:
            pd.DataFrame: Cechy order book dla każdego timestamp
        """
        print("Przetwarzam surowe dane order book na cechy...")
        
        # Konwertuj timestamp na datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Grupuj po timestamp (każdy pomiar order book)
        grouped = df.groupby('timestamp')
        
        features_list = []
        
        for timestamp, group in grouped:
            # Podstawowe cechy presji kupna/sprzedaży
            buy_pressure = group[group['percentage'] > 0]['depth'].sum()
            sell_pressure = group[group['percentage'] < 0]['depth'].sum()
            
            # Stosunek presji
            pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 1.0
            
            # Asymetria order book (-1 do +1)
            total_pressure = buy_pressure + sell_pressure
            asymmetry = (buy_pressure - sell_pressure) / total_pressure if total_pressure > 0 else 0
            
            # Płynność na różnych poziomach
            liquidity_near = group[abs(group['percentage']) <= 1]['depth'].sum()
            liquidity_far = group[abs(group['percentage']) >= 3]['depth'].sum()
            
            # Płynność w USD
            buy_notional = group[group['percentage'] > 0]['notional'].sum()
            sell_notional = group[group['percentage'] < 0]['notional'].sum()
            total_notional = buy_notional + sell_notional
            
            # Stosunek notional
            notional_ratio = buy_notional / sell_notional if sell_notional > 0 else 1.0
            
            # Koncentracja zleceń (odchylenie standardowe głębokości)
            depth_std = group['depth'].std()
            depth_mean = group['depth'].mean()
            depth_cv = depth_std / depth_mean if depth_mean > 0 else 0  # Coefficient of variation
            
            # Cechy dla różnych poziomów cenowych
            features = {
                'timestamp': timestamp,
                'datetime': group['datetime'].iloc[0],
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'pressure_ratio': pressure_ratio,
                'asymmetry': asymmetry,
                'liquidity_near': liquidity_near,
                'liquidity_far': liquidity_far,
                'buy_notional': buy_notional,
                'sell_notional': sell_notional,
                'total_notional': total_notional,
                'notional_ratio': notional_ratio,
                'depth_std': depth_std,
                'depth_cv': depth_cv,
                'total_depth': group['depth'].sum(),
                'num_levels': len(group)
            }
            
            # Dodaj cechy dla konkretnych poziomów cenowych
            for level in [-5, -3, -1, 1, 3, 5]:
                level_data = group[group['percentage'] == level]
                if len(level_data) > 0:
                    features[f'depth_level_{level}'] = level_data['depth'].iloc[0]
                    features[f'notional_level_{level}'] = level_data['notional'].iloc[0]
                else:
                    features[f'depth_level_{level}'] = 0
                    features[f'notional_level_{level}'] = 0
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Sortuj chronologicznie
        features_df = features_df.sort_values('timestamp')
        
        # Dodaj cechy zmiany (różnice między kolejnymi pomiarami)
        features_df['depth_change_rate'] = features_df['total_depth'].diff()
        features_df['notional_change_rate'] = features_df['total_notional'].diff()
        features_df['asymmetry_change'] = features_df['asymmetry'].diff()
        
        # Wypełnij NaN wartościami 0
        features_df = features_df.fillna(0)
        
        print(f"Przetworzono {len(features_df)} wierszy cech order book")
        
        return features_df
    
    def merge_with_ohlc(self, orderbook_df: pd.DataFrame, ohlc_file_path: str) -> pd.DataFrame:
        """
        Łączy dane order book z danymi OHLC
        
        Args:
            orderbook_df: DataFrame z cechami order book
            ohlc_file_path: Ścieżka do pliku OHLC (.feather)
            
        Returns:
            pd.DataFrame: Połączone dane OHLC + order book
        """
        print(f"Łączę dane order book z OHLC z pliku: {ohlc_file_path}")
        
        # Wczytaj dane OHLC
        ohlc_df = pd.read_feather(ohlc_file_path)
        
        # Konwertuj timestamp na datetime jeśli potrzeba
        if 'timestamp' not in ohlc_df.columns and 'datetime' in ohlc_df.columns:
            ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['datetime']).astype(np.int64) // 10**6
        
        # Połącz dane po timestamp (najbliższy match)
        merged_df = pd.merge_asof(
            ohlc_df.sort_values('timestamp'),
            orderbook_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(minutes=1)  # Tolerancja 1 minuty
        )
        
        print(f"Połączono {len(merged_df)} wierszy danych")
        print(f"Pokrycie order book: {merged_df['buy_pressure'].notna().sum()}/{len(merged_df)} wierszy")
        
        return merged_df

def parse_arguments():
    """Parsuje argumenty z linii komend"""
    parser = argparse.ArgumentParser(
        description="Pobiera dane order book z Binance Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  Podstawowe pobieranie:
    python binance_orderbook_downloader.py BTCUSDT 2025-01-01 2025-01-15

  Z niestandardowym katalogiem:
    python binance_orderbook_downloader.py ETHUSDT 2024-12-01 2024-12-31 --output-dir ./data

  Łączenie z OHLC:
    python binance_orderbook_downloader.py BTCUSDT 2025-01-01 2025-01-15 --merge-ohlc ./BTCUSDT_1m_raw.feather
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
        '--merge-ohlc',
        type=str,
        help='Ścieżka do pliku OHLC do połączenia (.feather)'
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
    downloader = BinanceOrderBookDownloader(market_type=args.market)
    
    try:
        # Pobierz dane order book
        orderbook_df = downloader.download_orderbook_data(
            symbol=symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
        
        # Łączenie z OHLC jeśli podano
        if args.merge_ohlc:
            if os.path.exists(args.merge_ohlc):
                print(f"\n🔗 Łączę z danymi OHLC: {args.merge_ohlc}")
                merged_df = downloader.merge_with_ohlc(orderbook_df, args.merge_ohlc)
                
                # Zapisz połączone dane
                merged_output_path = args.output_dir / f"{symbol}_merged_ohlc_orderbook.feather"
                merged_df.to_feather(merged_output_path)
                print(f"✅ Zapisano połączone dane: {merged_output_path}")
            else:
                print(f"⚠️  Plik OHLC nie istnieje: {args.merge_ohlc}")
        
        print("\n" + "="*60)
        print("🎉 Pobieranie zakończone pomyślnie!")
        print(f"   Symbol: {symbol}")
        print(f"   Zakres: {args.start_date} do {args.end_date}")
        print(f"   Wierszy: {len(orderbook_df):,}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Pobieranie przerwane przez użytkownika")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 