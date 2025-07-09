"""
Binance Data Downloader - Pobieranie danych historycznych dla modułu validation_and_labeling

Skrypt pobiera dane historyczne z Binance Vision i zapisuje w formacie kompatybilnym 
z modułem validation_and_labeling.

Użycie:
    python binance_data_downloader.py BTCUSDT 1m
    python binance_data_downloader.py ETHUSDT 5m --months 6
    python binance_data_downloader.py ADAUSDT 1h --start-date 2023-01-01
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

# Obsługa importów
try:
    from . import config
    from .utils import setup_logging
except ImportError:
    # Standalone script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from utils import setup_logging

class BinanceDataDownloader:
    """Klasa do pobierania danych historycznych z Binance Vision"""
    
    # Mapowanie timeframes na okresy wspomagane przez Binance
    SUPPORTED_TIMEFRAMES = {
        '1m', '3m', '5m', '15m', '30m', 
        '1h', '2h', '4h', '6h', '8h', '12h', 
        '1d', '3d', '1w', '1M'
    }
    
    # Bazowy URL dla Binance Vision
    BASE_URL = "https://data.binance.vision/data"
    
    def __init__(self, market_type: str = "futures"):
        """
        Args:
            market_type: 'spot' lub 'futures' (domyślnie futures)
        """
        self.logger = setup_logging("BinanceDataDownloader")
        self.market_type = market_type
        self.session = requests.Session()
        
        # Konfiguruj session dla lepszej wydajności
        self.session.headers.update({
            'User-Agent': 'BinanceDataDownloader/1.0'
        })
        
        self.logger.info(f"Inicjalizacja BinanceDataDownloader (market: {market_type})")
    
    def download_historical_data(
        self, 
        symbol: str, 
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        months_back: int = 12
    ) -> pd.DataFrame:
        """
        Pobiera kompletne dane historyczne dla symbolu
        
        Args:
            symbol: Para walutowa (np. 'BTCUSDT')
            interval: Timeframe (np. '1m', '1h')
            start_date: Data początkowa (YYYY-MM-DD) lub None
            end_date: Data końcowa (YYYY-MM-DD) lub None  
            months_back: Ile miesięcy wstecz jeśli brak start_date
            
        Returns:
            pd.DataFrame: Kompletne dane OHLCV
        """
        self.logger.info(f"Rozpoczynam pobieranie danych {symbol} {interval}")
        
        # Walidacja timeframe
        if interval not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Nieobsługiwany timeframe: {interval}. Obsługiwane: {self.SUPPORTED_TIMEFRAMES}")
        
        # Określ zakres dat
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"Zakres dat: {start_date} do {end_date}")
        
        # Pobierz dane dla każdego dnia
        all_dataframes = []
        dates = self._generate_date_range(start_date, end_date)
        
        self.logger.info(f"Będę pobierać dane dla {len(dates)} dni")
        
        for i, date in enumerate(dates):
            try:
                daily_df = self._download_daily_data(symbol, interval, date)
                if daily_df is not None and len(daily_df) > 0:
                    all_dataframes.append(daily_df)
                
                # Progress reporting
                if (i + 1) % 10 == 0 or i == len(dates) - 1:
                    self.logger.info(f"Postęp: {i + 1}/{len(dates)} dni, zebrano {len(all_dataframes)} plików")
                    
            except Exception as e:
                self.logger.warning(f"Błąd dla daty {date}: {e}")
                continue
                
            # Krótka pauza żeby nie przeciążać serwera
            time.sleep(0.1)
        
        if not all_dataframes:
            raise Exception(f"Nie pobrano żadnych danych dla {symbol} {interval}")
        
        # Połącz wszystkie dane
        self.logger.info("Łączę wszystkie pobrane dane...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Przetwórz do standardowego formatu
        processed_df = self._process_raw_data(combined_df)
        
        self.logger.info(f"Pobrano łącznie {len(processed_df):,} wierszy danych")
        
        return processed_df
    
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
    
    def _download_daily_data(self, symbol: str, interval: str, date: str) -> Optional[pd.DataFrame]:
        """
        Pobiera dane dla jednego dnia z Binance Vision
        
        Args:
            symbol: Para walutowa
            interval: Timeframe  
            date: Data w formacie YYYY-MM-DD
            
        Returns:
            pd.DataFrame lub None jeśli nie ma danych
        """
        # Konstruuj URL dla Binance Vision
        if self.market_type == "futures":
            url = f"{self.BASE_URL}/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
        else:
            url = f"{self.BASE_URL}/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Rozpakuj ZIP i wczytaj CSV
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_filename = f"{symbol}-{interval}-{date}.csv"
                    
                    if csv_filename in zip_file.namelist():
                        csv_data = zip_file.read(csv_filename)
                        
                        # Wczytaj jako DataFrame
                        df = pd.read_csv(
                            io.StringIO(csv_data.decode('utf-8')), 
                            header=None,
                            names=[
                                'open_time', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_asset_volume', 'number_of_trades',
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                            ]
                        )
                        
                        self.logger.debug(f"Pobrano {len(df)} wierszy dla {date}")
                        return df
                        
            elif response.status_code == 404:
                self.logger.debug(f"Brak danych dla {date} (404)")
                return None
            else:
                self.logger.warning(f"HTTP {response.status_code} dla {date}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Błąd pobierania {date}: {e}")
            return None
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Przetwarza surowe dane Binance do formatu wymaganego przez validation_and_labeling
        
        Args:
            df: Surowe dane z Binance
            
        Returns:
            pd.DataFrame: Dane w standardowym formacie
        """
        self.logger.info("Przetwarzam surowe dane do standardowego formatu")
        
        # Konwertuj timestamp na datetime - obsłuż różne formaty
        if df['open_time'].dtype == 'object':
            # Jeśli string, spróbuj konwersji na numeric
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        
        # Konwertuj z milliseconds na datetime
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
        
        # Sprawdź czy mamy prawidłowe daty
        if df['datetime'].isna().all():
            self.logger.warning("Problemy z konwersją timestamp, próbuję alternatywną metodę")
            # Spróbuj bezpośredniej konwersji
            df['datetime'] = pd.to_datetime(df['open_time'], errors='coerce')
        
        # Wybierz tylko potrzebne kolumny i konwertuj typy
        processed_df = pd.DataFrame({
            'datetime': df['datetime'],
            'open': pd.to_numeric(df['open'], errors='coerce'),
            'high': pd.to_numeric(df['high'], errors='coerce'),
            'low': pd.to_numeric(df['low'], errors='coerce'),
            'close': pd.to_numeric(df['close'], errors='coerce'),
            'volume': pd.to_numeric(df['volume'], errors='coerce')
        })
        
        # Usuń duplikaty i posortuj chronologicznie
        processed_df = processed_df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        
        # Usuń wiersze z NaN
        initial_rows = len(processed_df)
        processed_df = processed_df.dropna()
        final_rows = len(processed_df)
        
        if initial_rows != final_rows:
            self.logger.warning(f"Usunięto {initial_rows - final_rows} wierszy z NaN")
        
        # Ustaw datetime jako indeks
        processed_df.set_index('datetime', inplace=True)
        
        # Podstawowa walidacja OHLC
        self._validate_ohlc_data(processed_df)
        
        return processed_df
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> None:
        """Podstawowa walidacja logiczności danych OHLC"""
        
        # Sprawdź czy high >= max(open, close)
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
        if invalid_high > 0:
            self.logger.warning(f"Znaleziono {invalid_high} nieprawidłowych wartości HIGH")
        
        # Sprawdź czy low <= min(open, close)  
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
        if invalid_low > 0:
            self.logger.warning(f"Znaleziono {invalid_low} nieprawidłowych wartości LOW")
        
        # Sprawdź ujemne ceny/volume
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        negative_volume = (df['volume'] < 0).sum()
        
        if negative_prices > 0:
            self.logger.warning(f"Znaleziono {negative_prices} wierszy z ujemnymi cenami")
        if negative_volume > 0:
            self.logger.warning(f"Znaleziono {negative_volume} wierszy z ujemnym volume")
    
    def save_to_file(self, df: pd.DataFrame, symbol: str, interval: str, 
                     output_dir: Optional[Path] = None) -> Path:
        """
        Zapisuje dane do pliku w formacie kompatybilnym z validation_and_labeling
        
        Args:
            df: DataFrame z danymi
            symbol: Para walutowa
            interval: Timeframe
            output_dir: Katalog docelowy (domyślnie config.INPUT_DATA_PATH)
            
        Returns:
            Path: Ścieżka do zapisanego pliku
        """
        if output_dir is None:
            output_dir = config.INPUT_DATA_PATH
        
        # Upewnij się że katalog istnieje
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Wygeneruj nazwę pliku kompatybilną z modułem
        filename = f"{symbol}_{interval}_raw.feather"
        output_path = output_dir / filename
        
        self.logger.info(f"Zapisuję dane do {output_path}")
        
        try:
            # Resetuj indeks dla zapisu .feather
            df_to_save = df.reset_index()
            
            # Zapisz jako .feather (szybki format)
            df_to_save.to_feather(output_path)
            
            # Zapisz też kopię jako .csv (backup)
            csv_path = output_path.with_suffix('.csv')
            df_to_save.to_csv(csv_path, index=False)
            
            self.logger.info(f"Zapisano {len(df):,} wierszy do {output_path}")
            self.logger.info(f"Backup CSV: {csv_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania pliku: {e}")
            raise

def parse_arguments():
    """Parsuje argumenty z linii komend"""
    parser = argparse.ArgumentParser(
        description="Pobiera dane historyczne z Binance dla modułu validation_and_labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  Podstawowe:
    python binance_data_downloader.py BTCUSDT 1m
    python binance_data_downloader.py ETHUSDT 5m
    python binance_data_downloader.py ADAUSDT 1h

  Z niestandardowym zakresem:
    python binance_data_downloader.py BTCUSDT 1m --months 6
    python binance_data_downloader.py ETHUSDT 1m --start-date 2023-01-01
    python binance_data_downloader.py BTCUSDT 1m --start-date 2023-01-01 --end-date 2023-12-31

  Różne rynki:
    python binance_data_downloader.py BTCUSDT 1m --market spot
    python binance_data_downloader.py ETHUSDT 1m --market futures (domyślne)

Obsługiwane timeframes:
  1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
    )
    
    # Argumenty obowiązkowe
    parser.add_argument('symbol', help='Para walutowa (np. BTCUSDT, ETHUSDT)')
    parser.add_argument('interval', help='Timeframe (np. 1m, 5m, 1h, 1d)')
    
    # Argumenty opcjonalne
    parser.add_argument(
        '--start-date', 
        help='Data początkowa w formacie YYYY-MM-DD'
    )
    parser.add_argument(
        '--end-date', 
        help='Data końcowa w formacie YYYY-MM-DD (domyślnie dziś)'
    )
    parser.add_argument(
        '--months', 
        type=float, 
        default=12.0, 
        help='Ile miesięcy wstecz pobierać (domyślnie 12, ignorowane jeśli podano --start-date)'
    )
    parser.add_argument(
        '--market', 
        choices=['spot', 'futures'], 
        default='futures',
        help='Typ rynku Binance (domyślnie futures)'
    )
    parser.add_argument(
        '--output-dir', 
        type=Path,
        help='Katalog docelowy (domyślnie validation_and_labeling/input/)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Szczegółowe logowanie (DEBUG level)'
    )
    
    return parser.parse_args()

def main():
    """Główna funkcja skryptu"""
    args = parse_arguments()
    
    # Ustaw poziom logowania
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Walidacja argumentów
    symbol = args.symbol.upper()
    interval = args.interval.lower()
    
    try:
        # Inicjalizuj downloader
        downloader = BinanceDataDownloader(market_type=args.market)
        
        # Pobierz dane
        print(f"🚀 Rozpoczynam pobieranie danych {symbol} {interval} z Binance {args.market.title()}")
        
        df = downloader.download_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=args.start_date,
            end_date=args.end_date,
            months_back=args.months
        )
        
        # Zapisz do pliku
        output_path = downloader.save_to_file(
            df=df,
            symbol=symbol,
            interval=interval,
            output_dir=args.output_dir
        )
        
        # Podsumowanie
        print(f"✅ Sukces! Pobrano {len(df):,} wierszy danych")
        print(f"📁 Plik zapisany: {output_path}")
        print(f"📊 Zakres dat: {df.index.min()} do {df.index.max()}")
        print(f"💾 Rozmiar pliku: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Informacja o dalszych krokach
        print("\n🎯 Następne kroki:")
        print(f"   cd validation_and_labeling")
        print(f"   python main.py  # Uruchom przetwarzanie")
        
    except KeyboardInterrupt:
        print("\n❌ Pobieranie przerwane przez użytkownika")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 