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

# ==============================================================================
# KONFIGURACJA SKRYPTU
# ==============================================================================
# W poniższej liście wpisz wszystkie pary walutowe, które chcesz pobrać.
# Skrypt przetworzy je wszystkie po kolei.
PAIRS_TO_DOWNLOAD = [
"ETHUSDT",
"BCHUSDT",
"XRPUSDT",
"LTCUSDT",
"TRXUSDT",
"ETCUSDT",
"LINKUSDT",
"XLMUSDT",
"ADAUSDT",
"XMRUSDT",
"DASHUSDT",
"ZECUSDT",
"XTZUSDT",
"ATOMUSDT",
"BNBUSDT",
"ONTUSDT",
"IOTAUSDT",
"BATUSDT",
"VETUSDT",
"NEOUSDT",
"QTUMUSDT",
"IOSTUSDT",
"THETAUSDT",
"ALGOUSDT",
"ZILUSDT",
"KNCUSDT",
"ZRXUSDT",
"COMPUSDT",
"DOGEUSDT",
"SXPUSDT",
"KAVAUSDT",
"BANDUSDT",
"RLCUSDT",
"MKRUSDT",
"SNXUSDT",
"DOTUSDT",
"DEFIUSDT",
"YFIUSDT",
"CRVUSDT",
"TRBUSDT",
"RUNEUSDT",
"SUSHIUSDT",
"EGLDUSDT",
"SOLUSDT",
"ICXUSDT",
"STORJUSDT",
"UNIUSDT",
"AVAXUSDT",
"ENJUSDT",
"FLMUSDT",
"KSMUSDT",
"NEARUSDT",
"AAVEUSDT",
"FILUSDT",
"LRCUSDT",
"RSRUSDT",
"BELUSDT",
"AXSUSDT",
"ALPHAUSDT",
"ZENUSDT",
"SKLUSDT",
"GRTUSDT",
"1INCHUSDT"
]
# ==============================================================================

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
        auto_detect_start: bool = True
    ) -> pd.DataFrame:
        """
        Pobiera kompletne dane historyczne dla symbolu
        
        Args:
            symbol: Para walutowa (np. 'BTCUSDT')
            interval: Timeframe (np. '1m', '1h')
            start_date: Data początkowa (YYYY-MM-DD) lub None
            end_date: Data końcowa (YYYY-MM-DD) lub None
            auto_detect_start: Czy automatycznie wykryć najwcześniejszą datę
            
        Returns:
            pd.DataFrame: Kompletne dane OHLCV
        """
        self.logger.info(f"Rozpoczynam pobieranie danych {symbol} {interval}")
        
        # Walidacja timeframe
        if interval not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Nieobsługiwany timeframe: {interval}. Obsługiwane: {self.SUPPORTED_TIMEFRAMES}")
        
        # Automatyczne wykrycie daty początkowej
        if auto_detect_start and start_date is None:
            start_date = self._find_first_available_date(symbol, interval)
        elif start_date is None:
            start_date = '2018-01-01'
        
        # Określ zakres dat
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
    
    def _find_first_available_date(self, symbol: str, interval: str) -> str:
        """
        Używa API Binance do znalezienia pierwszej dostępnej daty dla pary
        
        Args:
            symbol: Para walutowa
            interval: Timeframe
            
        Returns:
            str: Pierwsza dostępna data w formacie YYYY-MM-DD
        """
        self.logger.info(f"🔍 Sprawdzam pierwszą dostępną datę dla {symbol} przez API")
        
        # Konstruuj URL API
        if self.market_type == "futures":
            api_url = "https://fapi.binance.com/fapi/v1/klines"
        else:
            api_url = "https://api.binance.com/api/v3/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': 0,  # Od samego początku
            'limit': 1       # Potrzebujemy tylko pierwszej świeczki
        }
        
        # Retry mechanism z exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Pauza między próbami
                if attempt > 0:
                    wait_time = 2 ** attempt  # 2, 4, 8 sekund
                    self.logger.info(f"Próba {attempt + 1}/{max_retries} dla {symbol}, czekam {wait_time}s...")
                    time.sleep(wait_time)
                
                # Zwiększony timeout
                response = self.session.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    # Pierwszy element to timestamp otwarcia w milisekundach
                    first_timestamp_ms = data[0][0]
                    # Konwertuj na czytelną datę
                    first_date = datetime.fromtimestamp(first_timestamp_ms / 1000).strftime('%Y-%m-%d')
                    self.logger.info(f"🎯 Pierwsza dostępna data dla {symbol}: {first_date}")
                    return first_date
                else:
                    raise Exception(f"Brak danych historycznych dla {symbol}")
                    
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    self.logger.error(f"Timeout dla {symbol} po {max_retries} próbach")
                    raise Exception(f"Timeout API dla {symbol} - spróbuj ponownie później")
                else:
                    self.logger.warning(f"Timeout dla {symbol}, próba {attempt + 1}/{max_retries}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Błąd API dla {symbol}: {e}")
                    raise Exception(f"Nie można sprawdzić danych dla {symbol}: {e}")
                else:
                    self.logger.warning(f"Błąd API dla {symbol} (próba {attempt + 1}/{max_retries}): {e}")
                    continue
        
        # Fallback (nie powinno się nigdy zdarzyć)
        raise Exception(f"Nieoczekiwany błąd dla {symbol}")
    
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

  Podstawowe (automatycznie wykrywa najwcześniejszą datę dla każdej pary):
    python binance_data_downloader.py

  Z niestandardowym interwałem:
    python binance_data_downloader.py --interval 5m

  Z niestandardowym zakresem końcowym:
    python binance_data_downloader.py --interval 1h --end-date 2023-12-31

  Różne rynki:
    python binance_data_downloader.py --market spot
    
  Wyłączenie auto-detekcji (używa 2018-01-01 jako start):
    python binance_data_downloader.py --no-auto-detect
"""
    )
    
    # Argumenty obowiązkowe
    parser.add_argument(
        '--interval', '-i',
        default='1m',
        help="Timeframe (np. 1m, 5m, 1h), wspólny dla wszystkich par. Domyślnie: '1m'"
    )
    parser.add_argument(
        '--end-date', 
        help='Data końcowa w formacie YYYY-MM-DD (domyślnie dziś)'
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
        '--no-auto-detect',
        action='store_true',
        help='Wyłącz automatyczne wykrywanie daty początkowej (użyje 2018-01-01)'
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
    interval = args.interval.lower()
    
    downloader = BinanceDataDownloader(market_type=args.market)
    
    auto_detect_mode = "🎯 INTELIGENTNY" if not args.no_auto_detect else "📅 STANDARDOWY"
    
    print(f"🚀 Rozpoczynam pobieranie danych dla {len(PAIRS_TO_DOWNLOAD)} par z interwałem {interval}")
    print(f"📊 Tryb: {auto_detect_mode} (auto-detekcja najwcześniejszych dat)")
    print("="*60)

    success_count = 0
    failed_pairs = []

    for i, symbol in enumerate(PAIRS_TO_DOWNLOAD):
        symbol = symbol.upper()
        print(f"\n[{i+1}/{len(PAIRS_TO_DOWNLOAD)}] Przetwarzanie: {symbol}")
        print("-" * 30)

        try:
            # Pobierz dane z automatyczną detekcją daty początkowej
            df = downloader.download_historical_data(
                symbol=symbol,
                interval=interval,
                start_date=None,  # Będzie automatycznie wykryte
                end_date=args.end_date,
                auto_detect_start=not args.no_auto_detect
            )
            
            # Zapisz do pliku
            output_path = downloader.save_to_file(
                df=df,
                symbol=symbol,
                interval=interval,
                output_dir=args.output_dir
            )
            
            # Podsumowanie dla pary
            print(f"✅ Sukces dla {symbol}!")
            print(f"   -> Zakres dat: {df.index.min()} do {df.index.max()}")
            print(f"   -> Plik zapisany: {output_path.name}")
            success_count += 1
            
        except KeyboardInterrupt:
            print("\n❌ Pobieranie przerwane przez użytkownika")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Błąd dla {symbol}: {e}")
            failed_pairs.append(symbol)
            continue
            
    print("\n" + "="*60)
    print("🎉 Zakończono całe zadanie.")
    print(f"   Pobrano pomyślnie: {success_count}/{len(PAIRS_TO_DOWNLOAD)} par.")
    if failed_pairs:
        print(f"   Nie udało się pobrać: {len(failed_pairs)} ({', '.join(failed_pairs)})")
    print("="*60)
    
    if success_count > 0:
        print("\n🎯 Następne kroki:")
        print(f"   cd validation_and_labeling")
        print(f"   python main.py  # Uruchom przetwarzanie")

if __name__ == "__main__":
    main() 