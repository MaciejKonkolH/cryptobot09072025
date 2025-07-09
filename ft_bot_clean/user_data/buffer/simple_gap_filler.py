"""
SimpleGapFiller - Prosty system gap filling dla strategii
Zastępuje kompleksowy ExternalDataCollector prostym API do uzupełniania brakujących świec

Cel: Tylko gap filling (pobieranie brakujących świec), bez heavy lifting
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time
import requests
from pathlib import Path

class SimpleGapFiller:
    """
    Prosty system gap filling - tylko brakujące świece od ostatniej daty
    
    Funkcjonalność:
    - Pobiera tylko brakujące świece od last_timestamp do teraz
    - Używa Binance API z rate limiting
    - Proste error handling
    - Brak kompleksowej logiki pobierania historii
    """
    
    def __init__(self, rate_limit_delay: float = 0.2):
        """
        Args:
            rate_limit_delay: Opóźnienie między wywołaniami API (sekundy)
        """
        self.logger = logging.getLogger(f"{__name__}.SimpleGapFiller")
        self.rate_limit_delay = rate_limit_delay
        self.binance_base_url = "https://api.binance.com/api/v3/klines"
        
        self.logger.info("SimpleGapFiller initialized - gap filling only")
    
    def fill_gap_to_now(self, pair: str, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Pobiera tylko brakujące świece od last_timestamp do teraz
        
        Args:
            pair: Para walutowa (np. BTCUSDT)
            last_timestamp: Ostatnia dostępna data w danych historycznych
            
        Returns:
            pd.DataFrame: Brakujące świece OHLCV lub pusty DataFrame
        """
        try:
            # Konwertuj timestamp do formatu dla Binance API
            start_time = int((last_timestamp + pd.Timedelta(minutes=1)).timestamp() * 1000)
            now = datetime.now()
            end_time = int(now.timestamp() * 1000)
            
            # Sprawdź czy gap jest istotny (więcej niż 2 minuty)
            gap_minutes = (now - last_timestamp.to_pydatetime()).total_seconds() / 60
            
            if gap_minutes <= 2:
                self.logger.debug(f"{pair}: Gap {gap_minutes:.1f}min - pomijam, za mały")
                return pd.DataFrame()
            
            self.logger.info(f"{pair}: Gap filling {gap_minutes:.1f} minut od {last_timestamp}")
            
            # Pobierz dane z Binance API
            gap_data = self._fetch_binance_klines(pair, start_time, end_time)
            
            if gap_data.empty:
                self.logger.warning(f"{pair}: Brak nowych danych z Binance API")
                return pd.DataFrame()
            
            self.logger.info(f"{pair}: Pobrano {len(gap_data)} nowych świec")
            return gap_data
            
        except Exception as e:
            self.logger.error(f"{pair}: Błąd gap filling: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_binance_klines(self, pair: str, start_time: int, end_time: int) -> pd.DataFrame:
        """
        Pobiera świece z Binance API
        
        Args:
            pair: Para walutowa
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)
            
        Returns:
            pd.DataFrame: Dane OHLCV
        """
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Konwertuj parę do formatu Binance API
            # BTC/USDT:USDT → BTCUSDT (usuń / i : oraz wszystko po :)
            if ':' in pair:
                # Format futures BTC/USDT:USDT → BTC/USDT → BTCUSDT
                base_pair = pair.split(':')[0]  # BTC/USDT
                binance_symbol = base_pair.replace('/', '')  # BTCUSDT
            else:
                # Format spot BTC/USDT → BTCUSDT
                binance_symbol = pair.replace('/', '')
            
            # Przygotuj parametry
            params = {
                'symbol': binance_symbol,  # BTCUSDT format
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000  # Maksymalnie 1000 świec na wywołanie
            }
            
            # Wywołanie API
            response = requests.get(self.binance_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Konwertuj do DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Zostaw tylko potrzebne kolumny i konwertuj typy
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Konwertuj timestamp
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            # Przemianuj na 'date' zamiast ustawiać jako index (kompatybilność z dataframe_extender)
            df.rename(columns={'datetime': 'date'}, inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            # Konwertuj do numerycznych
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Usuń ewentualne NaN
            df.dropna(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Błąd HTTP podczas pobierania danych: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Błąd podczas przetwarzania danych Binance: {str(e)}")
            raise
    
    def get_last_available_timestamp(self, data_file_path: Path) -> Optional[pd.Timestamp]:
        """
        Pobiera ostatnią datę z pliku danych historycznych
        
        Args:
            data_file_path: Ścieżka do pliku .feather
            
        Returns:
            pd.Timestamp lub None jeśli błąd
        """
        try:
            if not data_file_path.exists():
                return None
            
            # Załaduj tylko ostatnie kilka wierszy dla wydajności
            df = pd.read_feather(data_file_path)
            
            if df.empty:
                return None
            
            # Ustaw datetime jako index jeśli nie jest
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            return df.index[-1]
            
        except Exception as e:
            self.logger.error(f"Błąd podczas czytania {data_file_path}: {str(e)}")
            return None
    
    def validate_gap_data(self, gap_data: pd.DataFrame, pair: str) -> bool:
        """
        Prosta walidacja pobranych danych
        
        Args:
            gap_data: Pobrane dane
            pair: Para walutowa
            
        Returns:
            bool: True jeśli dane są poprawne
        """
        try:
            if gap_data.empty:
                return True  # Pusty DataFrame jest OK
            
            # Sprawdź kolumny OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in gap_data.columns for col in required_cols):
                self.logger.error(f"{pair}: Brak wymaganych kolumn OHLCV")
                return False
            
            # Sprawdź czy brak NaN
            if gap_data[required_cols].isnull().any().any():
                self.logger.warning(f"{pair}: Znaleziono NaN w gap data")
                return False
            
            # Sprawdź logiczność OHLC
            invalid_ohlc = (
                (gap_data['high'] < gap_data['low']) |
                (gap_data['high'] < gap_data['open']) |
                (gap_data['high'] < gap_data['close']) |
                (gap_data['low'] > gap_data['open']) |
                (gap_data['low'] > gap_data['close'])
            ).any()
            
            if invalid_ohlc:
                self.logger.warning(f"{pair}: Niepoprawne relacje OHLC w gap data")
                return False
            
            # Sprawdź volume > 0
            if (gap_data['volume'] <= 0).any():
                self.logger.warning(f"{pair}: Volume <= 0 w gap data")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"{pair}: Błąd walidacji gap data: {str(e)}")
            return False

class GapFillerConfig:
    """Konfiguracja SimpleGapFiller"""
    
    # Rate limiting
    BINANCE_API_DELAY = 0.2  # Sekundy między wywołaniami
    REQUEST_TIMEOUT = 10     # Timeout dla HTTP requests
    
    # Gap filling
    MIN_GAP_MINUTES = 2      # Minimalny gap do wypełnienia
    MAX_GAP_HOURS = 24       # Maksymalny gap do wypełnienia (safety)
    
    # Retry logic
    MAX_RETRIES = 3          # Maksymalne próby przy błędzie API
    RETRY_DELAY = 1.0        # Opóźnienie między retry (sekundy)
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Zwraca konfigurację jako słownik"""
        return {
            'BINANCE_API_DELAY': cls.BINANCE_API_DELAY,
            'REQUEST_TIMEOUT': cls.REQUEST_TIMEOUT,
            'MIN_GAP_MINUTES': cls.MIN_GAP_MINUTES,
            'MAX_GAP_HOURS': cls.MAX_GAP_HOURS,
            'MAX_RETRIES': cls.MAX_RETRIES,
            'RETRY_DELAY': cls.RETRY_DELAY
        } 