#!/usr/bin/env python3
"""
Szybki downloader danych OHLC z Binance Futures
Używa CCXT library dla maksymalnej prędkości
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import ccxt
from tqdm import tqdm

# Import konfiguracji
from config import PAIRS, DOWNLOAD_CONFIG, LOGGING_CONFIG, FILE_CONFIG

class FastOHLCDownloader:
    """Szybki downloader danych OHLC używający CCXT"""
    
    def __init__(self):
        self.exchange = ccxt.binanceusdm({
            'timeout': DOWNLOAD_CONFIG['timeout'] * 1000,  # CCXT używa milisekund
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Konfiguruje system logowania"""
        log_file = Path(LOGGING_CONFIG['file'])
        
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicjalizacja FastOHLCDownloader")
        
    def setup_directories(self):
        """Tworzy wymagane katalogi"""
        Path(FILE_CONFIG['output_dir']).mkdir(exist_ok=True)
        self.logger.info(f"Katalog wyjściowy: {FILE_CONFIG['output_dir']}")
        
    def get_available_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Sprawdza dostępny zakres dat dla pary"""
        self.logger.info(f"Sprawdzam dostępny zakres dat dla {symbol}")
        
        try:
            # Ustaw stały zakres od 1 stycznia 2022
            start_date = datetime(2022, 1, 1)
            
            # Pobierz najnowsze dane (ostatnie 1000 świec)
            latest_ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                DOWNLOAD_CONFIG['interval'], 
                limit=1000
            )
            
            if not latest_ohlcv:
                raise Exception(f"Brak danych dla {symbol}")
            
            # Najnowszy timestamp
            latest_timestamp = latest_ohlcv[-1][0]
            latest_date = datetime.fromtimestamp(latest_timestamp / 1000)
            
            self.logger.info(f"{symbol}: {start_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}")
            return start_date, latest_date
            
        except Exception as e:
            self.logger.error(f"❌ Błąd sprawdzania zakresu dat dla {symbol}: {e}")
            raise
    
    def load_progress(self) -> Dict:
        """Ładuje postęp pobierania z pliku"""
        progress_file = Path(FILE_CONFIG['progress_file'])
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                self.logger.info("Załadowano postęp pobierania")
                return progress
            except Exception as e:
                self.logger.warning(f"Nie udało się załadować postępu: {e}")
        
        return {}
    
    def save_progress(self, progress: Dict):
        """Zapisuje postęp pobierania do pliku"""
        progress_file = Path(FILE_CONFIG['progress_file'])
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.logger.error(f"Nie udało się zapisać postępu: {e}")
    
    def get_existing_data_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """Sprawdza zakres danych już pobranych lokalnie"""
        output_file = Path(FILE_CONFIG['output_dir']) / f"{symbol}_{DOWNLOAD_CONFIG['interval']}.csv"
        
        if not output_file.exists():
            return None
        
        try:
            df = pd.read_csv(output_file)
            if len(df) == 0:
                return None
            
            # Konwertuj timestamp na datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            oldest_date = df['timestamp'].min()
            latest_date = df['timestamp'].max()
            
            self.logger.info(f"Istniejące dane {symbol}: {oldest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}")
            return oldest_date, latest_date
            
        except Exception as e:
            self.logger.warning(f"Błąd odczytu istniejących danych dla {symbol}: {e}")
            return None
    
    def download_ohlc_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Pobiera dane OHLC dla określonego zakresu dat"""
        self.logger.info(f"Pobieram dane {symbol} od {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")
        
        all_data = []
        current_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # Oblicz całkowitą liczbę requestów dla progress bar
        total_requests = ((end_timestamp - current_timestamp) // (DOWNLOAD_CONFIG['chunk_size'] * 60 * 1000)) + 1
        
        with tqdm(total=total_requests, desc=f"Pobieranie {symbol}", unit="chunk") as pbar:
            while current_timestamp < end_timestamp:
                try:
                    # Pobierz chunk danych
                    ohlcv_chunk = self.exchange.fetch_ohlcv(
                        symbol,
                        DOWNLOAD_CONFIG['interval'],
                        since=current_timestamp,
                        limit=DOWNLOAD_CONFIG['chunk_size']
                    )
                    
                    if not ohlcv_chunk:
                        break
                    
                    all_data.extend(ohlcv_chunk)
                    
                    # Aktualizuj timestamp dla następnego requestu
                    last_timestamp = ohlcv_chunk[-1][0]
                    current_timestamp = last_timestamp + 1
                    
                    pbar.update(1)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Błąd pobierania {symbol} od {datetime.fromtimestamp(current_timestamp/1000)}: {e}")
                    time.sleep(DOWNLOAD_CONFIG['retry_delay'])
                    continue
        
        # Konwertuj na DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Usuń duplikaty i posortuj
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            self.logger.info(f"Pobrano {len(df):,} świec dla {symbol}")
            return df
        else:
            self.logger.warning(f"Brak danych dla {symbol}")
            return pd.DataFrame()
    
    def merge_with_existing_data(self, symbol: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Łączy nowe dane z istniejącymi"""
        output_file = Path(FILE_CONFIG['output_dir']) / f"{symbol}_{DOWNLOAD_CONFIG['interval']}.csv"
        
        if not output_file.exists():
            return new_data
        
        try:
            existing_data = pd.read_csv(output_file)
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'], unit='ms')
            
            # Połącz dane
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Usuń duplikaty i posortuj
            combined_data = combined_data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            self.logger.info(f"Połączono dane: {len(existing_data):,} + {len(new_data):,} = {len(combined_data):,} świec")
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Błąd łączenia danych dla {symbol}: {e}")
            return new_data
    
    def save_data(self, symbol: str, data: pd.DataFrame):
        """Zapisuje dane do pliku CSV"""
        if data.empty:
            self.logger.warning(f"Brak danych do zapisania dla {symbol}")
            return
        
        output_file = Path(FILE_CONFIG['output_dir']) / f"{symbol}_{DOWNLOAD_CONFIG['interval']}.csv"
        
        try:
            # Konwertuj timestamp z powrotem na milisekundy
            data_to_save = data.copy()
            data_to_save['timestamp'] = data_to_save['timestamp'].astype('int64') // 10**6
            
            data_to_save.to_csv(output_file, index=False)
            
            file_size = output_file.stat().st_size
            self.logger.info(f"Zapisano {symbol}: {len(data):,} świec, {file_size:,} bajtów")
            
        except Exception as e:
            self.logger.error(f"❌ Błąd zapisywania danych dla {symbol}: {e}")
    
    def download_pair(self, symbol: str) -> bool:
        """Pobiera dane dla jednej pary"""
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Rozpoczynam pobieranie dla {symbol}")
            self.logger.info(f"{'='*60}")
            
            # Sprawdź dostępny zakres dat
            available_start, available_end = self.get_available_date_range(symbol)
            
            # Sprawdź istniejące dane
            existing_range = self.get_existing_data_range(symbol)
            
            if existing_range:
                existing_start, existing_end = existing_range
                
                # Sprawdź czy potrzebujemy pobrać nowsze dane
                if existing_end < available_end:
                    self.logger.info(f"Pobieram nowsze dane od {existing_end.strftime('%Y-%m-%d')} do {available_end.strftime('%Y-%m-%d')}")
                    new_data = self.download_ohlc_data(symbol, existing_end, available_end)
                    
                    if not new_data.empty:
                        # Połącz z istniejącymi danymi
                        combined_data = self.merge_with_existing_data(symbol, new_data)
                        self.save_data(symbol, combined_data)
                else:
                    self.logger.info(f"{symbol}: Wszystkie dane są aktualne")
                    return True
            else:
                # Pobierz wszystkie dostępne dane
                self.logger.info(f"Pobieram wszystkie dostępne dane dla {symbol}")
                all_data = self.download_ohlc_data(symbol, available_start, available_end)
                self.save_data(symbol, all_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd pobierania {symbol}: {e}")
            return False
    
    def run(self):
        """Główna funkcja pobierania"""
        self.logger.info("Rozpoczynam szybkie pobieranie danych OHLC")
        self.logger.info(f"Pary: {', '.join(PAIRS)}")
        self.logger.info(f"Interwał: {DOWNLOAD_CONFIG['interval']}")
        self.logger.info(f"Rynek: {DOWNLOAD_CONFIG['market']}")
        
        start_time = time.time()
        success_count = 0
        
        for symbol in PAIRS:
            if self.download_pair(symbol):
                success_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Pobieranie zakończone!")
        self.logger.info(f"Udało się: {success_count}/{len(PAIRS)} par")
        self.logger.info(f"Czas: {duration:.1f} sekund")
        self.logger.info(f"{'='*60}")
        
        # Zapisz metadane
        self.save_metadata(success_count, duration)
    
    def save_metadata(self, success_count: int, duration: float):
        """Zapisuje metadane pobierania"""
        metadata = {
            'download_date': datetime.now().isoformat(),
            'pairs': PAIRS,
            'config': DOWNLOAD_CONFIG,
            'results': {
                'success_count': success_count,
                'total_pairs': len(PAIRS),
                'duration_seconds': duration,
                'success_rate': f"{success_count/len(PAIRS)*100:.1f}%"
            }
        }
        
        metadata_file = Path(FILE_CONFIG['metadata_file'])
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Metadane zapisane: {metadata_file}")
        except Exception as e:
            self.logger.error(f"Błąd zapisywania metadanych: {e}")

def main():
    """Główna funkcja"""
    try:
        downloader = FastOHLCDownloader()
        downloader.run()
    except KeyboardInterrupt:
        print("\nPobieranie przerwane przez użytkownika")
    except Exception as e:
        print(f"❌ Błąd krytyczny: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 