#!/usr/bin/env python3
"""
Szybki downloader danych Orderbook z Binance Futures
Używa Binance Vision API dla pobierania historycznych danych orderbook
"""

import os
import sys
import json
import time
import logging
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Import konfiguracji
from config import PAIRS, DOWNLOAD_CONFIG, LOGGING_CONFIG, FILE_CONFIG

class FastOrderbookDownloader:
    """Szybki downloader danych Orderbook używający Binance Vision API"""
    
    def __init__(self):
        self.base_url = "https://data.binance.vision/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FastOrderbookDownloader/1.0'
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
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicjalizacja FastOrderbookDownloader")
        
    def setup_directories(self):
        """Tworzy wymagane katalogi"""
        Path(FILE_CONFIG['output_dir']).mkdir(exist_ok=True)
        self.logger.info(f"Katalog wyjściowy: {FILE_CONFIG['output_dir']}")
        
    def check_file_exists_on_server(self, url: str) -> bool:
        """Sprawdza czy plik istnieje na serwerze"""
        try:
            response = self.session.head(url, timeout=DOWNLOAD_CONFIG['timeout'])
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Błąd sprawdzania {url}: {e}")
            return False
    
    def download_and_extract_file(self, url: str, local_zip_path: str, local_csv_path: str) -> bool:
        """Pobiera plik ZIP i rozpakowuje go do CSV"""
        try:
            # Pobierz ZIP
            resp = self.session.get(url, stream=True, timeout=DOWNLOAD_CONFIG['timeout'])
            if resp.status_code == 200:
                # Zapisz ZIP
                with open(local_zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Rozpakuj ZIP
                with zipfile.ZipFile(local_zip_path) as zip_file:
                    csv_filename = zip_file.namelist()[0]
                    zip_file.extract(csv_filename, os.path.dirname(local_csv_path))
                    # Zmień nazwę na standardową
                    old_path = os.path.join(os.path.dirname(local_csv_path), csv_filename)
                    os.rename(old_path, local_csv_path)
                
                # Usuń ZIP
                os.remove(local_zip_path)
                
                file_size = os.path.getsize(local_csv_path)
                # Loguj tylko do pliku, nie do konsoli podczas progress bar
                self.logger.debug(f"[OK] Pobrano i rozpakowano -> {local_csv_path} ({file_size:,} bajtów)")
                return True
            else:
                self.logger.debug(f"[ERROR] Błąd pobierania ({resp.status_code}): {url}")
                return False
        except Exception as e:
            self.logger.debug(f"[ERROR] Błąd pobierania {url}: {e}")
            return False
    
    def get_available_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Sprawdza dostępny zakres dat dla pary orderbook"""
        self.logger.info(f"Sprawdzam dostępny zakres dat orderbook dla {symbol}")
        
        # Testuj różne daty wstecz - sprawdź wszystkie aby znaleźć najstarszą dostępną
        test_dates = [
            datetime.now() - timedelta(days=30),   # 1 miesiąc
            datetime.now() - timedelta(days=90),   # 3 miesiące
            datetime.now() - timedelta(days=180),  # 6 miesięcy
            datetime.now() - timedelta(days=365),  # 1 rok
            datetime.now() - timedelta(days=730),  # 2 lata
            datetime.now() - timedelta(days=1095), # 3 lata
            datetime.now() - timedelta(days=1460), # 4 lata
            datetime.now() - timedelta(days=1825), # 5 lat
        ]
        
        oldest_date = None
        latest_date = datetime.now()
        
        # Sprawdź wszystkie daty aby znaleźć najstarszą dostępną
        for test_date in test_dates:
            date_str = test_date.strftime('%Y-%m-%d')
            url = f"{self.base_url}/{DOWNLOAD_CONFIG['market']}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
            
            if self.check_file_exists_on_server(url):
                oldest_date = test_date
                self.logger.info(f"Znaleziono dane orderbook dla {symbol} od {date_str}")
                # NIE PRZERYWAJ - sprawdź wszystkie daty aby znaleźć najstarszą
        
        if oldest_date is None:
            # Jeśli nie znaleziono starszych danych, spróbuj ostatnie 7 dni
            for i in range(7):
                test_date = datetime.now() - timedelta(days=i)
                date_str = test_date.strftime('%Y-%m-%d')
                url = f"{self.base_url}/{DOWNLOAD_CONFIG['market']}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
                
                if self.check_file_exists_on_server(url):
                    oldest_date = test_date
                    latest_date = test_date
                    self.logger.info(f"Znaleziono dane orderbook dla {symbol} od {date_str}")
                    break
        
        if oldest_date is None:
            raise Exception(f"Nie znaleziono żadnych danych orderbook dla {symbol}")
        
        self.logger.info(f"{symbol} orderbook: {oldest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}")
        return oldest_date, latest_date
    
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
        output_dir = Path(FILE_CONFIG['output_dir'])
        existing_files = list(output_dir.glob(f"{symbol}-bookDepth-*.csv"))
        
        if not existing_files:
            return None
        
        try:
            dates = []
            for file_path in existing_files:
                # Wyciągnij datę z nazwy pliku: BTCUSDT-bookDepth-2023-01-01.csv
                filename = file_path.stem
                date_str = filename.split('-bookDepth-')[-1]
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            
            if dates:
                oldest_date = min(dates)
                latest_date = max(dates)
                
                self.logger.info(f"Istniejące dane orderbook {symbol}: {oldest_date.strftime('%Y-%m-%d')} - {latest_date.strftime('%Y-%m-%d')}")
                return oldest_date, latest_date
            
        except Exception as e:
            self.logger.warning(f"Błąd odczytu istniejących danych dla {symbol}: {e}")
        
        return None
    
    def download_orderbook_data(self, symbol: str, date_str: str) -> bool:
        """Pobiera dane orderbook dla określonej daty"""
        url = f"{self.base_url}/{DOWNLOAD_CONFIG['market']}/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
        zip_path = os.path.join(FILE_CONFIG['output_dir'], f"{symbol}-bookDepth-{date_str}.zip")
        csv_path = os.path.join(FILE_CONFIG['output_dir'], f"{symbol}-bookDepth-{date_str}.csv")
        
        # Sprawdź czy CSV już istnieje i jest kompletny
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            if file_size > 1000:  # Sprawdź czy plik ma sensowny rozmiar
                self.logger.debug(f"[OK] {date_str}: Orderbook CSV już istnieje ({file_size:,} bajtów)")
                return True
            else:
                self.logger.warning(f"[WARN] {date_str}: Orderbook CSV istnieje ale jest za mały ({file_size} bajtów) - usuwam")
                os.remove(csv_path)
        
        # Sprawdź czy plik istnieje na serwerze
        if not self.check_file_exists_on_server(url):
            self.logger.debug(f"[ERROR] {date_str}: Orderbook niedostępny na serwerze")
            return False
        
        # Pobierz plik
        success = self.download_and_extract_file(url, zip_path, csv_path)
        
        # Rate limiting
        time.sleep(DOWNLOAD_CONFIG['chunk_delay'])
        
        return success
    
    def download_date_range(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[List[str], List[str]]:
        """Pobiera dane orderbook dla zakresu dat"""
        self.logger.info(f"Pobieram orderbook {symbol} od {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")
        
        successful_dates = []
        failed_dates = []
        
        # Generuj listę dat
        current_date = start_date
        all_dates = []
        while current_date <= end_date:
            all_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # Wyłącz logowanie podczas progress bar
        original_handlers = self.logger.handlers[:]
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                self.logger.removeHandler(handler)
        
        with tqdm(total=len(all_dates), desc=f"Pobieranie orderbook {symbol}", unit="dzień") as pbar:
            for date_str in all_dates:
                if self.download_orderbook_data(symbol, date_str):
                    successful_dates.append(date_str)
                else:
                    failed_dates.append(date_str)
                
                pbar.update(1)
        
        # Przywróć logowanie
        for handler in original_handlers:
            self.logger.addHandler(handler)
        
        self.logger.info(f"Pobrano orderbook {symbol}: {len(successful_dates)} udanych, {len(failed_dates)} nieudanych")
        return successful_dates, failed_dates
    
    def download_pair(self, symbol: str) -> bool:
        """Pobiera dane orderbook dla jednej pary"""
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Rozpoczynam pobieranie orderbook dla {symbol}")
            self.logger.info(f"{'='*60}")
            
            # Sprawdź dostępny zakres dat
            available_start, available_end = self.get_available_date_range(symbol)
            
            # Sprawdź istniejące dane
            existing_range = self.get_existing_data_range(symbol)
            
            if existing_range:
                existing_start, existing_end = existing_range
                
                # Sprawdź czy potrzebujemy pobrać nowsze dane
                if existing_end < available_end:
                    self.logger.info(f"Pobieram nowsze dane orderbook od {existing_end.strftime('%Y-%m-%d')} do {available_end.strftime('%Y-%m-%d')}")
                    successful, failed = self.download_date_range(symbol, existing_end + timedelta(days=1), available_end)
                else:
                    self.logger.info(f"{symbol}: Wszystkie dane orderbook są aktualne")
                    return True
            else:
                # Pobierz wszystkie dostępne dane
                self.logger.info(f"Pobieram wszystkie dostępne dane orderbook dla {symbol}")
                successful, failed = self.download_date_range(symbol, available_start, available_end)
            
            if failed:
                self.logger.warning(f"Nie udało się pobrać {len(failed)} dni orderbook dla {symbol}: {failed}")
            
            return len(successful) > 0
            
        except Exception as e:
            self.logger.error(f"[ERROR] Błąd pobierania orderbook {symbol}: {e}")
            return False
    
    def run(self):
        """Główna funkcja pobierania"""
        self.logger.info("Rozpoczynam szybkie pobieranie danych Orderbook")
        self.logger.info(f"Pary: {', '.join(PAIRS)}")
        self.logger.info(f"Rynek: {DOWNLOAD_CONFIG['market']}")
        
        start_time = time.time()
        success_count = 0
        
        for symbol in PAIRS:
            if self.download_pair(symbol):
                success_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Pobieranie orderbook zakończone!")
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
        downloader = FastOrderbookDownloader()
        downloader.run()
    except KeyboardInterrupt:
        print("\nPobieranie orderbook przerwane przez użytkownika")
    except Exception as e:
        print(f"[ERROR] Błąd krytyczny: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 