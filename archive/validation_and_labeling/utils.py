"""
Funkcje pomocnicze dla modułu validation_and_labeling
"""
import logging
import time
import psutil
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

# Obsługa importów
try:
    from . import config
except ImportError:
    # Standalone script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

def setup_logging(name: str = "validation_and_labeling") -> logging.Logger:
    """
    Konfiguruje system logowania zgodnie z ustawieniami w config
    
    Args:
        name: Nazwa loggera
        
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    # Usuń istniejące handlery jeśli istnieją
    if logger.handlers:
        logger.handlers.clear()
    
    # Ustaw poziom logowania
    level = getattr(logging, config.LOG_LEVEL.upper())
    logger.setLevel(level)
    
    # Stwórz handler dla konsoli
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Stwórz formatter
    formatter = logging.Formatter(
        config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    console_handler.setFormatter(formatter)
    
    # Dodaj handler do loggera
    logger.addHandler(console_handler)
    
    return logger

def find_input_files(input_dir: Path) -> List[Tuple[str, Path]]:
    """
    Znajduje wszystkie pliki wejściowe w katalogu input
    
    Args:
        input_dir: Ścieżka do katalogu input
        
    Returns:
        List[Tuple[str, Path]]: Lista (pair_name, file_path)
    """
    files_found = []
    
    if not input_dir.exists():
        return files_found
    
    # Sprawdź pliki .feather (priorytet)
    for feather_file in input_dir.glob("*.feather"):
        pair_name = extract_pair_from_filename(feather_file.name)
        if pair_name:
            files_found.append((pair_name, feather_file))
    
    # Sprawdź pliki .csv tylko jeśli nie ma odpowiadającego .feather
    found_pairs = {pair for pair, _ in files_found}
    
    for csv_file in input_dir.glob("*.csv"):
        pair_name = extract_pair_from_filename(csv_file.name)
        if pair_name and pair_name not in found_pairs:
            files_found.append((pair_name, csv_file))
    
    return sorted(files_found)

def extract_pair_from_filename(filename: str) -> Optional[str]:
    """
    Wyciąga nazwę pary z nazwy pliku
    
    Args:
        filename: Nazwa pliku (np. "BTCUSDT_1m_raw.feather")
        
    Returns:
        Optional[str]: Nazwa pary lub None jeśli nie znaleziono
    """
    # Usuń rozszerzenie
    name_without_ext = Path(filename).stem
    
    # Najprostsze podejście - nazwa pary to pierwsza część przed underscore
    parts = name_without_ext.split('_')
    if parts:
        return parts[0]
    
    return None

def load_data_file(file_path: Path) -> pd.DataFrame:
    """
    Ładuje plik danych (feather lub csv) i zwraca DataFrame
    
    Args:
        file_path: Ścieżka do pliku
        
    Returns:
        pd.DataFrame: Załadowane dane
        
    Raises:
        FileNotFoundError: Jeśli plik nie istnieje
        Exception: Jeśli błąd podczas ładowania
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.feather':
            df = pd.read_feather(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Nieobsługiwany format pliku: {file_path.suffix}")
            
        return df
        
    except Exception as e:
        raise Exception(f"Błąd podczas ładowania pliku {file_path}: {str(e)}")

def save_data_file(df: pd.DataFrame, file_path: Path, overwrite: bool = True) -> None:
    """
    Zapisuje DataFrame do pliku .feather
    
    Args:
        df: DataFrame do zapisania
        file_path: Ścieżka docelowa
        overwrite: Czy nadpisać istniejący plik
        
    Raises:
        FileExistsError: Jeśli plik istnieje i overwrite=False
        Exception: Jeśli błąd podczas zapisywania
    """
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"Plik już istnieje: {file_path}")
    
    try:
        # Upewnij się że katalog istnieje
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Zapisz jako .feather
        df.to_feather(file_path)
        
    except Exception as e:
        # Jeśli zapis się nie powiódł, usuń częściowy plik
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        raise Exception(f"Błąd podczas zapisywania pliku {file_path}: {str(e)}")

def get_memory_usage_mb() -> float:
    """
    Zwraca aktualne użycie pamięci procesu w MB
    
    Returns:
        float: Użycie pamięci w MB
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Konwersja z bajtów na MB
    except:
        return 0.0

class PerformanceTimer:
    """Klasa do mierzenia czasu wykonania operacji"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Rozpocznij pomiar czasu"""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self):
        """Zakończ pomiar czasu"""
        if self.start_time is None:
            raise RuntimeError("Timer nie został uruchomiony")
        self.end_time = time.time()
    
    def elapsed_seconds(self) -> float:
        """Zwraca czas trwania w sekundach"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def rows_per_second(self, row_count: int) -> float:
        """Oblicza prędkość przetwarzania wierszy na sekundę"""
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return 0.0
        return row_count / elapsed

class ProgressReporter:
    """Klasa do raportowania postępu długich operacji"""
    
    def __init__(self, total_rows: int, logger: logging.Logger, 
                 report_every_n_rows: int = None):
        self.total_rows = total_rows
        self.logger = logger
        self.report_every_n_rows = report_every_n_rows or config.PROGRESS_REPORT_EVERY_N_ROWS
        self.last_reported_row = 0
        self.timer = PerformanceTimer()
        self.timer.start()
    
    def update(self, current_row: int, pair_name: str = ""):
        """
        Aktualizuje postęp i loguje jeśli potrzeba
        
        Args:
            current_row: Numer aktualnego wiersza
            pair_name: Nazwa pary (opcjonalnie)
        """
        if (current_row - self.last_reported_row) >= self.report_every_n_rows:
            percentage = (current_row / self.total_rows) * 100
            elapsed = self.timer.elapsed_seconds()
            speed = self.timer.rows_per_second(current_row)
            
            self.logger.info(
                f"Processing {pair_name} - {current_row:,}/{self.total_rows:,} "
                f"({percentage:.1f}%) - Speed: {speed:.0f} rows/s - "
                f"Elapsed: {elapsed:.1f}s"
            )
            
            self.last_reported_row = current_row
    
    def finish(self, pair_name: str = ""):
        """Loguje zakończenie operacji z finalnymi statystykami"""
        self.timer.stop()
        elapsed = self.timer.elapsed_seconds()
        speed = self.timer.rows_per_second(self.total_rows)
        memory_mb = get_memory_usage_mb()
        
        self.logger.info(
            f"Processing {pair_name} completed: {self.total_rows:,} rows "
            f"in {elapsed:.2f}s - Speed: {speed:.0f} rows/s - "
            f"Memory: {memory_mb:.1f} MB"
        )

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """
    Sprawdza czy DataFrame zawiera wymagane kolumny
    
    Args:
        df: DataFrame do sprawdzenia
        required_columns: Lista wymaganych kolumn
        
    Returns:
        List[str]: Lista brakujących kolumn (pusta jeśli wszystko OK)
    """
    missing_columns = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    return missing_columns

def cleanup_partial_file(file_path: Path, logger: logging.Logger) -> None:
    """
    Usuwa częściowo przetworzony plik (strategia all-or-nothing)
    
    Args:
        file_path: Ścieżka do pliku do usunięcia
        logger: Logger do raportowania
    """
    if file_path.exists():
        try:
            file_path.unlink()
            logger.warning(f"Usunięto częściowy plik po błędzie: {file_path}")
        except Exception as e:
            logger.error(f"Nie udało się usunąć częściowego pliku {file_path}: {e}")

def format_number(number: float, decimals: int = 2) -> str:
    """
    Formatuje liczbę do wyświetlania z separatorami tysięcy
    
    Args:
        number: Liczba do sformatowania
        decimals: Liczba miejsc po przecinku
        
    Returns:
        str: Sformatowana liczba
    """
    return f"{number:,.{decimals}f}" 