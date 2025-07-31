# Konfiguracja dla szybkiego pobierania danych OHLC
# Autor: AI Assistant
# Data: 2025-07-30

# Lista par kryptowalut do pobierania (futures)
PAIRS = [
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
    "NEOUSDT"
]

# Konfiguracja pobierania
DOWNLOAD_CONFIG = {
    "interval": "1m",           # Interwał danych
    "market": "futures",        # Typ rynku
    "chunk_size": 1000,         # Liczba świec na request (maksymalna)
    "max_retries": 3,          # Maksymalna liczba prób dla failed request
    "retry_delay": 1,          # Opóźnienie między próbami (sekundy)
    "timeout": 30,             # Timeout dla requestów (sekundy)
}

# Konfiguracja logowania
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "download.log"
}

# Konfiguracja plików
FILE_CONFIG = {
    "output_dir": "ohlc_raw",
    "metadata_file": "download_metadata.json",
    "progress_file": "download_progress.json"
} 