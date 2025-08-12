# Konfiguracja dla szybkiego pobierania danych Orderbook z Binance Futures
# Autor: AI Assistant
# Data: 2025-01-30

# Lista par kryptowalut do pobierania (futures)
PAIRS = [
     "BTCUSDT",
#    "ETHUSDT",
#    "BCHUSDT",
#    "XRPUSDT",
#    "LTCUSDT",
#    "TRXUSDT",
#    "ETCUSDT",
#    "LINKUSDT",
#    "XLMUSDT",
#    "ADAUSDT",
#    "XMRUSDT",
#    "DASHUSDT",
#    "ZECUSDT",
#    "XTZUSDT",
#    "ATOMUSDT",
#    "BNBUSDT",
#    "ONTUSDT",
#    "IOTAUSDT",
#    "BATUSDT",  
#    "VETUSDT",
#    "NEOUSDT"
]

# Konfiguracja pobierania
DOWNLOAD_CONFIG = {
    "market": "futures",        # Typ rynku
    "max_retries": 3,          # Maksymalna liczba prób dla failed request
    "retry_delay": 1,          # Opóźnienie między próbami (sekundy)
    "timeout": 30,             # Timeout dla requestów (sekundy)
    "chunk_delay": 0.1,        # Opóźnienie między requestami (sekundy)
}

# Konfiguracja logowania
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "download.log"
}

# Konfiguracja plików
FILE_CONFIG = {
    "output_dir": "orderbook_raw",
    "metadata_file": "download_metadata.json",
    "progress_file": "download_progress.json"
} 