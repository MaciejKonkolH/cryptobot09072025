# Szybki Downloader Danych OHLC

## 🚀 Opis

Szybki downloader danych OHLC z Binance Futures używający biblioteki CCXT. 
Znacznie szybszy niż oryginalny `data_downloader.py` - pobiera dane w minutach zamiast godzin.

## 📋 Funkcjonalności

- ✅ **Ultra-szybkie pobieranie** używając CCXT library
- ✅ **Auto-detect zakresu dat** dla każdej pary
- ✅ **Resume download** (wznawianie przerwanego pobierania)
- ✅ **Progress bar** z szczegółowymi logami
- ✅ **Walidacja** pobranych danych
- ✅ **Automatyczne łączenie** z istniejącymi danymi
- ✅ **Rate limiting** zgodny z limitami Binance

## 🛠️ Instalacja

### Wymagane biblioteki:
```bash
pip install ccxt pandas tqdm
```

### Struktura katalogów:
```
download2/OHLC/
├── config.py                    # Konfiguracja par
├── fast_ohlc_downloader.py      # Główny skrypt
├── ohlc_raw/                    # Surowe dane OHLC
├── download_metadata.json       # Metadane pobierania
├── download.log                 # Logi
└── README.md                    # Ten plik
```

## ⚙️ Konfiguracja

Edytuj `config.py` aby zmienić:

### Lista par do pobierania:
```python
PAIRS = [
    "BNBUSDT",
    "ETHUSDT"
    # Dodaj więcej par...
]
```

### Parametry pobierania:
```python
DOWNLOAD_CONFIG = {
    "interval": "1m",           # Interwał (1m, 5m, 15m, 1h)
    "market": "futures",        # Typ rynku
    "chunk_size": 1000,         # Świec na request (max 1000)
    "max_retries": 3,          # Maksymalne próby
    "retry_delay": 1,          # Opóźnienie między próbami
    "timeout": 30,             # Timeout requestów
}
```

## 🚀 Uruchomienie

```bash
cd download2/OHLC
python fast_ohlc_downloader.py
```

## 📊 Przykład wyjścia

```
🚀 Inicjalizacja FastOHLCDownloader
📁 Katalog wyjściowy: ohlc_raw
🚀 Rozpoczynam szybkie pobieranie danych OHLC
📋 Pary: BNBUSDT, ETHUSDT
📊 Interwał: 1m
🏪 Rynek: futures

============================================================
🎯 Rozpoczynam pobieranie dla BNBUSDT
============================================================
🔍 Sprawdzam dostępny zakres dat dla BNBUSDT
📅 BNBUSDT: 2020-01-01 - 2025-07-30
📥 Pobieram wszystkie dostępne dane dla BNBUSDT
Pobieranie BNBUSDT: 100%|██████████| 5000/5000 [02:15<00:00, 37.04chunk/s]
✅ Pobrano 2,880,000 świec dla BNBUSDT
💾 Zapisano BNBUSDT: 2,880,000 świec, 123,456,789 bajtów

============================================================
🎯 Rozpoczynam pobieranie dla ETHUSDT
============================================================
...

🎉 Pobieranie zakończone!
✅ Udało się: 2/2 par
⏱️ Czas: 245.3 sekund
============================================================
💾 Metadane zapisane: download_metadata.json
```

## 📁 Pliki wyjściowe

### Dane OHLC:
- `ohlc_raw/BNBUSDT_1m.csv` - Dane BNBUSDT
- `ohlc_raw/ETHUSDT_1m.csv` - Dane ETHUSDT

### Format danych:
```csv
timestamp,open,high,low,close,volume
1640995200000,462.5,463.2,461.8,462.9,12345.67
1640995260000,462.9,463.5,462.7,463.1,9876.54
...
```

### Metadane:
- `download_metadata.json` - Informacje o pobieraniu
- `download.log` - Szczegółowe logi
- `download_progress.json` - Postęp pobierania (dla resume)

## 🔄 Resume Download

Skrypt automatycznie:
1. **Sprawdza istniejące dane** lokalnie
2. **Pobiera tylko nowsze dane** jeśli są dostępne
3. **Łączy nowe dane** z istniejącymi
4. **Usuwa duplikaty** i sortuje

## ⚡ Prędkość

**Porównanie z oryginalnym `data_downloader.py`:**

| Metryka | Oryginalny | Nowy (CCXT) |
|---------|------------|-------------|
| **30 dni BTCUSDT** | ~30-60 minut | ~2-5 minut |
| **1 rok BTCUSDT** | ~6-12 godzin | ~15-30 minut |
| **Metoda** | Pojedyncze pliki ZIP | Bulk API requests |
| **Równoległość** | Sekwencyjne | Zoptymalizowane |

## 🛡️ Zabezpieczenia

- **Rate limiting** - respektuje limity Binance API
- **Retry logic** - automatyczne ponowne próby przy błędach
- **Walidacja danych** - sprawdza integralność pobranych danych
- **Error handling** - szczegółowe logowanie błędów

## 🔧 Rozwiązywanie problemów

### Błąd: "No module named 'ccxt'"
```bash
pip install ccxt
```

### Błąd: "Rate limit exceeded"
- Skrypt automatycznie obsługuje rate limiting
- Jeśli nadal występuje, zwiększ `retry_delay` w config.py

### Błąd: "Connection timeout"
- Sprawdź połączenie internetowe
- Zwiększ `timeout` w config.py

## 📝 Licencja

Ten skrypt jest częścią projektu crypto trading system. 