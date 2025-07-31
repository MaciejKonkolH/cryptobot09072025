# Moduł Orderbook Downloader

Szybki downloader danych Orderbook z Binance Futures używający Binance Vision API.

## Struktura modułu

```
download2/orderbook/
├── config.py                           # Konfiguracja par i parametrów
├── fast_orderbook_downloader.py        # Główny downloader orderbook
├── analyze_orderbook_pairs.py          # Analizator dostępnych par
├── merge_orderbook_to_feather.py       # Łączenie danych w format feather (OPTYMALIZOWANY)
├── merge_orderbook_parallel.py         # Równoległe przetwarzanie par
├── fill_orderbook_gaps.py              # Wypełnianie luk w danych
├── orderbook_raw/                      # Katalog z pobranymi danymi
├── merged_raw/                         # Pliki feather (merged/filled)
├── download.log                        # Logi pobierania
├── download_metadata.json              # Metadane pobierania
└── README.md                           # Ten plik
```

## Funkcjonalności

### 1. Szybkie pobieranie danych Orderbook
- Pobiera historyczne dane orderbook z Binance Vision API
- Automatycznie sprawdza dostępny zakres dat dla każdej pary
- Inteligentnie łączy nowe dane z istniejącymi
- Rate limiting i error handling
- Progress bar z tqdm

### 2. Analiza dostępnych par
- Sprawdza zakres dat dla wszystkich popularnych par
- Generuje raporty w JSON/CSV/TXT
- Sortuje pary według długości historii

### 3. Konfiguracja
- Lista 20 popularnych par futures
- Konfigurowalne parametry pobierania
- System logowania

## Użycie

### 1. Analiza dostępnych par orderbook

```bash
cd download2/orderbook
python analyze_orderbook_pairs.py
```

To wygeneruje:
- `orderbook_pairs_analysis.json` - szczegółowe dane w JSON
- `orderbook_pairs_analysis.csv` - dane w formacie CSV
- `orderbook_pairs_analysis.txt` - raport tekstowy

### 2. Pobieranie danych orderbook

```bash
cd download2/orderbook
python fast_orderbook_downloader.py
```

To pobierze dane orderbook dla wszystkich par z konfiguracji.

### 3. Łączenie danych w format feather

#### **Wersja sekwencyjna (optymalizowana):**
```bash
cd download2/orderbook
python merge_orderbook_to_feather.py
```

#### **Wersja równoległa (NAJLEPSZA):**
```bash
cd download2/orderbook
python merge_orderbook_parallel.py
```

**Opcje równoległe:**
- `--max-workers 4` - liczba procesów równoległych (domyślnie: 4)
- Bez argumentów - automatyczna detekcja liczby rdzeni

**Zalety wersji równoległej:**
- **5-8x szybsze** niż wersja sekwencyjna
- Wykorzystuje wszystkie rdzenie CPU
- Automatyczne zarządzanie procesami
- Timeout 2h na parę

**Opcje sekwencyjne:**
- `--symbol ETHUSDT` - przetwórz tylko jedną parę
- Bez argumentów - przetwórz wszystkie pary z konfiguracji

### 4. Wypełnianie luk w danych

```bash
cd download2/orderbook
python fill_orderbook_gaps.py
```

To inteligentnie wypełni luki w danych orderbook używając różnych metod:
- **Interpolacja liniowa** - dla małych luk (≤ 5 min)
- **Rolling average** - dla średnich luk (≤ 60 min) z małą zmianą ceny
- **Forward fill** - dla dużych luk lub znaczących zmian ceny

**Opcje:**
- `--symbol ETHUSDT` - przetwórz tylko jedną parę
- `--max-small-gap 5` - maksymalna luka dla interpolacji (minuty)
- `--max-medium-gap 60` - maksymalna luka dla rolling average (minuty)
- `--price-threshold 2.0` - próg zmiany ceny (%)

### 5. Konfiguracja par

Edytuj `config.py` aby zmienić listę par:

```python
PAIRS = [
    "ETHUSDT",
    "BCHUSDT",
    "XRPUSDT",
    # ... dodaj więcej par
]
```

## Format danych

### Pliki surowe (orderbook_raw/)
- Nazwa: `{symbol}-bookDepth-{date}.csv`
- Przykład: `BTCUSDT-bookDepth-2023-01-01.csv`

### Pliki feather (połączenie)
- Nazwa: `orderbook_merged_{symbol}.feather`
- Przykład: `orderbook_merged_ETHUSDT.feather`
- Format: Wide format z dwoma snapshotami na minutę

### Pliki feather (po wypełnieniu luk)
- Nazwa: `orderbook_filled_{symbol}.feather`
- Przykład: `orderbook_filled_ETHUSDT.feather`
- Format: Ciągłe dane z wypełnionymi lukami

### Struktura danych orderbook
Dane zawierają snapshots orderbook z różnych momentów dnia:
- `timestamp` - czas snapshotu (YYYY-MM-DD HH:MM:SS)
- `bids` - zlecenia kupna (cena, ilość)
- `asks` - zlecenia sprzedaży (cena, ilość)

## Różnice między OHLC a Orderbook

| Aspekt | OHLC | Orderbook |
|--------|------|-----------|
| **Format** | Ciągłe dane czasowe | Snapshots w różnych momentach |
| **Zakres** | Długi (kilka lat) | Krótszy (od późniejszej daty) |
| **Rozmiar** | Mniejszy | Większy (pełny orderbook) |
| **Złożoność** | Proste (OHLCV) | Złożone (bids/asks) |

## Uwagi techniczne

1. **Dane historyczne** - Orderbook ma krótszą historię niż OHLC
2. **Rozmiar plików** - Pliki orderbook są znacznie większe
3. **Rate limiting** - 0.1s opóźnienie między requestami
4. **Walidacja** - Sprawdzanie rozmiaru plików (>1000 bajtów)

## Przykłady użycia

### Sprawdzenie dostępności danych
```bash
python analyze_orderbook_pairs.py
```

### Pobranie danych dla konkretnej pary
```bash
# Edytuj config.py aby zawierał tylko jedną parę
PAIRS = ["BTCUSDT"]
python fast_orderbook_downloader.py
```

### Przetworzenie konkretnej pary
```bash
# Przetwórz tylko ETHUSDT
python merge_orderbook_to_feather.py --symbol ETHUSDT
```

### Wypełnienie luk dla konkretnej pary
```bash
# Wypełnij luki tylko dla ETHUSDT
python fill_orderbook_gaps.py --symbol ETHUSDT
```

### Monitorowanie postępu
```bash
# Sprawdź logi
tail -f download.log

# Sprawdź metadane
cat download_metadata.json
```

## Integracja z systemem

Ten moduł jest kompatybilny z istniejącym systemem:
- Używa tej samej struktury co moduł OHLC
- Kompatybilny z `download_and_merge_orderbook.py`
- Może być używany w pipeline'ie przetwarzania danych

## Kompletny pipeline orderbook

```
1. fast_orderbook_downloader.py → orderbook_raw/*.csv
2. merge_orderbook_parallel.py → merged_raw/orderbook_merged_*.feather
3. fill_orderbook_gaps.py → merged_raw/orderbook_filled_*.feather
```

## Optymalizacje wydajności

### **Wersja 2.0 - Optymalizowana:**
- **Równoległe wczytywanie plików** (ThreadPoolExecutor)
- **Optymalizowana transformacja** (pivot_table zamiast groupby)
- **Równoległe przetwarzanie par** (ProcessPoolExecutor)
- **Wczytywanie wsadowe** (batch loading)

### **Prędkości:**
- **Stara wersja**: ~8-9 godzin dla 13 par
- **Nowa wersja sekwencyjna**: ~2-3 godziny dla 13 par
- **Nowa wersja równoległa**: ~30-60 minut dla 13 par

### **Zalecenia:**
- **Użyj `merge_orderbook_parallel.py`** dla najlepszej wydajności
- **4 procesy równoległe** na typowym komputerze
- **8 procesów** na mocnym komputerze z dużą ilością RAM

## Troubleshooting

### Błąd: "Nie znaleziono żadnych danych orderbook"
- Sprawdź czy para ma dostępne dane orderbook
- Użyj `analyze_orderbook_pairs.py` aby sprawdzić dostępność

### Błąd: "Timeout"
- Zwiększ `timeout` w `config.py`
- Sprawdź połączenie internetowe

### Błąd: "Rate limit"
- Zwiększ `chunk_delay` w `config.py`
- Poczekaj chwilę i spróbuj ponownie 