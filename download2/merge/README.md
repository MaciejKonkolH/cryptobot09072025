# Moduł Merge - Łączenie danych OHLC z Orderbook

## 📋 Opis

Moduł `merge` służy do łączenia danych OHLC (Open, High, Low, Close) z danymi orderbook w jeden zintegrowany dataset. Skrypt pobiera dane z modułów `download2/OHLC` i `download2/orderbook` i tworzy kompletne pliki feather gotowe do treningu modeli ML.

## 🗂️ Struktura

```
download2/merge/
├── merge_ohlc_orderbook.py    # Główny skrypt łączenia
├── README.md                  # Ta dokumentacja
└── merged_data/              # Katalog z wynikami (tworzony automatycznie)
    ├── merged_ETHUSDT.feather
    ├── merged_BTCUSDT.feather
    └── ...
```

## 🚀 Użycie

### Przetwarzanie wszystkich par

```bash
cd download2/merge
python merge_ohlc_orderbook.py
```

### Przetwarzanie jednej pary

```bash
python merge_ohlc_orderbook.py --symbol ETHUSDT
```

### Własny katalog wyjściowy

```bash
python merge_ohlc_orderbook.py --output-dir moje_dane
```

## 📊 Dane wejściowe

### OHLC (z `download2/OHLC/ohlc_raw/`)
- Format: CSV
- Nazwy plików: `{SYMBOL}_1m.csv`
- Kolumny: `timestamp`, `open`, `high`, `low`, `close`, `volume`

### Orderbook (z `download2/orderbook/orderbook_completed/`)
- Format: Feather
- Nazwy plików: `orderbook_filled_{SYMBOL}.feather`
- Kolumny: `timestamp`, `snapshot1_*`, `snapshot2_*`, metadane

## 📈 Dane wyjściowe

### Format: Feather
- Nazwy plików: `merged_{SYMBOL}.feather`
- Lokalizacja: `merged_data/` (domyślnie)

### Struktura danych:
```
timestamp              # Czas główny (z OHLC)
open, high, low, close # Ceny OHLC
volume                 # Wolumen
snapshot1_timestamp    # Czas pierwszego snapshotu orderbook
snapshot2_timestamp    # Czas drugiego snapshotu orderbook
snapshot1_depth_-5     # Głębokość -5% dla snapshot1
snapshot1_depth_-4     # Głębokość -4% dla snapshot1
...
snapshot2_depth_-5     # Głębokość -5% dla snapshot2
snapshot2_depth_-4     # Głębokość -4% dla snapshot2
...
snapshot1_notional_-5  # Notional -5% dla snapshot1
...
snapshot2_notional_-5  # Notional -5% dla snapshot2
...
```

## ⚙️ Funkcjonalności

### 1. Wczytywanie danych
- **OHLC**: Wczytuje pliki CSV z `ohlc_raw/`
- **Orderbook**: Wczytuje pliki feather z `orderbook_completed/`
- Automatyczna konwersja timestampów

### 2. Wyrównywanie timestampów
- Znajduje wspólny zakres czasowy
- Filtruje dane do wspólnego zakresu
- Zapewnia kompatybilność czasową

### 3. Łączenie danych
- LEFT JOIN (zachowuje wszystkie wiersze OHLC)
- Dodaje kolumny orderbook do każdego wiersza OHLC
- Usuwa konfliktowe kolumny

### 4. Czyszczenie danych
- Usuwa metadane z `fill_orderbook_gaps.py`
- Usuwa niepotrzebne kolumny timestamp
- Optymalizuje strukturę danych

### 5. Analiza wyników
- Sprawdza pokrycie danych
- Analizuje brakujące wartości
- Wyświetla statystyki

## 📋 Obsługiwane pary

Skrypt automatycznie przetwarza wszystkie pary z `config.py`:

- ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT
- BCHUSDT, LTCUSDT, LINKUSDT, TRXUSDT
- ETCUSDT, XLMUSDT, XMRUSDT, DASHUSDT
- ZECUSDT, XTZUSDT, ATOMUSDT, BATUSDT
- IOTAUSDT, NEOUSDT, VETUSDT, ONTUSDT

## 🔧 Wymagania

- Python 3.8+
- pandas
- numpy
- pathlib
- logging

## 📝 Logi

Skrypt generuje szczegółowe logi w pliku `merge_ohlc_orderbook.log`:
- Postęp przetwarzania
- Statystyki danych
- Błędy i ostrzeżenia
- Podsumowanie wyników

## 🎯 Zastosowanie

Połączone dane są gotowe do:
- **Treningu modeli ML** (OHLC + orderbook jako features)
- **Analizy technicznej** (ceny + głębokość rynku)
- **Backtestingu strategii** (pełne dane rynkowe)
- **Feature engineering** (obliczanie dodatkowych wskaźników)

## ⚠️ Uwagi

1. **Wymagane pliki**: Przed uruchomieniem upewnij się, że:
   - Pliki OHLC istnieją w `download2/OHLC/ohlc_raw/`
   - Pliki orderbook istnieją w `download2/orderbook/orderbook_completed/`

2. **Pamięć**: Przetwarzanie wszystkich par może wymagać dużo pamięci RAM

3. **Czas**: Proces może trwać kilka godzin dla wszystkich par

4. **Dysk**: Upewnij się, że masz wystarczająco miejsca na dysku dla plików wynikowych 