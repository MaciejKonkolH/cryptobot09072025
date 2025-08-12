## Moduł download3 – dokumentacja techniczna

### Cel
- End-to-end pipeline do przygotowania danych: OHLC 1m, Orderbook (Binance Vision), normalizacja Orderbook do dokładnie 2 snapshotów na minutę, scalanie dzienne → plik zbiorczy, merge OHLC+Orderbook do formatu wide (kolumny per poziom i snapshot).
- Nacisk na: porządek I/O, wznawialność, deterministykę i spójność czasową (UTC, pełne minuty), brak NaN w finalnym zbiorze.

### Struktura katalogów (`crypto/download3/`)
- `config/`
  - `config.py` – konfiguracja (pary, ścieżki, limity, retry, logi) + `ensure_directories_exist()`
- `OHLC/`
  - `ohlc_downloader.py` – pobieranie OHLC 1m (ccxt binanceusdm)
  - `raw/` – `SYMBOL_1m.parquet`
  - `logs/`, `metadata/`, `progress/`
- `orderbook/`
  - `orderbook_downloader.py` – pobieranie ZIP (Binance Vision) → CSV per dzień
  - `raw_zip/`, `raw_csv/`
  - `logs/`, `metadata/`, `progress/`
  - `validate_normalized.py` – walidacja po normalizacji (dystrybucja 0/1/2/≥3 snapshotów/min)
  - `raw_coverage_check.py` – opcjonalna walidacja kompletności poziomów w surowych CSV
- `orderbook_fill/`
  - `normalize_daily_csv.py` – normalizacja dziennych CSV do dokładnie 2 snapshotów na minutę
  - `normalized_csv/` – wyjściowe dzienne pliki CSV po normalizacji
- `orderbook_merge/`
  - `merge_normalized_csv.py` – scalanie wszystkich dni po normalizacji do jednego Feather
  - `normalized_merged/` – `{SYMBOL}_normalized_merged.feather`
  - `logs/`, `metadata/`
- `merge/`
  - `merge_ohlc_orderbook.py` – pivot Orderbook do formatu wide i merge z OHLC po minucie
  - `validate_merged_data.py` – walidacja finalnego zbioru (ciągłość, NaN, kolumny)
  - `merged_data/` – `merged_{SYMBOL}.parquet`
  - `logs/`, `metadata/`

Uwaga (legacy): ścieżki `orderbook_merge/merged_raw` i `orderbook_fill/completed` są historyczne i nie są używane w bieżącym pipeline.

### Kroki pipeline (wysoki poziom)
- Krok 1: OHLC (`OHLC/ohlc_downloader.py`)
  - Źródło: ccxt `binanceusdm.fetch_ohlcv` 1m
  - Backfill: domyślnie 30 dni (konfigurowalne), wznowienie z ostatniego `timestamp`
  - Wyjście: `OHLC/raw/{SYMBOL}_1m.parquet`
- Krok 2: Orderbook download (`orderbook/orderbook_downloader.py`)
  - Źródło: Binance Vision ZIP → rozpakowane CSV per dzień
  - Retry/backoff, cache zakresu dat, pomijanie istniejących plików
  - Wyjście: `orderbook/raw_csv/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
- Krok 3: Normalizacja per dzień (`orderbook_fill/normalize_daily_csv.py`)
  - Cel: dokładnie 2 snapshoty/minutę, komplet poziomów w każdym snapshotcie
  - Reguły:
    - ≥2 snapshoty w minucie: wybieramy pierwszy i ostatni (po czasie)
    - 1 snapshot: duplikujemy pełny snapshot w tej minucie z unikalnym czasem (`:15s`/`:45s`)
    - 0 snapshotów: kopiujemy CAŁY zestaw poziomów z najbliższego snapshotu „przed” (→ `:15s`) i „po” (→ `:45s`); jeśli istnieje tylko jedna strona, duplikujemy ją, aby uzyskać 2 snapshoty
  - Implementacja: pełna wektoryzacja (groupby/agg, merge_asof), wieloprocesowo (ProcessPoolExecutor)
  - Parametry: `--workers N`, `--force` (nadpisanie istniejących outputów)
  - Wyjście: `orderbook/normalized_csv/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
- Krok 4: Scalanie normalizowanych dni (`orderbook_merge/merge_normalized_csv.py`)
  - Łączy wszystkie dzienne CSV w jeden plik Feather
  - Wyjście: `orderbook/normalized_merged/{SYMBOL}_normalized_merged.feather`
- Krok 5: Merge OHLC + Orderbook wide (`merge/merge_ohlc_orderbook.py`)
  - Z `normalized_merged.feather` buduje per-minutowy pivot:
    - `ts_1`, `ts_2` – rzeczywiste czasy dwóch snapshotów w minucie
    - Kolumny: `depth_{1|2}_{m1..m5|p1..p5}`, `notional_{1|2}_{m1..m5|p1..p5}`
  - Łączenie z OHLC po minucie (wspólny zakres), wynik bez NaN
  - Wyjście: `merge/merged_data/merged_{SYMBOL}.parquet`
- Walidacja:
  - `orderbook/validate_normalized.py` – rozkład liczby snapshotów/min po normalizacji (powinno być 2)
  - `merge/validate_merged_data.py` – ciągłość (brak brakujących minut), NaN (powinno być 0), negative checks, przykładowy pełny wiersz

### Konfiguracja (`download3/config/config.py`)
- `PAIRS`, `INTERVAL` (np. "1m"), `OHLC_HISTORY_BACK_MINUTES`
- `GAPS` (progi pod przyszłe zaawansowane fill), `CONCURRENCY`, `RETRY`
- `PATHS` – ścieżki I/O; katalogi tworzone automatycznie
- `LOGGING` – poziom i format logów

### Konwencje I/O i nazewnictwo plików
- OHLC: `OHLC/raw/{SYMBOL}_1m.parquet`
- Orderbook raw: `orderbook/raw_csv/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
- Orderbook po normalizacji (dzień): `orderbook/normalized_csv/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
- Orderbook po scalen iu: `orderbook/normalized_merged/{SYMBOL}_normalized_merged.feather`
- Finalny merge: `merge/merged_data/merged_{SYMBOL}.parquet`
- Logi: `.../logs/*.log`; metadane: `.../metadata/*.json`

### Reguły „dokładnie 2 snapshoty na minutę” (obecna implementacja)
- Deterministyczna selekcja w minucie: pierwszy i ostatni snapshot (jeśli ≥2)
- Duplikacja pełnego snapshotu w minucie (jeśli 1)
- Kopiowanie pełnych zestawów poziomów z najbliższych snapshotów (prev/next) dla minut pustych (0), z osadzeniem czasów `:15s` i `:45s` – przy braku jednej strony, duplikacja drugiej
- Gwarancja unikalnych timestampów w obrębie minuty i kompletności poziomów → brak NaN po pivocie/merge

### Wydajność, wznawialność i metadane
- Wektoryzacja + wieloprocesowość per-dzień, szybkie I/O (Feather/Parquet)
- Skrypty są idempotentne; `--force` w normalizacji pozwala przeliczyć cały zakres
- Każdy krok zapisuje logi i metadane w swoich katalogach

### Stan jakości (BTCUSDT – przykład)
- Po pełnym przebiegu: `Missing minutes: 0`, `Rows with any NaN: 0` w `validate_merged_BTCUSDT.json`

---
Autor: download3 module
Data: aktualne