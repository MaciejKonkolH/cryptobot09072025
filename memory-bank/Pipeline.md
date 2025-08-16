## download3/orderbook — moduły pobierania i walidacji księgi zleceń

Ten rozdział opisuje moduły pipeline odpowiedzialne za pobieranie surowych danych orderbook (snapshoty), sprawdzanie ich kompletności i jakość przed dalszą normalizacją/mergem.

### Wspólne założenia i konfiguracja
- **Konfiguracja**: `download3/config/config.py` — klucze `PATHS`, `PAIRS`, `LOGGING`, `RETRY` (timeouty, retraje). Wszystkie skrypty dodają `DOWNLOAD3_ROOT` do `sys.path` i korzystają z tej konfiguracji.
- **Ścieżki kluczowe** (wg `PATHS`):
  - `ob_raw_zip` — archiwa .zip pobrane z Binance
  - `ob_raw_csv` — rozpakowane CSV (plik per dzień, nazwa: `{SYMBOL}-bookDepth-YYYY-MM-DD.csv`)
  - `ob_metadata` — raporty JSON (zasięg, pokrycie, dystrybucje, walidacje)
  - `ob_logs` — logi skryptów
- **Logowanie**: poziom i format zgodnie z `LOGGING`; FileHandler + StreamHandler (stdout).
- **Źródło danych**: `BASE_URL = "https://data.binance.vision/data"` (UM futures, ścieżka dzienna `.../futures/um/daily/bookDepth/{symbol}/...`).

---

### orderbook_downloader.py — pobieranie surowych CSV
- **Cel**: pobranie paczek ZIP z Binance dla zakresu dostępnych dni i ich rozpakowanie do `PATHS["ob_raw_csv"]`.
- **Główne funkcje**:
  - `get_available_date_range(symbol, session, logger)` — sondowanie najstarszej i najnowszej dostępnej daty (HEAD na plikach), cache w `ob_metadata/available_ranges.json`.
  - `download_and_extract(session, url, zip_path, csv_path, timeout, logger)` — pobieranie ZIP z retrajami, ekstrakcja do CSV, usunięcie ZIP; zwraca `True/False`.
  - `download_symbol(symbol, logger)` — iteracja po dniach (z pominięciem istniejących CSV), pobranie i pasek postępu (ETA, tempo), zwraca `True` gdy pobrano cokolwiek.
  - `main()` — iteruje po `PAIRS` i raportuje powodzenie.
- **We/Wy**:
  - We: `PAIRS`, `RETRY`, `PATHS`
  - Wy: CSV w `ob_raw_csv`, log w `ob_logs/orderbook_downloader.log`, cache zakresu w `ob_metadata/available_ranges.json`.
- **Uruchomienie**: `python -m download3.orderbook.orderbook_downloader`

---

### validate_normalized.py — walidacja znormalizowanych dni (2 snapshoty/min)
- **Cel**: dla CSV w `PATHS["ob_normalized_csv"]` (produkt kroku normalizacji) sprawdzić: liczność snapshotów/min, NaN, wartości ujemne itp.
- **Metryki per dzień**: `minutes_with_0/1/2/3/4+`, `nan_rows`, `negative_depth_rows`, `negative_notional_rows`, przykładowe minuty ≠ 2.
- **Wyjście**: `ob_metadata/validate_normalized_{symbol}.json` z agregatem `totals` i `per_day`.
- **Argumenty**: `--symbol`, opcjonalnie `--start`, `--end` (YYYY-MM-DD) do filtrowania zakresu dni.
- **Uruchomienie**: `python -m download3.orderbook.validate_normalized --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31`

---

### raw_coverage_check.py — pokrycie poziomów procentowych w surowych snapshotach
- **Cel**: sprawdzić, czy w każdej próbce czasu występuje komplet poziomów `[-5..-1, 1..5]` oraz czy oba snapshoty/min są kompletne i bez NaN.
- **Zakres obliczeń**:
  - Per snapshot: kompletność poziomów; lista kilku przykładów braków
  - Per minuta: czy pierwszy i ostatni snapshot w minucie są kompletne (`minutes_both_complete`)
- **Implementacja**: szybkie czytanie `pandas.read_csv` (preferowany silnik pyarrow), czyszczenie duplikatów po `(timestamp, percentage)`, równoległość `ProcessPoolExecutor` (do 8 workerów).
- **Wyjście**: `ob_metadata/raw_coverage_{symbol}.json` (totals i per_day).
- **Uruchomienie**: `python -m download3.orderbook.raw_coverage_check --symbol BTCUSDT`

---

### snapshot_distribution.py — dystrybucja liczby snapshotów/min w surowych CSV
- **Cel**: policzyć na ilu minutach wystąpiło 0/1/2/3/4+ unikalnych timestampów (zliczane per dzień, sumowane globalnie).
- **Wyjście**: `ob_metadata/snapshot_distribution_{symbol}.json` z polami `overall` i `per_day`.
- **Argumenty**: `--symbol`, `--start`, `--end` (filtrowanie dni).
- **Uruchomienie**: `python -m download3.orderbook.snapshot_distribution --symbol BTCUSDT`

---

### check_missing_csv.py — raport brakujących surowych dni i dostępności online
- **Cel**: dla zakresu dostępności (z cache lub sondowania) porównać listę **wszystkich dni** z dniami istniejącymi lokalnie i sprawdzić, które brakujące są nadal dostępne online (HEAD na ZIP) vs. brak na serwerze.
- **Główne kroki**:
  1) `load_cached_range` lub `probe_range_if_needed` (wywołuje `orderbook_downloader.get_available_date_range`)
  2) `list_existing_dates` → zbiory dni lokalnych
  3) `check_online_availability` → podział braków na „dostępne” i „not found”
  4) `write_report` → `ob_metadata/missing_report_{symbol}.json`
- **Uruchomienie**: `python -m download3.orderbook.check_missing_csv`

---

### Przepływ danych (wysoki poziom)
1) `orderbook_downloader.py` → `PATHS["ob_raw_csv"]` (surowe CSV per dzień)
2) (poza tym rozdziałem) normalizacja → `PATHS["ob_normalized_csv"]` (2 snapshoty/min)
3) Walidacje/raporty jakości:
   - `raw_coverage_check.py` i `snapshot_distribution.py` na surowych CSV
   - `validate_normalized.py` na znormalizowanych CSV
4) Kontrola braków: `check_missing_csv.py`

### Dobre praktyki uruchomieniowe
- Najpierw uruchom `orderbook_downloader.py`, potem raporty jakości; zakres dat ograniczaj argumentami `--start/--end`, aby skrócić czas.
- Jeśli trzeba odświeżyć zasięg, usuń `ob_metadata/available_ranges.json` (skrypt przeliczy cache).
- Raporty JSON są idempotentne — można uruchamiać wielokrotnie; skrypty nie modyfikują danych wejściowych poza pobieraniem/rozpakowaniem.

---

## download3/orderbook_fill — moduł uzupełniania i normalizacji orderbooku

### normalize_daily_csv.py — normalizacja do 2 snapshotów/min (per dzień)
- **Cel**: z surowych CSV (`PATHS["ob_raw_csv"]`) zbudować znormalizowany plik dzienny z dokładnie dwoma snapshotami na minutę (snap=1 przy :15s, snap=2 przy :45s), kompletne zestawy poziomów.
- **Zasady**:
  - minuty z ≥2 snapshotami → wybierz pierwszy i ostatni (oryginalne timestampy), przypisz snap=1/2
  - minuty z 1 snapshotem → duplikuj w tej minucie (domyślnie drugi na :45s; jeśli oryginał był :45s, syntetyczny na :15s)
  - minuty bez snapshotów → skopiuj PEŁNE zestawy poziomów z najbliższych sąsiadów (prev → :15s, next → :45s). Jeśli występuje tylko jedna strona, zduplikuj ją, by były 2 snapy
- **We/Wy**:
  - We: `PATHS["ob_raw_csv"]/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
  - Wy: `PATHS["ob_normalized_csv"]/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
  - Log: `ob_logs/normalize_daily_csv.log`
- **Równoległość**: per‑dzień (`--workers`), domyślnie 1
- **Argumenty**: `--symbol`, `--days` (lista), `--start`, `--end`, `--workers`, `--force`
- **Uruchomienie**: `python -m download3.orderbook_fill.normalize_daily_csv --symbol BTCUSDT --workers 4 --force`

### synthesize_missing_days.py — synteza brakujących surowych dni
- **Cel**: uzupełnić brakujące dni surowego orderbooku w `ob_raw_csv` w ramach dostępnego zakresu poprzez syntezę z dnia‑dawcy i sąsiadów czasowych.
- **Logika**:
  - Ustal zakres dostępności (cache `ob_metadata/available_ranges.json` lub sonda przez `orderbook_downloader.get_available_date_range`)
  - Wyznacz listę brakujących dni oraz ich sekwencje ciągłe
  - Dla sekwencji: wybierz dzień‑dawcę (najbliższy; preferencja tego samego dnia tygodnia)
  - Skalowanie głębokości per poziom: interpolacja geometryczna między medianami sąsiadów (dzień przed/po) względem median dawcy; `t=(i+1)/(N+1)` dla i‑tego dnia w sekwencji
  - Notional: wyliczane przez `depth * npd` (npd=notional per depth, mediany per poziom jako fallback)
  - Dodaj deterministyczny jitter (`jitter_sigma≈0.01`) dla ograniczenia artefaktów
- **We/Wy**:
  - We: istniejące dni w `ob_raw_csv`, zakres dostępności, cache
  - Wy: syntetyczne dni w `ob_raw_csv/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
  - Log: `ob_fill_logs/synthesize_missing_days.log`
- **Uwagi**:
  - Nie modyfikuje istniejących dni; po każdej sekwencji dopisuje nowe dni do listy `existing`
  - Wymaga przynajmniej jednego dnia dawcy (oraz opcjonalnie sąsiadów) — w przeciwnym wypadku pomija sekwencję
- **Uruchomienie**: `python -m download3.orderbook_fill.synthesize_missing_days`

---

## download3/orderbook_merge — moduł łączenia danych orderbook

### merge_normalized_csv.py — sklejanie znormalizowanych dni w jedną paczkę
- **Cel**: po normalizacji (2 snapshoty/min per dzień) złożyć wszystkie dni w jeden plik kolumnowy dla danego symbolu.
- **We/Wy**:
  - We: `PATHS["ob_normalized_csv"]/{SYMBOL}-bookDepth-YYYY-MM-DD.csv`
  - Wy: `PATHS["ob_normalized_merged"]/{SYMBOL}_normalized_merged.feather`
  - Log: `ob_merge_logs/merge_normalized_csv.log`
- **Szczegóły**: czyta CSV per dzień, konkatenacja, `timestamp` konwertowany do `datetime` przed zapisem do Feather.
- **Uruchomienie**: `python -m download3.orderbook_merge.merge_normalized_csv`

### merge_orderbook_to_wide.py — z surowych CSV do „wide” minutowych
- **Cel**: przekształcić surowe snapshoty (długa postać) w ramkę minutową „wide” z dwiema kolumnowymi fotografiami orderbooku na minutę (snapshot1/snapshot2) z metadanymi czasów.
- **We/Wy**:
  - We: `PATHS["ob_raw_csv"]/{SYMBOL}-bookDepth-YYYY-MM-DD.csv` (surowe)
  - Wy: `PATHS["ob_merged_raw"]/orderbook_wide_raw_{SYMBOL}.feather`
  - Log: `ob_merge_logs/merge_orderbook_to_wide.log`; metadane w `ob_merge_metadata` (tworzenie katalogów)
- **Algorytm (per minuta)**:
  - ≥2 snapshoty: wybierz pierwszy i ostatni (oryginalne czasy)
  - 1 snapshot: wygeneruj drugi przez interpolację wartości między tym snapshotem a najbliższym późniejszym (ratio 0.5; czas pośredni)
  - 0 snapshotów: użyj najbliższych „przed” i „po”, wyinterpoluj 25% i 75%, przypisz czasy na początek minuty i tuż przed jej końcem
- **Transformacja „long→wide”**: dla każdej minuty pivot `depth` i `notional` po `percentage` osobno dla `snapshot1_` i `snapshot2_`; dodatkowo kolumny `snapshot1_timestamp`, `snapshot2_timestamp`.
- **Uruchomienie**: `python -m download3.orderbook_merge.merge_orderbook_to_wide`

---

## download3/OHLC — moduł pobierania danych OHLC

### ohlc_downloader.py — pobieranie 1‑min OHLC z Binance USDT‑M Futures (ccxt)
- **Cel**: pobrać świece 1‑min (`INTERVAL` z configu) dla symboli z `PAIRS` i zapisać do jednego pliku Parquet na symbol.
- **We/Wy**:
  - We: konfiguracja `PAIRS`, `INTERVAL`, `OHLC_HISTORY_BACK_MINUTES`, `PATHS`, `LOGGING`, `RETRY` w `download3/config/config.py`
  - Wy: `PATHS["ohlc_raw"]/{SYMBOL}_{INTERVAL}.parquet` (kolumny: `timestamp, open, high, low, close, volume`)
  - Log: `PATHS["ohlc_logs"]/ohlc_downloader.log`
- **Główne kroki**:
  1) Inicjalizacja `ccxt.binanceusdm` z `timeout`, `enableRateLimit`, `options.defaultType='future'`.
  2) Konwersja symbolu do formatu ccxt, np. `BTCUSDT -> BTC/USDT:USDT`.
  3) Wyznaczenie zakresu czasu: argumenty `--start/--end` lub domyślnie `now - OHLC_HISTORY_BACK_MINUTES` do teraz (UTC). Daty bez czasu traktowane jako UTC; `--end` jako dzień włącznie (dodawany +1 dzień o północy).
  4) Pobieranie porcjami (limit=1000) metodą `exchange.fetch_ohlcv(...)` z retry wg `RETRY`.
  5) Re‑kalibracja startu do pierwszej dostępnej świecy (jeśli giełda pominie pusty zakres na początku).
  6) Pasek postępu w jednej linii (liczba chunków, minuty, ETA). Po pobraniu: deduplikacja timestampów, sortowanie, zapis Parquet.
- **Uruchomienie**:
  - Wszystkie z configu: `python -m download3.OHLC.ohlc_downloader`
  - Pojedynczy symbol i zakres: `python -m download3.OHLC.ohlc_downloader --symbol BTCUSDT --start 2023-01-01 --end 2023-03-01`
- **Uwagi**:
  - Gdy brak `--start`, skrypt może wznowić od ostatniego timestampu istniejącego Parquet (resume).
  - Plik wynikowy jest nadpisywany dla danego symbolu i interwału.

### check_ohlc_continuity.py — kontrola spójności i pokrycia danych OHLC
- **Cel**: sprawdzić, czy plik OHLC ma wszystkie minuty bez duplikatów w zakresie od minimum do maksimum timestampu.
- **We/Wy**:
  - We: `PATHS["ohlc_raw"]/{SYMBOL}_{INTERVAL}.parquet`
  - Wy: log `PATHS["ohlc_logs"]/ohlc_continuity.log`, opcjonalnie CSV: `PATHS["ohlc_metadata"]/missing_minutes_{SYMBOL}.csv`
- **Główne kroki**:
  1) Wczytanie Parquet, konwersja `timestamp` do UTC, deduplikacja, sortowanie.
  2) Budowa pełnego indeksu minutowego `start..end` i porównanie z obserwowanymi minutami.
  3) Raport: liczba oczekiwanych/obecnych minut, pokrycie %, duplikaty, liczba braków oraz pierwszych 5 luk (jako przedziały).
  4) Zapis listy brakujących minut do CSV (jeśli występują).
- **Uruchomienie**:
  - Wszystkie: `python -m download3.OHLC.check_ohlc_continuity`
  - Pojedynczy symbol: `python -m download3.OHLC.check_ohlc_continuity --symbol BTCUSDT`

## download3/merge — moduł łączenia danych OHLC oraz orderbook

### merge_ohlc_orderbook.py — połączenie 1‑min OHLC z wide‑orderbookiem (2 snapshoty/min)
- **Cel**: skleić ramkę OHLC minutową z ramką orderbook „wide” (głębokości/notionale dla snapshot1/snapshot2 i poziomów ±1..±5, plus znaczniki czasów `ts_1`, `ts_2`).
- **We/Wy**:
  - We: `PATHS["ohlc_raw"]/{SYMBOL}_{INTERVAL}.parquet`, `PATHS["ob_normalized_merged"]/{SYMBOL}_normalized_merged.feather`
  - Wy: `PATHS["merged_data"]/merged_{SYMBOL}.parquet`
  - Log: `PATHS["merge_logs"]/merge_ohlc_orderbook.log`
- **Główne kroki**:
  1) Wczytaj OHLC → `timestamp` do UTC → utwórz indeks minutowy (`minute`) i ustaw go jako index (usuń surowy `timestamp`).
  2) Wczytaj orderbook merged → `timestamp` do UTC → wyznacz `minute` oraz `ts_1=min(timestamp)`, `ts_2=max(timestamp)` per minuta.
  3) Oznacz rangę snapshotu (`rank`=1 dla `ts_1`, `rank`=2 dla `ts_2`), odfiltruj tylko te dwa.
  4) Zrób pivot `depth` i `notional` po indeksie `minute` i kolumnach (`rank`, `percentage`), spłaszcz nazwy do `depth_{rank}_{p|m}{level}`, `notional_{rank}_{p|m}{level}`; dołącz `ts_1`, `ts_2`.
  5) Zestrój wspólny zakres minut OHLC i OB (cięcie do części wspólnej) i złącz po indeksie.
  6) Dodaj aliasy kompatybilne z training3: `snapshot1_depth_{±1..±5}`, `snapshot2_depth_{±1..±5}`, oraz agregaty: `snapshot1_bid_volume/ask_volume`, `snapshot1_spread` (+ opcjonalnie snapshot2).
  7) Zresetuj indeks do kolumny `timestamp` i zapisz Parquet do `merged_data`.
- **Uruchomienie**: `python -m download3.merge.merge_ohlc_orderbook`

### validate_merged_data.py — walidacja połączonych danych OHLC+Orderbook
- **Cel**: weryfikacja spójności i kompletności po merge, ze szczególnym naciskiem na obecność obu snapshotów i brak braków minutowych.
- **We/Wy**:
  - We: `PATHS["merged_data"]/merged_{SYMBOL}.parquet`
  - Wy: `PATHS["merge_metadata"]/validate_merged_{SYMBOL}.json`
  - Log: `PATHS["merge_logs"]/validate_merged_data.log`
- **Sprawdzane elementy**:
  - Grupy kolumn: OHLC (`open/high/low/close/...`), orderbook (`depth_*`, `notional_*`), inne.
  - Wymagane: obecność `open/high/low/close` oraz kolumn `depth_1_*`, `depth_2_*`, `notional_1_*`, `notional_2_*` (dowolne poziomy).
  - Ciągłość czasu: pełna siatka minut `start..end`, liczba brakujących minut i przykłady.
  - NaN: liczba wierszy z jakimkolwiek NaN (globalnie, w OB, w OHLC) i rozbicie na kolumny.
  - Wartości ujemne: detekcja w `depth_*` i `notional_*`.
  - Przykładowy wiersz kompletny (bez NaN) dla wglądu w schemat.
- **Uruchomienie**:
  - Wszystkie: `python -m download3.merge.validate_merged_data`
  - Pojedynczy symbol: `python -m download3.merge.validate_merged_data --symbol BTCUSDT`

---

## feature_calculator_4 — moduł kalkulatora cech

### Przegląd
- **Cel**: z danych po merge (`download3/merge/merged_data/merged_{symbol}.parquet`) obliczyć zestaw uzgodnionych cech dla treningu i zapisać do Feather + metadata JSON.
- **We/Wy**:
  - We: `download3/merge/merged_data/merged_{SYMBOL}.parquet`
  - Wy: `feature_calculator_4/output/features_{SYMBOL}.feather`, metadata: `feature_calculator_4/metadata/features_{SYMBOL}.json`, log: `feature_calculator_4/logs/feature_calculator_4.log`

### Struktura modułu
- `feature_calculator_4/config.py` — ścieżki, parametry okien (kanały 240/180/120), przełączniki (`ENABLE_T3_FEATURES`), itp.
- `feature_calculator_4/logger.py` — konfiguracja logowania (plik + stdout).
- `feature_calculator_4/utils.py` — pomocnicze wskaźniki TA: EMA, ATR, RSI, Stochastic, MACD histogram, Bollinger, Donchian, ADX/DI, bezpieczne dzielenie, itp.
- `feature_calculator_4/feature_builder.py` — główna logika `compute_features(df, progress=False)`.
- `feature_calculator_4/main.py` — wczytanie merged, obliczenie cech, zapis wyników + metadata.
- `feature_calculator_4/validate_features.py` — walidacja pliku cech (NaN/Inf, stałe kolumny, brakujące minuty, itp.).
- `feature_calculator_4/check_output_columns.py` — szybkie sprawdzenie listy spodziewanych kolumn (kanały, OHLC‑TA, OB, interakcje).

### Główne kroki obliczeń (`compute_features`)
1) **Kanały cenowe** dla okien `CHANNEL_WINDOWS = [240, 180, 120]`:
   - `pos_in_channel_{w}`, `width_over_ATR_{w}`, `slope_over_ATR_window_{w}`, `channel_fit_score_{w}`.
2) **OHLC‑TA** (wybrane):
   - Geometria świecy: `body_ratio`, `wick_up_ratio`, `wick_down_ratio`.
   - Zwroty/trend: `r_1`, `r_5`, `r_15`, `slope_return_120`.
   - Zmienność/regresja: `vol_regime_120`, `vol_of_vol_120`, `r2_trend_120`.
   - Oscylatory/TA: `RSI_14`, `RSI_30`, `StochK_14_3`, `StochD_14_3`, `MACD_hist_over_ATR`, `ADX_14`, `di_diff_14`, `CCI_20_over_ATR`.
   - Bollinger/Donchian: `bb_pos_20`, `bb_width_over_ATR_20`, `donch_pos_60`, `donch_width_over_ATR_60`.
   - EMA: `close_vs_ema_60`, `close_vs_ema_120`, `slope_ema_60_over_ATR`.
   - Wolumenowe: `MFI_14`, `OBV_slope_over_ATR`.
3) **Orderbook (OB)** z danych wide lub fallbacków na notional:
   - `imbalance_1pct_notional`, log‑ratio 2/1 dla ask/bid, `ask_near_ratio`, `bid_near_ratio`, `concentration_near_mkt`.
   - `ask_com`, `bid_com`, `com_diff`; `pressure_12`, `pressure_12_norm`; `side_skew`.
   - Trwałość/impulsy: `persistence_imbalance_1pct_ema{5,10}`, `dA1`, `dB1`, `dImb1`, `ema_dImb1_{5,10}`.
   - (Opcjonalnie) jeśli ustawione `TP_PARAM/SL_PARAM`: `reach_TP_notional`, `reach_SL_notional`, `resistance_vs_support`.
4) **Training3 core (opcjonalnie)** jeśli `ENABLE_T3_FEATURES=True` — zgodne nazwy rdzeniowych 37 cech (trend/pozycja/vol/OB/regime/volatility cluster itp.).
5) **Interakcje kanał × OB** (jeśli dostępny `imbalance_1pct_notional`):
   - `{pos_in_channel|slope_over_ATR_window|width_over_ATR}_{w}_x_imbalance_1pct` dla `w ∈ {240,180,120}`.

### Normalizacja i sanity checks
- Po obliczeniach: odcięcie rozgrzewki `iloc[240:]`, zamiana Inf→NaN i wypełnienie 0.0; dociągnięcie surowych `open/high/low/close/volume` dla zgodności z labelerem.

### Uruchomienie
- Obliczenia: `python -m feature_calculator_4.main --symbol BTCUSDT`
- Walidacja: `python -m feature_calculator_4.validate_features --symbol BTCUSDT`
- Spis kolumn: `python -m feature_calculator_4.check_output_columns BTCUSDT`

---

## labeler5 — moduł etykietowania danych

### Przegląd
- **Cel**: dodać etykiety 3‑klasowe (0=LONG, 1=SHORT, 2=NEUTRAL) dla wielu par TP/SL, na bazie ścieżki ceny w oknie przyszłości `FUTURE_WINDOW`.
- **We/Wy**:
  - We: `feature_calculator_4/output/features_{SYMBOL}.feather` (wymaga kolumn `high`, `low`, `close`; indeks czasowy z `timestamp`).
  - Wy: `labeler5/output/labeled_{SYMBOL}.feather` (oryginalne cechy + kolumny `label_tp{tp}_sl{sl}`), metadata: `labeler5/metadata/labeled_{SYMBOL}.json`, log: `labeler5/logs/labeler5.log`.

### Pliki modułu
- `labeler5/config.py` — ścieżki, `FUTURE_WINDOW=120`, lista `TP_SL_LEVELS` (15 kombinacji), log format/poziom.
- `labeler5/logger.py` — konfiguracja logowania (plik + stdout).
- `labeler5/main.py` — funkcje `load_features`, `label_single_level`, `run` i CLI `main`.

### Parametry
- `FUTURE_WINDOW` (minuty): horyzont, w którym badamy, czy TP lub SL wystąpi jako pierwsze.
- `TP_SL_LEVELS`: lista par procentowych, np. `(0.6, 0.2)`, `(1.4, 0.7)` itp. Jednostki w % (0.6 oznacza 0.6%).

### Logika etykietowania (label_single_level)
1) Dla każdego czasu wejścia `i` przyjmujemy `entry = close[i]`.
2) LONG: sprawdzamy świeca po świecy `j=1..FUTURE_WINDOW`, czy wystąpi `TP` (high ≥ entry*(1+tp)) lub `SL` (low ≤ entry*(1−sl)).
3) SHORT: analogicznie, `TP` gdy `low ≤ entry*(1−tp)`, `SL` gdy `high ≥ entry*(1+sl)`.
4) Reguła wyboru etykiety:
   - Jeśli tylko LONG osiąga TP przed SL → etykieta 0 (LONG).
   - Jeśli tylko SHORT osiąga TP przed SL → etykieta 1 (SHORT).
   - Jeśli oba osiągają TP przed SL → wygrywa kierunek z wcześniejszym TP; remis (ten sam bar) → 2 (NEUTRAL, konflikt).
   - Jeśli żaden nie spełnia warunku w oknie → 2 (NEUTRAL).
5) Zliczane są „konflikty” (intra‑bar BOTH lub remis czasowy); ostatnie `FUTURE_WINDOW` wierszy są odrzucane (brak pełnego okna).

### Wyjściowy format
- Kolumny etykiet: `label_tp{tp}_sl{sl}` z kropką zamienioną na `p`, np. `label_tp0p8_sl0p3`.
- Plik wynikowy zawiera wszystkie oryginalne cechy + kolumny etykiet; indeks czasowy zachowany (po zapisie do Feather z kolumną `timestamp`).
- Metadata JSON: liczba wierszy/kolumn, liczba poziomów, `future_window`, suma konfliktów, lista nazw kolumn etykiet.

### Uruchomienie
- `python -m labeler5.main --symbol BTCUSDT`

---

## training5 — moduł treningowy (XGBoost, multi:softprob)

### Przegląd
- **Cel**: trenować 15 niezależnych modeli (po jednym na każdy poziom TP/SL) klasyfikacji 3‑klasowej (LONG/SHORT/NEUTRAL) oraz wygenerować raporty i artefakty do inferencji.
- **We/Wy**:
  - We: `labeler5/output/labeled_{SYMBOL}.feather`
  - Wy: w `training5/output/`:
    - `models/{SYMBOL}/model_{1..15}.json`, `scaler.pkl`, `metadata.json`
    - `reports/{SYMBOL}/results_{...}.md` i `results_{...}.json`, `metrics_{SYMBOL}.json`, `feature_importance_{...}.csv`, `predictions_{SYMBOL}_{level}_{...}.csv`, `predictions_trades_{SYMBOL}_{level}_{...}.csv`
    - `reports/{SYMBOL}/features_used_{SYMBOL}_{ts}.txt`
    - log: `logs/training5.log`

### Pliki modułu
- `training5/config.py` — ścieżki, lista poziomów `TP_SL_LEVELS`, podział 70/15/15, parametry XGB, klasy wag, tryb wyboru cech.
- `training5/data_loader.py` — wczytanie, wybór/filtr cech, inferencja nazw etykiet, skalowanie `RobustScaler` i podział danych.
- `training5/model_builder.py` — klasa `MultiOutputXGB` (15 modeli XGB, early‑stopping, opcjonalne wagi klas na zbiorze treningowym).
- `training5/main.py` — trening, zapis artefaktów, metryk, predykcji i raportów.
- `training5/eval_report.py` — raport bez treningu (ładuje zapisane modele), opcjonalny eksport transakcji JSON dla wybranego poziomu i progu.
- `training5/report.py` — zapis Markdown i JSON (użyte cechy, parametry, metryki, bloki „pewności”), obliczenia metryk handlowych.
- `training5/utils.py` — logowanie do pliku i stdout.

### Wybór cech (FEATURE_SELECTION_MODE)
- `'all'`: wszystkie kolumny numeryczne poza OHLC (`open/high/low/close/volume`) i etykietami.
- `'t3_37'`: lista 37 rdzeniowych cech (zgodna z training3), użyta gdy dostępne.
- `'custom'`: część wspólna z `CUSTOM_FEATURE_LIST` (brakujące raportowane jako ostrzeżenie).
- `'custom_strict'`: dokładnie `CUSTOM_FEATURE_LIST`; jeśli jakiejś brakuje → błąd i przerwanie treningu.

### Kroki treningu (`main.py`)
1) Wczytaj dane, zrób podział 70/15/15 i skalowanie (RobustScaler; brakujące wypełnione medianą z train).
2) Zbuduj 15 modeli XGB (po jednym per etykieta) z `objective=multi:softprob`, `tree_method=hist`, early‑stopping na walidacji.
3) (Opcjonalnie) zastosuj wagi klas z `CLASS_WEIGHTS` wyłącznie na zbiorze treningowym.
4) Zapisz modele, scaler i `metadata.json` (nazwa cech, poziomy TP/SL), `metrics_{symbol}.json` oraz listę cech `features_used_*.txt`.
5) Zapisz predykcje testowe do CSV oraz plik „tylko transakcje” (bez NEUTRAL) z flagą poprawności (`WIN/LOSS`).
6) Zapisz ważność cech `feature_importance_{ts}.csv` (sumaryczny gain, znormalizowany do 1).
7) Wygeneruj raporty: Markdown + JSON (sekcja użytych cech na początku obu formatów).

### Metryki i raporty (`report.py` i `eval_report.py`)
- Klasy: `['LONG','SHORT','NEUTRAL']` (odpowiednio etykiety 0/1/2).
- Standard: accuracy, `classification_report`, macierz pomyłek.
- Bloki „pewności” na progach: 0.30, 0.40, 0.45, 0.50 (spójne w MD i JSON).
- Metryki handlowe (z macierzy pomyłek):
  - **Accuracy SHORT+LONG (ważona)**: ważona precyzja po stronach według liczby predykcji na LONG/SHORT; jeśli brak predykcji po którejś stronie → strona nie wpływa na wynik.
  - **Próg opłacalności**: `SL / (SL + TP) * 100%`.
  - **Marża bezpieczeństwa**: `100% * Accuracy_SHORT+LONG (ważona) − Próg`.
  - **Dochód netto per trade**: `p_win*TP − (1−p_win)*SL` per strona i łączny (ważony liczbą trade’ów).

### Pliki predykcji
- `predictions_{SYMBOL}_{level}_{ts}.csv`: sygnał z progiem `0.5` na `max(prob)` (LONG/SHORT/NEUTRAL) oraz pełne prawdopodobieństwa.
- `predictions_trades_{SYMBOL}_{level}_{ts}.csv`: tylko wiersze LONG/SHORT (bez NEUTRAL) + kolumny `true_label`, `correct`, `result`.

### Uruchomienie
- Trening: `python -m training5.main --symbol BTCUSDT`
- Raport z zapisanych modeli: `python -m training5.eval_report --symbol BTCUSDT`
- Eksport transakcji (JSON) dla wybranego poziomu i progu (bez generowania raportu):
  - `python -m training5.eval_report --symbol BTCUSDT --save_trades --trades_only --tp 1.0 --sl 0.4 --conf_thr 0.5`
  - lub przez indeks poziomu: `--level_idx 7` (0‑based)




