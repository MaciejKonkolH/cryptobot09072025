# Dziennik Śledczy: Analiza Przesunięcia Czasowego o 120 Minut

**Status:** Rozpoczęto analizę modułu `validation_and_labeling`.

---

## GŁÓWNA HIPOTEZA (WSTĘPNA)

Po pierwszej, szczegółowej analizie kodu, źródłem problemu wydaje się być **fundamentalna rozbieżność w obsłudze stref czasowych** między danymi przygotowywanymi dla **backtestu w Freqtrade** a danymi przygotowywanymi dla **procesu treningu/walidacji modelu**.

-   **Dane dla Freqtrade:** Zapisywane jako **UTC-aware** (jawnie przypisana strefa czasowa UTC).
-   **Dane dla Treningu:** Zapisywane jako **timezone-naive** (pozbawione informacji o strefie czasowej).

Ta rozbieżność powoduje, że środowisko treningowe może błędnie interpretować czas, co prowadzi do generowania predykcji z przesunięciem.

---

## SZCZEGÓŁOWA ANALIZA PLIKÓW

### 1. `binance_data_downloader.py` - Pobieranie Danych

-   **Spostrzeżenie:** Dane są pobierane z Binance, gdzie timestampy są w formacie milisekund od epoki (UTC). Skrypt poprawnie konwertuje je za pomocą `pd.to_datetime(..., unit='ms')`.
-   **Kluczowy fakt:** `pd.to_datetime` domyślnie tworzy obiekty `datetime` **bez przypisanej strefy czasowej (naive)**, ale ich wartość numeryczna odpowiada czasowi UTC.
-   **Status:** **Potencjalna niejednoznaczność.** Dane są poprawne, ale ich naiwność czasowa wymaga spójnej obsługi w dalszych krokach.

### 2. `data_validator.py` - Walidacja i Standaryzacja Danych

-   **Spostrzeżenie:** W metodzie `_fix_datetime_index` znajduje się kluczowy fragment kodu.
-   **Kod (`_fix_datetime_index`):**
    ```python
    if config.STANDARDIZE_TIMESTAMP_FORMAT and config.TRAINING_COMPATIBILITY_MODE:
        if df['datetime'].dt.tz is not None:
            # Usuń timezone info dla consistency
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            self.logger.debug("Usunięto timezone info dla training compatibility")
    ```
-   **Kluczowy fakt:** Ten kod **celowo i jawnie usuwa wszelkie informacje o strefie czasowej**, aby zapewnić "kompatybilność z treningiem". Oznacza to, że od tego momentu wszystkie dane w potoku są oficjalnie **timezone-naive**.
-   **Status:** **Krytyczny punkt kontrolny.** To tutaj dane tracą swoją jawność UTC.

### 3. `main.py` - Orchestrator i KRYTYCZNA ROZBIEŻNOŚĆ

-   **Spostrzeżenie:** Ten plik jest sercem modułu i to tutaj następuje rozdzielenie danych na dwie różne ścieżki.
-   **Ścieżka A: Zapis danych dla Freqtrade (`raw_validated`)**
    -   **Kod (`process_single_pair`):**
        ```python
        # Dodaj timezone UTC (wymagane przez FreqTrade)
        freqtrade_data['date'] = pd.to_datetime(freqtrade_data['date'], utc=True)
        # Zapisz w formacie kompatybilnym z FreqTrade
        freqtrade_data.to_feather(raw_validated_path)
        ```
    -   **Kluczowy fakt:** Naiwny timestamp (reprezentujący UTC) jest tutaj poprawnie konwertowany na **świadomy (aware) obiekt datetime ze strefą czasową UTC**. To jest format, którego Freqtrade oczekuje.
-   **Ścieżka B: Zapis danych dla Treningu (`output/*_training_ready.feather`)**
    -   **Kod (`process_single_pair`):**
        ```python
        # ZACHOWAJ TIMESTAMP - konwertuj datetime index na kolumnę przed zapisem
        labeled_data_with_timestamp = labeled_data.reset_index()
        # ...
        # Zapisuję dane z timestamp
        save_data_file(labeled_data_with_timestamp, output_file_path, ...)
        ```
    -   **Kluczowy fakt:** Tutaj dane są zapisywane z kolumną `timestamp`, która **pozostaje timezone-naive**. Nie ma żadnej konwersji do strefy czasowej.
-   **Status:** **SMOKING GUN.** Znaleziono bezpośrednią przyczynę problemu. Pipeline tworzy dwa różne zestawy danych z fundamentalnie inną obsługą czasu.

### 4. `feature_calculator.py` - Obliczanie Cech

-   **Spostrzeżenie:** Metoda `calculate_features` odrzuca pierwsze 43200 świec (`warmup_period`).
-   **Kod (`calculate_features`):**
    ```python
    warmup_period = config.MA_LONG_WINDOW
    if len(df_final) > warmup_period:
        df_final = df_final.iloc[warmup_period:].copy()
    ```
-   **Kluczowy fakt:** To działanie jest **prawidłowe i zamierzone**. Zapewnia, że długoterminowe średnie kroczące są obliczane na pełnym oknie danych. **To nie jest źródło błędu przesunięcia**, ale wyjaśnia, dlaczego dla najwcześniejszego okresu w danych nie ma dostępnych predykcji.

### 5. `competitive_labeler.py` - Etykietowanie Danych

-   **Spostrzeżenie:** Algorytm etykietowania jest zoptymalizowany do działania na indeksach liczbowych, a nie na obiektach `datetime`.
-   **Kod (`_execute_competitive_labeling_algorithm`):**
    ```python
    # ...
    timestamps_map, ohlc_data_array, sufficient_data_mask = self._prepare_optimized_data_access(...)
    # ...
    for t in range(total_rows):
        # ...
        current_pos = timestamps_map.get(current_timestamp)
        # ...
        for future_pos in range(current_pos + 1, future_window_end + 1):
            # ...
    ```
-   **Kluczowy fakt:** Moduł ten otrzymuje dane z **timezone-naive** indeksem, wykonuje na nich operacje bazujące na pozycjach w tablicy, a następnie zwraca wynik z **nienaruszonym, timezone-naive indeksem**.
-   **Status:** ** ogniwo w łańcuchu.** Moduł działa poprawnie i nie wprowadza przesunięcia, ale utrwala problem, przekazując dane pozbawione strefy czasowej do finalnego pliku treningowego.

---

## Analiza Modułu `Kaggle` (Trening)

### 6. `data_loader.py` - Ładowanie Danych Treningowych

-   **Spostrzeżenie:** Metoda `load_training_data` wczytuje plik `.feather` (np. `...__single_label.feather`). Jak ustaliliśmy w poprzednim kroku, ten plik zawiera daty w formacie **timezone-naive**.
-   **Kod (`_filter_by_date_range`):**
    ```python
    start_dt = datetime.strptime(config.START_DATE, '%Y-%m-%d')
    # ...
    df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]
    ```
-   **Kluczowy fakt:** Operacje filtrowania i podziału chronologicznego (`_chronological_split`) są wykonywane na tych "naiwnych" datach. Cały `DataLoader` jest spójny wewnętrznie, ale operuje na danych, które są już "skażone" brakiem informacji o strefie czasowej.

### 7. `trainer.py` - Zapisywanie Wyników Walidacji

-   **Spostrzeżenie:** Metoda `generate_validation_positions_report` jest odpowiedzialna za tworzenie pliku `validation_analysis_...csv`, który był używany do tworzenia wykresów.
-   **Kod:**
    ```python
    analysis_df = pd.DataFrame({
        'timestamp': timestamps, # Te timestamps są timezone-naive
        # ...
    })
    analysis_df.to_csv(analysis_filename, index=False)
    ```
-   **Kluczowy fakt:** Skrypt bierze **naiwne znaczniki czasu** i zapisuje je do pliku CSV. W pliku pojawiają się one jako zwykły tekst (np. `2024-12-20 00:00:00`), utrwalając tym samym błąd. To właśnie ten plik był źródłem danych dla skryptu wizualizacyjnego.

---

## WNIOSEK KOŃCOWY

**Główna hipoteza została w pełni potwierdzona. Źródłem przesunięcia o 120 minut jest fundamentalna i systemowa rozbieżność w obsłudze stref czasowych.**

1.  **Pipeline Walidacji:** Tworzy **dwa rodzaje danych**:
    -   **Dla Freqtrade:** Poprawne, świadome strefy czasowej **UTC**.
    -   **Dla Treningu:** Błędne, **nieświadome strefy czasowej (naive)**, które są niejawnie przesunięte o czas lokalny serwera (UTC+2).

2.  **Pipeline Treningu:** Konsekwentnie operuje na danych **timezone-naive**, trenując model na przesuniętych danych. Następnie zapisuje wyniki walidacji z tymi samymi błędnymi, naiwnymi datami.

3.  **Skrypt Wizualizacji:** Czytając naiwne daty z CSV (`2024-12-20 00:00:00`) i błędnie je lokalizując do strefy `Europe/Warsaw`, "przypadkowo" korygował błąd na wykresie, przesuwając dane o 2 godziny wstecz do czasu UTC (`2024-12-19 22:00:00 UTC`). To zaciemniło prawdziwy problem i sprawiło, że dane z walidacji pasowały do wykresu.

4.  **Freqtrade:** Jako jedyny komponent w całym systemie, od początku operował na **poprawnych danych UTC**. Dlatego bezlitośnie ujawnił przesunięcie, pokazując predykcje o 2 godziny (120 minut) później niż oczekiwano.

**Problem nie leży w logice `window_size` ani w algorytmie strategii, ale w przygotowaniu danych na samym początku procesu.**
