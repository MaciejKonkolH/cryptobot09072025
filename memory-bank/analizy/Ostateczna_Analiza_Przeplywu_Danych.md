# Ostateczna Analiza Przepływu Danych i Źródła Błędu (Wersja Poprawiona)

**Data:** 2025-07-05
**Status:** Zidentyfikowano ostateczne, precyzyjne źródło potencjalnego błędu.

## Streszczenie Problemu

Po długiej analizie i odrzuceniu poprzednich hipotez, ustalono, że kluczowe jest zrozumienie, jak dokładnie `DataFrameExtender` (bufor) interaguje z danymi dostarczanymi przez silnik Freqtrade. Użytkownik słusznie zauważył, że opis tego procesu był nieprecyzyjny.

## Poprawiony, Definitywny Przepływ Danych w Strategii

### ETAP 1: Freqtrade Wywołuje Strategię

*   Freqtrade wywołuje metodę `populate_indicators` w strategii.
*   Przekazuje do niej ramkę danych (`dataframe`). Wielkość tej ramki zależy od trybu:
    *   **Backtesting:** `dataframe` zawiera **cały okres** testowy (tysiące świec).
    *   **Dry-run/Live:** `dataframe` zawiera tylko **niewielki, najnowszy fragment danych** (np. ostatnie kilkaset świec).
*   Dane te są świadome strefy czasowej UTC (format `datetime64[ns, UTC]`).

### ETAP 2: Wejście do Bufora (`DataFrameExtender`)

*   Strategia przekazuje całą otrzymaną ramkę (`dataframe`) do metody `extend_and_calculate_features` w buforze.

### ETAP 3: Budowanie Kontekstu Historycznego (SERCE OPERACJI)

1.  **Analiza Ramki od Freqtrade:** Bufor pobiera datę **pierwszej** świecy z ramki otrzymanej od Freqtrade.
2.  **Wczytanie Danych Historycznych:** Bufor otwiera plik `..._raw_validated.feather` i wczytuje z niego **ogromną ilość danych historycznych** (np. 43200+ świec) sprzed daty pierwszej świecy. Dane te, po naszych poprawkach, również są świadome strefy czasowej UTC.
3.  **ŁĄCZENIE DANYCH (`pd.concat`) - KRYTYCZNY PUNKT:**
    *   Bufor łączy wczytaną historię z pliku z "teraźniejszością" otrzymaną od Freqtrade.
    *   `work_df = pd.concat([GIGANTYCZNA_HISTORIA_Z_PLIKU, RAMKA_OD_FREQTRADE])`
    *   **To jest jedyne miejsce w całym systemie, gdzie dane z dwóch różnych źródeł (plik `.feather` vs. ramka z silnika Freqtrade) są ze sobą sklejane.**
4.  **Obliczenia i Przycinanie:** Na tej połączonej, ogromnej ramce danych obliczane są wskaźniki (`ma43200` etc.), a następnie dołożona historia jest odrzucana.

## Ostateczny Wniosek

Problem **nie leży** w danych treningowych, modelu, ani w logice strategii jako takiej.

Problem musi powstawać w **Kroku 3.3 (`pd.concat`)**. Pomimo że oba źródła danych (`.feather` i ramka z Freqtrade) są w formacie UTC, istnieje bardzo wysokie prawdopodobieństwo, że występuje między nimi jakaś **subtelna niezgodność**, która powoduje, że biblioteka `pandas` podczas operacji łączenia gubi lub błędnie interpretuje strefę czasową.

Możliwe przyczyny tej niezgodności:
*   Różna precyzja (`datetime64[ns, UTC]` vs `datetime64[ms, UTC]`).
*   Różnice w metadanych obiektu `DataFrame`.
*   Wewnętrzna optymalizacja w `pandas` lub `pyarrow`, która prowadzi do nieoczekiwanego zachowania.

**To wyjaśnia wszystkie obserwowane zjawiska:**
*   **Dlaczego daty są ważne, mimo że model ich nie używa:** Bo błąd powstaje podczas przygotowywania cech na podstawie dat.
*   **Dlaczego problem pojawia się tylko w Freqtrade:** Bo tylko tam następuje to specyficzne łączenie danych z dwóch źródeł.
*   **Dlaczego przesunięcie o 120 minut "naprawiało" problem:** Bo przypadkowo kompensowało błąd powstały podczas łączenia danych.

Następnym krokiem jest dodanie precyzyjnych logów diagnostycznych do `DataFrameExtender`, aby "złapać na gorącym uczynku" `pandas.concat` i zobaczyć, co dokładnie dzieje się ze strefami czasowymi w tym konkretnym momencie. 