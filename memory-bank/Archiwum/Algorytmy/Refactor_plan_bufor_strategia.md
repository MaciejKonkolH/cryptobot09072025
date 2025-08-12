# Plan Refaktoryzacji Bufora i Strategii

## Cel
Zmiana architektury w celu wyeliminowania problemu przesunięcia predykcji, uproszczenia przepływu danych i zwiększenia wydajności.

## Nowa Architektura

### 1. Bufor (`DataFrameExtender`) - Serwis Cech (Feature Service)

- **Główna zmiana**: Bufor nie zwraca już gigantycznego `DataFrame`. Staje się "serwisem", który na żądanie dostarcza gotowe, obliczone cechy dla konkretnego punktu w czasie.
- **Nowe API**:
    - `get_features_for_timestamp(pair, timestamp)`:
        - **Wejście**: `para`, pojedynczy `timestamp`.
        - **Logika**:
            1. Wczytuje dane historyczne potrzebne do obliczenia cech dla danego `timestamp` (np. 43,200 świec *przed* tym timestampem).
            2. Oblicza wszystkie 8 cech (MA, stosunki, zmiany procentowe).
            3. **Wyjście**: Zwraca **jedną linię** (słownik lub `pd.Series`) zawierającą 8 obliczonych cech dla zadanego `timestamp`.
- **Cache**:
    - Bufor będzie posiadał wewnętrzny cache (`features_cache`), aby przechowywać już obliczone cechy.
    - Klucz cache: `(pair, timestamp)`.
    - W trybie backtest, cechy dla danej świecy będą obliczane tylko raz.

### 2. Strategia (`Enhanced_ML_MA43200_Buffer_Strategy`) - Konsument Cech

- **Główna zmiana**: Strategia nie wykonuje już skomplikowanych operacji na `DataFrame`. Zamiast tego, dla każdej świecy prosi bufor o gotowe cechy.
- **Nowa logika w `populate_indicators`**:
    1. Otrzymuje `dataframe` od FreqTrade (długi w backtest, krótki w live).
    2. Tworzy puste kolumny dla 8 cech w `dataframe`.
    3. **Iteruje po każdej świecy** w `dataframe`.
    4. Wewnątrz pętli, dla każdej świecy (`row`):
        - Wywołuje `features = buffer.get_features_for_timestamp(pair, row['date'])`.
        - Wstawia otrzymane `features` do odpowiednich kolumn w bieżącym `row`.
    5. Po zakończeniu pętli, `dataframe` jest w pełni uzupełniony o poprawne cechy, bez żadnych przesunięć.
    6. `dataframe` jest następnie przekazywany do `add_ml_signals`.

### 3. Generator Sygnałów (`signal_generator.py`) - Czysta Predykcja

- **Główna zmiana**: Z `signal_generator` zostanie usunięta cała logika obliczania cech.
- **Nowa odpowiedzialność**:
    1. Otrzymuje `dataframe` z **już obliczonymi** cechami.
    2. Skaluje cechy.
    3. Tworzy sekwencje (windowing).
    4. Generuje predykcje z modelu.
    5. Mapuje predykcje z powrotem do `dataframe`.
    - Logika mapowania będzie teraz trywialna i poprawna, ponieważ istnieje bezpośredni związek 1:1 między świecą, jej cechami i predykcją.

## Oczekiwane Rezultaty
- **Problem rozwiązany**: Całkowite wyeliminowanie przesunięcia o 119 minut.
- **Kod uproszczony**: Jasny podział odpowiedzialności (Bufor=dostawca, Strategia=konsument).
- **Wydajność poprawiona**: Znacznie mniejsze zużycie pamięci i potencjalnie szybsze działanie dzięki cache.
- **Łatwiejsze debugowanie**: Prosty przepływ danych ułatwia analizę problemów. 