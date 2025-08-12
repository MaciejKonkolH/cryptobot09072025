# Proste Wyjaśnienie Problemu z Przesunięciem Czasu

**Data:** 2025-07-05
**Status:** Ostateczne, uproszczone wyjaśnienie problemu.

### 1. Jaki jest główny problem?

- W testach historycznych (backtesting) musimy sztucznie przesuwać dane o 120 minut.
- Bez tego przesunięcia wyniki są złe.
- Oznacza to, że dane, na których ostatecznie pracuje strategia, są przesunięte w czasie.

### 2. Gdzie powstaje ten błąd?

- Błąd powstaje **tylko w jednym, konkretnym miejscu**.
- Tym miejscem jest **Bufor Danych** (`DataFrameExtender`).

### 3. Co dokładnie robi Bufor? (Krok po kroku)

- **Krok A: Dostaje dane "na żywo" od Freqtrade**
    - Freqtrade (silnik strategii) daje Buforowi mały kawałek najnowszych danych (np. 500 ostatnich świec).
    - Te dane mają **poprawną strefę czasową (UTC)**.

- **Krok B: Wczytuje DANE HISTORYCZNE z pliku**
    - Bufor otwiera duży plik z danymi historycznymi (`..._raw_validated.feather`).
    - Te dane również mają **poprawną strefę czasową (UTC)**.

- **Krok C: ŁĄCZY DANE (TUTAJ JEST PROBLEM!)**
    - Bufor musi połączyć te dwa zestawy danych:
        - Dokleja starą historię (z pliku) **przed** danymi "na żywo" (od Freqtrade).
    - Używa do tego jednej komendy: `pd.concat`.
    - **To jest ten jeden, jedyny moment, w którym czas się "psuje" i powstaje przesunięcie.**

### 4. Dlaczego czas psuje się podczas łączenia?

- Pomyśl o tym jak o sklejaniu dwóch kawałków papieru:
    - Niby oba są białe, ale jeden jest matowy, a drugi błyszczący.
    - Gdy je skleisz, w miejscu łączenia widać różnicę.
- Podobnie jest z naszymi danymi:
    - Mimo że oba zestawy danych są w UTC, **minimalnie się od siebie różnią** (np. mają inną wewnętrzną precyzję zapisu daty).
    - Gdy biblioteka `pandas` próbuje skleić te "trochę inne" dane, **gubi informację o strefie czasowej**.
    - Efekt: powstaje przesunięcie o równe 2 godziny.

### 5. Podsumowanie w jednym zdaniu

Problem nie leży w modelu, danych treningowych ani w logice strategii, ale w **technicznym błędzie podczas operacji sklejania dwóch różnych źródeł danych w buforze**. 