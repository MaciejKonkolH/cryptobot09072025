# Finalny, Zoptymalizowany Zestaw Cech do Treningu Modelu

Poniższa lista zawiera finalny zestaw cech, na którym będziemy trenować model. Strategia została uproszczona, aby zredukować redundancję i szum, a jednocześnie wzmocniona o kluczowy mechanizm normalizacji wewnątrzsekwencyjnej.

---

### Grupa A: Cechy Podstawowe (Normalizacja Wewnątrzsekwencyjna)

**Cel:** Ta grupa jest fundamentem całego modelu. Jej zadaniem jest uniezależnienie danych od absolutnego poziomu ceny (np. czy BTC kosztuje $20k czy $70k) poprzez wyrażenie wszystkich cen jako **procentowej zmiany względem ceny otwarcia pierwszej świecy w analizowanej sekwencji.**

*Ta transformacja będzie wykonywana "w locie" w `DataGeneratorze` modułu treningowego.*

*   **`open_norm`**
    *   **Opis:** Znormalizowana cena otwarcia.
    *   **Obliczenia:** `(open - open_anchor) / open_anchor`

*   **`high_norm`**
    *   **Opis:** Znormalizowana cena maksymalna.
    *   **Obliczenia:** `(high - open_anchor) / open_anchor`

*   **`low_norm`**
    *   **Opis:** Znormalizowana cena minimalna.
    *   **Obliczenia:** `(low - open_anchor) / open_anchor`

*   **`close_norm`**
    *   **Opis:** Znormalizowana cena zamknięcia.
    *   **Obliczenia:** `(close - open_anchor) / open_anchor`

---

### Grupa B: Cechy Zmienności (Volatility Features)

*   **`bb_width_20`**: **Szerokość Wstęg Bollingera (20 okresów)**
    *   **Opis:** Mierzy znormalizowaną odległość między górną a dolną wstęgą. Wąskie wstęgi sygnalizują niską zmienność i potencjalne wybicie.
    *   **Obliczenia:** 1. Obliczana jest prosta średnia krocząca z 20 okresów (środkowa wstęga). 2. Obliczane jest odchylenie standardowe z 20 okresów. 3. Górna wstęga to środkowa wstęga + 2 * odchylenie; dolna to środkowa - 2 * odchylenie. 4. Cecha to `(Górna - Dolna) / Środkowa`.

---

### Grupa C: Cechy Pędu/Siły Ruchu (Momentum Features)

*   **`rsi_14`**: **Relative Strength Index (14 okresów)**
    *   **Opis:** Oscylator pędu (0-100) wskazujący na siłę i szybkość zmian cen. Pomaga identyfikować stany wykupienia (>70) i wyprzedania (<30).
    *   **Obliczenia:** 1. Obliczane są średnie zyski i średnie straty dla cen zamknięcia z ostatnich 14 okresów. 2. Stosunek średniego zysku do średniej straty (RS) jest następnie przekształcany wzorem matematycznym na skalę 0-100.

*   **`macd_hist_norm`**: **Histogram MACD znormalizowany**
    *   **Opis:** Różnica między linią MACD a linią sygnałową, znormalizowana przez cenę zamknięcia. Pokazuje pęd trendu; rosnący histogram sugeruje wzmacnianie się trendu.
    *   **Obliczenia:** 1. Obliczana jest różnica między wykładniczą średnią kroczącą z 12 i 26 okresów (linia MACD). 2. Obliczana jest wykładnicza średnia krocząca z 9 okresów samej linii MACD (linia sygnałowa). 3. Cecha to `(linia MACD - linia sygnałowa) / cena zamknięcia`.

---

### Grupa D: Cechy Siły i Kierunku Trendu (Trend Features)

*   **`adx_14`**: **Average Directional Index (14 okresów)**
    *   **Opis:** Absolutna miara siły trendu (0-100), niezależnie od jego kierunku. Wartości > 25 wskazują na silny trend (warunki dla LONG/SHORT). Wartości < 20 wskazują na rynek w konsolidacji (warunki dla HOLD).
    *   **Obliczenia:** 1. Oblicza się "Ruch Kierunkowy" w górę (+DI) i w dół (-DI), które mierzą siłę ruchów wzrostowych i spadkowych w danym okresie. 2. ADX jest wygładzoną średnią z matematycznie przetworzonej różnicy między +DI a -DI, co daje ostateczną miarę siły trendu w skali 0-100.

*   **`price_to_ma_60`**: **Stosunek ceny do MA(60)**
    *   **Opis:** Pozycja ceny względem trendu z ostatniej godziny.
    *   **Obliczenia:** Obliczana jest prosta średnia krocząca z 60 ostatnich cen zamknięcia (`MA60`). Cecha to `cena zamknięcia / MA60`.

*   **`price_to_ma_240`**: **Stosunek ceny do MA(240)**
    *   **Opis:** Pozycja ceny względem trendu z ostatnich 4 godzin.
    *   **Obliczenia:** Obliczana jest prosta średnia krocząca z 240 ostatnich cen zamknięcia (`MA240`). Cecha to `cena zamknięcia / MA240`.

*   **`ma_60_to_ma_240`**: **Stosunek MA(60) do MA(240)**
    *   **Opis:** Relacja trendu krótko- do średnioterminowego. Rosnący stosunek oznacza przyspieszanie trendu krótkoterminowego.
    *   **Obliczenia:** Obliczane są proste średnie kroczące MA(60) i MA(240). Cecha to `MA(60) / MA(240)`.

---

### Grupa E: Cechy Wolumenu i Kontekstu Długoterminowego

*   **`volume_change_norm`**: **Znormalizowana procentowa zmiana wolumenu**
    *   **Opis:** Procentowa zmiana wolumenu względem poprzedniej świecy, znormalizowana przez odchylenie standardowe.
    *   **Obliczenia:** 1. Obliczana jest procentowa zmiana wolumenu (`(wolumen - wolumen poprzedni) / wolumen poprzedni`). 2. Obliczane jest odchylenie standardowe tych zmian z ostatnich 14 okresów. 3. Cecha to `zmiana procentowa / odchylenie standardowe`.

*   **`price_to_ma_1440`**: **Stosunek ceny do MA(1440)**
    *   **Opis:** Kontekst trendu dziennego.
    *   **Obliczenia:** `cena zamknięcia / prosta średnia krocząca z 1440 okresów`

*   **`price_to_ma_43200`**: **Stosunek ceny do MA(43200)**
    *   **Opis:** Kontekst trendu miesięcznego.
    *   **Obliczenia:** `cena zamknięcia / prosta średnia krocząca z 43200 okresów`

*   **`volume_to_ma_1440`**: **Stosunek wolumenu do jego średniej dziennej**
    *   **Opis:** Porównuje obecny wolumen do średniego wolumenu z ostatniej doby.
    *   **Obliczenia:** `bieżący wolumen / prosta średnia krocząca z wolumenu z 1440 okresów`

*   **`volume_to_ma_43200`**: **Stosunek wolumenu do jego średniej miesięcznej**
    *   **Opis:** Porównuje obecny wolumen do średniego wolumenu z ostatniego miesiąca.
    *   **Obliczenia:** `bieżący wolumen / prosta średnia krocząca z wolumenu z 43200 okresów`

---
### Usunięte Cechy (dla zachowania kontekstu)

*   **`atr_14_norm`**: Usunięto z powodu redundancji z `bb_width_20`.
*   **`stoch_k_14`**: Usunięto z powodu redundancji z `rsi_14`.
*   **Grupa cech struktury świecy**: Usunięto z powodu bardzo niskiego stosunku sygnału do szumu na interwale 1-minutowym.
*   Stare cechy `high_change`, `low_change`, `close_change`: Zastąpione przez znacznie lepszy mechanizm normalizacji wewnątrzsekwencyjnej.
