# Algorytm Strategii OCO (One-Cancels-the-Other) w Freqtrade

## 🎯 Cel Strategii
Celem jest stworzenie w pełni zautomatyzowanej strategii, w której:
1.  **Model ML decyduje wyłącznie o otwarciu pozycji.**
2.  **Pozycja jest zamykana tylko przez zlecenia Stop Loss (SL) lub Take Profit (TP).**
3.  **Oba zlecenia (SL i TP) są natychmiast po otwarciu pozycji wysyłane na giełdę.**
4.  Realizacja jednego zlecenia automatycznie anuluje drugie (mechanizm OCO).

Poniższy algorytm opisuje, jak ten cel jest realizowany w ramach platformy Freqtrade.

---

### Krok 1: Konfiguracja i Inicjalizacja
*Lokalizacja w kodzie: `__init__`, `bot_start`, plik `config.json`*

1.  **Wczytanie Parametrów:** Przy starcie, strategia wczytuje z pliku `config.json` kluczowe parametry:
    *   `take_profit` (np. 0.01 dla +1.0%)
    *   `stoploss` (np. -0.005 dla -0.5%)
2.  **Ustawienia Strategii:** W kodzie strategii ustawione są flagi kluczowe dla logiki OCO:
    *   `use_exit_signal = True`: Informuje Freqtrade, że będziemy używać niestandardowej logiki wyjścia (`custom_exit`).
    *   `stoploss_on_exchange = True`: Nakazuje Freqtrade, aby zlecenie Stop Loss było wysyłane bezpośrednio na giełdę, a nie zarządzane wirtualnie.

---

### Krok 2: Generowanie Sygnału Wejścia
*Lokalizacja w kodzie: `populate_entry_trend`*

1.  **Analiza Świecy:** Dla każdej nowej świecy (np. co 1 minutę), Freqtrade uruchamia funkcję `populate_entry_trend`.
2.  **Predykcja Modelu:** Model ML analizuje najnowsze dane i generuje predykcję: `LONG`, `SHORT` lub `HOLD`.
3.  **Decyzja:**
    *   Jeśli model zwróci `LONG` lub `SHORT` z wymaganą pewnością, strategia ustawia flagę `enter_long = 1` lub `enter_short = 1`.
    *   Jeśli model zwróci `HOLD`, strategia nie podejmuje żadnych działań.

---

### Krok 3: Otwarcie Pozycji i Złożenie Zleceń OCO
*Lokalizacja w kodzie: Logika Freqtrade, `custom_exit`*

1.  **Wykrycie Sygnału:** Główna pętla Freqtrade wykrywa flagę (`enter_long` lub `enter_short`) i natychmiast wykonuje zlecenie otwarcia pozycji (np. `MARKET` lub `LIMIT`).
2.  **Potwierdzenie Otwarcia:** Giełda potwierdza, że pozycja została otwarta. Freqtrade zapisuje szczegóły transakcji w lokalnej bazie danych.
3.  **Wysłanie Zleceń Zabezpieczających (Logika OCO):**
    *   **Zlecenie Stop Loss:** Dzięki `stoploss_on_exchange=True`, Freqtrade **automatycznie** wysyła na giełdę zlecenie `STOP_MARKET` z ceną wyliczoną na podstawie parametru `stoploss`. Giełda zwraca **`SL_order_id`**, który jest zapisywany.
    *   **Zlecenie Take Profit:** Ponieważ `use_exit_signal=True`, Freqtrade wywołuje funkcję `custom_exit`. Ta funkcja oblicza cenę docelową TP (na podstawie ceny wejścia i parametru `take_profit`) i zwraca sygnał, który nakazuje Freqtrade wysłanie na giełdę zlecenia `LIMIT` na zamknięcie pozycji. Giełda zwraca **`TP_order_id`**, który również jest zapisywany.

**Efekt:** Na giełdzie znajdują się teraz dwa aktywne, powiązane z transakcją zlecenia.

---

### Krok 4: Monitorowanie i Zamknięcie Pozycji
*Lokalizacja w kodzie: Główna pętla Freqtrade (logika wewnętrzna)*

1.  **Ciągłe Odpytywanie:** W swojej głównej pętli, Freqtrade cyklicznie (co kilka sekund) komunikuje się z API giełdy, pytając o status zleceń na podstawie zapisanych **`SL_order_id`** i **`TP_order_id`**.
2.  **Scenariusz A: Realizacja Take Profit**
    *   Cena rynkowa osiąga poziom TP. Giełda realizuje zlecenie `LIMIT`.
    *   W kolejnym zapytaniu, Freqtrade otrzymuje od giełdy status **`filled`** dla `TP_order_id`.
    *   Freqtrade odnotowuje zamknięcie pozycji z zyskiem.
    *   **Natychmiast** wysyła polecenie **anulowania** zlecenia Stop Loss (używając `SL_order_id`), ponieważ nie jest już potrzebne.
3.  **Scenariusz B: Realizacja Stop Loss**
    *   Cena rynkowa osiąga poziom SL. Giełda realizuje zlecenie `STOP_MARKET`.
    *   Freqtrade otrzymuje od giełdy status **`filled`** dla `SL_order_id`.
    *   Freqtrade odnotowuje zamknięcie pozycji ze stratą.
    *   **Natychmiast** wysyła polecenie **anulowania** zlecenia Take Profit (używając `TP_order_id`).

---

## 🔑 Kluczowe Zasady Implementacji

1.  **Separacja Odpowiedzialności:** Model ML decyduje **TYLKO** o wejściu. Mechanizmy Freqtrade (`stoploss`, `custom_exit`) decydują **TYLKO** o wyjściu.
2.  **Wyjście przez OCO:** Wyjście z pozycji jest realizowane **WYŁĄCZNIE** przez zlecenia SL/TP na giełdzie, nigdy przez sygnał z modelu.
3.  **Pełna Automatyzacja:** Proces od otwarcia, przez zabezpieczenie, aż po zamknięcie jest w pełni zautomatyzowany i oparty na stałej komunikacji z giełdą.
4.  **Niezależność Pozycji:** Każda pozycja jest zarządzana niezależnie, z własnym, unikalnym zestawem zleceń OCO.
5.  **Brak Działania przy HOLD:** Sygnał `HOLD` z modelu jest sygnałem neutralnym – nie powoduje otwarcia ani zamknięcia żadnych pozycji.
