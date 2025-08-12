# Ostateczny Plan Przetwarzania Danych w Trybie Backtestu (Wersja 2.0 - Zgodna z Uściśleniami)

**CEL:** 100% Poprawność, Zero Przesunięć, Pełna Kontrola dzięki Precyzyjnemu, Dwuetapowemu Przycinaniu Danych.

---

### Faza 1: Konfiguracja Strategii

*   **Działanie:** W pliku strategii ustawiamy `startup_candle_count: int = 0`.
*   **Uzasadnienie:** Przejmujemy pełną, manualną kontrolę nad przygotowaniem danych i buforów. Freqtrade nie dodaje automatycznie żadnych dodatkowych świec.

---

### Faza 2: Działanie `DataFrameExtender` (Główna Logika)

Metoda `extend_and_calculate_features` będzie realizować następujące kroki:

1.  **Otrzymanie Czystych Danych:** Funkcja dostaje od Freqtrade "czystą" ramkę danych (`df_backtest`) zawierającą **tylko** okres backtestu.

2.  **Dodanie Pełnej Historii:** Na początku tej ramki doklejane jest `43200` świec z pliku `raw_validated.feather`.

3.  **Obliczenie Wskaźników Długoterminowych:** Na tej bardzo dużej, połączonej ramce obliczane są **tylko** wskaźniki wymagające długiej historii (np. `ma43200`, `volume_ma43200`).

4.  **KROK KLUCZOWY - PIERWSZE CIĘCIE (Logika Użytkownika):** Z ramki usuwane jest `43200 - 121 = 43079` pierwszych wierszy.
    *   **Rezultat:** Otrzymujemy **ramkę pośrednią** o wielkości `okres backtestu + 121 świec`. Ten jeden dodatkowy wiersz na początku jest niezbędny do poprawnego obliczenia cech (`pct_change`) dla pierwszej świecy z właściwego bufora.

5.  **Obliczenie Pozostałych Cech:** Na tej **ramce pośredniej** (`okres backtestu + 121`) obliczane są wszystkie pozostałe, krótkoterminowe cechy (`price_to_ma`, `pct_change` itd.). Pierwszy wiersz tej ramki poprawnie posłuży jako `t-1` dla drugiego wiersza.

6.  **KROK KLUCZOWY - DRUGIE CIĘCIE (Logika Użytkownika):** Po obliczeniu wszystkich cech, usuwamy pierwszy wiersz z ramki pośredniej.
    *   **Rezultat:** Otrzymujemy **ramkę finalną** o wielkości `okres backtestu + 120 świec bufora`.

7.  **Zwrot Wyniku:** Cała **ramka finalna** (`okres backtestu + 120`) jest zwracana do strategii.

---

### Faza 3: Przetwarzanie wewnątrz Freqtrade

*Proces pozostaje bez zmian, ale teraz opiera się na ramce przygotowanej nową, poprawną metodą.*

1.  **`populate_indicators`**: Wywołuje logikę z Fazy 2, a następnie `SignalGenerator`. Pierwsza predykcja jest poprawnie generowana dla wiersza o indeksie `119` (pierwsza świeca z okresu backtestu).
2.  **`populate_entry_trend`**: Tworzy sygnały. Dla pierwszych 119 wierszy bufora nie powstaną sygnały, ponieważ `ml_signal` będzie tam `NaN`.
3.  **`populate_exit_trend`**: Dodaje sygnały wyjścia.

---

### Faza 4: Ostateczna Analiza przez Silnik Freqtrade

*Bez zmian.* Freqtrade filtruje sygnały na podstawie daty z kolumny `date` i porównuje ją z `--timerange`, co stanowi ostateczne zabezpieczenie przed przetworzeniem sygnałów z bufora.
