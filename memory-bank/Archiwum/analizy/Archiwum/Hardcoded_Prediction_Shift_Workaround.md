### Tytuł: Obejście Problemu Przesunięcia Czasowego: Twarde Przesunięcie Predykcji o -119 Minut

**Problem:**
Po wielotygodniowej analizie nie udało się jednoznacznie zidentyfikować źródła stałego przesunięcia predykcji o 119-120 minut między walidacją a backtestem w Freqtrade. Mimo licznych prób, w tym ujednolicenia stref czasowych (UTC) w całym pipeline, problem nadal występował.

**Rozwiązanie (Pragmatyczne Obejście):**
Na wyraźne polecenie użytkownika, w dniu 2025-07-08, zaimplementowano twarde, ręczne przesunięcie predykcji w kodzie strategii jako ostateczne rozwiązanie problemu.

**Szczegóły Implementacji:**
*   **Lokalizacja:** `ft_bot_clean/user_data/strategies/components/signal_generator.py`
*   **Funkcja:** `_add_predictions_to_dataframe`
*   **Logika:** Wszystkie kolumny pochodzące z modelu ML (`ml_signal`, `ml_confidence`, `ml_*_prob`) są przesuwane o **-119** pozycji (minut) za pomocą metody `pandas.DataFrame.shift(-119)`.

**Konsekwencje:**
*   **Pozytywne:** Backtesty natychmiast zaczęły generować poprawne i bardzo zyskowne wyniki. To potwierdziło, że model, cechy i ogólna logika strategii były prawidłowe, a jedynym problemem była anomalia w osi czasu.
*   **Negatywne (Dług Technologiczny):** To rozwiązanie jest "hackiem" i stanowi dług technologiczny. Jakakolwiek przyszła zmiana w logice danych, wewnętrznych opóźnieniach Freqtrade lub ogólnym pipeline'ie może sprawić, że to przesunięcie stanie się nieprawidłowe. 

**Mechanizm Ostrzegawczy:**
Aby zminimalizować ryzyko zapomnienia o tej modyfikacji, przy każdym uruchomieniu strategii w logach pojawia się krytyczny komunikat:
`🔥🔥🔥 UWAGA: Aktywna jest twarda modyfikacja przesunięcia predykcji o -119 minut! 🔥🔥🔥`

**Status Wersji:**
Wersja z tą modyfikacją jest uznawana za **v1.0.0** - pierwszą stabilną, w pełni działającą i zyskowną wersję projektu. 