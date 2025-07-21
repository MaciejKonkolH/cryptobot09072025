# Algorytm Działania Strategii i Bufora

Dokument ten opisuje ostateczną, obowiązującą logikę przepływu danych dla trybu Backtest i Live.

---

## 1. Tryb Backtestingu (Testy Historyczne)

W tym trybie system operuje na pojedynczych świecach, symulując działanie krok po kroku.

### Strategia (Głównodowodzący)
- **Trigger:** Dostaje sygnał o jednej, nowej, zamkniętej świecy.
- **Zlecenie:** Wyciąga z tej świecy tylko jej **datę (timestamp)** i przekazuje ją do Bufora.
- **Rezultat:** Odbiera od Bufora gotową tabelkę z cechami **(120, 8)** i podaje ją do modelu.

### Bufor (Analityk Historyczny)
- **Wejście:** Otrzymuje od Strategii jedną **datę (timestamp)**.
- **Krok 1: Wczytanie Historii:** Na podstawie daty, wczytuje z pliku `.feather` **43,200+ historycznych świec**.
- **Krok 2: Obliczenie Długich Średnich:** Na całej wczytanej historii oblicza 4 wskaźniki: `MA43200`, `MA1440` i ich odpowiedniki dla wolumenu.
- **Krok 3: Przycięcie do Okna Pracy:** Zostawia tylko **ostatnie 121 świec**.
- **Krok 4: Obliczenie 8 Cech:** Na tych 121 świecach oblicza **8 kluczowych cech** (zmiany procentowe, stosunek ceny do średnich itp.).
- **Krok 5: Finalne Cięcie:** Usuwa najstarszą świecę.
- **Wyjście:** Zwraca do Strategii finalną tabelkę cech o wymiarach **(120, 8)**.

---

## 2. Tryb Live Tradingu (Działanie na Żywo)

Działanie w tym trybie jest dwuetapowe: najpierw jednorazowa synchronizacja, potem normalna pętla pracy.

### Etap 1: Inicjalizacja i Synchronizacja (Jednorazowo na starcie)

Ten proces zapewnia, że bot startuje z aktualnymi danymi i wypełnia ewentualną "lukę" w historii.

- **Krok 1: Wczytanie Historii z Pliku:** Bufor wczytuje do pamięci ostatnie 43,200 świec z pliku `.feather`.
- **Krok 2: Wykrycie "Luki":** Porównuje datę ostatniej świecy w pamięci z aktualnym czasem giełdy.
- **Krok 3: Wypełnienie "Luki":** Jeśli jest przerwa, bot łączy się z giełdą i pobiera wszystkie brakujące świece.
- **Krok 4: Aktualizacja Pamięci i Dysku:** Dołącza pobrane świece do bufora w pamięci i zapisuje całość z powrotem do pliku `.feather`, nadpisując go.
- **Krok 5: Zapis Daty:** Bufor zapamiętuje datę ostatniej, teraz już w pełni aktualnej świecy jako `ostatnia_zapisana_data`.

### Etap 2: Normalna Praca (Pętla co minutę)

Po udanej synchronizacji, bot przechodzi do standardowej pętli operacyjnej.

#### Strategia (Głównodowodzący)
- **Trigger:** Dostaje od Freqtrade paczkę **60 najnowszych, zamkniętych świec**.
- **Zlecenie:** Od razu wysyła całą paczkę 60 świec do Bufora.
- **Rezultat:** Odbiera od Bufora gotową tabelkę z cechami **(120, 8)** i podaje ją do modelu.

#### Bufor (Inteligentny Analityk Danych)
- **Wejście:** Otrzymuje od Strategii paczkę 60 świec.
- **Krok 1: Logika Zapisu na Dysk:**
    - Porównuje najstarszą datę z nowej paczki 60 świec z `ostatnia_zapisana_data`.
    - Jeśli różnica jest **mniejsza niż 30 minut**, oznacza to ciągłość pracy. Wtedy:
        1. Łączy nowe dane z historią w pamięci.
        2. Zapisuje całą zaktualizowaną historię do pliku `.feather` na dysku.
        3. Aktualizuje `ostatnia_zapisana_data` na nową, ostatnią datę.
- **Krok 2: Aktualizacja Pamięci:** Zawsze inteligentnie łączy nową paczkę z danymi w pamięci (usuwa duplikaty, dba o ciągłość).
- **Krok 3: Obliczenie Długich Średnich:** Na całej historii w pamięci oblicza 4 długie wskaźniki.
- **Krok 4: Przycięcie do Okna Pracy:** Zostawia tylko **ostatnie 121 świec**.
- **Krok 5: Obliczenie 8 Cech:** Na tych 121 świecach oblicza **8 kluczowych cech**.
- **Krok 6: Finalne Cięcie:** Usuwa najstarszą świecę.
- **Wyjście:** Zwraca do Strategii finalną tabelkę cech o wymiarach **(120, 8)**.
