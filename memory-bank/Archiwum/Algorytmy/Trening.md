### Algorytm Modułu Treningowego (`trainer`) - Wersja Finalna

**Cel Główny:** Stworzyć w pełni konfigurowalny, powtarzalny i solidny proces treningowy, który na podstawie historycznych cech wytrenuje model sekwencyjny, uczciwie go oceni i zapisze wszystkie niezbędne artefakty do dalszej analizy i użytku.

---

**Krok 1: Wczytanie i Wstępne Przetworzenie Danych**
*   Moduł wczytuje plik z cechami (`..._features.feather`).
*   **Filtrowanie po Dacie (Opcjonalne):** Na podstawie ustawień w konfiguracji, dane mogą być filtrowane do określonego zakresu dat (`START_DATE`, `END_DATE`).
*   **Walidacja:** Skrypt oblicza i wyświetla rozkład etykiet (np. % `HOLD`, `LONG`, `SHORT`), aby potwierdzić jakość danych wejściowych.

**Krok 2: Chronologiczny Podział na Zbiory**
*   Dane (cała tabela) są dzielone chronologicznie na trzy osobne zbiory:
    1.  **Zbiór Treningowy** (np. pierwsze 70%)
    2.  **Zbiór Walidacyjny** (np. następne 15%)
    3.  **Zbiór Testowy** (np. ostatnie 15%)

**Krok 3: Skalowanie Cech (Kluczowy Krok)**
*   Tworzony jest `RobustScaler`.
*   Scaler jest "uczony" (`fit`) **tylko i wyłącznie na cechach ze zbioru treningowego**. To zapobiega przeciekowi informacji z przyszłości.
*   Za pomocą nauczonego scalera, transformowane (`transform`) są wszystkie trzy zbiory danych.
*   Nauczony obiekt scalera jest **zapisywany do pliku** (`scaler.pkl`), aby można go było użyć w strategii na żywo.

**Krok 4: Przygotowanie Generatorów Danych**
*   Na podstawie jednej, uniwersalnej klasy `SequenceGenerator`, tworzone są trzy instancje, po jednej dla każdego (już przeskalowanego) zbioru danych.
*   Każdy generator operuje na swojej części danych i wie, jakie są dla niego prawidłowe "cele" predykcji. Pierwszym możliwym celem jest indeks `119`.

**Krok 5: Produkcja Sekwencji "W Locie"**
*   Logika pozostaje bez zmian: generator dla celu `X` dostarcza sekwencję cech od `X-119` do `X` i etykietę z wiersza `X`.

**Krok 6: Konfiguracja i Trening Modelu**
*   **Architektura Modelu:** Na podstawie konfiguracji budowany jest model LSTM, odzwierciedlający sprawdzoną architekturę (np. 3 warstwy LSTM, 2 warstwy Dense, Dropout).
*   **Callbacki:** Przygotowywana jest pełna lista "asystentów" treningu:
    *   `BalancedUndersamplingCallback`: Przed każdą epoką tasuje i balansuje zbiór treningowy.
    *   `ReduceLROnPlateau`: Automatycznie zmniejsza tempo uczenia, gdy postęp zwalnia.
    *   `EarlyStopping`: Przerywa trening, jeśli nie ma postępu, zapobiegając stracie czasu.
    *   `ModelCheckpoint`: Zapisuje najlepszą wersję modelu w trakcie treningu.
*   **Trening:** Uruchamiana jest funkcja `model.fit()`, która wykorzystuje generatory i callbacki do przeprowadzenia inteligentnego procesu uczenia.

**Krok 7: Ostateczny Egzamin i Analiza Predykcji**
*   Po treningu, najlepsza wersja modelu jest wczytywana.
*   Model jest oceniany na **zbiorze testowym** (serwowanym chronologicznie), aby uzyskać finalne metryki (dokładność, precyzja, etc.).
*   **Zapis Predykcji:** Model generuje predykcje dla całego zbioru testowego. Wynik (prawdziwa etykieta, przewidziana etykieta, prawdopodobieństwa dla każdej klasy) jest **zapisywany do pliku .csv** w celu dalszej, dogłębnej analizy.

**Krok 8: Zapisanie Wszystkich Artefaktów**
*   Moduł zapisuje kompletny zestaw wyników pracy:
    1.  **Plik Modelu** (`model.h5`): Najlepsza wersja wytrenowanego modelu.
    2.  **Plik Scalera** (`scaler.pkl`): Nauczony scaler, niezbędny do strategii.
    3.  **Raport Tekstowy**: Podsumowanie metryk z egzaminu.
    4.  **Raport z Predykcjami** (`predictions.csv`): Plik CSV z surowymi predykcjami.
    5.  **Plik Metadanych** (`metadata.json`): "Akt urodzenia" modelu, zawierający wszystkie użyte parametry konfiguracyjne.
