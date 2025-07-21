# 🎯 **Wizja Kompletnej Strategii: Multi-Pair, Multi-Position, Model-Driven**

Celem jest stworzenie w pełni zautomatyzowanej strategii, która potrafi handlować na **wielu parach walutowych** jednocześnie, używając **dedykowanego modelu ML dla każdej pary**. Strategia musi być w stanie otwierać **wiele niezależnych pozycji** na tej samej parze (tzw. "position stacking"). Absolutnie kluczowe jest to, aby cena wejścia w pozycję była precyzyjnie kontrolowana, co gwarantuje 100% spójność z danymi treningowymi modelu.

## **Krok 1: Inicjalizacja i System Multi-Pair (Przy starcie bota)**

1.  **Centralna Konfiguracja (`pair_config.json`):**
    *   System startuje od wczytania pliku konfiguracyjnego, który definiuje, które pary są aktywne (np. `BTC/USDT`, `ETH/USDT`).
    *   Dla każdej aktywnej pary, plik ten wskazuje na katalog z jej unikalnymi "artefaktami" (np. `user_data/strategies/inputs/BTCUSDT/`).

2.  **Dynamiczne Ładowanie Modeli (`ModelLoader`):**
    *   Dla każdej **aktywnej pary** z konfiguracji, system automatycznie ładuje jej dedykowane komponenty:
        *   `best_model.h5` (model neuronowy)
        *   `scaler.pkl` (skaler do normalizacji cech)
        *   `metadata.json` (metadane, np. `window_size`, które określa długość sekwencji, np. 120)
    *   Wszystkie załadowane modele i skalery są przechowywane w pamięci (w "cache"), aby uniknąć ich ponownego wczytywania przy każdej świecy.

3.  **Zarządzanie Parami (`PairManager`):**
    *   Jeśli model dla jakiejś pary nie zostanie wczytany (np. z powodu braku pliku), para ta jest oznaczana jako "nieaktywna", a bot nie będzie na niej handlował. Bot kontynuuje pracę z pozostałymi, poprawnie skonfigurowanymi parami.

## **Krok 2: Cykl Życia Predykcji (Dla każdej nowej świecy)**

1.  **Pobranie Danych i Bufor:**
    *   Freqtrade dostarcza nową świecę dla konkretnej pary (np. `BTC/USDT`).
    *   System bufora (`MA43200 Buffer`) rozszerza te dane o historię, aby umożliwić poprawne obliczenie długoterminowych średnich kroczących.

2.  **Obliczenie Cech i Predykcja:**
    *   Na pełnym, rozszerzonym zbiorze danych obliczane jest 8 cech wejściowych dla modelu (takich jak `high_change`, `price_to_ma43200` itp.).
    *   System pobiera z pamięci **model i skaler specyficzny dla tej pary**.
    *   Cechy są skalowane, a następnie tworzona jest **sekwencja 120 ostatnich świec** (długość z `metadata.json`).
    *   Sekwencja `[świeca_1, świeca_2, ..., świeca_120]` jest przekazywana do modelu, który zwraca prawdopodobieństwa dla trzech klas: `[SHORT, HOLD, LONG]`.

## **Krok 3: Logika Wejścia i Precyzyjne Otwieranie Pozycji**

1.  **Sygnał Wejścia:**
    *   Gdy prawdopodobieństwo dla `LONG` lub `SHORT` przekroczy zdefiniowany próg, strategia generuje sygnał wejścia (`enter_long=1` lub `enter_short=1`).

2.  **Kluczowa Logika Ceny Wejścia (`custom_entry_price`):**
    *   **To jest najważniejszy element gwarantujący spójność z treningiem.**
    *   W momencie generowania sygnału, Freqtrade pyta strategię: "Po jakiej cenie mam otworzyć pozycję?".
    *   Strategia **MUSI** zaimplementować funkcję `custom_entry_price`.
    *   Funkcja ta zagląda do danych i zwraca **cenę zamknięcia (`close`) ostatniej świecy w 120-elementowej sekwencji użytej do predykcji**. Innymi słowy, jest to cena zamknięcia świecy na indeksie `t-1`, gdzie predykcja jest robiona na początku świecy `t`.
    *   Freqtrade używa tej konkretnej ceny do złożenia zlecenia `LIMIT`.

## **Krok 4: Obsługa Wielu Pozycji na Jednej Parze (Position Stacking)**

1.  **Konfiguracja Freqtrade:**
    *   W głównym pliku `config.json` parametr `max_open_trades` musi być ustawiony na wartość większą niż liczba par, lub na `-1` (nieskończoność).
    *   W strategii musi być ustawione: `position_adjustment_enable = True`.

2.  **Jak to działa w praktyce:**
    *   Strategia jest na pozycji `LONG` dla `BTC/USDT`.
    *   Po kilku godzinach pojawia się kolejny, silny sygnał `LONG`.
    *   Dzięki `position_adjustment_enable = True`, strategia może zasygnalizować `enter_long=1` **ponownie**.
    *   Freqtrade otworzy **drugą, niezależną pozycję** `LONG` na `BTC/USDT`. Każda pozycja będzie miała własną cenę wejścia (zgodną z `custom_entry_price` w momencie jej otwarcia), własny stop-loss i take-profit.

## **Krok 5: Zarządzanie Niewypełnionymi Zleceniami Wejścia (Timeout)**

1.  **Problem Starych Zleceń Wejścia:**
    *   Zlecenie `LIMIT` na otwarcie pozycji, złożone po cenie `close` poprzedniej świecy, może nigdy nie zostać zrealizowane, jeśli cena rynkowa natychmiast odjedzie w przeciwnym kierunku.
    *   Takie "wiszące" zlecenie wejścia blokuje kapitał i uniemożliwia wykorzystanie innych okazji.

2.  **Rozwiązanie: Wbudowany Timeout w Freqtrade:**
    *   Strategia wykorzystuje natywną funkcjonalność Freqtrade do automatycznego anulowania niezrealizowanych zleceń wejścia po upływie określonego czasu.

3.  **Konfiguracja (`config.json`):**
    *   W głównym pliku `config.json` sekcja `unfilledtimeout` pozwala zdefiniować czas oczekiwania.

    ```json
    "unfilledtimeout": {
        "entry": 5
    }
    ```
    *   **`"entry": 5`**: Oznacza, że jeśli zlecenie wejścia w pozycję nie zostanie zrealizowane w ciągu **5 minut**, Freqtrade automatycznie je anuluje. Wartość `5` jest tu przykładem i może być dowolnie skonfigurowana.

## **Krok 6: Zarządzanie Pozycją: Natychmiastowy Stop-Loss i Take-Profit (Logika OCO)**

1.  **Cel: Natychmiastowe Zabezpieczenie Pozycji**
    *   Zamiast czekać, aż bot w kolejnym cyklu zdecyduje o zamknięciu pozycji, celem jest, aby zlecenia `STOP_LOSS` i `TAKE_PROFIT` zostały wysłane na giełdę **natychmiast** po otwarciu transakcji.
    *   Daje to maksymalne bezpieczeństwo i szybkość reakcji, niezależnie od ewentualnych opóźnień w działaniu bota.

2.  **Problem: Brak Natywnego OCO i Ryzyko "Wiszących Zleceń"**
    *   Standardowe API Binance nie pozwala na proste powiązanie dwóch oddzielnie złożonych zleceń (Stop Loss i Take Profit) w jedno zlecenie typu OCO (One-Cancels-the-Other).
    *   **Ryzyko:** Gdybyśmy po prostu wysłali na giełdę zlecenie Stop Loss, a następnie Take Profit, zrealizowanie jednego z nich **nie anulowałoby drugiego**. Jeśli Stop Loss zostałby aktywowany, zlecenie Take Profit nadal pozostałoby aktywne, co mogłoby prowadzić do otwarcia nowej, niezamierzonej pozycji.

3.  **Rozwiązanie: Aktywne Zarządzanie Zleceniami przez Strategię**
    *   Strategia implementuje w pełni bezpieczną, ręczną symulację zlecenia OCO, wykorzystując do tego dwie specjalne funkcje Freqtrade:

    *   **a) Umieszczenie Zleceń (`confirm_trade_entry`):**
        *   Ta funkcja jest wywoływana **dokładnie raz**, tuż po tym, jak pozycja zostanie otwarta.
        *   **Stop Loss:** Standardowa logika Freqtrade (`stoploss_on_exchange=True`) natychmiast wysyła na giełdę zlecenie `STOP`.
        *   **Take Profit:** Niestandardowa logika w `confirm_trade_entry` oblicza cenę docelową i **bezpośrednio wysyła na giełdę zlecenie `LIMIT`**, które ma zrealizować zysk.
        *   **Kluczowy element:** ID tego zlecenia Take Profit jest zapisywane w pamięci transakcji (`trade.custom_info`), aby bot wiedział, które zlecenie "należy do niego".

    *   **b) Anulowanie Zleceń (`custom_exit`):**
        *   Ta funkcja jest wywoływana, gdy transakcja jest zamykana **z dowolnego powodu** (np. przez Stop Loss).
        *   Jej zadaniem jest posprzątanie. Logika sprawdza, czy istnieje zapisane ID zlecenia Take Profit.
        *   Jeśli tak, strategia **wysyła na giełdę polecenie anulowania tego konkretnego zlecenia `LIMIT`**, zanim transakcja zostanie finalnie zamknięta.

4.  **Konfiguracja i Deaktywacja Standardowego ROI:**
    *   Aby uniknąć konfliktu z wbudowanym mechanizmem Freqtrade, standardowy parametr `minimal_roi` jest deaktywowany poprzez ustawienie go na bardzo wysoką, nierealną wartość. Całą logikę realizacji zysku przejmuje niestandardowe rozwiązanie.
    *   Z tego samego powodu, w pliku `config.json` **celowo usuwamy parametr `"exit"`** z sekcji `unfilledtimeout`. Nasze zlecenie Take Profit ma czekać na realizację bez limitu czasowego, a ten parametr mógłby je niepotrzebnie anulować.

## **Krok 7: Przykładowe Scenariusze**

### **Scenariusz 1: Otwarcie Pozycji i Zabezpieczenie OCO**

1.  **Godzina 10:00:** Kończy się świeca na indeksie `t-1`. Jej cena zamknięcia `close` to **$50,000**. Model generuje sygnał `LONG` z TP na 1% i SL na 0.5%.
2.  **Godzina 10:01:** Zaczyna się świeca `t`.
    *   Strategia ustawia `enter_long = 1`.
    *   `custom_entry_price` zwraca cenę **$50,000**. Freqtrade składa zlecenie `LIMIT BUY`.
3.  **Godzina 10:02:** Zlecenie `LIMIT BUY` zostaje wypełnione. Pozycja jest otwarta.
4.  **Natychmiast (`confirm_trade_entry`):**
    *   Freqtrade (standardowo) wysyła na giełdę zlecenie `STOP LOSS` na cenę $49,750.
    *   Strategia (niestandardowo) wysyła na giełdę zlecenie `TAKE PROFIT` (typu `LIMIT`) na cenę $50,500 i zapisuje jego ID.
    *   Na giełdzie są teraz dwa aktywne zlecenia chroniące pozycję.
5.  **Przypadek A (Take Profit):** Cena rośnie do $50,500. Zlecenie `LIMIT` jest realizowane, pozycja zamknięta z zyskiem. Bot wykrywa, że pozycja zniknęła z konta i aktualizuje swój stan.
6.  **Przypadek B (Stop Loss):** Cena spada do $49,750. Funkcja `custom_exit` jest wywoływana, **anuluje wiszące zlecenie `TAKE PROFIT`** na $50,500. Następnie Freqtrade zamyka pozycję przez `STOP LOSS`. Nie ma żadnych "wiszących" zleceń.

### **Scenariusz 2: Niewypełnione Zlecenie Wejścia (Timeout)**

1.  **Godzina 11:00:** Model generuje kolejny sygnał `LONG`. Cena wejścia to **$51,000**. Freqtrade składa zlecenie `LIMIT BUY`.
2.  Cena rynkowa natychmiast rośnie do $51,010 i nie wraca do poziomu zlecenia.
3.  **Godzina 11:05:** Mija 5 minut (zgodnie z `unfilledtimeout.entry: 5`).
4.  Zlecenie `LIMIT BUY` wciąż jest niezrealizowane.
5.  Freqtrade, zgodnie z konfiguracją, automatycznie **anuluje zlecenie**. Kapitał zostaje odblokowany i jest gotowy na kolejny sygnał.

### **Scenariusz 3: Otwarcie Drugiej Pozycji (Position Stacking)**

1.  Pozycja ze Scenariusza 1 jest wciąż otwarta.
2.  **Godzina 13:30:** Model ponownie generuje silny sygnał `LONG` przy cenie wejścia **$52,000**.
3.  Dzięki `position_adjustment_enable = True`, strategia może otworzyć kolejną transakcję.
4.  Freqtrade składa nowe zlecenie `LIMIT BUY` na **$52,000**.
5.  Po wypełnieniu, Freqtrade otwiera **drugą, niezależną pozycję** na tej samej parze. Ta nowa pozycja ma własną cenę wejścia, własny Stop Loss i własny Take Profit (zarządzane logiką OCO).

W ten sposób system jest w 100% spójny z logiką treningu, a jednocześnie elastyczny i bezpieczny, pozwalając na zaawansowane zarządzanie pozycjami na wielu rynkach.
