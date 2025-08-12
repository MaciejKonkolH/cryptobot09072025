<!-- filepath: c:\Users\macie\OneDrive\Python\Binance\BinanceBot\memory-bank\documentation\general\binancebot_complete_description.md -->
# BinanceBot - Kompletny opis działania

*Data ostatniej aktualizacji: 29 kwietnia 2025*

## Czym jest BinanceBot?

BinanceBot to zaawansowany system automatycznego handlu kryptowalutami na giełdzie Binance, działający wyłącznie na rynku futures (kontraktów terminowych). Umożliwia zajmowanie zarówno pozycji długich (LONG - zysk przy wzroście ceny), jak i krótkich (SHORT - zysk przy spadku ceny) w oparciu o precyzyjne sygnały generowane przez modele uczenia maszynowego.

## Główny cel BinanceBota

Celem BinanceBota jest generowanie regularnych, drobnych zysków z krótkoterminowego handlu kryptowalutami poprzez otwieranie pozycji na kilka minut do kilku godzin, co pozwala zarabiać niezależnie od długofalowego trendu rynkowego.

## Jak działa BinanceBot?

### 1. Pozyskiwanie danych rynkowych
- Nieustannie (24/7) pobiera aktualne dane cenowe z giełdy Binance poprzez API WebSocket
- Przetwarza dane z różnych par walutowych jednocześnie (np. BTCUSDT, ETHUSDT)
- Śledzi dane w różnych interwałach czasowych (np. 15-minutowe, 1-godzinne)

### 2. Analiza przez model ML
- System wykorzystuje specjalnie wytrenowane modele uczenia maszynowego, specyficzne dla każdej pary walutowej
- Modele analizują dane historyczne ostatnich 90 świec (domyślnie, można konfigurować)
- Każdy model generuje konkretny sygnał transakcyjny: 0, 1 lub 2
  - 0 = Sygnał SHORT (otworzenie pozycji krótkiej)
  - 1 = Sygnał HOLD (brak akcji)
  - 2 = Sygnał LONG (otwarcie pozycji długiej)

### 3. Podejmowanie decyzji handlowych
- Po otrzymaniu sygnału 0 (SHORT) lub 2 (LONG), bot sprawdza warunki zarządzania ryzykiem:
  - Dostępny kapitał
  - Maksymalna liczba jednoczesnych pozycji
  - Limity ryzyka na parę walutową
- Jeśli wszystkie warunki są spełnione, bot przystępuje do otwarcia pozycji

### 4. Zarządzanie pozycją
- Po otwarciu pozycji, bot automatycznie ustawia dwa zlecenia:
  - Stop Loss (SL) - aby ograniczyć potencjalne straty (np. 1% od ceny wejścia)
  - Take Profit (TP) - aby zrealizować zyski po osiągnięciu określonego poziomu (np. 2% od ceny wejścia)
- Wartości SL i TP są konfigurowalne dla każdej pary walutowej
- Gdy jedno z tych zleceń zostanie zrealizowane, drugie jest automatycznie anulowane

### 5. Dokumentowanie transakcji
- Każda transakcja jest szczegółowo zapisywana:
  - Data i czas otwarcia/zamknięcia
  - Rodzaj pozycji (LONG/SHORT)
  - Cena wejścia/wyjścia
  - Wielkość pozycji
  - Zysk/strata
  - Poziomy SL i TP
- System prowadzi dziennik wszystkich operacji dla późniejszej analizy i optymalizacji

### 6. Zarządzanie kapitałem
- Bot automatycznie oblicza wielkość pozycji w oparciu o:
  - Dostępny kapitał
  - Ustawiony procent kapitału na transakcję (np. 5%)
  - Specyficzne ustawienia dla danej pary walutowej
- Pozwala to na utrzymanie dyscypliny inwestycyjnej i rozłożenie ryzyka

## Bezpieczeństwo i kontrola ryzyka

1. **Dźwignia finansowa**
   - System operuje na rynku futures z niską dźwignią (domyślnie 1x)
   - Wyższe dźwignie są możliwe, ale niezalecane ze względu na zwiększone ryzyko

2. **Automatyczne zabezpieczenia**
   - Każda pozycja ma zdefiniowany poziom maksymalnej akceptowalnej straty (Stop Loss)
   - System monitoruje stan połączenia z Binance i może zamknąć wszystkie pozycje w razie problemów

3. **Limity pozycji**
   - Ustalona maksymalna liczba jednocześnie otwartych pozycji
   - Limit pozycji na poszczególne pary walutowe
   - Procentowe ograniczenie zaangażowania kapitału

## Tryby działania

1. **Tryb testowy (Testnet)**
   - Łączenie z testowym środowiskiem Binance
   - Możliwość przetestowania strategii na wirtualnych środkach
   - Identyczne funkcjonalności jak w trybie produkcyjnym

2. **Tryb read-only**
   - Generowanie sygnałów bez faktycznego otwierania pozycji
   - Przydatny do weryfikacji działania modelu przed użyciem prawdziwych środków

3. **Tryb produkcyjny**
   - Pełna automatyzacja - otwieranie i zarządzanie pozycjami bez ingerencji człowieka
   - Ciągłe monitorowanie rynku i reagowanie na zmiany 24/7

## Konfiguracja i dostosowywanie

1. **Konfiguracja modeli**
   - Dla każdej pary walutowej można określić:
     - Model ML do użycia
     - Interwał czasowy (np. 15m, 1h, 4h)
     - Poziomy Stop Loss i Take Profit
     - Wielkość pozycji jako procent kapitału

2. **Konfiguracja zarządzania ryzykiem**
   - Maksymalna liczba równoczesnych pozycji
   - Maksymalny procent kapitału w użyciu
   - Warunki zabezpieczające (np. blokada handlu przy dużej zmienności)

## Weryfikacja skuteczności

1. **Testy historyczne (backtesting)**
   - Weryfikacja strategii na danych historycznych
   - Symulacja wyników bez ryzyka finansowego
   - Generowanie raportów skuteczności

2. **Testy krzyżowe w środowisku testowym**
   - Sprawdzenie działania w czasie rzeczywistym
   - Weryfikacja stabilności i niezawodności systemu

3. **Monitoring wyników**
   - Śledzenie skuteczności strategii w czasie rzeczywistym
   - Regularne raporty wyników

## Kluczowe różnice od tradycyjnych botów tradingowych

1. BinanceBot NIE wykorzystuje tradycyjnych wskaźników technicznych (RSI, MACD, itp.), lecz opiera się wyłącznie na modelach uczenia maszynowego.

2. System zawsze generuje konkretny sygnał (0, 1, 2), co przekłada się na jasne działania handlowe.

3. Działa wyłącznie na rynku futures, umożliwiając zarabianie zarówno na wzrostach, jak i spadkach.

4. Skupia się na krótkoterminowym handlu, generując małe ale częste zyski.

5. Każda para walutowa ma dedykowany model ML, specjalnie wytrenowany dla jej charakterystyki.

BinanceBot to narzędzie skierowane do osób, które chcą wykorzystać potencjał uczenia maszynowego w handlu kryptowalutami, eliminując czynnik emocjonalny i umożliwiając całodobowe śledzenie rynku z automatycznym reagowaniem na zmiany.

## Historia zmian

| Data | Wersja | Opis zmian |
|------|--------|------------|
| 2025-04-29 | 1.0 | Utworzenie kompletnej dokumentacji systemu BinanceBot |
