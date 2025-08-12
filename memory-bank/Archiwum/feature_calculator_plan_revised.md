# PLAN PRAC - WERSJA POPRAWIONA: MODUŁ FEATURE_CALCULATOR_OHLC_SNAPSHOT

## 🎯 CEL PROJEKTU
Stworzenie nowego modułu do obliczania zaawansowanych cech na podstawie połączonych danych OHLC + Orderbook.

## ⚠️ KRYTYCZNA OCENA ORYGINALNEGO PLANU

### PROBLEMY ZIDENTYFIKOWANE:
1. **Zbyt ambitny zakres:** 75 cech to za dużo na raz
2. **Nierealistyczny czas:** 3-4 tygodnie to za mało
3. **Brak priorytetyzacji:** Wszystkie cechy traktowane równo
4. **Ryzyko wydajności:** 1.3M wierszy × 75 cech = problemy z pamięcią
5. **Złożoność testowania:** Trudno przetestować 75 cech jednocześnie

## 🚀 POPRAWIONY PLAN - WERSJA SKRÓCONA

### FAZA 1: MINIMALNA WERSJA DZIAŁAJĄCA (1 tydzień)
**CEL:** Stworzenie podstawowej wersji z 30-35 najważniejszymi cechami

#### 1.1 Infrastruktura (2 dni)
- Struktura katalogów
- Konfiguracja podstawowa
- System logowania
- Obsługa błędów

#### 1.2 Cechy OHLC - Podstawowe (2 dni)
- **Tradycyjne wskaźniki:** Wstęgi Bollingera, RSI, MACD, ADX
- **Średnie kroczące:** 1h, 4h, 1d
- **Cechy cenowe:** Price-to-MA ratios, candle patterns
- **ŁĄCZNIE:** 15 cech OHLC

#### 1.3 Cechy Orderbook - Podstawowe (2 dni)
- **Cechy podstawowe:** Buy/sell ratios, imbalances
- **Cechy głębokości:** TP/SL depths, ratios
- **Cechy dynamiczne:** Total depth/notional changes
- **ŁĄCZNIE:** 12 cech Orderbook

#### 1.4 Cechy Hybrydowe - Podstawowe (1 dzień)
- **Korelacje:** Price-orderbook, volume-depth
- **Cechy czasowe:** Hour of day patterns
- **ŁĄCZNIE:** 5 cech Hybrydowe

**WYNIK FAZY 1:** 32 cechy, działający moduł

### FAZA 2: ROZSZERZENIE (1 tydzień)
**CEL:** Dodanie kolejnych 20-25 cech

#### 2.1 Dodatkowe wskaźniki OHLC (3 dni)
- Stochastic, Williams %R, CCI, ROC
- ATR, Historical volatility
- **ŁĄCZNIE:** +8 cech OHLC

#### 2.2 Zaawansowane cechy Orderbook (3 dni)
- Order flow imbalance, large order detection
- Depth trends, orderbook volatility
- **ŁĄCZNIE:** +8 cech Orderbook

#### 2.3 Dodatkowe cechy hybrydowe (1 dzień)
- Z-scores, outlier detection
- **ŁĄCZNIE:** +4 cechy Hybrydowe

**WYNIK FAZY 2:** 52 cechy

### FAZA 3: OPTYMALIZACJA I TESTOWANIE (3-5 dni)
- Testy funkcjonalne
- Optymalizacja wydajności
- Dokumentacja
- Walidacja wyników

## 📋 FINALNE PODSUMOWANIE

### **WERSJA MINIMALNA (FAZA 1):**
- **OHLC:** 15 cech
- **Orderbook:** 12 cech  
- **Hybrydowe:** 5 cech
- **ŁĄCZNIE:** 32 cechy

### **WERSJA ROZSZERZONA (FAZA 2):**
- **OHLC:** +8 cech = 23 cechy
- **Orderbook:** +8 cech = 20 cech
- **Hybrydowe:** +4 cechy = 9 cech
- **ŁĄCZNIE:** 52 cechy

## ⏱️ REALISTYCZNY CZAS REALIZACJI
- **Faza 1:** 1 tydzień (5 dni roboczych)
- **Faza 2:** 1 tydzień (5 dni roboczych)  
- **Faza 3:** 3-5 dni
- **ŁĄCZNY CZAS:** 2.5-3 tygodnie

## 🎯 NOWE PRIORYTETY
1. **WYSOKI:** Faza 1 (minimalna wersja działająca)
2. **ŚREDNI:** Faza 2 (rozszerzenie)
3. **NISKI:** Dodatkowe zaawansowane cechy (w przyszłości)

## ✅ ZALETY POPRAWIONEGO PLANU
1. **Realistyczny zakres:** 52 cechy zamiast 75
2. **Iteracyjny rozwój:** Działająca wersja po 1 tygodniu
3. **Lepsze testowanie:** Mniej cech = łatwiejsze testy
4. **Mniejsze ryzyko:** Mniej złożoności = mniej błędów
5. **Szybszy feedback:** Możliwość testowania wcześniej

## 🔄 STRATEGIA ROZWOJU
1. **MVP first:** Minimalna wersja działająca
2. **Test early:** Sprawdzanie każdej fazy
3. **Iterate:** Rozszerzanie na podstawie feedbacku
4. **Optimize:** Optymalizacja po każdej fazie

## ⚠️ WAŻNE UWAGI TECHNICZNE

### **OKRES ROZGRZEWANIA (WARMUP PERIOD):**
- **MA_43200 (30-dniowa średnia)** wymaga 30 dni danych do rozgrzania
- **Pierwszy miesiąc** (30 dni) służy tylko do obliczeń
- **Po obliczeniach** pierwszy miesiąc jest obcinany z wyników
- **Efekt:** Utrata 30 dni danych, ale poprawa jakości cech

### **ZAKRES DANYCH:**
- **Dane wejściowe:** 2023-01-01 do 2025-06-30 (2.5 roku)
- **Dane wyjściowe:** 2023-02-01 do 2025-06-30 (2.4 roku po obcięciu)
- **Utrata:** ~30 dni (1 miesiąc) na początku

### **IMPLEMENTACJA:**
- Wszystkie cechy z oknami czasowymi > 30 dni muszą uwzględniać warmup
- Funkcja `trim_warmup_period()` do obcinania pierwszego miesiąca
- Logowanie informacji o utracie danych 