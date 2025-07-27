# PLAN PRAC: MODUŁ FEATURE_CALCULATOR_OHLC_SNAPSHOT

## 🎯 CEL PROJEKTU
Stworzenie nowego modułu do obliczania zaawansowanych cech na podstawie połączonych danych OHLC + Orderbook, który zastąpi obecny `feature_calculator_snapshot`.

## 📊 ANALIZA OBECNEGO STANU

### Obecny moduł `feature_calculator_snapshot`:
- **50 cech** (20 OHLC + 30 Orderbook)
- **Struktura:** 7 grup cech orderbook + tradycyjne wskaźniki OHLC
- **Format danych:** Feather z kolumnami `snapshot1_*` i `snapshot2_*`
- **Zakres czasowy:** 2.5 roku danych (2023-2025)

### Nowe dane wejściowe:
- **Plik:** `merged_ohlc_orderbook.feather` (410MB, 1.3M wierszy)
- **Kolumny:** 58 (5 OHLC + 42 Orderbook + 11 inne)
- **Format:** Ciągłe dane bez luk czasowych

## 🚀 PLAN ROZWOJU NOWEGO MODUŁU

### FAZA 1: PODSTAWOWA INFRASTRUKTURA (1-2 dni)
1. **Struktura katalogów:**
   ```
   feature_calculator_ohlc_snapshot/
   ├── main.py              # Główna logika
   ├── config.py            # Konfiguracja
   ├── feature_groups/      # Moduły cech
   │   ├── __init__.py
   │   ├── ohlc_features.py
   │   ├── orderbook_features.py
   │   └── hybrid_features.py
   ├── utils/               # Narzędzia pomocnicze
   │   ├── __init__.py
   │   ├── indicators.py
   │   └── statistics.py
   ├── output/              # Wyniki
   └── requirements.txt
   ```

2. **Konfiguracja podstawowa:**
   - Ścieżki do plików wejściowych/wyjściowych
   - Parametry obliczeń (okna czasowe, okresy)
   - System logowania
   - Obsługa błędów

### FAZA 2: CECHY OHLC - PODSTAWOWE (2-3 dni)
1. **Tradycyjne wskaźniki techniczne:**
   - Wstęgi Bollingera (3 cechy)
   - RSI, MACD (2 cechy)
   - ADX, Choppiness Index (2 cechy)
   - Średnie kroczące (4 cechy: 1h, 4h, 1d, 30d)

2. **Cechy cenowe:**
   - Price-to-MA ratios (4 cechy)
   - Candle patterns (5 cech: doji, hammer, engulfing, pin bar, inside/outside)
   - Wick ratios (2 cechy)

3. **Cechy wolumenu:**
   - Volume change (1 cecha)
   - Volume momentum (1 cecha)

**ŁĄCZNIE FAZA 2: 24 cechy OHLC**

### FAZA 3: CECHY ORDERBOOK - PODSTAWOWE (3-4 dni)
1. **Cechy podstawowe (z 2 snapshotów):**
   - Buy/sell ratios (2 cechy)
   - Imbalances (2 cechy)
   - Pressure change (1 cecha)

2. **Cechy głębokości:**
   - TP/SL depths (3 cechy)
   - TP/SL ratios (2 cechy)

3. **Cechy dynamiczne:**
   - Total depth change (1 cecha)
   - Total notional change (1 cecha)

4. **Cechy historyczne:**
   - Depth trends (3 cechy)
   - Orderbook volatility (1 cecha)
   - Pressure trends (2 cechy)

5. **Cechy korelacyjne:**
   - Depth-price correlation (1 cecha)
   - Pressure-volume correlation (1 cecha)

**ŁĄCZNIE FAZA 3: 18 cech Orderbook**

### FAZA 4: CECHY HYBRYDOWE (2-3 dni)
1. **Cechy OHLC + Orderbook:**
   - Price-orderbook divergence (1 cecha)
   - Volume-depth correlation (1 cecha)
   - Momentum-liquidity ratio (1 cecha)

2. **Cechy czasowe:**
   - Hour of day patterns (1 cecha)
   - Day of week patterns (1 cecha)

3. **Cechy statystyczne:**
   - Z-scores dla kluczowych metryk (3 cechy)
   - Outlier detection (2 cechy)

**ŁĄCZNIE FAZA 4: 10 cech Hybrydowe**

### FAZA 5: ZAAWANSOWANE CECHY OHLC (3-4 dni)
1. **Dodatkowe wskaźniki techniczne:**
   - Stochastic Oscillator (3 cechy)
   - Williams %R (1 cecha)
   - CCI (1 cecha)
   - ROC (1 cecha)
   - MFI (1 cecha)
   - OBV (1 cecha)
   - VWAP (1 cecha)

2. **Cechy zmienności:**
   - ATR (1 cecha)
   - Historical volatility (1 cecha)
   - Volatility ratio (1 cecha)

**ŁĄCZNIE FAZA 5: 11 cech OHLC**

### FAZA 6: ZAAWANSOWANE CECHY ORDERBOOK (4-5 dni)
1. **Cechy mikrostruktury:**
   - Order flow imbalance (1 cecha)
   - Large order detection (1 cecha)
   - Order book resilience (1 cecha)
   - Market impact estimation (1 cecha)
   - Liquidity score (1 cecha)

2. **Cechy koncentracji:**
   - Herfindahl-Hirschman Index (1 cecha)
   - Gini coefficient (1 cecha)
   - Entropy measures (1 cecha)

3. **Cechy dynamiki:**
   - Order book velocity (1 cecha)
   - Spread dynamics (1 cecha)
   - Mid-price volatility (1 cecha)

**ŁĄCZNIE FAZA 6: 12 cech Orderbook**

### FAZA 7: TESTOWANIE I OPTYMALIZACJA (2-3 dni)
1. **Testy funkcjonalne:**
   - Sprawdzenie wszystkich cech
   - Walidacja obliczeń
   - Testy na próbce danych

2. **Optymalizacja wydajności:**
   - Profilowanie kodu
   - Optymalizacja obliczeń
   - Zarządzanie pamięcią

3. **Dokumentacja:**
   - Opis wszystkich cech
   - Przykłady użycia
   - Instrukcje instalacji

## 📋 PODSUMOWANIE CECH

### **FAZA 2-3 (PODSTAWOWE):**
- **OHLC:** 24 cechy
- **Orderbook:** 18 cech
- **Hybrydowe:** 10 cech
- **ŁĄCZNIE:** 52 cechy

### **FAZA 5-6 (ZAAWANSOWANE):**
- **OHLC:** +11 cech = 35 cech
- **Orderbook:** +12 cech = 30 cech
- **Hybrydowe:** 10 cech
- **ŁĄCZNIE:** 75 cech

## ⏱️ SZACUNKOWY CZAS REALIZACJI
- **Faza 1:** 1-2 dni
- **Faza 2:** 2-3 dni
- **Faza 3:** 3-4 dni
- **Faza 4:** 2-3 dni
- **Faza 5:** 3-4 dni
- **Faza 6:** 4-5 dni
- **Faza 7:** 2-3 dni

**ŁĄCZNY CZAS:** 17-24 dni (3-4 tygodnie)

## 🎯 PRIORYTETY ROZWOJU
1. **WYSOKI:** Faza 1-3 (podstawowa funkcjonalność)
2. **ŚREDNI:** Faza 4 (cechy hybrydowe)
3. **NISKI:** Faza 5-6 (zaawansowane cechy)

## ⚠️ POTENCJALNE WYZWANIA
1. **Wydajność:** 75 cech może być wolne do obliczenia
2. **Pamięć:** Duże dane mogą wymagać optymalizacji
3. **Złożoność:** Wiele zależności między cechami
4. **Testowanie:** Trudność w walidacji wszystkich cech

## 🔄 STRATEGIA ROZWOJU
1. **Iteracyjny rozwój:** Faza po fazie
2. **Testowanie na próbkach:** Sprawdzanie każdej fazy
3. **Optymalizacja ciągła:** Poprawki wydajności
4. **Dokumentacja na bieżąco:** Opisywanie każdej cechy 