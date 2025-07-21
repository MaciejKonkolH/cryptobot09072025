# 🔄 RESET PROJEKTU - KOMPLETNE INFORMACJE

**Data utworzenia:** 25.05.2025  
**Cel:** Dokumentacja umożliwiająca kontynuację prac przez nowego agenta AI

---

## 🎯 **KONTEKST PROJEKTU**

### **GŁÓWNY CEL:**
Freqtrade ML Trading Bot z Enhanced Strategy wymagającą MA43200 (30 dni danych 1-minutowych). System musi trenować modele LSTM do predykcji kierunku ruchu cen BTC/USDT.

### **KLUCZOWY PROBLEM:**
- **Oryginalny problem:** Freqtrade wymaga startup_candle_count: 43300, ale Binance API ogranicza pobieranie do ~7-14 dni
- **Rozwiązanie:** Zbudowano kompletny system bufora MA43200 z zewnętrznymi źródłami danych
- **Nowy problem:** Model trenuje się z accuracy ~1.3% z powodu błędnego algorytmu symulacji giełdy

---

## 📁 **STRUKTURA PROJEKTU**

### **GŁÓWNE KATALOGI:**
```
C:\Users\macie\OneDrive\Python\Binance\Freqtrade\
├── ft_bot_docker_compose\          # Główny katalog roboczy
│   ├── memory-bank\                # Dokumentacja i plany
│   │   ├── plans\                  # Plany rozwoju
│   │   └── documentation\          # Dokumentacja (ten plik)
│   ├── user_data\                  # Dane użytkownika Freqtrade
│   │   ├── strategies\             # Strategie tradingowe
│   │   ├── buffer\                 # System bufora MA43200
│   │   ├── training\               # System treningu ML
│   │   ├── data\                   # Dane historyczne
│   │   └── ml_artifacts\           # Modele i artefakty ML
│   ├── models\                     # Modele zewnętrzne
│   └── scripts\                    # Skrypty pomocnicze
├── memory-bank\                    # Dokumentacja główna
└── ml_artifacts\                   # Artefakty ML (backup)
```

### **KLUCZOWE PLIKI:**

#### **STRATEGIE:**
- `Enhanced_ML_MA43200_Buffer_Strategy.py` - Główna strategia z buforem MA43200
- `EnhancedBinanceBotSignalStrategy.py` - Strategia z sygnałami
- `Enhanced_ML_Backtest_Strategy.py` - Strategia do backtestów

#### **SYSTEM BUFORA MA43200:**
- `buffer_manager.py` (23KB) - Lazy loading + LRU cache, kompresja .feather
- `external_data_collector.py` (18KB) - Pobieranie z Yahoo Finance & CoinGecko
- `binance_realtime_sync.py` (19KB) - Synchronizacja real-time co minutę
- `dataframe_extender.py` (14KB) - Integracja z Freqtrade (singleton)

#### **SYSTEM TRENINGU ML:**
- `user_data/training/` - Kompletny system treningu
- `train_gpu.py` - Skrypt treningu z GPU
- `core/` - Główne moduły (data_loader, feature_engineering, models, sequence_generator)

#### **DANE HISTORYCZNE:**
- `user_data/data/binanceusdm/futures/BTC_USDT-1m-futures.feather` (79MB)
- **Zakres:** 2020-01-01 → 2025-04-30 (5.3 lat, 2,803,656 rekordów)
- **Kompletność:** 100.1% pokrycia danych

---

## 🚨 **AKTUALNY STAN I PROBLEMY**

### **SYSTEM BUFORA MA43200 - ✅ DZIAŁAJĄCY**
- **Status:** Kompletnie zaimplementowany i działający
- **Funkcjonalności:** 8-stopniowa auto-recovery, lazy loading, LRU cache
- **Integracja:** Pomyślnie zintegrowany z Freqtrade
- **Bot:** Uruchomiony w Docker z startup_candle_count: 43300

### **SYSTEM TRENINGU ML - ⚠️ PROBLEM**
- **Status:** Zaimplementowany ale z krytycznym błędem
- **Problem:** Accuracy ~1.3% (powinno być >40%)
- **Przyczyna:** Błędny algorytm symulacji giełdy w module trenującym

### **ZIDENTYFIKOWANY BŁĄD ALGORYTMU:**
```python
# BŁĘDNY KOD (obecny):
long_tp_hit = (future_candles['high'] >= long_tp).any()
long_sl_hit = (future_candles['low'] <= long_sl).any()

# PROBLEM: .any() sprawdza CZY KIEDYKOLWIEK, nie KTÓRE PIERWSZE
```

### **KONSEKWENCJE BŁĘDU:**
- **Ekstremalna nierównowaga klas:** 97.4% HOLD, 1.2% SHORT, 1.4% LONG
- **Nierealistyczne etykiety:** Brak chronologii zdarzeń
- **Model nie może się nauczyć:** Accuracy ~1.3%

---

## 📋 **PLAN NAPRAWY - GOTOWY DO IMPLEMENTACJI**

### **LOKALIZACJA PLANU:**
`Freqtrade/memory-bank/plans/Plan_modyfikacji_modulu_trenujacego_symulacacja_gieldy.md`

### **KLUCZOWE ZMIANY:**

#### **1. NAPRAW SYMULACJĘ CHRONOLOGICZNĄ:**
```python
# NOWY ALGORYTM:
FOR każda świeca w future_window:
├── Sprawdź gap otwarcia → TP/SL hit?
├── Sprawdź low → SL hit?
├── Sprawdź high → TP hit?
├── Sprawdź close → TP/SL hit?
├── Jeśli hit → zapisz (typ, świeca_nr) i BREAK
└── Jeśli konflikt → LOSOWY WYBÓR
```

#### **2. EDGE CASES - LOSOWY WYBÓR:**
- Gap przekracza oba poziomy → `random.choice(['TP', 'SL'])`
- W świecy hit zarówno TP jak SL → `random.choice(['TP', 'SL'])`
- LONG_TP i SHORT_TP równocześnie → `random.choice(['LONG', 'SHORT'])`

#### **3. BALANSOWANIE KLAS:**
- Zachowaj WSZYSTKIE SHORT i LONG (najcenniejsze)
- Z HOLD wybierz losowo ~62.5% (dla rozkładu 25/50/25)
- Wagi czasowe: szybkie sygnały × 1.5, wolne × 0.8

### **OCZEKIWANE REZULTATY:**
- **Rozkład klas:** 25% SHORT, 50% HOLD, 25% LONG
- **Accuracy:** >40% (poprawa z 1.3%)
- **Realistyczne etykiety:** Chronologiczna kolejność zdarzeń

---

## 🔧 **ŚRODOWISKO TECHNICZNE**

### **HARDWARE:**
- **GPU:** NVIDIA GeForce GTX 1660 SUPER
- **OS:** Windows 10 (win32 10.0.22631)
- **Shell:** PowerShell

### **SOFTWARE STACK:**
- **Freqtrade:** Docker container
- **Python:** ML training system
- **TensorFlow/Keras:** Deep LSTM models
- **Data:** Feather format (compressed)

### **PARAMETRY TRENINGU:**
- **Para:** BTC_USDT
- **Timeframe:** 1m
- **Window Past:** 60 candles
- **Window Future:** 240 candles
- **TP:** 0.8%, **SL:** 0.4%
- **Model:** Deep LSTM (128→64→32 units, ~133k parameters)

---

## 📊 **DANE I METRYKI**

### **DANE HISTORYCZNE:**
- **Lokalizacja:** `user_data/data/binanceusdm/futures/BTC_USDT-1m-futures.feather`
- **Rozmiar:** 79MB, 2,803,656 rekordów
- **Zakres:** 2020-01-01 → 2025-04-30
- **Jakość:** 100.1% kompletności

### **OSTATNI TRENING:**
- **Data:** 25.05.2025
- **Sekwencje:** 661,010 training samples
- **Features:** 8 technical indicators
- **Epochs:** 16/100 (przerwany z powodu niskiej accuracy)
- **Czas/epoch:** ~420 sekund
- **Accuracy:** 1.34% training, 1.28% validation

### **ROZKŁAD KLAS (BŁĘDNY):**
- **SHORT:** 6,593 samples (1.2%)
- **HOLD:** 514,603 samples (97.4%) ← PROBLEM!
- **LONG:** 7,372 samples (1.4%)

---

## 🎯 **NASTĘPNE KROKI - PRIORYTET**

### **FAZA 1: PROOF OF CONCEPT (1-2 dni)**
1. **Znajdź moduł symulacji** w `user_data/training/core/`
2. **Zaimplementuj nowy algorytm** chronologiczny
3. **Test na małej próbce** (1000 sekwencji)
4. **Porównaj rozkład klas** przed/po

### **FAZA 2: PEŁNA IMPLEMENTACJA (2-3 dni)**
1. **Optymalizuj wydajność** (vectorization, parallel processing)
2. **Dodaj balansowanie klas** (losowa redukcja HOLD)
3. **Implementuj wagi czasowe**
4. **Pełny test** na całym datasecie

### **FAZA 3: WALIDACJA (1-2 dni)**
1. **Trening nowego modelu** z poprawionymi etykietami
2. **Porównanie accuracy** (oczekiwane >40%)
3. **Analiza jakości sygnałów**
4. **Dokumentacja rezultatów**

---

## 🔍 **KLUCZOWE LOKALIZACJE KODU**

### **DO NAPRAWY:**
- `user_data/training/core/sequence_generator/` - Główny moduł symulacji
- Szukaj funkcji zawierających `.any()` na future_candles
- Prawdopodobnie w pliku odpowiedzialnym za generowanie etykiet

### **DO TESTOWANIA:**
- `user_data/training/scripts/train_gpu.py` - Skrypt treningu
- `user_data/training/test_implementation.py` - Testy

### **KONFIGURACJA:**
- `user_data/training/config/` - Parametry treningu
- Sprawdź TP/SL settings, window sizes

---

## 📝 **WAŻNE USTALENIA**

### **FILOZOFIA PROJEKTU:**
- **Zachowaj wszystkie dane** - crypto jest volatile, nie filtruj
- **Losowy wybór dla konfliktów** - odzwierciedla nieprzewidywalność rynku
- **Proste rozwiązania** - unikaj overengineering
- **Kontakt z rzeczywistością** - nie "cherry pick" idealnych sygnałów

### **WAGI KLAS (OBECNE):**
- SHORT: ~53 (rzadkie, wysokie znaczenie)
- HOLD: ~0.3 (częste, niskie znaczenie)
- LONG: ~48 (rzadkie, wysokie znaczenie)

### **REPRODUKOWALNOŚĆ:**
- Użyj fixed seed dla random.choice()
- Loguj wszystkie losowe decyzje
- Wersjonuj algorytm i parametry

---

## 🚀 **GOTOWOŚĆ DO KONTYNUACJI**

### **WSZYSTKO PRZYGOTOWANE:**
- ✅ **Kompletny plan naprawy** w `Plan_modyfikacji_modulu_trenujacego_symulacacja_gieldy.md`
- ✅ **Działający system bufora MA43200**
- ✅ **Kompletne dane historyczne** (5.3 lat)
- ✅ **Zidentyfikowany błąd** i sposób naprawy
- ✅ **Środowisko techniczne** gotowe

### **PIERWSZY KROK:**
Znajdź i napraw funkcję symulacji w `user_data/training/core/` - zamień `.any()` na chronologiczną iterację przez future_window.

### **OCZEKIWANY REZULTAT:**
Model z accuracy >40% gotowy do praktycznego zastosowania w tradingu BTC/USDT.

---

**Ten plik zawiera wszystkie informacje potrzebne do kontynuacji prac. Projekt jest w 90% gotowy - pozostaje tylko naprawić jeden krytyczny błąd w algorytmie symulacji.**
