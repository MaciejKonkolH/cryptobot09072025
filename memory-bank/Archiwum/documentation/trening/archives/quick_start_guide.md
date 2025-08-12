# Quick Start Guide - Dwuokienny Moduł Trenujący

*Wersja: 2.0.0 | Status: ✅ Gotowy do użycia*

## 🚀 Szybki Start (5 minut)

### 1. Lokalizacja
```
C:\Users\macie\OneDrive\Python\Binance\Freqtrade\ft_bot_docker_compose\user_data\training\
```

### 2. Podstawowe Uruchomienie
```bash
cd "C:\Users\macie\OneDrive\Python\Binance\Freqtrade\ft_bot_docker_compose\user_data\training"

# Szybki test (3 epoki, 7 dni)
python scripts\train_dual_window_model.py --pair BTC_USDT --start-date 2024-01-01 --end-date 2024-01-07 --epochs 3 --batch-size 16

# Pełny trening (100 epok, 6 miesięcy)  
python scripts\train_dual_window_model.py --pair BTC_USDT --start-date 2024-01-01 --end-date 2024-06-30
```

### 3. Sprawdź Wyniki
```
outputs\models\
├── best_model_BTC_USDT_YYYYMMDD_YYYYMMDD.keras     ← Najlepszy model
└── training_BTC_USDT_YYYYMMDD_HHMMSS\
    ├── dual_window_lstm_model.keras                ← Model końcowy
    ├── evaluation_results.json                     ← Metryki
    └── training_config.json                        ← Konfiguracja
```

## 📊 Jak Działają Wyniki

### Kluczowe Metryki
```
📈 TRADING SIGNALS AVG F1: 0.171 ⭐  ← NAJWAŻNIEJSZA METRYKA
SHORT Recall: 29.7%                  ← Znajdzie 30% sygnałów SHORT  
LONG Recall: 35.6%                   ← Znajdzie 36% sygnałów LONG
Test Accuracy: 60.34%                ← Ogólna dokładność
```

### Interpretacja
- **F1 0.0-0.2**: Słabe sygnały
- **F1 0.2-0.4**: Przeciętne sygnały  
- **F1 0.4+**: Dobre sygnały
- **Recall 30%**: Z 10 prawdziwych sygnałów znajdzie 3

## ⚙️ Podstawowa Konfiguracja

### Domyślne Parametry
```python
WINDOW_SIZE = 60           # 60 świec historii (dane dla modelu)
FUTURE_WINDOW = 60         # 60 świec weryfikacji (sprawdzenie sygnału)
LONG_TP_PCT = 0.007       # 0.7% Take Profit  
LONG_SL_PCT = 0.007       # 0.7% Stop Loss
EPOCHS = 100              # Liczba epok treningu
BATCH_SIZE = 32           # Rozmiar batcha
```

### Szybkie Modyfikacje
```bash
# Krótsze okna (szybszy trening)
--window-size 30 --future-window 30

# Więcej sygnałów (niższe progi)
--tp-pct 0.005 --sl-pct 0.005  

# Szybki test
--epochs 3 --batch-size 16
```

## 🔧 Rozwiązywanie Problemów

### Test Komponentów
```bash
python test_implementation.py
# Powinno pokazać: 📊 WYNIKI TESTÓW: 5/5 przeszło ✅
```

### Sprawdź Dane
```bash
python scripts\train_dual_window_model.py --pair BTC_USDT --start-date 2024-01-01 --end-date 2024-01-07 --validate-data
# Powinno pokazać: ✅ Dane wystarczające
```

### Typowe Błędy
```
❌ "Niewystarczające dane historyczne" 
✅ Potrzebujesz 33 dni buffera przed okresem treningu

❌ "Invalid comparison between dtype=datetime64[ns, UTC]"
✅ Naprawione automatycznie w EnhancedFeatherLoader

❌ "Incompatible shapes"  
✅ Sprawdź czy WINDOW_SIZE i NUM_FEATURES są spójne
```

## 📈 Interpretacja Wyników

### Confusion Matrix
```
       SHORT    HOLD    LONG
SHORT    239     204     361    ← 30% poprawnych SHORT
HOLD    1889    6418    1592    ← 65% poprawnych HOLD  
LONG     191     328     287    ← 36% poprawnych LONG
```

### Co oznacza?
- **Model najlepiej przewiduje HOLD** (65% poprawność)
- **SHORT i LONG** mają niższą poprawność, ale to normalne
- **Precision 10-13%** - z 10 sygnałów modelu, 1-2 będą skuteczne
- **Recall 30-36%** - model znajdzie około 1/3 prawdziwych sygnałów

## 🎯 Poprawa Wyników

### 1. Więcej Danych
```bash
# Zamiast 7 dni, użyj minimum 3-6 miesięcy
--start-date 2024-01-01 --end-date 2024-06-30
```

### 2. Bardziej Agresywne TP/SL  
```bash
# Generuje więcej sygnałów SHORT/LONG
--tp-pct 0.005 --sl-pct 0.005
```

### 3. Dłuższy Trening
```bash
# Więcej epok dla lepszej konwergencji
--epochs 50
```

## 📁 Architektura Systemu

### Dwuokienne Podejście
```
[świece 940-999] ←── HISTORICAL WINDOW (model input)
     ↓
[świeca 1000] ←── PREDICTION POINT (decyzja)
     ↓  
[świece 1001-1060] ←── FUTURE WINDOW (weryfikacja)
```

### Kluczowa Zasada
**Model NIE widzi przyszłości** - tylko przeszłość służy do predykcji, przyszłość tylko do weryfikacji skuteczności sygnału.

### 8 Cech Wejściowych
```python
FEATURE_COLUMNS = [
    'high_change',      # Zmiana ceny max vs open
    'low_change',       # Zmiana ceny min vs open  
    'close_change',     # Zmiana ceny close vs open
    'volume_change',    # Zmiana wolumenu
    'price_to_ma1440',  # Cena vs średnia 24h
    'price_to_ma43200', # Cena vs średnia 30 dni
    'volume_to_ma1440', # Wolumen vs średnia 24h  
    'volume_to_ma43200' # Wolumen vs średnia 30 dni
]
```

## 🎭 Tryby Użycia

### Development (szybkie testy)
```bash
python scripts\train_dual_window_model.py \
    --pair BTC_USDT \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --epochs 3 \
    --window-size 30 \
    --future-window 30 \
    --batch-size 16
```

### Production (pełny trening)
```bash
python scripts\train_dual_window_model.py \
    --pair BTC_USDT \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --epochs 100 \
    --batch-size 32
```

### Validation Only
```bash
python scripts\train_dual_window_model.py \
    --pair BTC_USDT \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --validate-data
```

---

**📞 Pomoc**: Przy problemach sprawdź pełną dokumentację w `dual_window_training_system.md`  
**🔄 Status**: System gotowy do użycia produkcyjnego  
**📈 Cel**: Trading F1 > 0.2 dla dobrych sygnałów handlowych 