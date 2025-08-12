# 📋 Kompletna Lista Plików Uruchomieniowych Treningu

**Data inwentaryzacji:** 2025-05-26  
**Projekt:** Freqtrade ML Trading Bot  
**Cel:** Usystematyzowanie chaosu w plikach treningowych

## 🎯 GŁÓWNE PLIKI URUCHOMIENIOWE (AKTYWNE)

### 1. **PROSTY SYSTEM** - Symulacja Treningu

#### `scripts/train_model.py` (294 linii)
**Lokalizacja:** `C:\Users\macie\OneDrive\Python\Binance\Freqtrade\scripts\`  
**Status:** ✅ **AKTYWNY** - używany w ostatnim treningu  
**Typ:** Symulacja treningu z analizą danych  

**Funkcjonalność:**
- ✅ Ładuje gotowe dane (feather/csv/parquet)
- ✅ Analizuje jakość danych po bridge strategy
- ❌ **TYLKO SYMULACJA** - nie trenuje prawdziwego modelu
- ✅ Zapisuje wyniki do JSON

**Parametry:**
```bash
--pair BTCUSDT              # Para walutowa
--input path/to/file        # Ręczna ścieżka do danych
--epochs 100                # Liczba epok (domyślnie: 100)
--window 60                 # Rozmiar okna (domyślnie: 60)
--output-dir models         # Katalog wyników
```

**Uruchomienie:**
```bash
# AKTUALNY SPOSÓB (Docker Compose):
docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_model.py --epochs 5 --input /freqtrade/user_data/training/data/processed/BTCUSDT_1m_clean.feather

# Bezpośrednio:
python scripts/train_model.py --pair BTCUSDT --epochs 100
```

#### `ft_bot_docker_compose/user_data/training/scripts/train_model.py` (294 linii)
**Status:** ✅ **KOPIA** - identyczny z powyższym  
**Lokalizacja:** `C:\Users\macie\OneDrive\Python\Binance\Freqtrade\ft_bot_docker_compose\user_data\training\scripts\`  
**Uwaga:** To jest ta sama implementacja co w `scripts/`

---

### 2. **ZAAWANSOWANY SYSTEM** - Prawdziwy Trening ML

#### `ft_bot_docker_compose/user_data/training/scripts/train_dual_window_model.py` (629 linii)
**Status:** ✅ **DOSTĘPNY** ale nieużywany  
**Typ:** Pełny system ML z Dual-Window approach  
**Poziom:** Zaawansowany

**Funkcjonalność:**
- ✅ **PRAWDZIWY TRENING** modelu LSTM
- ✅ Dual-Window approach (eliminuje data leakage)
- ✅ Zaawansowana architektura ML pipeline
- ✅ Prawdziwe dane Freqtrade (EnhancedFeatherLoader)
- ✅ TrainingConfig integration
- ✅ Class balancing, Feature scaling

**Parametry (pełna lista):**
```bash
# Podstawowe:
--pair BTC_USDT             # Para krypto
--epochs 50                 # Liczba epok
--window-past 60            # Okno przeszłości
--window-future 60          # Okno przyszłości

# Czasowe:
--date-from 2024-01-01      # Data początkowa
--date-to 2024-01-31        # Data końcowa

# Risk Management:
--take-profit 1.0           # Take Profit %
--stop-loss 0.5             # Stop Loss %

# Model:
--batch-size 32             # Rozmiar batcha
--learning-rate 0.001       # Szybkość uczenia
--validation-split 0.2      # Podział train/val
--early-stopping 3          # Early stopping patience

# Presety:
--preset {quick,standard,production,test}

# Walidacja:
--validate-data             # Tylko sprawdź dostępność danych
--dry-run                   # Dry run - nie trenuj modelu
```

**Presety:**
- `quick`: 5 epok, ostatnie 30 dni
- `standard`: 50 epok, cały 2024
- `production`: 100 epok, od 2020 do dziś
- `test`: 2 epoki, ostatnie 7 dni

**Uruchomienie:**
```bash
# Docker Compose:
docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_dual_window_model.py --preset quick

# Bezpośrednio:
python user_data/training/scripts/train_dual_window_model.py --preset production
```

---

### 3. **WRAPPERY DOCKER**

#### `ft_bot_docker_compose/train_gpu.py` (265 linii)
**Status:** ⚠️ **CZĘŚCIOWO DZIAŁAJĄCY** (błąd argparse)  
**Funkcja:** Docker wrapper dla `train_dual_window_model.py`

**Funkcjonalność:**
- ✅ Automatyczna detekcja GPU
- ✅ Montowanie katalogów (`ml_artifacts`, `user_data`)
- ✅ Przekazywanie wszystkich parametrów
- ❌ **BŁĄD** - nie działa `--help` (ValueError w argparse)
- ✅ Używa obrazu `ft_bot_docker_compose-freqtrade-gpu`

**Uruchomienie (gdyby działał):**
```bash
python train_gpu.py --preset quick
python train_gpu.py --pair ETH_USDT --epochs 100
```

#### `ft_bot_docker_compose/train_gpu.bat` (38 linii)
**Status:** ✅ **DZIAŁAJĄCY**  
**Funkcja:** Windows launcher dla `train_gpu.py`

```cmd
train_gpu.bat --preset quick
train_gpu.bat --pair ETH_USDT --epochs 100
```

---

### 4. **UNIWERSALNY SYSTEM**

#### `ft_bot_docker_compose/train_universal.py` (662 linii)
**Status:** ✅ **DOSTĘPNY** ale nieużywany  
**Typ:** Uniwersalny skrypt z mock data  

**Funkcjonalność:**
- ✅ Pełna parametryzacja CLI
- ✅ System presets (quick/standard/production/test)
- ✅ Real-time monitoring z GPU metrics
- ❌ **MOCK DATA** - generuje sztuczne dane
- ✅ Automatyczne nazewnictwo wyników

**Parametry:** Podobne do `train_dual_window_model.py`

**Uruchomienie:**
```bash
python train_universal.py --preset production
python train_universal.py --pair BTC_USDT --epochs 100
```

---

## 🗂️ PLIKI ARCHIWALNE (NIEAKTYWNE)

### `ft_bot_docker_compose/user_data/training/archives/`

#### `train_model.py` (111 linii)
**Status:** 🗄️ **ARCHIWUM** - stara wersja prostego systemu

#### `train_4_years_140_epochs.py` (281 linii)
**Status:** 🗄️ **ARCHIWUM** - specjalistyczny skrypt dla długiego treningu

#### `train_may2020_dec2024_100epochs.py` (321 linii)
**Status:** 🗄️ **ARCHIWUM** - skrypt dla konkretnego okresu

#### `train_30days_5epochs_gpu_test.py` (326 linii)
**Status:** 🗄️ **ARCHIWUM** - test GPU na krótkich danych

---

## 🔧 PLIKI KONFIGURACYJNE

#### `ft_bot_docker_compose/user_data/training/config/training_config.py` (227 linii)
**Status:** ✅ **AKTYWNY**  
**Funkcja:** Centralna konfiguracja TrainingConfig dla Dual-Window

**Kluczowe parametry:**
```python
WINDOW_SIZE: int = 60           # Historical window
FUTURE_WINDOW: int = 60         # Future window
LONG_TP_PCT: float = 0.008      # 0.8% Take Profit
LONG_SL_PCT: float = 0.004      # 0.4% Stop Loss
LSTM_UNITS: [128, 64, 32]       # LSTM layers
EPOCHS: 10
BATCH_SIZE: 32
```

---

## 🚀 AKTUALNY STAN URUCHAMIANIA

### ✅ **CO DZIAŁA OBECNIE:**

1. **Docker Compose + Prosty System:**
   ```bash
   docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_model.py --epochs 5 --input /freqtrade/user_data/training/data/processed/BTCUSDT_1m_clean.feather
   ```
   - ✅ **UŻYWANE** - ostatni trening 26.05.2025
   - ❌ Tylko symulacja, nie prawdziwy trening

2. **Docker Compose + Zaawansowany System:**
   ```bash
   docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_dual_window_model.py --preset quick
   ```
   - ✅ **DOSTĘPNE** - prawdziwy trening ML
   - ❌ **NIEUŻYWANE** - nie testowane

### ⚠️ **CO NIE DZIAŁA:**

1. **Docker Wrapper:**
   ```bash
   python train_gpu.py --help  # ValueError w argparse
   ```

2. **Bezpośrednie uruchomienie:**
   ```bash
   python scripts/train_model.py  # Problemy z ścieżkami
   ```

---

## 🎯 REKOMENDACJE UPORZĄDKOWANIA

### 1. **GŁÓWNE PLIKI DO ZACHOWANIA:**
- ✅ `train_dual_window_model.py` - **GŁÓWNY SYSTEM TRENINGOWY**
- ✅ `training_config.py` - konfiguracja
- ✅ `train_model.py` - prosty system do testów

### 2. **DO NAPRAWY:**
- 🔧 `train_gpu.py` - naprawić błąd argparse
- 🔧 Dokumentacja - zaktualizować instrukcje

### 3. **DO USUNIĘCIA/ARCHIWIZACJI:**
- 🗄️ `train_universal.py` - duplikuje funkcjonalność
- 🗄️ Stare pliki w `archives/` - przenieść do backup

### 4. **PRIORYTET UŻYCIA:**
1. **`train_dual_window_model.py`** - główny system produkcyjny
2. **`train_model.py`** - testy i analiza danych
3. **`train_gpu.py`** - po naprawie jako wrapper

---

## 📊 PODSUMOWANIE CHAOSU

**Łącznie plików treningowych:** 10+ aktywnych + 4 archiwalne  
**Systemy treningowe:** 3 różne (prosty, dual-window, universal)  
**Wrappery:** 2 (Python + Batch)  
**Stan:** **CHAOS** - potrzeba uporządkowania

**Główny problem:** Brak jasności który plik jest "właściwy" i aktualny.

**Rozwiązanie:** Ustalić `train_dual_window_model.py` jako główny system i zaktualizować dokumentację.
