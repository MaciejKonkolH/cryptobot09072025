# 🎯 Kompletna Dokumentacja Modułu Treningowego Freqtrade

*Data utworzenia: 27 stycznia 2025*  
*Wersja: 3.0*  
*Status: ✅ Implementacja zakończona z Confidence Thresholding*  
*Scalono z dwóch dokumentów: Medul_treningowy_dokumentacja.md + Modul_trenowania_modeli.md*

## 📋 Spis Treści

1. [Przegląd Systemu](#1-przegląd-systemu)
2. [Architektura Dwuokiennego Podejścia](#2-architektura-dwuokiennego-podejścia)
3. [🆕 Confidence Thresholding](#3-confidence-thresholding)
4. [Komponenty Systemu](#4-komponenty-systemu)
5. [Konfiguracja](#5-konfiguracja)
6. [🐳 Docker Wrapper - ZALECANY SPOSÓB](#6-docker-wrapper---zalecany-sposób)
7. [Instrukcje Użycia](#7-instrukcje-użycia)
8. [System Presets](#8-system-presets)
9. [Analiza Wyników](#9-analiza-wyników)
10. [Rozwiązywanie Problemów](#10-rozwiązywanie-problemów)
11. [Zalecenia](#11-zalecenia)

---

## 1. Przegląd Systemu

### 1.1. Cel i Zastosowanie

Dwuokienny moduł trenujący to zaawansowany system uczenia maszynowego dla Freqtrade, który:

- **Generuje sygnały handlowe** (SHORT/HOLD/LONG) na podstawie analizy technicznej
- **Eliminuje data leakage** poprzez ścisłą separację czasową
- **🆕 Implementuje confidence thresholding** dla selektywnych predykcji
- **Symuluje rzeczywiste warunki handlowe** gdzie trader ma dostęp tylko do przeszłości
- **Używa hybrydowego etykietowania** bazującego na Take Profit/Stop Loss

### 1.2. 🆕 Najnowsze Funkcjonalności (v3.0)

✅ **Confidence Thresholding System**
- Conservative/Aggressive/Balanced modes
- Osobne progi dla SHORT (60%), LONG (60%), HOLD (40%)
- Automatyczne przełączanie na HOLD przy niskiej pewności

✅ **Zaawansowane Zarządzanie Pamięcią**
- Batch processing dla dużych zbiorów danych
- Memory monitoring z automatycznym cleanup
- Vectorized labeling (300% szybsze)

✅ **Deterministyczny Trening**
- Fixed seed dla reprodukowalności
- Eliminacja randomowych wyników między treningami

### 1.3. Kluczowe Innowacje

1. **Temporal Separation**: Rozdzielenie danych wejściowych od weryfikacji etykiet
2. **🆕 Confidence-Based Predictions**: Model wymaga 60%+ pewności dla sygnałów
3. **Realistic Trading Simulation**: Model nie ma dostępu do przyszłości
4. **Advanced Class Balancing**: Inteligentne wagi dla niezbalansowanych danych
5. **Production-Ready Pipeline**: Kompletny system z callbacks i artefaktami
6. **🐳 Docker Integration**: Pełna integracja z wrapper do GPU treningu

---

## 2. Architektura Dwuokiennego Podejścia

### 2.1. Fundamentalna Zasada

**🔑 KLUCZOWE**: Model NIE może widzieć przyszłości podczas predykcji

```
TIMELINE dla świecy i=1000:

[świece 940-999] ←── HISTORICAL WINDOW (60 świec)
     ↓              Dane wejściowe dla modelu
[świeca 1000] ←── PREDICTION POINT
     ↓              Punkt decyzji handlowej  
[świece 1001-1060] ←── FUTURE WINDOW (60 świec)
                      Weryfikacja skuteczności sygnału
```

### 2.2. Historical Window (Input)

```python
# Co model widzi:
WINDOW_SIZE = 90  # 90 ostatnich świec (zaktualizowane w v3.0)
INPUT_FEATURES = 8  # Cechy techniczne

# Shape: (batch_size, 90, 8)
historical_features = [
    'high_change', 'low_change', 'close_change', 'volume_change',
    'price_to_ma1440', 'price_to_ma43200', 
    'volume_to_ma1440', 'volume_to_ma43200'
]

# Cel: "Na podstawie ostatnich 90 minut - jaki sygnał?"
```

### 2.3. Future Window (Verification)

```python
# Jak weryfikujemy skuteczność:
FUTURE_WINDOW = 90  # 90 następnych świec (zaktualizowane w v3.0)
TP_THRESHOLD = 1.0%  # Take Profit (zaktualizowane)
SL_THRESHOLD = 0.5%  # Stop Loss (zaktualizowane)

# Hybrydowa klasyfikacja:
if long_tp_hit and not long_sl_hit:
    label = 2  # LONG
elif short_tp_hit and not short_sl_hit:
    label = 0  # SHORT  
else:
    label = 1  # HOLD
```

### 2.4. Eliminacja Data Leakage

```python
# ✅ PRAWIDŁOWE: Model nie widzi przyszłości
X[i] = historical_features[i-90:i]      # Przeszłość
y[i] = verify_signal[i+1:i+91]          # Przyszłość (tylko weryfikacja)

# ❌ BŁĘDNE: Model widziałby przyszłość  
X[i] = features[i-45:i+45]              # ZAWIERA PRZYSZŁOŚĆ!
```

---

## 3. 🆕 Confidence Thresholding

### 3.1. Zasada Działania

Confidence Thresholding to nowa funkcjonalność w v3.0, która pozwala modelowi na:
- **Selektywne predykcje**: Model otwiera pozycję tylko gdy jest wystarczająco pewny
- **Automatyczne HOLD**: Przy niskiej pewności model wybiera bezpieczną opcję HOLD
- **Różne tryby**: Conservative, Aggressive, Balanced dla różnych strategii

### 3.2. Tryby Confidence

#### 🛡️ Conservative Mode (domyślny)
```python
CONFIDENCE_THRESHOLD_SHORT = 0.70  # 70% pewności dla SHORT
CONFIDENCE_THRESHOLD_LONG = 0.70   # 70% pewności dla LONG  
CONFIDENCE_THRESHOLD_HOLD = 0.30   # 30% wystarczy dla HOLD
```

#### ⚡ Aggressive Mode
```python
CONFIDENCE_THRESHOLD_SHORT = 0.45  # 45% pewności dla SHORT
CONFIDENCE_THRESHOLD_LONG = 0.45   # 45% pewności dla LONG
CONFIDENCE_THRESHOLD_HOLD = 0.60   # 60% potrzeba dla HOLD
```

#### ⚖️ Balanced Mode
```python
CONFIDENCE_THRESHOLD_SHORT = 0.55  # 55% pewności dla SHORT
CONFIDENCE_THRESHOLD_LONG = 0.55   # 55% pewności dla LONG
CONFIDENCE_THRESHOLD_HOLD = 0.45   # 45% wystarczy dla HOLD
```

### 3.3. Implementacja w Treningu

```python
# W dual_window_lstm_model.py - ModelEvaluator
def _apply_confidence_thresholding(self, y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Aplikuje confidence thresholding zamiast standardowego argmax.
    
    Args:
        y_pred_proba: Macierz prawdopodobieństw shape (n_samples, 3)
        
    Returns:
        np.ndarray: Predykcje z confidence thresholding
    """
    predictions = []
    
    for proba in y_pred_proba:
        short_confidence = proba[0]  # SHORT
        hold_confidence = proba[1]   # HOLD  
        long_confidence = proba[2]   # LONG
        
        # Sprawdź czy któryś sygnał przekracza próg
        if short_confidence >= self.config.CONFIDENCE_THRESHOLD_SHORT:
            predictions.append(0)  # SHORT
        elif long_confidence >= self.config.CONFIDENCE_THRESHOLD_LONG:
            predictions.append(2)  # LONG
        elif hold_confidence >= self.config.CONFIDENCE_THRESHOLD_HOLD:
            predictions.append(1)  # HOLD
        else:
            # Fallback - wybierz najwyższą pewność
            predictions.append(np.argmax(proba))
    
    return np.array(predictions)
```

### 3.4. Implementacja w Strategii

```python
# W Enhanced_ML_MA43200_Buffer_Strategy.py
confidence_threshold = self.confidence_threshold_ml.value  # 0.60 domyślnie

# Znajdź najwyższą pewność i klasę
max_confidence = np.max(pred)
predicted_class = np.argmax(pred)

# Sprawdź czy pewność przekracza próg
if max_confidence >= confidence_threshold:
    if predicted_class == 0:    # SHORT
        dataframe.loc[dataframe.index[-1], 'ml_signal'] = 'SHORT'
    elif predicted_class == 2:  # LONG
        dataframe.loc[dataframe.index[-1], 'ml_signal'] = 'LONG'
    else:  # HOLD lub niska pewność
        dataframe.loc[dataframe.index[-1], 'ml_signal'] = 'HOLD'
else:
    # Niska pewność - zawsze HOLD
    dataframe.loc[dataframe.index[-1], 'ml_signal'] = 'HOLD'
```

### 3.5. Konfiguracja CLI

```bash
# Confidence thresholding parameters w train_dual_window_model.py
--confidence-short 0.70     # Próg pewności SHORT
--confidence-long 0.70      # Próg pewności LONG  
--confidence-hold 0.30      # Próg pewności HOLD
--confidence-mode conservative  # Tryb: conservative/aggressive/balanced
--disable-confidence        # Wyłącz confidence thresholding
```

### 3.6. Porównanie Wyników

System automatycznie porównuje wyniki argmax vs confidence thresholding:

```
🔍 PORÓWNANIE ARGMAX vs CONFIDENCE THRESHOLDING:

📊 ARGMAX (standardowe):
   Accuracy: 60.34%
   SHORT Precision: 36.7%, Recall: 29.7%
   LONG Precision: 41.3%, Recall: 35.6%

🎯 CONFIDENCE THRESHOLDING (conservative):
   Accuracy: 58.12%
   SHORT Precision: 42.1%, Recall: 24.8% 
   LONG Precision: 45.7%, Recall: 28.9%
   
💡 RÓŻNICA:
   +5.4pp precision SHORT, +4.4pp precision LONG
   -4.9pp recall SHORT, -6.7pp recall LONG
   Mniej błędnych sygnałów, więcej konserwatywnych HOLD
```

---

## 4. Komponenty Systemu

### 4.1. Struktura Katalogów

```
user_data/training/
├── 📁 config/                     # Konfiguracje systemu
│   ├── __init__.py
│   └── training_config.py         ← 🆕 TrainingConfig z Confidence Thresholding
├── 📁 core/                       # Komponenty główne
│   ├── __init__.py
│   ├── 📁 data_loaders/           # Ładowanie danych
│   │   ├── __init__.py
│   │   └── enhanced_feather_loader.py  ← Inteligentne ładowanie z bufferami
│   ├── 📁 models/                 # Modele ML
│   │   ├── __init__.py
│   │   └── dual_window_lstm_model.py   ← 🆕 Model z Confidence Thresholding
│   ├── 📁 sequence_builders/      # Tworzenie sekwencji
│   │   ├── __init__.py
│   │   └── dual_window_sequence_builder.py ← 🆕 Batch processing + vectorized labeling
│   └── 📁 feature_engineering/    # Engineering cech (pusty)
├── 📁 scripts/                    # Skrypty uruchomieniowe
│   └── train_dual_window_model.py ← 🆕 CLI z confidence parameters
├── 📁 outputs/                    # Wyniki treningu
│   ├── 📁 models/                 # Wytrenowane modele
│   └── 📁 scalers/                # Skalowanie danych
├── 📁 utilities/                  # Narzędzia pomocnicze
├── 📁 archives/                   # Archiwa starych wersji
└── 📄 test_implementation.py      # Testy systemu
```

### 4.2. 🔧 TrainingConfig - Centralna Konfiguracja

**Plik**: `config/training_config.py`

```python
@dataclass
class TrainingConfig:
    """Centralna konfiguracja dla dwuokiennego systemu"""
    
    # === TEMPORAL WINDOWS ===
    WINDOW_SIZE: int = 90            # Historical window (zaktualizowane w v3.0)
    FUTURE_WINDOW: int = 90          # Future window (zaktualizowane w v3.0)
    
    # === LABELING PARAMETERS ===  
    LONG_TP_PCT: float = 0.01        # 1.0% Take Profit LONG (zaktualizowane)
    LONG_SL_PCT: float = 0.005       # 0.5% Stop Loss LONG (zaktualizowane)
    SHORT_TP_PCT: float = 0.01       # 1.0% Take Profit SHORT
    SHORT_SL_PCT: float = 0.005      # 0.5% Stop Loss SHORT
    
    # === 🆕 CONFIDENCE THRESHOLDING ===
    USE_CONFIDENCE_THRESHOLDING: bool = True
    CONFIDENCE_THRESHOLD_SHORT: float = 0.60
    CONFIDENCE_THRESHOLD_LONG: float = 0.60  
    CONFIDENCE_THRESHOLD_HOLD: float = 0.40
    CONFIDENCE_MODE: str = "conservative"
    
    # === MODEL PARAMETERS ===
    LSTM_UNITS: List[int] = [128, 64, 32]
    DENSE_UNITS: List[int] = [32, 16]
    DROPOUT_RATE: float = 0.2
    RECURRENT_DROPOUT_RATE: float = 0.2  # 🆕 Dodany parametr
    
    # === TRAINING ===
    EPOCHS: int = 100                # Zaktualizowane z 10 do 100
    BATCH_SIZE: int = 256            # Zwiększone z 32 do 256
    LEARNING_RATE: float = 0.002     # Zaktualizowane
```

**🆕 Nowe metody w v3.0**:
- `apply_confidence_mode()` - aplikuje tryby confidence
- `validate_confidence_params()` - walidacja progów confidence
- `from_cli_args()` - tworzenie config z parametrów CLI

### 4.3. 📊 EnhancedFeatherLoader - Ładowanie Danych

**Plik**: `core/data_loaders/enhanced_feather_loader.py`

**Funkcjonalności**:
- **🔄 Automatyczny buffer**: Oblicza wymagane 33 dni buffera
- **📁 Multi-format support**: Obsługa różnych struktur katalogów
- **🕐 Timezone handling**: Konwersja na timezone-naive
- **🔗 Multi-file loading**: Łączenie wielu plików .feather
- **✅ Data validation**: Sprawdzanie kompletności danych

**Kluczowe metody**:
- `load_training_data()` - główna metoda ładowania z bufferem
- `_calculate_total_buffer_days()` - obliczanie buffera (33 dni)
- `_compute_all_features()` - obliczanie wszystkich cech technicznych

### 4.4. 🔄 DualWindowSequenceBuilder - Tworzenie Sekwencji

**Plik**: `core/sequence_builders/dual_window_sequence_builder.py`

**🆕 Funkcjonalności v3.0**:
- **Batch Processing**: Eliminuje memory exhaustion na dużych zbiorach
- **Vectorized Labeling**: 300% szybsze przetwarzanie
- **Memory Monitoring**: Real-time monitoring z automatycznym cleanup
- **Deterministyczny seed**: Reprodukowalne wyniki

**Kluczowe metody**:
- `create_training_sequences()` - 🆕 batch processing version
- `_create_label_vectorized()` - 🆕 vectorized labeling (300% szybsze)
- `_apply_confidence_thresholding()` - 🆕 confidence-based predictions

**Pipeline procesu**:
```python
# FAZA 1: Batch processing configuration
BATCH_SIZE = 100000  # Process 100k sequences at a time

# FAZA 2: Memory monitoring
memory_monitor = MemoryMonitor(max_memory_gb=10)

# FAZA 3: Vectorized labeling
if USE_VECTORIZED_LABELING:
    label, simulation_results = self._create_label_vectorized(...)
else:
    label, simulation_results = self._create_label_from_future_chronological(...)

# FAZA 4: Batch saving with metadata
batch_file, metadata_file = self._save_batch(...)
```

### 4.5. 🧠 DualWindowLSTM - Model Architecture

**Plik**: `core/models/dual_window_lstm_model.py`

**🆕 Funkcjonalności v3.0**:
- **Confidence Thresholding**: `_apply_confidence_thresholding()`
- **Deterministyczny setup**: `setup_deterministic_training()`
- **Porównanie wyników**: `_print_confidence_comparison()`

**Architektura modelu**:
```python
# Input: (None, 90, 8) - 90 świec x 8 cech (zaktualizowane)
# LSTM Stack: 128 → 64 → 32 units (3 warstwy)
# Dense Stack: 32 → 16 units z Dropout
# Output: 3 klasy (softmax) - SHORT/HOLD/LONG

# 🆕 Confidence evaluation
def _apply_confidence_thresholding(self, y_pred_proba):
    for proba in y_pred_proba:
        if proba[0] >= self.config.CONFIDENCE_THRESHOLD_SHORT:
            predictions.append(0)  # SHORT
        elif proba[2] >= self.config.CONFIDENCE_THRESHOLD_LONG:
            predictions.append(2)  # LONG
        else:
            predictions.append(1)  # HOLD (default)
```

---

## 5. Konfiguracja

### 5.1. Podstawowa Konfiguracja

```python
# Utwórz domyślną konfigurację
from config.training_config import TrainingConfig

config = TrainingConfig()
config.print_summary()
```

### 5.2. 🆕 Konfiguracja Confidence Thresholding

```python
# Aplikuj tryb confidence
config.apply_confidence_mode("conservative")  # conservative/aggressive/balanced

# Lub ręczne ustawienie progów
config.USE_CONFIDENCE_THRESHOLDING = True
config.CONFIDENCE_THRESHOLD_SHORT = 0.70
config.CONFIDENCE_THRESHOLD_LONG = 0.70
config.CONFIDENCE_THRESHOLD_HOLD = 0.30

# Walidacja parametrów
config.validate_confidence_params()
```

### 5.3. Dostosowanie Parametrów

```python
# Modyfikacja okien czasowych
config.WINDOW_SIZE = 90       # Historical window (zaktualizowane w v3.0)
config.FUTURE_WINDOW = 90     # Future window (zaktualizowane w v3.0)

# Modyfikacja progów TP/SL
config.LONG_TP_PCT = 0.01     # 1.0% TP (zaktualizowane)
config.LONG_SL_PCT = 0.005    # 0.5% SL (zaktualizowane)

# Modyfikacja treningu
config.EPOCHS = 100           # Więcej epok (zaktualizowane)
config.BATCH_SIZE = 256       # Większy batch size (zaktualizowane)
config.LEARNING_RATE = 0.002  # Zaktualizowany learning rate
```

### 5.4. Zapis/Odczyt Konfiguracji

```python
# Zapis do pliku
config.save_to_file("my_config.json")

# Odczyt z pliku
config = TrainingConfig.from_config_file("my_config.json")

# 🆕 Tworzenie z parametrów CLI
config = TrainingConfig.from_cli_args(args)
```

### 5.5. Walidacja Konfiguracji

```python
# Automatyczna walidacja
config.validate_config()  # 🆕 Nowa metoda walidacji

# Indywidualne walidacje
config.validate_windows()           # Walidacja okien czasowych  
config.validate_trading_params()    # Walidacja parametrów handlowych
config.validate_confidence_params() # 🆕 Walidacja confidence thresholding

# Buffer calculation
buffer_days = config.calculate_required_buffer_days()
# 📊 Required data buffer: 33 days
```

### 5.6. 🆕 Parametry CLI z Confidence

```bash
# Podstawowe parametry
python train_dual_window_model.py \
    --pair BTC_USDT \
    --epochs 100 \
    --window-past 90 \
    --window-future 90 \
    --take-profit 1.0 \
    --stop-loss 0.5

# 🆕 Confidence thresholding parameters
python train_dual_window_model.py \
    --confidence-short 0.70 \
    --confidence-long 0.70 \
    --confidence-hold 0.30 \
    --confidence-mode conservative

# 🆕 Wyłączenie confidence thresholding
python train_dual_window_model.py \
    --disable-confidence
```

---

## 6. 🐳 Docker Wrapper - ZALECANY SPOSÓB

### 6.1. ⭐ DLACZEGO DOCKER WRAPPER?

**🎯 NAJLEPSZY SPOSÓB uruchamiania treningu to użycie Docker Wrapper `train_gpu.py`**

**Zalety Docker Wrapper:**
- ✅ **Automatyczna konfiguracja środowiska** - nie musisz instalować zależności
- ✅ **GPU Support** - automatyczne wykorzystanie GPU jeśli dostępne
- ✅ **Izolacja środowiska** - brak konfliktów z innymi projektami
- ✅ **🆕 Pełna kompatybilność z confidence thresholding** - wszystkie nowe parametry
- ✅ **Wszystkie aliasy parametrów** - obsługuje wszystkie warianty nazw
- ✅ **Łatwość użycia** - jeden plik, wszystkie funkcje

### 6.2. Lokalizacja i Wymagania

```bash
# Wrapper znajduje się w:
ft_bot_docker_compose/train_gpu.py

# Wymagania:
- Docker i Docker Compose
- Serwis 'freqtrade' w docker-compose.yml
- Uruchamianie z katalogu ft_bot_docker_compose/
```

### 6.3. 🚀 Podstawowe Użycie

```bash
# Przejdź do katalogu Docker Compose
cd ft_bot_docker_compose

# SZYBKI TEST (5 epok, ostatnie 30 dni)
python train_gpu.py --preset quick

# STANDARDOWY TRENING (50 epok, cały 2024)
python train_gpu.py --preset standard

# PRODUKCYJNY TRENING (100 epok, wszystkie dane)
python train_gpu.py --preset production

# TEST ROZWOJOWY (2 epoki, ostatnie 7 dni)
python train_gpu.py --preset test
```

### 6.4. 🆕 Przykłady z Confidence Thresholding

```bash
# Trening z conservative confidence mode
python train_gpu.py --preset standard --confidence-mode conservative

# Aggressive confidence thresholding
python train_gpu.py --pair ETH_USDT --confidence-mode aggressive \
    --confidence-short 0.45 --confidence-long 0.45

# Wyłączenie confidence thresholding
python train_gpu.py --preset production --disable-confidence

# Custom confidence thresholds
python train_gpu.py --pair BTC_USDT \
    --confidence-short 0.75 \
    --confidence-long 0.75 \
    --confidence-hold 0.25 \
    --confidence-mode custom
```

### 6.5. Pełna Lista Parametrów

| Parametr | Alias | Domyślna | Opis |
|----------|-------|----------|------|
| `--preset` | - | - | System presets (quick/standard/production/test) |
| `--pair` | - | `BTC_USDT` | Para krypto |
| `--date-from` | `--start-date` | `2024-01-01` | Data początkowa YYYY-MM-DD |
| `--date-to` | `--end-date` | `2024-01-07` | Data końcowa YYYY-MM-DD |
| `--window-past` | `--window-size` | `90` | Okno przeszłości (🆕 zaktualizowane) |
| `--window-future` | - | `90` | Okno przyszłości (🆕 zaktualizowane) |
| `--take-profit` | `--tp-pct` | `1.0` | Take Profit % (🆕 zaktualizowane) |
| `--stop-loss` | `--sl-pct` | `0.5` | Stop Loss % (🆕 zaktualizowane) |
| `--epochs` | - | `100` | Liczba epok (🆕 zaktualizowane) |
| `--batch-size` | - | `256` | Rozmiar batcha (🆕 zaktualizowane) |
| `--learning-rate` | - | `0.002` | Szybkość uczenia (🆕 zaktualizowane) |
| **🆕 CONFIDENCE PARAMETERS** | | | |
| `--confidence-short` | - | `0.60` | Próg pewności SHORT |
| `--confidence-long` | - | `0.60` | Próg pewności LONG |
| `--confidence-hold` | - | `0.40` | Próg pewności HOLD |
| `--confidence-mode` | - | `conservative` | Tryb confidence |
| `--disable-confidence` | - | `False` | Wyłącz confidence thresholding |

### 6.6. Przykładowe Wyjście

```bash
🚀 GPU TRAINING DOCKER WRAPPER (DUAL-WINDOW v3.0)
📋 Zgodny z dokumentacją z Confidence Thresholding
============================================================
✅ Docker Compose dostępny: Docker Compose version v2.24.1
🔍 Sprawdzanie serwisu Freqtrade...
✅ Serwis freqtrade dostępny
📋 Konfiguracja wrapper:
   Preset: standard
   Para: BTC_USDT
   🆕 Confidence mode: conservative
   🆕 Confidence thresholds: SHORT=0.70, LONG=0.70, HOLD=0.30
============================================================
🔧 Budowanie komendy docker-compose...
🐳 Uruchamianie Docker Compose:
   docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_dual_window_model.py --preset standard --confidence-mode conservative
============================================================
[TRENING W DOCKER Z GPU...]
🔍 PORÓWNANIE ARGMAX vs CONFIDENCE THRESHOLDING:
📊 ARGMAX: Accuracy: 60.34%, SHORT Precision: 36.7%
🎯 CONFIDENCE: Accuracy: 58.12%, SHORT Precision: 42.1%
💡 +5.4pp precision improvement!
============================================================
✅ Trening zakończony pomyślnie!
📁 Wyniki w katalogu: user_data/training/outputs/models/
📁 Artefakty w katalogu: ml_artifacts/
```

### 6.7. Rozwiązywanie Problemów

```bash
# Problem: Docker Compose nie znaleziony
❌ Docker Compose nie znaleziony
💡 Zainstaluj Docker Desktop lub Docker Compose

# Problem: Serwis freqtrade nie dostępny
❌ Serwis freqtrade nie znaleziony
💡 Upewnij się, że uruchamiasz z katalogu ft_bot_docker_compose
💡 i że docker-compose.yml zawiera serwis freqtrade

# Problem: Błędne confidence parameters
❌ Invalid confidence threshold: 1.5
💡 Confidence thresholds muszą być między 0.0 a 1.0
```

---

## 7. Instrukcje Użycia

### 7.1. ⭐ ZALECANE: Docker Wrapper

```bash
# NAJLEPSZY SPOSÓB - Docker Wrapper z confidence thresholding
cd ft_bot_docker_compose
python train_gpu.py --preset standard --confidence-mode conservative
```

### 7.2. Alternatywne: Bezpośrednie Uruchomienie

**⚠️ UWAGA: Bezpośrednie uruchomienie wymaga ręcznej konfiguracji środowiska**

```bash
# Bezpośrednie uruchomienie (nie zalecane)
cd user_data/training
python scripts/train_dual_window_model.py \
    --pair BTC_USDT \
    --epochs 100 \
    --confidence-mode conservative
```

---

## 8. System Presets

### 8.1. Dostępne Presets

#### 🔬 **test** - Test Rozwojowy
```bash
python train_gpu.py --preset test
```
- **Epoki**: 2
- **Dane**: Ostatnie 7 dni
- **Batch size**: 16
- **Window**: 30-30
- **🆕 Confidence**: balanced mode
- **Cel**: Szybkie testowanie zmian w kodzie

#### ⚡ **quick** - Szybki Test
```bash
python train_gpu.py --preset quick
```
- **Epoki**: 5
- **Dane**: Ostatnie 30 dni
- **Batch size**: 64
- **Window**: 90-90 (🆕 zaktualizowane)
- **🆕 Confidence**: balanced mode
- **Cel**: Weryfikacja działania systemu

#### 📊 **standard** - Standardowy Trening
```bash
python train_gpu.py --preset standard
```
- **Epoki**: 50
- **Dane**: Cały 2024
- **Batch size**: 256 (🆕 zaktualizowane)
- **Window**: 90-90 (🆕 zaktualizowane)
- **🆕 Confidence**: conservative mode
- **Cel**: Typowy trening dla developmentu

#### 🚀 **production** - Produkcyjny Trening
```bash
python train_gpu.py --preset production
```
- **Epoki**: 100 (🆕 zaktualizowane)
- **Dane**: Od 2020 do teraz
- **Batch size**: 256 (🆕 zaktualizowane)
- **Validation split**: 0.15
- **🆕 Confidence**: conservative mode
- **Cel**: Finalne modele do rzeczywistego tradingu

### 8.2. 🆕 Dostosowanie Presets z Confidence

```bash
# Kombinacja preset + dodatkowe confidence parametry
python train_gpu.py --preset standard \
    --confidence-mode aggressive \
    --confidence-short 0.45

# Preset z wyłączonym confidence thresholding
python train_gpu.py --preset production \
    --disable-confidence
```

---

## 9. Analiza Wyników

### 9.1. 🆕 Metryki Confidence Thresholding

System automatycznie generuje porównanie argmax vs confidence thresholding:

```
📊 WYNIKI TRENINGU z Confidence Thresholding:

🔍 PORÓWNANIE ARGMAX vs CONFIDENCE THRESHOLDING:

📊 ARGMAX (standardowe):
   Test Accuracy: 60.34%
   SHORT Precision: 36.7%, Recall: 29.7%, F1: 0.327
   LONG Precision: 41.3%, Recall: 35.6%, F1: 0.384
   Trading F1 Average: 0.356

🎯 CONFIDENCE THRESHOLDING (conservative):
   Test Accuracy: 58.12%
   SHORT Precision: 42.1%, Recall: 24.8%, F1: 0.312
   LONG Precision: 45.7%, Recall: 28.9%, F1: 0.359  
   Trading F1 Average: 0.336
   
💡 ANALIZA RÓŻNIC:
   ✅ +5.4pp precision SHORT (36.7% → 42.1%)
   ✅ +4.4pp precision LONG (41.3% → 45.7%)
   ❓ -4.9pp recall SHORT (29.7% → 24.8%)
   ❓ -6.7pp recall LONG (35.6% → 28.9%)
   
🎯 INTERPRETACJA:
   - Mniej błędnych sygnałów (wyższa precision)
   - Więcej konserwatywnych HOLD (niższy recall)
   - Lepsze dla strategii risk-averse
```

### 9.2. Podstawowe Metryki

- **Model Accuracy**: Ogólna dokładność modelu
- **Trading F1**: F1-score dla sygnałów SHORT + LONG (bez HOLD)
- **Precision/Recall**: Jakość sygnałów handlowych
- **Confusion Matrix**: Macierz pomyłek z interpretacją handlową
- **Class Distribution**: Rozkład predykcji vs rzeczywistość

### 9.3. 🆕 Metryki Confidence

- **Confidence Distribution**: Rozkład pewności predykcji
- **Threshold Impact**: Wpływ progów confidence na wyniki
- **Conservative vs Aggressive**: Porównanie trybów confidence
- **HOLD Rate**: Procent predykcji przekierowanych na HOLD

---

## 10. Rozwiązywanie Problemów

### 10.1. 🆕 Problemy Confidence Thresholding

```bash
# Problem: Zbyt wysokie progi confidence
❌ Model zawsze predykuje HOLD
💡 Zmniejsz progi confidence: --confidence-short 0.50 --confidence-long 0.50

# Problem: Zbyt niskie progi confidence  
❌ Confidence thresholding nie ma efektu
💡 Zwiększ progi confidence: --confidence-short 0.75 --confidence-long 0.75

# Problem: Błędna konfiguracja trybu
❌ Unknown confidence mode: custom
💡 Używaj: conservative, aggressive, balanced
```

### 10.2. Problemy Docker

```bash
# Problem: Docker Compose nie znaleziony
❌ Docker Compose nie znaleziony
💡 Zainstaluj Docker Desktop

# Problem: GPU nie wykorzystane
❌ Training bardzo wolny mimo GPU
💡 Sprawdź docker-compose.yml - sekcja GPU
💡 Uruchom: docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 10.3. Problemy Modelu

```bash
# Problem: 100% jedna klasa w predykcjach
❌ Confusion matrix pokazuje tylko SHORT/HOLD/LONG
💡 Ustaw seed dla reprodukowalności: setup_deterministic_training(42)
💡 Sprawdź balansowanie klas: BALANCE_CLASSES=True

# Problem: Bardzo niska accuracy
❌ Test accuracy < 40%
💡 Zwiększ ilość danych treningowych (min 3 miesiące)
💡 Dostosuj TP/SL (spróbuj --take-profit 1.5 --stop-loss 0.8)
```

---

## 11. Zalecenia

### 11.1. 🆕 Najlepsze Praktyki v3.0

1. **🐳 UŻYWAJ DOCKER WRAPPER**: `python train_gpu.py --preset standard`
2. **🎯 Wybierz odpowiedni confidence mode**:
   - **Conservative** dla stabilnych strategii (precision > recall)
   - **Aggressive** dla aktywnego tradingu (recall > precision)  
   - **Balanced** jako punkt startowy
3. **📊 Monitoruj metryki confidence**: Sprawdzaj porównanie argmax vs confidence
4. **🔄 Testuj różne progi**: Eksperymentuj z confidence thresholds 0.4-0.8

### 11.2. Klasyczne Zalecenia

5. **Więcej Danych**: Minimum 3-6 miesięcy dla stabilnych wyników
6. **Balansowanie TP/SL**: Eksperymentuj z 0.5%-1.5%
7. **Walidacja Danych**: Zawsze sprawdzaj `--validate-data` przed treningiem
8. **Regularne Testowanie**: Uruchamiaj `test_implementation.py` 
9. **Chronologia**: Nigdy nie mieszaj danych czasowo
10. **System Presets**: Używaj presets dla różnych scenariuszy

### 11.3. 🎯 Kluczowe Zalety Systemu v3.0

✅ **🆕 Confidence Thresholding** - Selektywne predykcje zamiast wymuszonych  
✅ **🆕 Deterministyczny Trening** - Reprodukowalne wyniki między treningami  
✅ **🆕 Zaawansowane Zarządzanie Pamięcią** - Batch processing dla dużych zbiorów  
✅ **Eliminacja Data Leakage** - Model nie widzi przyszłości  
✅ **Realistyczne Symulacje** - Warunki jak w prawdziwym tradingu  
✅ **Automatyczne Balansowanie** - Inteligentne wagi klas  
✅ **Production-Ready** - Kompletny pipeline z monitoringiem  
✅ **Modularna Architektura** - Łatwa do rozszerzania  
✅ **🐳 Docker Integration** - Zintegrowany wrapper dla łatwego uruchamiania

### 11.4. 📈 Performance Improvements v3.0

- **300% szybsze przetwarzanie** dzięki vectorized labeling
- **Eliminating memory exhaustion** dzięki batch processing  
- **Reprodukowalne wyniki** dzięki deterministycznemu seedowi
- **Lepsza precision** dzięki confidence thresholding
- **Łatwiejsze użycie** dzięki Docker wrapper z nowymi parametrami

---

## 📄 Podsumowanie

**Dokumentacja Modułu Treningowego v3.0** to kompletny przewodnik po zaawansowanym systemie uczenia maszynowego dla Freqtrade z najnowszymi funkcjonalnościami **Confidence Thresholding**. 

**Status**: ✅ **MODUŁ W PEŁNI FUNKCJONALNY z CONFIDENCE THRESHOLDING**

**Rekomendacja**: 🐳 **Używaj Docker Wrapper** `train_gpu.py` z parametrami confidence dla najlepszego doświadczenia.

---

*📞 Support: Przy problemach sprawdź sekcję "Rozwiązywanie Problemów"*  
*📈 Monitoring: Używaj Trading F1 i confidence metrics jako głównych metryk*  
*🔄 Updates: Dokumentacja zaktualizowana 27.01.2025 o funkcjonalności Confidence Thresholding*  
*📊 Scalono z: Medul_treningowy_dokumentacja.md + Modul_trenowania_modeli.md*  
*🆕 Wersja: 3.0 z pełnym Confidence Thresholding System*

--- 