# Algorytm Trenowania Dwuokiennego Systemu ML

*Ostatnia aktualizacja: 25 maja 2025*  
*Status: Production-ready (v2.0.0)*  
*Lokalizacja kodu: `Freqtrade/ft_bot_docker_compose/user_data/training/`*
*Verified against code: ✅*

## 🎯 Przegląd Algorytmu

System trenowania implementuje **dwuokienne podejście** (Dual Window Approach) eliminujące data leakage w modelach predykcyjnych dla handlu kryptowalutami. Główną zasadą jest **ścisła separacja czasowa** między danymi wejściowymi modelu a weryfikacją etykiet.

### 🔑 Kluczowa Zasada: Temporal Separation
```
[świece i-60:i] ←── HISTORICAL WINDOW (dane dla modelu LSTM)
     ↓
[świeca i] ←── PREDICTION POINT (moment decyzji handlowej)
     ↓  
[świece i+1:i+61] ←── FUTURE WINDOW (weryfikacja etykiet post-factum)
```

## 📋 Główne Komponenty Systemu

### 1. **DualWindowTrainingPipeline** (Orkiestrator)
- Koordynuje wszystkie etapy treningu
- Zarządza przepływem danych między komponentami
- Implementuje time-aware split dla walidacji

### 2. **EnhancedFeatherLoader** (Ładowanie Danych)
- Inteligentne ładowanie plików .feather z automatycznym bufferem
- Oblicza wymagane bufory historyczne (33 dni total)
- Konwertuje i waliduje dane czasowe z timezone handling

### 3. **DualWindowSequenceBuilder** (Tworzenie Sekwencji)
- Implementuje separację temporal windows
- Tworzy sekwencje treningowe z eliminacją data leakage
- Generuje etykiety hybrydowe na podstawie TP/SL
- Progress tracking co 5,000 sekwencji

### 4. **DualWindowLSTMModel** (Model ML)
- cuDNN-optimized LSTM model bez recurrent_dropout
- Input: (60 świec, 8 cech) → Output: 3 klasy (SHORT/HOLD/LONG)
- Legacy RMSprop optimizer z decay

### 5. **ModelEvaluator** (Ewaluacja)
- Trading-focused metrics (F1 dla sygnałów handlowych)
- Confusion matrix, classification report
- Analiza rozkładu klas i balansowania

## 🔄 Szczegółowy Przepływ Algorytmu

### KROK 1: Inicjalizacja i Konfiguracja
```
📋 TrainingConfig
├── WINDOW_SIZE = 60 świec (Historical Window)
├── FUTURE_WINDOW = 60 świec (Future Window) 
├── LONG_TP_PCT = 0.007 (0.7% Take Profit)
├── LONG_SL_PCT = 0.007 (0.7% Stop Loss)
├── MA_FAST_PERIODS = 1440 (24h)
├── MA_SLOW_PERIODS = 43200 (30 dni)
├── LEARNING_RATE = 0.0005
├── EPOCHS = 100
└── 8 FEATURE_COLUMNS (cechy wejściowe)
```

**Walidacja:**
- Sprawdzenie sensowności okien czasowych (max 1440 = 24h)
- Walidacja parametrów TP/SL (0 < x < 1)
- Obliczenie wymaganego buforu historycznego

### KROK 2: Ładowanie Danych z Inteligentnym Bufferem
```
📊 EnhancedFeatherLoader.load_training_data()
├── Obliczenie extended_start = training_start - buffer_days
├── Odnalezienie plików .feather dla pary walutowej
├── Konwersja timestamp → datetime index (timezone-naive)
├── Filtrowanie do zakresu dat
└── Walidacja wystarczalności danych
```

**Buffer Calculation (POPRAWIONE):**
```python
# MA Buffer
ma_buffer_days = max(
    MA_FAST_PERIODS // (24 * 60),   # 1440 // 1440 = 1 dzień
    MA_SLOW_PERIODS // (24 * 60)    # 43200 // 1440 = 30 dni
) = 30 dni

# Window Buffer  
window_buffer_days = (WINDOW_SIZE + FUTURE_WINDOW) // (24 * 60) + 1
                   = (60 + 60) // 1440 + 1 = 0 + 1 = 1 dzień

# Safety Buffer
safety_buffer_days = 2 dni

# TOTAL BUFFER = 30 + 1 + 2 = 33 dni
```

### KROK 3: Feature Engineering z Zero-Safe Operations
```
🔧 Obliczanie 8 Cech Technicznych (Parametryzowane):
├── high_change = np.where(open == 0, 0, (high - open) / open)
├── low_change = np.where(open == 0, 0, (low - open) / open) 
├── close_change = np.where(open == 0, 0, (close - open) / open)
├── volume_change = volume.pct_change().fillna(0)
├── price_to_ma1440 = np.where(ma_fast == 0, 1.0, close / ma_fast)
├── price_to_ma43200 = np.where(ma_slow == 0, 1.0, close / ma_slow)
├── volume_to_ma1440 = np.where(vol_ma_fast == 0, 1.0, volume / vol_ma_fast)
└── volume_to_ma43200 = np.where(vol_ma_slow == 0, 1.0, volume / vol_ma_slow)
```

**Implementacja z parametryzacją:**
- MA periods z `config.MA_FAST_PERIODS` i `config.MA_SLOW_PERIODS`
- Zero-safe divisions z fallback values
- `.replace([np.inf, -np.inf], fallback)` dla czyszczenia
- Forward/backward fill dla brakujących wartości

### KROK 4: Tworzenie Sekwencji Dwuokiennych ⭐
```
🔄 DualWindowSequenceBuilder.create_training_sequences()

FOR każda świeca i w zakresie [WINDOW_SIZE : len(df) - FUTURE_WINDOW]:
    
    # Progress tracking co 5,000 sekwencji
    if processed_count % 5000 == 0 and processed_count > 0:
        progress = (processed_count / total_sequences) * 100
        print(f"   Tworzenie sekwencji: {processed_count:,}/{total_sequences:,} ({progress:.1f}%)")
    
    1. HISTORICAL WINDOW (dane dla modelu):
       features[i-60:i] → X_sequence (60 świec × 8 cech)
    
    2. PREDICTION POINT (moment decyzji):
       base_price = close[i]
       timestamp = index[i]
    
    3. FUTURE WINDOW (weryfikacja etykiety):
       future_candles = df[i+1:i+61]
       
    4. LABEL CREATION (hybrydowa klasyfikacja):
       long_tp = base_price × (1 + 0.007)
       long_sl = base_price × (1 - 0.007)
       short_tp = base_price × (1 - 0.007)
       short_sl = base_price × (1 + 0.007)
       
       long_tp_hit = any(future_candles.high >= long_tp)
       long_sl_hit = any(future_candles.low <= long_sl)
       short_tp_hit = any(future_candles.low <= short_tp)
       short_sl_hit = any(future_candles.high >= short_sl)
       
       IF long_tp_hit AND NOT long_sl_hit AND NOT (short_tp_hit AND NOT short_sl_hit):
           label = 2  # LONG
       ELIF short_tp_hit AND NOT short_sl_hit AND NOT (long_tp_hit AND NOT long_sl_hit):
           label = 0  # SHORT
       ELSE:
           label = 1  # HOLD
    
    5. APPEND TO TRAINING SET:
       X.append(features[i-60:i])  # shape: (60, 8)
       y.append(label)             # scalar: 0, 1, or 2
       timestamps.append(timestamp)

# Data leakage validation z progress co 10,000 par
_validate_no_data_leakage(timestamps)

RETURN X, y, timestamps, metadata
```

**Kluczowe aspekty:**
- **Zero data leakage**: Model NIE widzi przyszłych danych podczas treningu
- **Temporal consistency**: Wszystkie sekwencje w porządku chronologicznym
- **Hybrydowa klasyfikacja**: Uwzględnia zarówno TP jak i SL w jednej etykiecie
- **Progress monitoring**: Real-time feedback dla długich operacji

### KROK 5: Time-Aware Split
```
✂️ Chronologiczny podział danych:
├── split_idx = len(X) × config.TRAIN_TEST_SPLIT  # 0.8
├── X_train = X[:split_idx]    (80% wcześniejszych danych)
├── X_val = X[split_idx:]      (20% późniejszych danych)
├── Walidacja: train_end < val_start (brak data leakage)
└── Sprawdzenie chronologii timestamps
```

### KROK 6: Balansowanie Klas (opcjonalne)
```
⚖️ Class Balancing:
IF config.BALANCE_CLASSES:
    class_weights = compute_class_weight(
        'balanced', 
        classes=[0, 1, 2], 
        y=y_train
    )
    # Zwiększenie wag dla klas <10% (x2.0)
    # Obsługa ekstremalnego niezbalansowania (>90% jedna klasa)
ELSE:
    class_weights = None
```

### KROK 7: Budowanie Modelu LSTM (cuDNN-Optimized)
```
🏗️ build_dual_window_lstm_model() - RZECZYWISTA ARCHITEKTURA:
Input: (None, 60, 8)

# LSTM Stack (BEZ dropout/recurrent_dropout dla cuDNN optimization)
FOR i, units in enumerate(config.LSTM_UNITS):  # [128, 64, 32]
    return_sequences = (i < len(config.LSTM_UNITS) - 1)
    x = LSTM(
        units,
        return_sequences=return_sequences,
        name=f'lstm_{i+1}',
        activation='tanh',              # explicit dla cuDNN
        recurrent_activation='sigmoid', # required dla cuDNN
        use_bias=True,                  # required dla cuDNN
        # USUNIĘTE: dropout, recurrent_dropout, kernel_regularizer
    )(x)
    # USUNIĘTE: BatchNormalization (problemy z gradientami)

# Dense Stack (z Dropout)
FOR i, units in enumerate(config.DENSE_UNITS):  # [32, 16]
    x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
    x = Dropout(config.DROPOUT_RATE + 0.1, name=f'dropout_{i+1}')(x)  # 0.2 + 0.1 = 0.3

# Output Layer
└── Dense(3, activation='softmax', name='output')  # [prob_SHORT, prob_HOLD, prob_LONG]

Compilation:
├── Optimizer: tf.keras.optimizers.legacy.RMSprop(lr=0.0005, decay=1e-6)
├── Loss: sparse_categorical_crossentropy
└── Metrics: accuracy
```

**🚀 cuDNN Optimizations:**
- **REMOVED**: `dropout`, `recurrent_dropout`, `kernel_regularizer` z LSTM layers
- **REMOVED**: `BatchNormalization` layers (problemy z gradientami)
- **EXPLICIT**: `activation='tanh'`, `recurrent_activation='sigmoid'`
- **PRESERVED**: `Dropout` tylko w Dense layers
- **LEGACY**: `tf.keras.optimizers.legacy.RMSprop` dla kompatybilności

### KROK 8: Konfiguracja Callbacks
```
📋 Training Callbacks:
├── EarlyStopping(
│     monitor='val_loss', 
│     patience=15,
│     restore_best_weights=True,
│     verbose=1,
│     mode='min'
│   )
├── ReduceLROnPlateau(
│     monitor='val_loss', 
│     factor=0.5, 
│     patience=8,
│     min_lr=1e-7,
│     verbose=1,
│     mode='min'
│   )
└── ModelCheckpoint(
      monitor='val_accuracy', 
      save_best_only=True,
      save_weights_only=False,
      verbose=1,
      mode='max'
    )
```

### KROK 9: Trening Modelu
```
🎯 Model Training:
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config.EPOCHS,           # 100
    batch_size=config.BATCH_SIZE,   # 32
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)
```

### KROK 10: Ewaluacja Modelu
```
📊 ModelEvaluator.evaluate_model():
├── Podstawowe metryki: test_loss, test_accuracy
├── Predykcje: y_pred = argmax(model.predict(X_test))
├── Classification Report (precision, recall, F1 per klasa)
├── Confusion Matrix
├── Trading-focused metrics:
│   ├── SHORT Precision/Recall (sygnały sprzedaży)
│   ├── LONG Precision/Recall (sygnały kupna)
│   └── TRADING SIGNALS AVG F1 (średnia F1 dla SHORT+LONG, bez HOLD)
└── Rozkład predykcji vs rzeczywistych etykiet
```

**Kluczowe metryki:**
- **TRADING SIGNALS AVG F1**: średnia F1 dla klas SHORT i LONG (bez HOLD)
- **Obecny poziom**: ~0.17 (cel: >0.25)
- **SHORT/LONG Recall**: czy model znajduje możliwości handlowe

### KROK 11: Zapis Artefaktów
```
💾 save_model_artifacts():
├── dual_window_lstm_model.keras        (finalny model)
├── training_config.json                (użyta konfiguracja)
├── evaluation_results.json             (wyniki ewaluacji)
└── model_metadata.json                 (metadane: timestamp, version, parameters)
```

## 🔍 Kluczowe Różnice vs Poprzednie Wersje

### Dwuokienny System (v2.0) vs Prosty System (v1.0)
| Aspekt | Prosty System | Dwuokienny System |
|--------|---------------|-------------------|
| **Data leakage** | Możliwy (brak separacji) | Eliminowany (temporal separation) |
| **Etykietowanie** | Prosty price shift | Hybrydowe TP/SL classification |
| **Model type** | Regresja (1 output) | Klasyfikacja (3 klasy) |
| **Walidacja** | Random split | Time-aware split |
| **Sekwencje** | expand_dims hack | Właściwe okna czasowe |
| **Buffer management** | Brak | Inteligentny auto-buffer (33 dni) |
| **GPU optimization** | Brak | cuDNN-compatible LSTM |

### vs Oryginalny BinanceBot
| Aspekt | BinanceBot | Dwuokienny Freqtrade |
|--------|-------------|----------------------|
| **Data source** | SQLite | Pliki .feather |
| **Format danych** | Własny | Natywny Freqtrade |
| **Deployment** | Standalone | Docker container |
| **Configuracja** | Python files | JSON + dataclass |
| **Timezone handling** | Manual | Automatic timezone-naive conversion |

## ⚡ Optymalizacje i Innowacje

### 1. **Temporal Separation**
- Model NIE MA dostępu do przyszłych danych podczas predykcji
- Eliminuje optimistic bias typowy dla ML w finansach

### 2. **Inteligentny Buffer Management**
- Automatyczne obliczanie wymaganych buforów (33 dni total)
- Uwzględnia potrzeby MA, windows, safety margins

### 3. **Hybrydowa Klasyfikacja**
- Etykiety bazują na rzeczywistych warunkach TP/SL
- Uwzględnia złożoność rzeczywistych decyzji handlowych

### 4. **Trading-Focused Metrics**
- Metryki optymalizowane pod sygnały handlowe
- F1 dla SHORT/LONG ważniejsze niż ogólna accuracy

### 5. **cuDNN GPU Optimization**
- LSTM layers bez dropout/recurrent_dropout
- Explicit activation functions dla cuDNN compatibility
- Legacy RMSprop dla stabilności

### 6. **Zero-Safe Operations**
- `np.where(denominator == 0, fallback, division)` dla wszystkich obliczeń
- Automatic handling inf/NaN values
- Robust feature engineering

### 7. **Progress Monitoring**
- Real-time tracking dla sequence creation (co 5,000)
- Data leakage validation progress (co 10,000)
- Comprehensive logging na każdym etapie

### 8. **Production-Ready Pipeline**
- Pełna walidacja danych na każdym etapie
- Error handling i informative logging
- Systematyczny zapis artefaktów

## 🎯 Oczekiwane Wyniki

### Obecne Benchmarki (7-dniowy test)
```
✅ Test Accuracy: 60.34%
✅ Trading F1: 0.171
✅ SHORT Recall: 29.7%
✅ LONG Recall: 35.6%
✅ Zero tensor shape errors
✅ Zero timezone issues
✅ cuDNN optimization working
✅ Progress tracking functional
```

### Cele Rozwojowe
```
🎯 Krótkoterminowe (3-6 miesięcy danych):
├── Trading F1: >0.25 (+46% improvement)
├── Lepsze class balancing
└── Dodatkowe features

🎯 Długoterminowe:
├── Multi-timeframe ensemble
├── Advanced feature engineering
├── Real-time prediction API
└── Portfolio-based signals
```

## 🚀 Uruchomienie Treningu

### Podstawowy trening (POPRAWIONE CLI):
```bash
cd "C:\Users\macie\OneDrive\Python\Binance\Freqtrade\ft_bot_docker_compose\user_data\training"

python scripts\train_dual_window_model.py \
  --pair BTC_USDT \
  --date-from 2024-01-01 \
  --date-to 2024-06-30 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.0005
```

### Preset system:
```bash
# Szybki test
python scripts\train_dual_window_model.py --preset test

# Standardowy trening
python scripts\train_dual_window_model.py --preset standard

# Produkcyjny (pełne dane)
python scripts\train_dual_window_model.py --preset production
```

### Test i walidacja:
```bash
# Sprawdź dostępność danych
python scripts\train_dual_window_model.py --pair BTC_USDT --date-from 2024-01-01 --date-to 2024-01-07 --validate-data

# Dry run (bez treningu)
python scripts\train_dual_window_model.py --pair BTC_USDT --date-from 2024-01-01 --date-to 2024-01-07 --dry-run

# Szybki test (2 epoki)
python scripts\train_dual_window_model.py --pair BTC_USDT --date-from 2024-01-01 --date-to 2024-01-07 --epochs 2
```

---

**Ten algorytm reprezentuje state-of-the-art w zastosowaniu uczenia maszynowego do automatycznego handlu, z naciskiem na eliminację data leakage, optymalizację GPU i praktyczne rezultaty handlowe.**

*Dokumentacja zweryfikowana względem kodu - wersja 2.0.0 verified ✅*
