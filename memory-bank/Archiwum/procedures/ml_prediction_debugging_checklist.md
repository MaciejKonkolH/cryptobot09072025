# 🚨 PLAN DEBUGOWANIA: Problem z predykcjami ML w strategii FreqTrade

## 📋 **PROBLEM:**
Model ML osiąga 65% accuracy w walidacji i przewiduje 26,752 sygnałów transakcyjnych na danych ze stycznia 2025, ale strategia FreqTrade na tych samych danych generuje 0 sygnałów transakcyjnych. **To jest logiczna sprzeczność!**

## 🎯 **CHECKLIST DEBUGOWANIA:**

### **FAZA 1: WERYFIKACJA DANYCH** ✅❌
- [ ] **1.1** Porównać dane źródłowe (trening vs backtest)
  - [ ] Sprawdzić źródło danych (Binance API vs pliki)
  - [ ] Porównać timeframe (1m vs 1m)
  - [ ] Porównać zakres dat (styczeń 2025)
  - [ ] Porównać liczbę świec i timestamp
- [ ] **1.2** Sprawdzić preprocessing danych
  - [ ] Porównać obliczanie OHLCV changes
  - [ ] Porównać obliczanie MA1440 i MA43200
  - [ ] Sprawdzić buffer system (czy działa poprawnie)
  - [ ] Porównać kolejność i nazwy kolumn

### **FAZA 2: WERYFIKACJA FEATURES** ✅❌
- [x] **2.1** Sprawdzić obliczanie features
  - [x] Porównać wzory features (trening vs strategia) ✅ NAPRAWIONE!
  - [x] Sprawdzić kolejność features (już naprawione, ale zweryfikować) ✅
  - [ ] Porównać wartości features dla tych samych dat
  - [x] Sprawdzić czy brak NaN/inf w features ✅ NAPRAWIONE!
- [ ] **2.2** Sprawdzić features scaling
  - [ ] Porównać parametry scalera (mean, scale, center)
  - [ ] Sprawdzić czy scaler jest poprawnie załadowany
  - [ ] Porównać scaled features (trening vs strategia)
  - [ ] Sprawdzić czy scaler.transform() działa poprawnie

### **FAZA 3: WERYFIKACJA MODELU** ✅❌
- [ ] **3.1** Sprawdzić ładowanie modelu
  - [ ] Porównać ścieżki do modelu
  - [ ] Sprawdzić czy model.h5 jest poprawnie załadowany
  - [ ] Porównać architekturę modelu (warstwy, parametry)
  - [ ] Sprawdzić czy wagi modelu są identyczne
- [ ] **3.2** Sprawdzić predykcje modelu
  - [ ] Porównać raw predictions (probabilities)
  - [ ] Sprawdzić kształt predykcji (shape)
  - [ ] Porównać confidence scores
  - [ ] Sprawdzić czy model.predict() działa identycznie

### **FAZA 4: WERYFIKACJA PIPELINE** ✅❌
- [ ] **4.1** Sprawdzić window_size i sequences
  - [ ] Porównać window_size (120 vs 120)
  - [ ] Sprawdzić tworzenie sequences
  - [ ] Porównać padding (pierwsze 120 wierszy)
  - [ ] Sprawdzić indeksowanie sequences
- [ ] **4.2** Sprawdzić konwersję predykcji
  - [ ] Porównać progi confidence (43% vs 42%)
  - [ ] Sprawdzić logikę konwersji probabilities → signals
  - [ ] Porównać klasyfikację (argmax)
  - [ ] Sprawdzić mapowanie klas (0=SHORT, 1=HOLD, 2=LONG)

### **FAZA 5: WERYFIKACJA KONFIGURACJI** ✅❌
- [x] **5.1** Sprawdzić parametry treningu vs strategii
  - [x] Porównać SEQUENCE_LENGTH (120 vs 120) ✅
  - [x] Porównać FUTURE_WINDOW (120 vs 120) ✅
  - [x] Porównać TP/SL parametry (1.0/0.5 vs 1.0/0.5) ✅
  - [ ] Porównać batch_size i inne parametry
- [ ] **5.2** Sprawdzić metadata modelu
  - [ ] Porównać metadata.json
  - [ ] Sprawdzić kompatybilność parametrów
  - [ ] Porównać feature_columns
  - [ ] Sprawdzić confidence thresholds

### **FAZA 6: TESTY DIAGNOSTYCZNE** ✅❌
- [ ] **6.1** Test na identycznych danych
  - [ ] Wziąć dokładnie te same dane (10 świec)
  - [ ] Przepuścić przez trening pipeline
  - [ ] Przepuścić przez strategię pipeline
  - [ ] Porównać wyniki krok po kroku
- [ ] **6.2** Test confidence scores
  - [ ] Wylogować wszystkie confidence scores
  - [ ] Sprawdzić rozkład confidence
  - [ ] Porównać z progami (42% vs 43%)
  - [ ] Sprawdzić czy są sygnały powyżej progu

### **FAZA 7: ROZWIĄZANIE** ✅❌
- [ ] **7.1** Identyfikacja root cause
  - [ ] Określić dokładną przyczynę problemu
  - [ ] Udokumentować różnice
  - [ ] Zaplanować naprawę
- [ ] **7.2** Implementacja fix
  - [ ] Naprawić zidentyfikowany problem
  - [ ] Przetestować naprawę
  - [ ] Zweryfikować że sygnały są generowane

## 🚨 **CZERWONE FLAGI DO SPRAWDZENIA:**

1. ~~**SEQUENCE_LENGTH: 300 (trening) vs 120 (strategia)**~~ ✅ ROZWIĄZANE
2. **Różne źródła danych** (Kaggle vs Binance API) ⚠️
3. **Buffer system** - czy poprawnie ładuje dane historyczne ⚠️
4. **Scaler parameters** - czy identyczne między treningiem a strategią ⚠️
5. **Model architecture** - czy model został poprawnie załadowany ⚠️

## 📊 **EXPECTED OUTCOME:**
Po przejściu przez checklist powinniśmy znaleźć przyczynę dlaczego:
- **Walidacja:** 26,752 sygnałów na styczniu 2025
- **Backtest:** 0 sygnałów na styczniu 2025

## 🎯 **NOWY PRIORITET:**
1. **FAZA 2.2** - Sprawdzić scaler parameters (mean, scale, center)
2. **FAZA 3.2** - Porównać raw predictions i confidence scores  
3. **FAZA 1.2** - Sprawdzić czy dane źródłowe są identyczne 

## ✅ POTWIERDZONY PROBLEM PRZESUNIĘCIA (2024-12-20)

### 🚨 DOWÓD PRZESUNIĘCIA O 119 MINUT:
```
WALIDACJA:  2024-12-20 00:00:00 → predykcja: 0.47290429,0.16065080,0.36644489
FREQTRADE:  2024-12-20 01:59:00 → predykcja: 0.47291863,0.16063705,0.36644438
```

**WNIOSEK**: Identyczne modele, scalery i features, ale FreqTrade mapuje predykcje o 119 minut później!

### 🔍 POTWIERDZONE FAKTY:
- ✅ System bufora dostarcza pełne 120x8 features dla każdej świecy
- ✅ Pierwsza świeca NIE JEST problemem (buffer rozwiązuje to)
- ✅ startup_candle_count = 0 vs 120 nie ma wpływu na backtesting
- ✅ startup_candle_count to parametr tylko live tradingu
- ❌ Problem JEST w mapowaniu predykcji do timestampów

### 🎯 LOKALIZACJA PROBLEMU:
Prawdopodobnie w `signal_generator.py` w funkcji `_add_predictions_to_dataframe()`:
```python
start_idx = window_size - 1  # = 119 - może to być przyczyna?
```

---

## General ML Debugging Checklist

### 1. Data Consistency
- [ ] Raw features identical between training and prediction
- [ ] Scaled features identical between training and prediction  
- [ ] Same scaler object used (check scaler parameters)
- [ ] Same model architecture and weights

### 2. Sequence Generation
- [ ] Window size matches training (120)
- [ ] Feature order matches training
- [ ] Sequence indexing correct ([t-120:t] not [t:t+120])
- [ ] No off-by-one errors in sequence creation

### 3. Timestamp Alignment
- [ ] **CRITICAL**: Check prediction timestamp mapping
- [ ] Verify first prediction timestamp matches validation
- [ ] Compare prediction outputs for same timestamp
- [ ] Check for timezone issues

### 4. Model Loading
- [ ] Model loads without errors
- [ ] Scaler loads without errors
- [ ] Model metadata matches training config
- [ ] Feature names and order preserved

### 5. Buffer System
- [ ] MA calculations identical to training
- [ ] Feature calculations match validation module
- [ ] No data leakage from future
- [ ] Proper handling of NaN/inf values

### 6. Signal Generation
- [ ] Threshold application correct
- [ ] Signal mapping (0=SHORT, 1=HOLD, 2=LONG) consistent
- [ ] Confidence calculation matches training
- [ ] CSV logging captures all predictions

### 7. Common Issues
- [ ] Off-by-one errors in indexing
- [ ] Timezone mismatches
- [ ] Different feature calculation methods
- [ ] Scaler state differences
- [ ] Model version mismatches
- [ ] **TIMESTAMP MAPPING ERRORS** ⚠️

### 8. Debugging Tools
- [ ] Save debug features to compare with training
- [ ] Log prediction timestamps and values
- [ ] Compare raw vs scaled features
- [ ] Verify sequence shapes and contents
- [ ] Cross-reference with validation module outputs 