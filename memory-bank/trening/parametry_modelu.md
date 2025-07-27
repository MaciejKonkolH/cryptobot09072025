# PARAMETRY MODELU XGBOOST - PRZEWODNIK

## 🎯 **PODSTAWOWE PARAMETRY TRENINGU**

### **1. LICZBA DRZEW (N_ESTIMATORS)**
```python
XGB_N_ESTIMATORS = 300  # Maksymalna liczba drzew
```
- **Co to:** Maksymalna liczba drzew w ensemble
- **Zakres:** 100-1000 (300 to dobry start)
- **Wpływ:** Więcej drzew = dłuższy trening, ryzyko overfitting
- **Optymalizacja:** Early stopping automatycznie wybiera najlepszą liczbę
- **Twój model:** Zatrzymał się na **147 drzewach** (early stopping)

### **2. LEARNING RATE (ETA)**
```python
XGB_LEARNING_RATE = 0.05  # Współczynnik uczenia
```
- **Co to:** Jak szybko model się uczy
- **Zakres:** 0.01 - 0.3 (0.05 to dobry start)
- **Wpływ:** 
  - Niższy = wolniejszy ale stabilniejszy
  - Wyższy = szybszy ale ryzykowny
- **Rekomendacja:** 0.05-0.1 dla większości przypadków
- **Twój model:** 0.05 - konserwatywne podejście

### **3. MAX DEPTH**
```python
XGB_MAX_DEPTH = 5  # Maksymalna głębokość drzewa
```
- **Co to:** Jak głębokie mogą być pojedyncze drzewa
- **Zakres:** 3-10 (5 to dobry kompromis)
- **Wpływ:** 
  - Głębsze = bardziej skomplikowane wzorce, ryzyko overfitting
  - Płytsze = prostsze wzorce, mniej overfitting
- **Rekomendacja:** 5-7 dla większości przypadków
- **Twój model:** 5 - umiarkowana złożoność

### **4. SUBSAMPLE**
```python
XGB_SUBSAMPLE = 0.8  # Procent próbek do każdego drzewa
```
- **Co to:** Jaką część danych używa każde drzewo
- **Zakres:** 0.5-1.0 (0.8 to standard)
- **Wpływ:** 
  - Niższy = mniej overfitting
  - Wyższy = lepsze dopasowanie
- **Rekomendacja:** 0.8-0.9
- **Twój model:** 0.8 - standardowe podejście

### **5. COLSAMPLE_BYTREE**
```python
XGB_COLSAMPLE_BYTREE = 0.8  # Procent cech do każdego drzewa
```
- **Co to:** Jaką część cech używa każde drzewo
- **Zakres:** 0.5-1.0 (0.8 to standard)
- **Wpływ:** 
  - Niższy = mniej overfitting
  - Wyższy = lepsze dopasowanie
- **Rekomendacja:** 0.8-0.9
- **Twój model:** 0.8 - standardowe podejście

## 🔄 **PARAMETRY EARLY STOPPING**

### **6. EARLY STOPPING ROUNDS**
```python
XGB_EARLY_STOPPING_ROUNDS = 15  # Rundy bez poprawy
```
- **Co to:** Ile rund bez poprawy przed zatrzymaniem
- **Zakres:** 10-50 (15 to dobry start)
- **Wpływ:** 
  - Niższy = szybsze zatrzymanie
  - Wyższy = więcej szans na poprawę
- **Twój model:** Zatrzymał się po **15 rundach** bez poprawy

### **7. EVAL METRIC**
```python
eval_metric = 'mlogloss'  # Metryka ewaluacji
```
- **Co to:** Jaką metrykę używa early stopping
- **Opcje:** 
  - 'mlogloss' - standardowa dla klasyfikacji wieloklasowej
  - 'merror' - błąd klasyfikacji
  - 'auc' - area under curve
- **Rekomendacja:** 'mlogloss' dla klasyfikacji wieloklasowej
- **Twój model:** 'mlogloss' - standardowa

## ⚖️ **PARAMETRY BALANSOWANIA**

### **8. CLASS WEIGHTS**
```python
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0}  # WYŁĄCZONE
```
- **Co to:** Wagi dla różnych klas
- **Zakres:** Dowolne wartości dodatnie
- **Wpływ:** Może pomóc z nierównomiernym rozkładem klas
- **Przykład:** {0: 2.0, 1: 2.0, 2: 0.5} - zwiększa wagę LONG/SHORT
- **Twój model:** Wszystkie klasy równe (wyłączone balansowanie)

### **9. SAMPLE WEIGHTS**
```python
ENABLE_WEIGHTED_LOSS = False  # WYŁĄCZONE
```
- **Co to:** Wagi dla poszczególnych próbek
- **Wpływ:** Może pomóc z problematycznymi próbkami
- **Rekomendacja:** Włącz dla nierównomiernych danych
- **Twój model:** Wyłączone

## 🎲 **PARAMETRY SPECYFICZNE DLA XGBOOST**

### **10. GAMMA**
```python
XGB_GAMMA = 0.1  # Minimalna redukcja straty
```
- **Co to:** Minimalna redukcja straty wymagana do podziału
- **Zakres:** 0-10 (0.1 to standard)
- **Wpływ:** 
  - Wyższy = mniej podziałów, prostszy model
  - Niższy = więcej podziałów, bardziej skomplikowany model
- **Rekomendacja:** 0.1-1.0
- **Twój model:** 0.1 - standardowe

### **11. RANDOM STATE**
```python
XGB_RANDOM_STATE = 42  # Ziarno losowości
```
- **Co to:** Ziarno dla powtarzalności wyników
- **Wpływ:** Ten sam seed = te same wyniki
- **Rekomendacja:** Ustaw dla powtarzalności eksperymentów
- **Twój model:** 42 - standardowe

### **12. N_JOBS**
```python
XGB_N_JOBS = -1  # Liczba procesów
```
- **Co to:** Ile procesów używa do treningu
- **Wartości:** 
  - -1 = wszystkie dostępne procesory
  - 1 = jeden proces
  - 4 = 4 procesy
- **Rekomendacja:** -1 dla maksymalnej wydajności
- **Twój model:** -1 - wszystkie procesory

## 📊 **ANALIZA TWOJEGO MODELU**

### **DODATNIE ASPEKTY:**
- ✅ **Early stopping działa** - model zatrzymał się na optymalnej iteracji
- ✅ **Konserwatywne parametry** - niskie ryzyko overfitting
- ✅ **Standardowe wartości** - sprawdzone podejście
- ✅ **Brak błędów infinity/NaN** - problem rozwiązany u źródła

### **MOŻLIWE ULEPSZENIA:**
```python
# Dla lepszych wyników LONG/SHORT:
XGB_LEARNING_RATE = 0.1      # Szybsze uczenie
XGB_MAX_DEPTH = 6            # Więcej złożoności
XGB_SUBSAMPLE = 0.9          # Więcej danych
XGB_COLSAMPLE_BYTREE = 0.9   # Więcej cech
XGB_GAMMA = 0.05             # Więcej podziałów
```

## 🔧 **REKOMENDACJE EKSPERYMENTÓW**

### **1. EKSPERYMENT 1 - SZYBSZE UCZENIE:**
```python
XGB_LEARNING_RATE = 0.1      # Zwiększ z 0.05
XGB_EARLY_STOPPING_ROUNDS = 20  # Więcej szans
```

### **2. EKSPERYMENT 2 - WIĘCEJ ZŁOŻONOŚCI:**
```python
XGB_MAX_DEPTH = 6            # Zwiększ z 5
XGB_GAMMA = 0.05             # Zmniejsz z 0.1
```

### **3. EKSPERYMENT 3 - WIĘCEJ DANYCH:**
```python
XGB_SUBSAMPLE = 0.9          # Zwiększ z 0.8
XGB_COLSAMPLE_BYTREE = 0.9   # Zwiększ z 0.8
```

### **4. EKSPERYMENT 4 - BALANSOWANIE KLAS:**
```python
CLASS_WEIGHTS = {0: 2.0, 1: 2.0, 2: 0.5}  # Włącz balansowanie
ENABLE_WEIGHTED_LOSS = True               # Włącz weighted loss
```

## 📈 **MONITOROWANIE TRENINGU**

### **1. EARLY STOPPING:**
- **Obserwuj:** Czy model zatrzymuje się za wcześnie
- **Sprawdzaj:** Liczba iteracji vs maksymalna
- **Analizuj:** Czy metryka się stabilizuje

### **2. OVERFITTING:**
- **Porównuj:** Train accuracy vs Validation accuracy
- **Sprawdzaj:** Czy różnica rośnie z czasem
- **Analizuj:** Czy validation accuracy spada

### **3. FEATURE IMPORTANCE:**
- **Analizuj:** Które cechy są najważniejsze
- **Sprawdzaj:** Czy ważność ma sens
- **Optymalizuj:** Usuń nieistotne cechy

## 🎯 **PODSUMOWANIE**

### **OBECNE PARAMETRY (KONSERWATYWNE):**
- N_ESTIMATORS: 300 (użyto 147)
- LEARNING_RATE: 0.05
- MAX_DEPTH: 5
- SUBSAMPLE: 0.8
- COLSAMPLE_BYTREE: 0.8
- EARLY_STOPPING_ROUNDS: 15

### **REKOMENDOWANE EKSPERYMENTY:**
1. **Zwiększ learning rate** do 0.1
2. **Zwiększ max_depth** do 6
3. **Włącz class weights** dla lepszego balansu
4. **Zwiększ subsample** do 0.9

### **KLUCZOWE METRYKI DO OBSERWOWANIA:**
- **Early stopping** - czy działa poprawnie
- **Overfitting** - różnica train/val
- **Feature importance** - które cechy są ważne
- **Class balance** - czy LONG/SHORT są rozpoznawane
