# USTALENIA - MODUŁ TRAINING3

## 🎯 **PODSTAWOWE ZASADY**

### **1. Podział danych - CHRONOLOGICZNY (OBOWIĄZKOWY)**
- **NIE WOLNO** zmieniać na losowy podział
- Chronologiczny podział zapobiega data leakage z przyszłości
- Train: 70% najstarszych danych (2023-2024)
- Val: 15% środkowych danych  
- Test: 15% najnowszych danych (2025)

### **2. Balansowanie klas - WYŁĄCZONE**
- `ENABLE_CLASS_BALANCING = False`
- `ENABLE_WEIGHTED_LOSS = False`
- Wszystkie klasy mają równe wagi: `CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0}`
- **NIE WOLNO** włączać balansowania bez zgody użytkownika

### **3. Early Stopping - ROZWIĄZANE**
- Przepisano kod na native XGBoost API
- `early_stopping_rounds` działa w `xgb.train()`
- Progress logs działają z `verbose_eval=True`
- Dodano `predict_proba()` dla prawdopodobieństw

### **4. Progress Logs - WYMAGANE**
- Użytkownik musi widzieć postęp treningu w czasie rzeczywistym
- `verbose=True` działa i pokazuje mlogloss
- To jest krytyczne dla użytkownika

## 🔍 **ZIDENTYFIKOWANE PROBLEMY**

### **1. Model Bias - SHORT > LONG - ROZWIĄZANE**
- **Dane z labeler3**: SHORT/LONG proporcja 1.0-1.1 (symetryczna)
- **Algorytm etykietowania**: POPRAWNY i symetryczny
- **Data drift**: Znikome różnice (1-6% w NEUTRAL, trend SHORT/LONG 0.001-0.011)
- **Problem**: NIE w data drift (różnice zbyt małe)
- **Model bias**: SHORT recall 0.001-0.151, LONG recall 0.000-0.001 (SHORT > LONG)
- **PRZYCZYNA**: RÓŻNICE W CECHACH między LONG a SHORT (RSI, MACD, MA, bb_position)
- **Status**: PRZYCZYNA ZNALEZIONA - różnice w charakterystyce cech

### **2. Data Drift - WYELIMINOWANE**
- Model trenuje się na 2023-2024, testuje na 2025
- **Różnice znikome**: 1-6% w NEUTRAL, trend SHORT/LONG 0.001-0.011
- **Wniosek**: Data drift NIE wyjaśnia katastrofalnych wyników
- **Status**: Problem NIE w data drift



## 📋 **ZADANIA DO WYKONANIA**

### **1. Early Stopping - ROZWIĄZANE**
- Przepisano na native XGBoost API
- Early stopping działa
- Progress logs działają

### **2. Sprawdzić algorytm etykietowania - ROZWIĄZANE**
- ✅ Zweryfikowano symetrię TP/SL w labeler3
- ✅ Algorytm jest poprawny i symetryczny
- ✅ SHORT/LONG proporcja 1.0-1.1 w normie
- ✅ Problem nie jest w etykietowaniu

### **3. Rozwiązać problem model bias - ROZWIĄZANE**
- **Etykietowanie OK**: SHORT/LONG proporcja 1.0-1.1
- **Data drift**: Znikome różnice (NIE wyjaśnia problemu)
- **Problem w treningu**: Model ma bias SHORT > LONG
- **Model**: Przewiduje głównie NEUTRAL (87% danych) - TO NORMALNE
- **Bias SHORT > LONG**: SHORT recall 0.001-0.151, LONG recall 0.000-0.001
- **PRZYCZYNA**: RÓŻNICE W CECHACH między LONG a SHORT
- **Szczegóły**: LONG (RSI=52, MACD=+, cena>MA) vs SHORT (RSI=48, MACD=-, cena<MA)
- **Rozwiązanie**: Sample weights lub balansowanie cech

## 🚫 **ZAKAZY**

### **1. NIE WOLNO:**
- Zmieniać chronologicznego podziału danych
- Włączać balansowania klas bez zgody
- Wprowadzać zmian bez pozwolenia użytkownika
- Zapominać o ustaleniach

### **2. NIE WOLNO:**
- Proponować losowego podziału danych
- Ignorować problem data leakage
- Zapominać o wymaganiu progress logs

## ✅ **CO DZIAŁA**

### **1. Progress Logs**
- `verbose=True` pokazuje postęp
- Widoczne wartości mlogloss w czasie rzeczywistym
- Użytkownik widzi że kod się wykonuje

### **2. Mapowanie klas**
- `LABEL_MAPPING = {'LONG': 0, 'SHORT': 1, 'NEUTRAL': 2}`
- Poprawne w labeler3 i training3
- Nie ma błędu w definicji klas

### **3. Struktura danych**
- 5 osobnych modeli XGBoost
- Każdy dla innego poziomu TP/SL
- Multi-output classification

### **4. Etykietowanie - ZWERYFIKOWANE**
- ✅ Algorytm etykietowania jest poprawny i symetryczny
- ✅ SHORT/LONG proporcja 1.0-1.1 w normie
- ✅ Wszystkie 5 poziomów TP/SL mają symetryczne rozkłady
- ✅ Problem nie jest w module labeler3

## 📝 **OSTATNIE USTALENIA**

### **Data**: 2025-07-26
### **Status**: Early stopping ROZWIĄZANE, etykietowanie ZWERYFIKOWANE, data drift WYELIMINOWANE, PRZYCZYNA BIAS ZNALEZIONA, CECHY ZIDENTYFIKOWANE JAKO PROBLEM
### **Ostatnie ustalenie**: Problem SHORT > LONG - PRZYCZYNA ZNALEZIONA
### **Wszystkie sprawdzone elementy OK**: etykietowanie, data drift, podział danych, konfiguracja
### **Przyczyna bias**: RÓŻNICE W CECHACH między LONG a SHORT (RSI, MACD, MA, bb_position)
### **Szczegóły**: LONG (RSI=52, MACD=+, cena>MA) vs SHORT (RSI=48, MACD=-, cena<MA)
### **NOWE USTALENIE**: Cechy są bezużyteczne do treningu (73 cechy absolutne zamiast względnych)
### **Następny krok**: Przeprojektować cechy na 25 względnych cech (patrz: idealne_cechy_treningowe.md) 