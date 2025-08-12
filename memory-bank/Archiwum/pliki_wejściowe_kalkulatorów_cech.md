# PLIKI WEJŚCIOWE KALKULATORÓW CECH - ANALIZA

**Data utworzenia:** 4 sierpnia 2025  
**Cel:** Sprawdzenie z jakich plików wejściowych korzystają oba kalkulatory cech

## 📊 **PODSUMOWANIE PLIKÓW WEJŚCIOWYCH**

### **Stary kalkulator (`feature_calculator_ohlc_snapshot`):**

#### **Konfiguracja:**
- **INPUT_DIR:** `PROJECT_ROOT / "merge"`
- **INPUT_FILENAME:** `"merged_ohlc_orderbook.feather"`
- **INPUT_FILE_PATH:** `merge/merged_ohlc_orderbook.feather`

#### **Pełna ścieżka:**
```
C:\Users\macie\OneDrive\Python\Binance\crypto\merge\merged_ohlc_orderbook.feather
```

#### **Plik wyjściowy:**
- **DEFAULT_OUTPUT_FILENAME:** `'ohlc_orderbook_features.feather'`
- **Ścieżka:** `feature_calculator_ohlc_snapshot/output/ohlc_orderbook_features.feather`

---

### **Nowy kalkulator (`feature_calculator_download2`):**

#### **Konfiguracja:**
- **INPUT_DIR:** `PROJECT_ROOT / "download2" / "merge" / "merged_data"`
- **INPUT_FILENAME:** `"merged_{symbol}.feather"`
- **INPUT_FILE_PATH:** `download2/merge/merged_data/merged_{symbol}.feather`

#### **Pełna ścieżka (dla BTCUSDT):**
```
C:\Users\macie\OneDrive\Python\Binance\crypto\download2\merge\merged_data\merged_BTCUSDT.feather
```

#### **Plik wyjściowy:**
- **DEFAULT_OUTPUT_FILENAME:** `'features_{symbol}.feather'`
- **Ścieżka:** `feature_calculator_download2/output/features_BTCUSDT.feather`

## 🔍 **RÓŻNICE W KONFIGURACJI**

### **1. Struktura katalogów:**
- **Stary:** Używa prostego katalogu `merge/`
- **Nowy:** Używa zagnieżdżonej struktury `download2/merge/merged_data/`

### **2. Nazewnictwo plików:**
- **Stary:** Stała nazwa `merged_ohlc_orderbook.feather`
- **Nowy:** Dynamiczna nazwa `merged_{symbol}.feather` (np. `merged_BTCUSDT.feather`)

### **3. Obsługa symboli:**
- **Stary:** Jeden plik dla wszystkich symboli
- **Nowy:** Osobny plik dla każdego symbolu

## 📁 **STRUKTURA KATALOGÓW**

### **Stary pipeline:**
```
crypto/
├── merge/
│   └── merged_ohlc_orderbook.feather  ← Plik wejściowy
└── feature_calculator_ohlc_snapshot/
    └── output/
        └── ohlc_orderbook_features.feather  ← Plik wyjściowy
```

### **Nowy pipeline:**
```
crypto/
├── download2/
│   └── merge/
│       └── merged_data/
│           └── merged_BTCUSDT.feather  ← Plik wejściowy
└── feature_calculator_download2/
    └── output/
        └── features_BTCUSDT.feather  ← Plik wyjściowy
```

## 🎯 **WYWOŁANIA KALKULATORÓW**

### **Stary kalkulator:**
```bash
python feature_calculator_ohlc_snapshot/main.py
# Domyślnie używa: merge/merged_ohlc_orderbook.feather
```

### **Nowy kalkulator:**
```bash
python feature_calculator_download2/main.py --input download2/merge/merged_data/merged_BTCUSDT.feather --output feature_calculator_download2/output/features_BTCUSDT.feather
```

## 📊 **PORÓWNANIE PLIKÓW WEJŚCIOWYCH**

### **Pliki porównywane w naszych testach:**
- **Stary:** `merge/merged_ohlc_orderbook.feather`
- **Nowy:** `download2/merge/merged_data/merged_BTCUSDT.feather`

### **Wyniki porównania (z poprzednich testów):**
- **Dane OHLC:** 100% identyczności
- **Dane Orderbook:** 99.8% identyczności
- **Korelacja:** 0.9998-1.0000

## 🔍 **KLUCZOWE WNIOSKI**

### **1. Różne źródła danych:**
- **Stary kalkulator** używa pliku z `merge/merged_ohlc_orderbook.feather`
- **Nowy kalkulator** używa pliku z `download2/merge/merged_data/merged_BTCUSDT.feather`

### **2. Różne moduły łączenia danych:**
- **Stary:** Moduł `merge/merge_ohlc_orderbook.py`
- **Nowy:** Moduł `download2/merge/merge_ohlc_orderbook.py`

### **3. Różne struktury danych:**
- **Stary:** Jeden plik dla wszystkich symboli
- **Nowy:** Osobny plik dla każdego symbolu

### **4. Dane są prawie identyczne:**
- **99.8-100%** zgodności między plikami wejściowymi
- **0.2% różnicy** w danych orderbook
- **Różnice w timestampach** orderbook

## 🎯 **WPŁYW NA PROBLEM**

### **Czy różnice w plikach wejściowych powodują problem?**

1. **Dane OHLC są identyczne** (100%) - nie powinny powodować problemów
2. **Dane orderbook mają 99.8% zgodności** - małe różnice
3. **Problem nie leży w danych wejściowych** - są one bardzo podobne

### **Wniosek:**
**Różnice w plikach wejściowych NIE są przyczyną problemu** z kalkulatorami cech. Dane są wystarczająco podobne, aby algorytmy powinny dawać identyczne wyniki.

## 📋 **REKOMENDACJE**

### **1. Standaryzacja ścieżek:**
- Rozważyć ujednolicenie struktury katalogów
- Użyć tych samych plików wejściowych dla obu kalkulatorów

### **2. Testowanie z identycznymi danymi:**
- Skopiować plik wejściowy ze starego do nowego kalkulatora
- Przetestować czy problem nadal istnieje

### **3. Debugowanie środowiska:**
- Skupić się na różnicach w środowisku wykonania
- Sprawdzić wersje bibliotek i precyzję floating point

---

**Status:** Pliki wejściowe są prawie identyczne, problem leży gdzie indziej.
**Następny krok:** Sprawdzenie środowiska wykonania (biblioteki, precyzja). 