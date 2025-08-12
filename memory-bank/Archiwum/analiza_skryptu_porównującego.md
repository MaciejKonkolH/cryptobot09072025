# ANALIZA SKRYPTU PORÓWNUJĄCEGO CECHY

**Data utworzenia:** 4 sierpnia 2025  
**Cel:** Sprawdzenie czy problem leży w skrypcie porównującym czy w kalkulatorach cech

## 🔍 **HIPOTEZA**

Użytkownik zasugerował, że problem może leżeć w skrypcie porównującym (`compare_feature_calculators.py`), a nie w samych kalkulatorach cech.

## 📊 **ANALIZA KODU SKRYPTU PORÓWNUJĄCEGO**

### **Kluczowe funkcje:**

#### **1. `align_data()` (linie 150-165):**
```python
def align_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wyrównuje dane do wspólnego zakresu czasowego."""
    # Znajdź wspólny zakres
    common_start = max(self.old_df.index.min(), self.new_df.index.min())
    common_end = min(self.old_df.index.max(), self.new_df.index.max())
    
    # Filtruj dane
    old_aligned = self.old_df[(self.old_df.index >= common_start) & (self.old_df.index <= common_end)]
    new_aligned = self.new_df[(self.new_df.index >= common_start) & (self.new_df.index <= common_end)]
    
    return old_aligned, new_aligned
```

#### **2. `generate_detailed_feature_report()` (linie 300-407):**
```python
def generate_detailed_feature_report(self, old_aligned: pd.DataFrame, new_aligned: pd.DataFrame, 
                                   common_features: List[str]) -> pd.DataFrame:
    # ...
    for feature in common_features:
        # Pobierz serie
        old_series = old_aligned[feature].dropna()
        new_series = new_aligned[feature].dropna()
        
        # Upewnij się, że mają ten sam indeks
        common_index = old_series.index.intersection(new_series.index)
        old_series = old_series.loc[common_index]
        new_series = new_series.loc[common_index]
        
        # Oblicz identyczne wartości
        identical_count = (old_series == new_series).sum()
        total_count = len(old_series)
        identical_pct = (identical_count / total_count) * 100 if total_count > 0 else 0
```

## 🧪 **TESTY WALIDACYJNE**

### **Test 1: Bezpośrednie porównanie**
- **Wynik:** ✅ Poprawny
- **Logika:** `(old_data[feature] == new_data[feature]).sum()`

### **Test 2: Z filtrowaniem NaN**
- **Wynik:** ✅ Poprawny
- **Logika:** Używa `dropna()` i `intersection()` poprawnie

### **Test 3: Z różnymi indeksami**
- **Wynik:** ✅ Poprawny
- **Logika:** Obsługuje różne długości danych poprawnie

### **Test 4: Rzeczywiste dane**
- **Wynik:** ✅ Poprawny
- **Wyrównanie:** 1,270,074 wierszy wspólnych
- **Indeksy:** Identyczne po wyrównaniu

## 📊 **WYNIKI TESTÓW RZECZYWISTYCH DANYCH**

### **Zakresy danych:**
- **Stary kalkulator:** 1,270,074 wierszy (2023-01-31 do 2025-06-30)
- **Nowy kalkulator:** 1,317,594 wierszy (2023-01-31 do 2025-08-02)
- **Wspólny zakres:** 1,270,074 wierszy (2023-01-31 do 2025-06-30)

### **Problematyczne cechy - rzeczywiste wyniki:**
| Cecha | Identyczne wartości | Procent | Status |
|-------|-------------------|---------|---------|
| **pressure_volume_corr** | 65,103/1,270,074 | **5.13%** | ❌ Problem |
| **depth_price_corr** | 65,113/1,270,074 | **5.13%** | ❌ Problem |
| **volume_price_correlation** | 408,437/1,270,074 | **32.16%** | ❌ Problem |

## 🎯 **WNIOSKI**

### **✅ SKRYPT PORÓWNUJĄCY DZIAŁA POPRAWNIE:**

1. **Logika wyrównania danych:** Poprawna
2. **Obliczanie identycznych wartości:** Poprawne
3. **Obsługa różnych indeksów:** Poprawna
4. **Wyniki są wiarygodne:** Tak

### **❌ PROBLEM LEŻY W KALKULATORACH CECH:**

1. **Dane wejściowe:** 99.8-100% zgodności
2. **Kod algorytmów:** Skopiowany identycznie
3. **Konfiguracje:** Podobne
4. **Ale wyniki:** Różne (5-32% identyczności)

## 🔍 **MOŻLIWE PRZYCZYNY PROBLEMU**

### **1. Różne wersje bibliotek:**
- **pandas:** Różne wersje mogą mieć różne implementacje `rolling().corr()`
- **numpy:** Różnice w precyzji floating point
- **bamboo_ta:** Różne wersje biblioteki technicznej

### **2. Różne środowiska Python:**
- **Precyzja floating point:** Różne między środowiskami
- **Optymalizacje kompilatora:** Różne na różnych maszynach

### **3. Różne parametry funkcji:**
- **min_periods:** Może być różne w `rolling().corr()`
- **Obsługa NaN:** Różne strategie w różnych wersjach
- **Algorytmy korelacji:** Różne implementacje

### **4. Różne dane wejściowe:**
- **0.2% różnicy** w danych orderbook może powodować **duże różnice** w korelacjach
- **Korelacje Pearsona** są bardzo wrażliwe na outliers

## 📋 **REKOMENDACJE**

### **1. Sprawdzenie wersji bibliotek:**
```bash
pip list | grep -E "(pandas|numpy|bamboo-ta)"
```

### **2. Testowanie w tym samym środowisku:**
- Uruchomić oba kalkulatory w tym samym środowisku Python
- Sprawdzić czy problem nadal istnieje

### **3. Debugowanie algorytmów:**
- Dodać szczegółowe logi do problematycznych funkcji
- Porównać wartości pośrednie w obliczeniach

### **4. Normalizacja danych:**
- Rozważyć normalizację danych przed obliczaniem korelacji
- Sprawdzić wpływ outliers na wyniki

---

**Status:** Skrypt porównujący działa poprawnie. Problem leży w kalkulatorach cech.
**Następny krok:** Sprawdzenie wersji bibliotek i środowiska wykonania. 