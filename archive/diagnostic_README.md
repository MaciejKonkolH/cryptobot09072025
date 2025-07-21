# 🔍 DIAGNOSTIC SYSTEM - System Diagnostyczny

System diagnostyczny do wykrywania rozbieżności między modułem treningu a FreqTrade backtesting.

## 🎯 Cel

Zapewnienie identycznej metodologii w obu modułach poprzez:
- **Model fingerprinting**: Zapisywanie hashy wag modelu, architektury, parametrów
- **Scaler auditing**: Zapisywanie statystyk scalera, parametrów normalizacji
- **Scaled data capture**: Zapisywanie próbek przeskalowanych features
- **Systematyczne porównanie**: Automatyczne wykrywanie różnic

## 📁 Struktura Plików

```
crypto/
├── diagnostic_utils.py          # Wspólny moduł diagnostyczny
├── test_diagnostic.py           # Skrypt testowy
├── diagnostic_README.md         # Ta instrukcja
├── raporty/                     # Wygenerowane raporty
│   ├── model_scaler_audit_trainer_*.json
│   ├── model_scaler_audit_freqtrade_*.json
│   ├── scaled_features_sample_trainer.json
│   ├── scaled_features_sample_freqtrade.json
│   ├── audit_comparison_report_*.json
│   └── scaled_features_comparison_*.json
├── Kaggle/
│   └── trainer.py               # Zmodyfikowany do używania diagnostic_utils
└── ft_bot_clean/
    └── user_data/strategies/components/
        └── signal_generator.py  # Zmodyfikowany do używania diagnostic_utils
```

## 🚀 Instrukcje Użycia

### Krok 1: Uruchom Trening
```bash
cd Kaggle
python trainer.py
```
**Rezultat**: Wygeneruje pliki `model_scaler_audit_trainer_*.json` i `scaled_features_sample_trainer.json`

### Krok 2: Uruchom Backtesting
```bash
cd ft_bot_clean
python -m freqtrade backtesting --config config.json --strategy Enhanced_ML_MA43200_Buffer_Strategy --timerange 20241220-20241221
```
**Rezultat**: Wygeneruje pliki `model_scaler_audit_freqtrade_*.json` i `scaled_features_sample_freqtrade.json`

### Krok 3: Uruchom Porównanie
```bash
python test_diagnostic.py
```
**Rezultat**: Wygeneruje raporty porównawcze i pokaże różnice

## 📊 Generowane Raporty

### Model & Scaler Audit
Plik: `model_scaler_audit_[module]_[timestamp].json`

Zawiera:
- **Model fingerprint**: hash wag, architektura, liczba parametrów
- **Scaler parameters**: typ scalera, statystyki (mean, scale, center)
- **Metadata**: wersje TensorFlow, numpy, timestamp

### Scaled Features Sample
Plik: `scaled_features_sample_[module].json`

Zawiera:
- **Próbkę przeskalowanych features** (pierwszych 100 rekordów)
- **Statystyki**: mean, std, min, max per feature
- **Timestamps**: dla weryfikacji chronologii

### Audit Comparison Report
Plik: `audit_comparison_report_[timestamp].json`

Zawiera:
- **Porównanie hashy wag modelu**
- **Porównanie parametrów scalera**
- **Lista wykrytych różnic**
- **Podsumowanie**: czy modele/scalery są identyczne

### Scaled Features Comparison
Plik: `scaled_features_comparison_[timestamp].json`

Zawiera:
- **Maksymalne różnice** między features
- **Średnie różnice** per feature
- **Tolerancje**: czy różnice są w akceptowalnych granicach

## 🔍 Interpretacja Wyników

### ✅ Sukces (Brak Rozbieżności)
```
Models identical: true
Scalers identical: true
Max difference: 0.0000000001
```

### ⚠️ Wykryte Różnice
```
Models identical: false
Differences found:
- Model weights hash mismatch
- Scaler mean mismatch
Max difference: 0.0001234567
```

## 🛠️ Troubleshooting

### Problem: Brak plików audit
**Rozwiązanie**: Uruchom trening i backtesting ponownie

### Problem: Diagnostic failed
**Rozwiązanie**: Sprawdź logi, może brakować zależności lub uprawnień do zapisu

### Problem: Różnice w wagach modelu
**Rozwiązanie**: Sprawdź czy oba moduły używają tego samego pliku `best_model.h5`

### Problem: Różnice w scalerze
**Rozwiązanie**: Sprawdź czy oba moduły używają tego samego pliku scalera

## 🔧 Konfiguracja

### Zmiana rozmiaru próbki
W `diagnostic_utils.py`:
```python
save_scaled_features_sample(..., sample_size=1000)  # Domyślnie 100
```

### Zmiana katalogu wyjściowego
W `trainer.py` i `signal_generator.py`:
```python
run_complete_diagnostic(..., output_dir="./custom_reports/")
```

## 📈 Workflow Diagnostyczny

1. **Uruchom trening** → Generuje audit trainera
2. **Uruchom backtesting** → Generuje audit FreqTrade
3. **Uruchom porównanie** → Analizuje różnice
4. **Napraw różnice** → Jeśli wykryto problemy
5. **Powtórz** → Aż do uzyskania identycznych wyników

## 💡 Wskazówki

- **Uruchamiaj w tej samej kolejności**: Trening → Backtesting → Porównanie
- **Sprawdzaj timestamps**: Upewnij się że porównujesz najnowsze pliki
- **Archiwizuj raporty**: Zachowaj historię dla analizy trendów
- **Monitoruj różnice**: Nawet małe różnice mogą powodować duże rozbieżności w predykcjach

## 🎯 Cele Diagnostyczne

- [x] **Identyczne wagi modelu** - Oba moduły używają tego samego modelu
- [x] **Identyczne parametry scalera** - Oba moduły używają tego samego scalera  
- [x] **Identyczne przeskalowane features** - Oba moduły generują te same dane wejściowe
- [ ] **Identyczne predykcje** - Ostateczny cel: identyczne wyniki ML

## ⚠️ WAŻNE: Która Wersja Modelu Jest Sprawdzana

### W Module Trainer:
- **Model diagnostyczny**: `best_model.h5` ładowany bezpośrednio z dysku
- **Powód**: Identyczna metodologia jak FreqTrade
- **Lokalizacja**: `generate_validation_positions_report()` → po manual restoration

### W Module FreqTrade:
- **Model diagnostyczny**: `best_model.h5` ładowany bezpośrednio z dysku
- **Powód**: Standardowe ładowanie modelu
- **Lokalizacja**: `_generate_predictions_batch()` → podczas backtesting

### Dlaczego Nie `self.model`?
Transfer wag `self.model.set_weights(best_model.get_weights())` może nie być identyczny z bezpośrednim ładowaniem pliku. System diagnostyczny sprawdza **identyczny plik `best_model.h5`** w obu modułach.

---

**Autor**: AI Assistant  
**Wersja**: 1.0  
**Data**: 2024-12-30 