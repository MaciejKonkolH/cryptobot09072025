# Freqtrade Project Structure Analysis

## 📁 **GŁÓWNA STRUKTURA PROJEKTU**

```
C:\Users\macie\OneDrive\Python\Binance\crypto\
├── ft_bot_clean/              # ← AKTUALNA INSTALACJA FREQTRADE (lokalna)
├── ft_bot_docker_compose/     # ← ORYGINALNA INSTALACJA (Docker)
├── Kaggle/                    # ← Dane/eksperymenty ML
├── memory-bank/               # ← Dokumentacja projektu
└── validation_and_labeling/   # ← Walidacja danych
```

## 🎯 **FT_BOT_CLEAN - STRUKTURA ROBOCZA**

```
ft_bot_clean/
├── user_data/                 # ← GŁÓWNY KATALOG KONFIGURACJI
│   ├── strategies/            # ← STRATEGIE TRADINGOWE
│   │   ├── components/        # ← Moduły: signal_generator.py
│   │   ├── config/           # ← Konfiguracje: pair_config.json
│   │   ├── inputs/           # ← Dane wejściowe ML
│   │   ├── utils/            # ← Narzędzia: model_loader.py, pair_manager.py
│   │   └── Enhanced_ML_MA43200_Buffer_Strategy.py  # ← GŁÓWNA STRATEGIA
│   ├── buffer/               # ← SYSTEM BUFORA MA43200
│   │   ├── data/            # ← Cache danych historycznych
│   │   └── dataframe_extender.py  # ← Rozszerzanie DataFrame
│   ├── backtest_results/     # ← Wyniki backtestingu
│   ├── data/                # ← Dane giełdowe (OHLCV)
│   ├── logs/                # ← Logi systemu
│   └── *.json               # ← Pliki konfiguracyjne
└── venv/                    # ← Virtual Environment Python
```

## 🔧 **KLUCZOWE PLIKI I MODUŁY**

### **Strategia ML:**
- `Enhanced_ML_MA43200_Buffer_Strategy.py` - główna strategia z ML + MA43200
- `utils/model_loader.py` - ładowanie modeli TensorFlow (.keras)
- `utils/pair_manager.py` - zarządzanie wieloma parami walut
- `components/signal_generator.py` - generowanie sygnałów ML

### **System Buffer MA43200:**
- `buffer/dataframe_extender.py` - rozszerzanie danych dla MA43200
- `buffer/data/` - cache historycznych danych

### **Konfiguracje:**
- `config_backtest.json` - konfiguracja backtestingu
- `config/pair_config.json` - konfiguracja par walutowych
- `*.json` - różne profile konfiguracyjne

### **Dane ML:**
- `*.feather` - dane historyczne (OHLCV)
- `*.keras` - modele TensorFlow
- `*.pkl` - scalery/preprocessory
- `metadata.json` - metadane modeli

## 🚨 **ZIDENTYFIKOWANE PROBLEMY**

### **1. Importy Linux vs Windows:**
```python
# OBECNE (błędne dla Windows):
sys.path.append('/freqtrade')
sys.path.append('/freqtrade/user_data/strategies')

# POTRZEBNE (Windows):
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
```

### **2. Brakujące __init__.py:**
```
user_data/buffer/__init__.py          # ← BRAKUJE
user_data/strategies/utils/__init__.py # ← BRAKUJE  
user_data/strategies/components/__init__.py # ← BRAKUJE
```

### **3. Konfiguracja strategii:**
```json
// config_backtest.json - TRZEBA ZMIENIĆ:
"strategy": "Enhanced_ML_MA43200_Buffer_Strategy"  // zamiast "BinanceBotSignalStrategy"
```

## ✅ **STAN INSTALACJI**

### **Freqtrade Core:**
- ✅ Freqtrade 2025.5 zainstalowany
- ✅ TA-Lib 0.6.4 skompilowany i działający
- ✅ TensorFlow, pandas, wszystkie dependencje
- ✅ Virtual environment aktywny

### **Pliki Strategii:**
- ✅ Wszystkie pliki Python skopiowane
- ✅ Dane ML (.feather, .keras, .pkl) dostępne
- ✅ Struktura katalogów kompletna
- ❌ Importy wymagają naprawy

### **Gotowość:**
- **Freqtrade:** ✅ Gotowy technicznie
- **Strategia:** ❌ Wymaga naprawy importów
- **Backtesting:** ❌ Czeka na naprawę strategii

## 📝 **NASTĘPNE KROKI**

1. **Napraw importy** w Enhanced_ML_MA43200_Buffer_Strategy.py
2. **Dodaj __init__.py** w katalogach modułów
3. **Zmień strategię** w config_backtest.json
4. **Test:** `freqtrade list-strategies`
5. **Uruchom:** `freqtrade backtesting`

---
*Analiza wykonana: 2025-06-17*
*Lokalizacja: C:\Users\macie\OneDrive\Python\Binance\crypto\* 