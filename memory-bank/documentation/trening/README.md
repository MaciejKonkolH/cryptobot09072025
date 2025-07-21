# 📖 DOKUMENTACJA MODUŁU TRENUJĄCEGO V3

## 🎯 PRZEGLĄD SYSTEMU

**Moduł Trenujący V3** to 100% standalone system trenowania modeli LSTM dla Freqtrade Enhanced ML Strategy. System jest zoptymalizowany pod kątem produkcji, pamięci i łatwości użycia.

### ✨ Kluczowe Innowacje V3

- ✅ **100% Standalone** - zero zależności od modułu walidacji
- ✅ **Config-driven** - pojedyncza sekcja konfiguracji zamiast hierarchii klas
- ✅ **Memory-efficient** - redukcja użycia pamięci z 81GB+ do 2-3GB
- ✅ **Feature scaling** - zaawansowany system skalowania z zero data leakage
- ✅ **Class balancing** - systematic undersampling i class weights
- ✅ **Pre-computed labels** - eliminacja competitive labeling (95+ linii kodu)
- ✅ **Docker-optimized** - explicit paths, production-ready

### 📁 Struktura Plików

```
ft_bot_docker_compose/user_data/trening2/
├── 📄 config.py          (276 linii) - Konfiguracja standalone
├── 📄 data_loader.py     (641 linii) - Ładowanie z explicit paths
├── 📄 sequence_generator.py (570 linii) - Memory-efficient generators  
├── 📄 model_builder.py   (503 linie) - LSTM architecture builder
├── 📄 trainer.py         (724 linie) - Główny pipeline trenowania
├── 📄 utils.py           (422 linie) - Funkcje pomocnicze
├── 📁 inputs/            - Training-ready files
├── 📁 outputs/           - Modele, metadata, scalers
└── 📁 temp/              - Pliki tymczasowe
```

## 📚 NAVIGATION - DOKUMENTACJA

### 🔍 **Ogólne Przeglądy**
- 📄 **[01_Przeglad_i_filozofia.md](./01_Przeglad_i_filozofia.md)** - Filozofia V3, problemy V2, architektura
- 📄 **[02_Struktura_projektu.md](./02_Struktura_projektu.md)** - File structure, dependencies, workflow

### ⚙️ **Dokumentacja Plików**
- 📄 **[03_Config.md](./03_Config.md)** - `config.py` - parametry, walidacja, CLI override
- 📄 **[04_Data_loader.md](./04_Data_loader.md)** - `data_loader.py` - explicit loading, scaling, balancing
- 📄 **[05_Sequence_generator.md](./05_Sequence_generator.md)** - `sequence_generator.py` - memory generators, pre-computed labels
- 📄 **[06_Model_builder.md](./06_Model_builder.md)** - `model_builder.py` - LSTM architecture, callbacks, GPU
- 📄 **[07_Trainer.md](./07_Trainer.md)** - `trainer.py` - pipeline trenowania, confusion matrix
- 📄 **[08_Utils.md](./08_Utils.md)** - `utils.py` - helper functions, monitoring

### 🧬 **Algorytmy i Integracja**
- 📄 **[09_Algorytmy_i_integracje.md](./09_Algorytmy_i_integracje.md)** - Systematic undersampling, scaling, optimization
- 📄 **[10_Praktyczne_przykłady.md](./10_Praktyczne_przykłady.md)** - Usage examples, troubleshooting, FAQ

## 🚀 QUICK START

### 1. **Podstawowe Użycie**
```bash
# 1. Przygotuj dane w module walidacji
# 2. Skopiuj training-ready file do inputs/
# 3. Skonfiguruj parametry w config.py
# 4. Uruchom trening
python trainer.py
```

### 2. **CLI Override**
```bash
# Override config parameters
python trainer.py --pair ETHUSDT --epochs 50 --batch-size 128
```

### 3. **Test Konfiguracji**
```bash
# Test konfiguracji bez trenowania
python trainer.py --config-test
```

## 📊 CHARAKTERYSTYKI PERFORMANCE

### 💾 **Wykorzystanie Pamięci**
- **V2:** 81GB+ (full dataset w pamięci)
- **V3:** 2-3GB (memory-efficient generators)
- **Redukcja:** ~95% mniej pamięci

### ⚡ **Szybkość Trenowania**
- **GPU (Tesla V100):** ~1-2 sekundy/epoch
- **CPU:** ~10-15 sekund/epoch
- **Inference:** <1ms pojedyncza predykcja

### 🎯 **Model Architecture**
- **Input:** (120, 8) - 120 timesteps, 8 features
- **LSTM Stack:** 128 → 64 → 32 units
- **Dense Stack:** 32 → 16 units
- **Output:** (3,) - SHORT, HOLD, LONG probabilities
- **Parameters:** ~50K-100K (configurable)

## 🔧 WYMAGANIA SYSTEMOWE

### 📦 **Dependencies**
```python
tensorflow>=2.10.0      # Model training
pandas>=1.5.0           # Data manipulation  
numpy>=1.21.0           # Numerical computing
scikit-learn>=1.1.0     # Feature scaling
pyarrow>=10.0.0         # Feather file support
```

### 🐳 **Docker Environment**
- **Paths:** `/freqtrade/user_data/trening2/`
- **Memory:** Minimum 4GB RAM
- **Storage:** ~1GB per model + data
- **GPU:** Optional (auto-detection)

## ⚖️ **Key Algorithms**

### 🔄 **Class Balancing**
```python
# Systematic Undersampling
- Znajdź minority class size
- Zastosuj systematic sampling (co N-ta próbka)
- Zachowaj temporal order
- Safety limit: min 50K samples/class

# Class Weights  
- Automatic: sklearn balanced weights
- Manual: custom weights per class
```

### 📏 **Feature Scaling**
```python
# Zero Data Leakage Prevention:
1. Split data chronologically (train/val)
2. Fit scaler ONLY on train data  
3. Transform train AND val using SAME scaler
4. Save scaler for production inference
```

### 🧠 **Memory Optimization**
```python
# Generator Pattern:
- Numpy memory views (zero-copy)
- Lazy sequence loading
- Batch-wise processing
- Automatic garbage collection
```

## 🎯 **Supported Features**

### ✅ **Training Features**
- Multi-layer LSTM (128→64→32)
- Dropout regularization (0.3)
- Class balancing (undersampling/weights)
- Feature scaling (standard/robust/minmax)
- Early stopping (patience=10)
- Learning rate reduction (factor=0.5)
- Model checkpointing (.keras format)

### ✅ **Data Features**
- Explicit path loading
- Filename parameter parsing
- Date range filtering
- Pre-computed labels
- Memory-efficient processing
- Feature scaling with train/val awareness

### ✅ **Production Features**
- Docker paths compatibility
- CLI parameter override
- Configuration validation
- Error handling with clear messages
- Metadata saving (JSON)
- Confusion matrix generation

## 🆚 **V2 vs V3 Porównanie**

| Aspekt | V2 | V3 |
|--------|----|----|
| **Configuration** | Hierarchia klas (100+ linii) | Single section (50 linii) |
| **Dependencies** | Validation module required | 100% standalone |
| **Memory Usage** | 81GB+ | 2-3GB |
| **Labeling** | Competitive (95+ linii) | Pre-computed (0 linii) |
| **Paths** | Auto-detection | Explicit configuration |
| **Scaling** | Basic/manual | Advanced with leakage prevention |
| **Error Handling** | Generic | Specific guidance |

## 🔗 **Links Zewnętrzne**

- **[Freqtrade Documentation](https://www.freqtrade.io/)**
- **[TensorFlow/Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)**
- **[Enhanced ML Strategy](../strategy/)** - Main trading strategy documentation

## 📝 **Changelog**

### V3.0 (Current)
- ✅ 100% standalone operation
- ✅ Config-driven approach
- ✅ Memory-efficient generators
- ✅ Feature scaling system
- ✅ Class balancing algorithms
- ✅ Pre-computed labels
- ✅ Docker optimization

### V2.0 (Legacy)
- ❌ Validation module dependencies
- ❌ Memory inefficient
- ❌ Complex configuration hierarchy
- ❌ Competitive labeling overhead

---

**📧 Autor:** Freqtrade Enhanced ML Strategy Project  
**📅 Ostatnia aktualizacja:** 2024  
**🔖 Wersja:** V3.0 Production Ready 