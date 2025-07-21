# 🚀 Crypto Trading System with ML Analysis

## 📋 Opis Projektu

Zaawansowany system tradingu kryptowalut wykorzystujący uczenie maszynowe do analizy i przewidywania ruchów cenowych. Projekt składa się z kilku zintegrowanych modułów, które razem tworzą kompletny pipeline od pobierania danych, przez trenowanie modeli, aż po backtesting strategii.

## 🏗️ Architektura Systemu

```
crypto/
├── 🤖 ft_bot_clean/           # FreqTrade Bot - główny silnik tradingu
├── 🧠 Kaggle/                 # Moduł trenowania modeli ML
├── 🔍 validation_and_labeling/ # Walidacja i etykietowanie danych
├── 📊 skrypty/                # Narzędzia do analizy i porównań
├── 📚 memory-bank/            # Dokumentacja i analizy
└── 🛠️ user_data/             # Konfiguracje użytkownika
```

## 🎯 Główne Funkcjonalności

### 🤖 FreqTrade Bot (`ft_bot_clean/`)
- **Strategia ML**: Enhanced_ML_MA43200_Buffer_Strategy
- **Backtesting**: Kompleksowe testowanie strategii
- **Buffer System**: Optymalizacja obliczeń MA
- **Export danych**: Eksport wyników do analizy

### 🧠 Kaggle Training Module (`Kaggle/`)
- **LSTM Model**: Zaawansowana architektura sieci neuronowej
- **Sequence-aware Undersampling**: Inteligentne balansowanie danych
- **Memory-efficient Processing**: Optymalizacja pamięci
- **Production-ready Pipeline**: Gotowy do wdrożenia

### 🔍 Validation & Labeling (`validation_and_labeling/`)
- **Competitive Labeling**: Zaawansowane etykietowanie danych
- **Feature Calculation**: 8 kluczowych cech technicznych
- **Data Quality Validation**: Kontrola jakości danych
- **Binance Integration**: Bezpośrednie pobieranie danych

### 📊 Analysis Tools (`skrypty/`)
- **Prediction Comparison**: Porównanie predykcji między modułami
- **Feature Analysis**: Analiza różnic w obliczaniu cech
- **Correlation Studies**: Badania korelacji
- **Automated Reports**: Automatyczne raporty

## 🔧 Kluczowe Cechy Techniczne

### 📈 8 Features Analysis
1. **high_change** - Zmiana maksimum
2. **low_change** - Zmiana minimum  
3. **close_change** - Zmiana ceny zamknięcia
4. **volume_change** - Zmiana wolumenu
5. **price_to_ma1440** - Stosunek ceny do MA1440
6. **price_to_ma43200** - Stosunek ceny do MA43200
7. **volume_to_ma1440** - Stosunek wolumenu do MA1440
8. **volume_to_ma43200** - Stosunek wolumenu do MA43200

### 🧮 MA Algorithms Analysis
- **Validation Module**: EXPANDING→ROLLING algorithm
- **FreqTrade**: PURE ROLLING algorithm
- **Impact**: Różnice w pierwszych 1440/43200 świecach
- **Solution**: Buffer system dla unifikacji

## 🚀 Quick Start

### 1. 📥 Instalacja
```bash
git clone https://github.com/MaciejKonkolH/crypto.git
cd crypto
```

### 2. 🔧 Konfiguracja FreqTrade
```bash
cd ft_bot_clean
# Skonfiguruj config.json z własnymi parametrami
```

### 3. 🧠 Trenowanie Modelu
```bash
cd Kaggle
python trainer.py
```

### 4. 🔍 Walidacja Danych
```bash
cd validation_and_labeling
python main.py
```

### 5. 📊 Analiza Porównawcza
```bash
cd skrypty
python run_comparison.py
```

## 📊 Wyniki Analizy

### 🎯 Prediction Comparison Results
- **Total Predictions**: 231,304
- **Identical**: 32 (0.0%)
- **Major Differences**: 73.8% (>5%)
- **Signal Changes**: 2.1%

### 🔍 Feature Correlation Analysis
- **Excellent**: price_to_ma43200 (1.000)
- **Good**: price_to_ma1440 (0.996), close_change (0.964)
- **Problematic**: volume_change (0.807), volume_to_ma1440 (0.742)

## 📚 Dokumentacja

### 📖 Główne Dokumenty
- [📋 Instrukcja FreqTrade](memory-bank/documentation/Instrukcja_obslugi_freqTrade.md)
- [🔍 Analiza 8 Cech](memory-bank/Plany/Plan_porownania_8_cech.md)
- [🧠 Dokumentacja Treningu](memory-bank/documentation/trening/)
- [🐳 Docker Setup](memory-bank/documentation/docker/Docker_instrukcja.md)

### 🛠️ Narzędzia
- [📊 Compare Predictions](skrypty/README.md)
- [🔧 Buffer System](ft_bot_clean/user_data/buffer/)
- [📈 Strategy Components](ft_bot_clean/user_data/strategies/components/)

## 🔬 Discoveries & Insights

### 🎯 Root Cause Analysis
1. **Different MA Algorithms**: Główna przyczyna różnic w predykcjach
2. **Volume Data Inconsistencies**: Różnice w źródłach danych wolumenu
3. **Column Order Issues**: Naprawione problemy z kolejnością kolumn
4. **Timestamp Synchronization**: Problemy z synchronizacją czasów

### 💡 Solutions Implemented
1. **Buffer System**: Unifikacja obliczeń MA
2. **Column Order Fix**: Naprawiona kolejność cech
3. **Timestamp Alignment**: Synchronizacja czasów między modułami
4. **Enhanced Logging**: Lepsze debugowanie i monitoring

## 🔧 Technical Stack

- **Python 3.10+**
- **FreqTrade**: Trading framework
- **TensorFlow/Keras**: ML models
- **Pandas/NumPy**: Data processing
- **TA-Lib**: Technical analysis
- **Binance API**: Data source
- **Docker**: Containerization

## 📈 Performance Metrics

### 🎯 Model Performance
- **Training Accuracy**: Optimized with sequence-aware sampling
- **Memory Efficiency**: Generator-based processing
- **Production Ready**: Comprehensive error handling

### ⚡ System Performance
- **Buffer System**: 3x faster MA calculations
- **Parallel Processing**: Multi-threaded data processing
- **Memory Optimization**: Efficient data handling

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FreqTrade Community** - Amazing trading framework
- **Binance API** - Reliable data source
- **TensorFlow Team** - Powerful ML framework
- **Open Source Community** - Inspiration and tools

## 📞 Contact

- **GitHub**: [@MaciejKonkolH](https://github.com/MaciejKonkolH)
- **Project Link**: [https://github.com/MaciejKonkolH/crypto](https://github.com/MaciejKonkolH/crypto)

---

## 🏆 Project Status

**Status**: ✅ **ACTIVE DEVELOPMENT**

**Last Major Update**: Complete ML Analysis System with Feature Comparison

**Next Steps**:
- [ ] Unify MA calculation algorithms
- [ ] Implement real-time trading
- [ ] Add more ML models
- [ ] Enhance documentation

---

*Made with ❤️ for the crypto trading community* 