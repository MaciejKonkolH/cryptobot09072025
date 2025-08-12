# Analiza Systemu Trenowania Modeli - Poprzedni BinanceBot

*Data analizy: 17 maja 2025*  
*Autor: Agent AI*  
*Status: Kompletna analiza kodu i dokumentacji*

## 1. Ogólna Architektura Systemu

Poprzedni system trenowania BinanceBot był zaprojektowany zgodnie z **4-warstwową architekturą**:

### Warstwy Systemu:
- **Warstwa Danych (Data Layer)**: Przechowywanie i zarządzanie danymi (SQLite + pliki)
- **Warstwa Core (Logika Biznesowa)**: Algorytmy ML, logika modeli
- **Warstwa Usług Zewnętrznych**: API Binance, repozytorium modeli
- **Warstwa Wykonawcza**: Skrypty uruchomieniowe, orkiestracja procesów

## 2. Struktura Katalogów i Organizacja Kodu

```
Trening/
├── config.py              # Centralna konfiguracja systemu
├── data_loader.py          # Ładowanie danych z SQLite
├── data_preprocessing.py   # Przetwarzanie i feature engineering
├── model_builder.py        # Budowanie architektury modeli LSTM
├── trainer.py              # Logika trenowania modeli
├── evaluator.py            # Ewaluacja modeli
├── training_pipeline.py    # Orkiestracja całego procesu
├── train_model.py          # Główny skrypt uruchomieniowy
├── hybrid_model_labeling.py # Etykietowanie danych (TP/SL)
├── feature_scaling.py      # Zaawansowane transformacje cech
├── data_updater.py         # Aktualizacja danych historycznych
├── utils.py                # Funkcje pomocnicze
├── visualization.py        # Wizualizacja wyników
├── data/                   # Baza danych SQLite
├── models/                 # Zapisane modele (.h5)
├── scalers/                # Zapisane skalery (.pkl)
├── results/                # Wyniki i raporty
└── logs/                   # Logi systemowe
```

## 3. Przepływ Danych i Procesów

### 3.1. Główny Pipeline Treningu (`training_pipeline.py`):

1. **Aktualizacja danych**: `DataUpdater` → API Binance → SQLite
2. **Ładowanie danych**: `DataLoader` → SQLite → DataFrame
3. **Feature Engineering**: `DataPreprocessor.compute_features()` → cechy techniczne
4. **Etykietowanie**: `label_data_hybrid()` → sygnały SHORT/HOLD/LONG
5. **Przygotowanie sekwencji**: `prepare_sequences()` → okna czasowe
6. **Skalowanie**: `MinMaxScaler` → normalizacja cech
7. **Trenowanie**: `ModelBuilder` + `ModelTrainer` → model LSTM
8. **Ewaluacja**: `ModelEvaluator` → metryki wydajności
9. **Zapis artefaktów**: modele (.h5) + skalery (.pkl)

### 3.2. Skrypt Uruchomieniowy (`train_model.py`):
- CLI z argumentami (symbol, interwał, okna czasowe, TP/SL)
- Konfiguracja treningu przez `TrainingConfig`
- Orchestracja pipeline'u
- Obsługa błędów i logowanie

## 4. Feature Engineering (Inżynieria Cech)

### 4.1. Standardowe 8 Cech (`data_preprocessing.py`):
1. **high_change**: (high - open) / open
2. **low_change**: (low - open) / open  
3. **close_change**: (close - open) / open
4. **volume_change**: procentowa zmiana wolumenu
5. **price_to_ma1440**: stosunek ceny do MA(1440) - 24h
6. **price_to_ma43200**: stosunek ceny do MA(43200) - 30 dni
7. **volume_to_ma1440**: stosunek wolumenu do MA(1440)
8. **volume_to_ma43200**: stosunek wolumenu do MA(43200)

### 4.2. Zaawansowane Transformacje (`feature_scaling.py`):
- **Analiza zmienności**: percentyle dla granic transformacji
- **Adaptacyjne skalowanie**: dynamiczne skalowanie cechy ceny
- **Transformacje nieliniowe**: log, sigmoid dla lepszej dystrybucji
- **Parametry zmienności**: zapisywane w `volatility_params/`

## 5. System Etykietowania (`hybrid_model_labeling.py`)

### 5.1. Logika Hybrydowa:
- **Ograniczone okno czasowe**: analiza tylko `future_window` świec (np. 60 min)
- **Take Profit / Stop Loss**: progi TP (1.0%) i SL (0.5%)
- **Trzy klasy sygnałów**:
  - **LONG (2)**: TP osiągnięte przed SL w oknie czasowym
  - **SHORT (0)**: TP osiągnięte przed SL w oknie czasowym
  - **HOLD (1)**: brak wyraźnej przewagi lub oba/żadne osiągnięte

### 5.2. Algorytm Etykietowania:
```python
for każda_świeca in range(dane - future_window):
    base_price = close[i]
    long_tp = base_price * (1 + tp_pct)
    long_sl = base_price * (1 - sl_pct)
    short_tp = base_price * (1 - tp_pct) 
    short_sl = base_price * (1 + sl_pct)
    
    # Analiza przyszłych `future_window` świec
    for future_candle in range(i+1, i+1+future_window):
        if high >= long_tp: long_tp_hit = True
        if low <= long_sl: long_sl_hit = True
        if low <= short_tp: short_tp_hit = True
        if high >= short_sl: short_sl_hit = True
    
    # Przypisanie etykiety na podstawie wyników
    if long_tp_hit and not long_sl_hit and not (short_tp_hit and not short_sl_hit):
        label = LONG (2)
    elif short_tp_hit and not short_sl_hit and not (long_tp_hit and not long_sl_hit):
        label = SHORT (0)
    else:
        label = HOLD (1)
```

## 6. Architektura Modeli (`model_builder.py`)

### 6.1. Model LSTM (domyślny):
```python
Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, 8)),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.2), 
    BatchNormalization(),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 klasy: SHORT, HOLD, LONG
])
```

### 6.2. Model Zoptymalizowany:
- **Większe jednostki LSTM**: 160 → 80 → 40
- **Więcej neuronów Dense**: 40 → 20
- **Optymalizator Adam**: learning_rate=0.0003, clipnorm=1.0
- **BatchNormalization** na każdej warstwie

## 7. Proces Trenowania (`trainer.py`)

### 7.1. Przygotowanie Danych:
- **Sekwencjonowanie**: okna `window_size` (np. 90 świec) × 8 cech
- **Balansowanie klas**: oversampling/undersampling dla równowagi SHORT/HOLD/LONG
- **Podział train/val**: 80/20 z `VALIDATION_SPLIT=0.1`
- **Skalowanie**: `MinMaxScaler` dopasowany na train, zastosowany na val/test

### 7.2. Callbacki TensorFlow:
- **EarlyStopping**: `monitor='val_loss'`, `patience=10`
- **ReduceLROnPlateau**: `factor=0.5`, `patience=5`
- **ModelCheckpoint**: zapis najlepszego modelu
- **TensorBoard**: wizualizacja treningu

### 7.3. Generatory Batchy:
- **Zbalansowane batche**: równa reprezentacja klas w każdym batchu
- **Dynamic batching**: `generate_balanced_batches()`
- **Memory efficient**: generator zamiast ładowania wszystkich danych

## 8. System Konfiguracji (`config.py`)

### 8.1. Parametry Globalne:
```python
# Ścieżki
DB_PATH = "data/crypto_data_futures.db"
MODELS_DIR = "models/"
SCALERS_DIR = "scalers/"

# Parametry modelu
DEFAULT_WINDOW_SIZE = 90      # okno wejściowe
DEFAULT_FUTURE_PREDICTION = 60 # okno predykcji
BATCH_SIZE = 32
EPOCHS = 10

# Progi TP/SL
LONG_TP_PCT = 0.01    # 1.0%
LONG_SL_PCT = 0.005   # 0.5%
SHORT_TP_PCT = 0.01   # 1.0%
SHORT_SL_PCT = 0.005  # 0.5%

# Parametry modelu
MODEL_PARAMS = {
    "lstm_units": [128, 64, 32],
    "dense_units": [32, 16],
    "dropout_rate": 0.2,
    "learning_rate": 0.0005
}
```

## 9. Zarządzanie Artefaktami

### 9.1. Konwencje Nazewnicze:
- **Modele**: `{symbol}_{interval}_ws{window_size}_fp{future_prediction}.h5`
- **Skalery**: `scaler_{symbol}_{interval}_ws{window_size}_{timestamp}.pkl`
- **Historia**: `history_{symbol}_{interval}_win{window_size}_fut{future_prediction}_{timestamp}.pkl`

### 9.2. Przechowywanie:
- **Modele**: pliki `.h5` (TensorFlow/Keras format)
- **Skalery**: pliki `.pkl` (joblib serialization)
- **Metadane**: JSONy z metrykami i parametrami treningu
- **Logi**: TensorBoard logs + zwykłe logi tekstowe

## 10. Ewaluacja Modeli (`evaluator.py`)

### 10.1. Metryki Podstawowe:
- **Accuracy**: ogólna dokładność klasyfikacji
- **Precision/Recall/F1**: per klasa (SHORT, HOLD, LONG)
- **Confusion Matrix**: macierz pomyłek
- **Classification Report**: szczegółowy raport sklearn

### 10.2. Metryki Biznesowe:
- **Profit Factor**: stosunek zysków do strat
- **Win Rate**: procent zyskownych transakcji
- **Average Win/Loss**: średni zysk/strata
- **Maximum Drawdown**: maksymalny spadek kapitału

### 10.3. Wizualizacje:
- **Learning curves**: accuracy/loss vs epochs
- **Confusion matrices**: heatmapy z seaborn
- **Feature importance**: analiza wpływu cech
- **Prediction distributions**: rozkłady predykcji per klasa

## 11. Integracja z Systemem Produkcyjnym

### 11.1. Sposób Użycia Modeli:
1. **ModelRegistry**: konfiguracja ścieżek do modeli/skalerów
2. **MLModelInterface**: abstrakcyjny interfejs do ładowania/predykcji
3. **LiveFeatureExtractor**: obliczanie cech w czasie rzeczywistym
4. **SignalGenerator**: konwersja predykcji na sygnały handlowe

### 11.2. Przepływ Inferencji:
```
Dane Live → LiveFeatureExtractor → Skalowanie → Model → Predykcja → SignalGenerator → Sygnał Handlowy
```

## 12. Mocne Strony Systemu

1. **Modularność**: Jasny podział na komponenty z określonymi zadaniami
2. **Spójność**: Identyczne przetwarzanie w treningu i inferencji  
3. **Konfigurowalność**: Centralna konfiguracja + argumenty CLI
4. **Balansowanie**: Zaawansowane techniki balansowania klas
5. **Monitoring**: TensorBoard + szczegółowe logowanie
6. **Feature Engineering**: Przemyślane cechy techniczne + transformacje
7. **Etykietowanie**: Realistyczna logika hybrydowa z TP/SL
8. **Architektura**: Zgodność z zasadami clean architecture

## 13. Obszary do Potencjalnych Ulepszeń

1. **Dependency Injection**: Ręczne tworzenie obiektów vs IoC
2. **Testing**: Brak testów jednostkowych dla komponentów
3. **Pipeline Orchestration**: Brak zaawansowanych narzędzi (Airflow, Kubeflow)
4. **Model Versioning**: Podstawowe wersjonowanie vs MLflow/DVC
5. **Hyperparameter Tuning**: Brak automatycznej optymalizacji (Optuna)
6. **Cross-Validation**: Brak walidacji krzyżowej dla modeli
7. **Feature Selection**: Brak automatycznej selekcji cech
8. **Real-time Monitoring**: Podstawowe logowanie vs zaawansowane monitorowanie

## Podsumowanie

Poprzedni system BinanceBot reprezentował **solidnie zaprojektowany i implementowany system trenowania modeli ML** z przemyślaną architekturą, realistycznym etykietowaniem i spójnym pipeline'em. System był gotowy do użycia produkcyjnego z dobrymi praktykami MLOps, chociaż można było go jeszcze bardziej rozbudować o zaawansowane narzędzia.
