# Moduł Training3 - Multi-Output XGBoost

## Opis

Moduł `training3` to nowy system treningowy wykorzystujący **XGBoost Multi-Output** do przewidywania 3-klasowych etykiet dla 5 poziomów TP/SL. Zastępuje poprzedni moduł `training2` i jest dostosowany do nowego formatu danych z `labeler3`.

## Architektura

### Model
- **XGBoost Multi-Output** - jeden model dla wszystkich 5 poziomów TP/SL
- **3 klasy**: LONG (0), SHORT (1), NEUTRAL (2)
- **5 wyjść**: po jednym dla każdego poziomu TP/SL

### Dane Wejściowe
- **Źródło**: `labeler3/output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather`
- **Cechy**: 85 cech z `feature_calculator_ohlc_snapshot`
- **Etykiety**: 5 kolumn 3-klasowych z `labeler3`

### Poziomy TP/SL
1. TP: 0.8%, SL: 0.2%
2. TP: 0.6%, SL: 0.3%
3. TP: 0.8%, SL: 0.4%
4. TP: 1.0%, SL: 0.5%
5. TP: 1.2%, SL: 0.6%

## Struktura Plików

```
training3/
├── config.py          # Konfiguracja parametrów
├── main.py            # Główny skrypt treningu
├── data_loader.py     # Wczytywanie i przygotowanie danych
├── model_builder.py   # Budowanie modelu XGBoost
├── utils.py           # Funkcje pomocnicze
├── __init__.py        # Inicjalizacja modułu
├── README.md          # Ten plik
└── output/
    ├── models/        # Zapisane modele
    ├── reports/       # Raporty i wykresy
    └── logs/          # Logi treningu
```

## Konfiguracja

### Główne Parametry (config.py)

```python
# Dane wejściowe
INPUT_FILENAME = "ohlc_orderbook_labeled_3class_fw60m_5levels.feather"

# Parametry XGBoost
XGB_N_ESTIMATORS = 500
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 5
XGB_EARLY_STOPPING_ROUNDS = 15

# Balansowanie klas
ENABLE_CLASS_BALANCING = True
CLASS_WEIGHTS = {0: 2.0, 1: 2.0, 2: 1.0}  # LONG, SHORT, NEUTRAL
```

## Użycie

### Uruchomienie Treningu

```bash
cd crypto
python -m training3.main
```

### Programowe Użycie

```python
from training3 import Trainer

# Uruchom trening
trainer = Trainer()
trainer.run()
```

## Wyniki

### Pliki Wyjściowe

1. **Model**: `output/models/model_multioutput.pkl`
2. **Scaler**: `output/models/scaler.pkl`
3. **Wyniki ewaluacji**: `output/reports/evaluation_results.json`
4. **Porównanie poziomów**: `output/reports/level_comparison.csv`
5. **Ważności cech**: `output/reports/feature_importance.csv`

### Wykresy (jeśli SAVE_PLOTS=True)

1. **Feature Importance**: `output/reports/feature_importance.png`
2. **Confusion Matrices**: `output/reports/confusion_matrices.png`

### Logi

- **Plik logów**: `output/logs/trainer_3class.log`
- **Poziom**: INFO
- **Format**: timestamp - module - level - message

## Metryki Ewaluacji

### Dla Każdego Poziomu TP/SL
- **Accuracy**: Ogólna dokładność
- **Precision/Recall/F1**: Dla każdej klasy (LONG, SHORT, NEUTRAL)
- **Confusion Matrix**: Macierz pomyłek

### Porównanie Poziomów
- Ranking poziomów według różnych metryk
- Identyfikacja najlepszych poziomów dla LONG/SHORT

## Różnice względem Training2

| Aspekt | Training2 | Training3 |
|--------|-----------|-----------|
| **Etykiety** | 6-klasowe (PROFIT_LONG, LOSS_SHORT, etc.) | 3-klasowe (LONG, SHORT, NEUTRAL) |
| **Model** | Osobne modele per poziom | Jeden Multi-Output model |
| **Dane** | Stare cechy | Nowe cechy z feature_calculator_ohlc_snapshot |
| **Balansowanie** | Stałe | Konfigurowalne |
| **Architektura** | Skomplikowana | Uproszczona |

## Wymagania

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- joblib

## Rozwiązywanie Problemów

### Błąd: "Plik wejściowy nie istnieje"
- Sprawdź czy `labeler3` został uruchomiony
- Sprawdź ścieżkę w `config.py`

### Błąd: "Brakuje wymaganych kolumn"
- Sprawdź czy używasz właściwych cech w `config.py`
- Upewnij się, że dane z `labeler3` są kompletne

### Błąd: "Model nie został wytrenowany"
- Sprawdź logi w `output/logs/`
- Upewnij się, że dane treningowe są poprawne

## Następne Kroki

1. **Uruchom trening** na pełnym zbiorze danych
2. **Przeanalizuj wyniki** w `output/reports/`
3. **Dostrój parametry** w `config.py` jeśli potrzeba
4. **Zintegruj model** z systemem tradingowym 