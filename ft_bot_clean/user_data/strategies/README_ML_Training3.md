### ML_Training3_Strategy — integracja z modułem training3

Ta strategia ładuje artefakty z `crypto/training3/output/models/` i korzysta z modelu dla poziomu `TP 0.8%, SL 0.3%` (`model_tp0p8_sl0p3.json`).

#### Pliki
- `ML_Training3_Strategy.py` — klasa strategii Freqtrade (czas 1m, long/short)
- `utils/t3_model_loader.py` — ładowanie modelu (xgboost Booster JSON), `scaler.pkl`, metadanych (lista cech)
- `components/t3_feature_adapter.py` — dostosowanie wejścia do dokładnie 37 cech
- `components/t3_signal_generator.py` — skalowanie, predykcja, decyzja sygnału z progiem pewności
- `configs/training3_strategy_config.json` — konfig: ścieżka modeli, progi pewności, nazwy plików

#### Konfiguracja progów
W `configs/training3_strategy_config.json` ustaw:
```
{
  "ml_long_threshold": 0.5,
  "ml_short_threshold": 0.5
}
```

#### Wymagane cechy
Strategia wymaga tych samych 37 cech, co w `training3` (kolejność i nazwy z pliku `metadata_tp0p8_sl0p3.json`). Brakujące kolumny zostaną wypełnione `0.0`, ale zalecane jest pełne dopasowanie.

