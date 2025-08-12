# Kompletny Pipeline Orderbook - Od Pobrania do Treningu Modeli

## Przegląd Pipeline

Pipeline orderbook składa się z 6 głównych etapów:
1. **Pobieranie danych orderbook** - z Binance API
2. **Łączenie danych** - z formatu long do wide + feather
3. **Uzupełnianie luk** - inteligentne wypełnianie brakujących minut
4. **Łączenie z OHLC** - merge z danymi OHLC
5. **Obliczanie cech** - zaawansowane cechy techniczne
6. **Etykietowanie danych** - 3-klasowe etykiety
7. **Trening modeli** - Multi-Output XGBoost

## Etap 1: Pobieranie Danych Orderbook

### Plik: `download2/orderbook/download_orderbook.py`
- **Cel**: Pobiera snapshoty orderbook z Binance API
- **Konfiguracja**: Pary zdefiniowane w `config.py`
- **Parametry**: 10 minut wcześniej (zgodnie z modyfikacją)
- **Wynik**: Pliki CSV w katalogu `orderbook_raw/`

### Uruchomienie:
```bash
cd crypto/download2/orderbook
python download_orderbook.py
```

## Etap 2: Łączenie Danych Orderbook

### Plik: `download2/orderbook/merge_orderbook_to_feather.py`
- **Cel**: Łączy wszystkie pliki CSV w jeden format feather
- **Funkcje**:
  - Wczytuje wszystkie pliki CSV dla każdej pary
  - Przekształca z formatu long do wide
  - Grupuje po minutach (2 snapshoty na minutę)
  - Interpoluje brakujące snapshoty
  - Zapisuje w formacie feather

### Uruchomienie:
```bash
cd crypto/download2/orderbook
python merge_orderbook_to_feather.py
```

### Wynik:
- Pliki feather w katalogu `merged_raw/`
- Format: `orderbook_merged_{SYMBOL}.feather`
- Struktura: timestamp + snapshot1_* + snapshot2_* kolumny

## Etap 3: Uzupełnianie Luk w Danych

### Plik: `download2/orderbook/fill_orderbook_gaps.py`
- **Cel**: Inteligentnie wypełnia luki w danych orderbook
- **Strategie wypełniania**:
  - **Małe luki (≤5 min)**: Interpolacja liniowa
  - **Średnie luki (≤60 min)**: Rolling average (30 min)
  - **Duże luki (>60 min)**: Forward fill
- **Inteligencja**: Analizuje zmianę ceny wokół luki

### Uruchomienie:
```bash
cd crypto/download2/orderbook
python fill_orderbook_gaps.py
```

### Parametry (opcjonalne):
```bash
python fill_orderbook_gaps.py --max-small-gap 5 --max-medium-gap 60 --price-threshold 2.0
```

### Wynik:
- Pliki feather w katalogu `orderbook_completed/`
- Format: `orderbook_filled_{SYMBOL}.feather`
- Dodatkowe kolumny: `fill_method`, `gap_duration_minutes`, `price_change_percent`

## Etap 4: Łączenie z OHLC

### Plik: `download2/merge/merge_ohlc_orderbook.py`
- **Cel**: Łączy dane orderbook z danymi OHLC
- **Wymagania**: 
  - Dane OHLC z `download2/OHLC/`
  - Dane orderbook z `orderbook_completed/`

### Uruchomienie:
```bash
cd crypto/download2/merge
python merge_ohlc_orderbook.py
```

### Wynik:
- Pliki feather w katalogu `merged_data/`
- Format: `merged_{SYMBOL}.feather`
- Struktura: timestamp + OHLC + orderbook kolumny

## Etap 5: Pobieranie Danych OHLC

### Plik: `download2/OHLC/fast_ohlc_downloader.py`
- **Cel**: Szybkie pobieranie danych OHLC z Binance Futures
- **Technologia**: Używa biblioteki CCXT dla maksymalnej prędkości
- **Konfiguracja**: Pary zdefiniowane w `config.py`
- **Parametry**: 43200 minut (30 dni) wcześniej (zgodnie z modyfikacją)

### Kluczowe Funkcje:
- **Inteligentne sprawdzanie zakresu**: Automatycznie wykrywa dostępny zakres dat
- **Resume capability**: Może kontynuować przerwane pobieranie
- **Merge z istniejącymi danymi**: Łączy nowe dane z już pobranymi
- **Rate limiting**: Automatyczne ograniczanie requestów
- **Progress tracking**: Szczegółowe logi i progress bars

### Uruchomienie:
```bash
cd crypto/download2/OHLC
python fast_ohlc_downloader.py
```

### Wynik:
- Pliki CSV w katalogu `ohlc_raw/`
- Format: `{SYMBOL}_{INTERVAL}.csv`
- Struktura: timestamp, open, high, low, close, volume
- Metadane: `download_metadata.json`

### Konfiguracja w `config.py`:
```python
DOWNLOAD_CONFIG = {
    'interval': '1m',           # Interwał czasowy
    'market': 'futures',        # Typ rynku
    'chunk_size': 1000,         # Rozmiar chunka danych
    'timeout': 30,              # Timeout w sekundach
    'retry_delay': 5            # Opóźnienie przy retry
}
```

### Funkcje Inteligentne:
1. **Sprawdzanie dostępnego zakresu**: Automatycznie wykrywa najstarsze i najnowsze dostępne dane
2. **Incremental download**: Pobiera tylko nowe dane, jeśli już istnieją
3. **Duplicate handling**: Automatycznie usuwa duplikaty
4. **Error recovery**: Graceful handling błędów z retry
5. **Memory optimization**: Przetwarza dane w chunkach

## Etap 6: Obliczanie Cech - Feature Calculator

### Plik: `feature_calculator_download2/main.py`
- **Cel**: Oblicza zaawansowane cechy na podstawie danych OHLC + Order Book
- **Technologia**: Używa biblioteki bamboo_ta i scipy dla obliczeń technicznych
- **Format wejściowy**: Pliki feather z połączonymi danymi OHLC + Orderbook
- **Format wyjściowy**: Pliki feather z obliczonymi cechami

### Kluczowe Funkcje:
- **113 cech w sumie**: 5 OHLC + 50+ Bamboo TA + 40+ Orderbook + 10+ Hybrid + 15+ Relative + 20+ Advanced
- **Inteligentne obliczenia**: Automatyczne handling błędów i wartości NaN
- **Warmup period**: Automatyczne obcinanie okresu rozgrzewania
- **Memory optimization**: Optymalizowane dla dużych zbiorów danych
- **Batch processing**: Obsługuje pojedynczą parę i wszystkie pary na raz

### Uruchomienie:
```bash
cd crypto/feature_calculator_download2
python main.py --symbol ETHUSDT
python main.py --all-pairs
```

### Parametry (opcjonalne):
```bash
python main.py --input "path/to/merged_data.feather" --output "path/to/features.feather"
python main.py --start-date "2024-01-01" --end-date "2024-12-31"
```

### Kategorie Cech:

#### 1. **Cechy OHLC (5 cech)**
- **Podstawowe**: open, high, low, close, volume

#### 2. **Cechy Bamboo TA (50+ cech)**
- **Wstęgi Bollingera** (3): bb_width, bb_position, bb_*
- **RSI** (1): rsi_14
- **MACD** (3): macd_hist, macd_signal, macd
- **Stochastyczny** (2): stoch_k, stoch_d
- **ADX** (1): adx
- **CCI, Williams %R, MFI, OBV** (4): cci, williams_r, mfi, obv
- **Średnie kroczące** (6): ema_12, ema_26, sma_20, sma_50, sma_200
- **ATR, NATR, TRange** (3): atr, natr, trange
- **VWAP** (3): vwap, vwap_upper, vwap_lower
- **Supertrend** (3): supertrend, supertrend_direction, supertrend_signal
- **Ichimoku** (4): ichimoku_a, ichimoku_b, ichimoku_base, ichimoku_conversion
- **KST, TSI, UO, AO, MOM, ROC** (12): kst, kst_sig, kst_diff, tsi, tsi_signal, tsi_diff, uo, uo_bull, uo_bear, ao, ao_signal, mom, mom_signal, roc, roc_signal
- **Stoch RSI** (2): stoch_rsi_k, stoch_rsi_d
- **Inne średnie** (6): wma, hma, dema, tema, kama, t3
- **TRIX, Aroon, PSAR** (6): trix, trix_signal, aroon_up, aroon_down, aroon_ind, psar, psar_up, psar_down
- **BBands, KC, DC** (9): bbands_upper, bbands_middle, bbands_lower, kc_upper, kc_middle, kc_lower, dc_upper, dc_middle, dc_lower

#### 3. **Cechy Orderbook (40+ cech)**
- **Buy/Sell Ratio** (5): buy_sell_ratio_s1/s2/s3/s4/s5
- **Depth Imbalance** (5): depth_imbalance_s1/s2/s3/s4/s5
- **Spread** (2): spread, spread_pct
- **Bid/Ask Volume** (10): bid_volume_s1/s2/s3/s4/s5, ask_volume_s1/s2/s3/s4/s5
- **Bid/Ask Price** (10): bid_price_s1/s2/s3/s4/s5, ask_price_s1/s2/s3/s4/s5
- **Total Volume** (3): total_bid_volume, total_ask_volume, total_volume
- **Imbalance** (2): volume_imbalance, price_imbalance
- **Order Flow** (2): order_flow_imbalance, order_flow_trend
- **Scores** (2): market_microstructure_score, liquidity_score

#### 4. **Cechy Hybrydowe (10+ cech)**
- **Trends** (2): price_volume_trend, volume_price_trend
- **Korelacje** (2): orderbook_price_correlation, orderbook_volume_correlation
- **Efficiency Ratios** (4): market_efficiency_ratio, price_efficiency_ratio, volume_efficiency_ratio, orderbook_efficiency_ratio

#### 5. **Cechy Względne (15+ cech)**
- **Price Changes** (4): price_change_1m, price_change_5m, price_change_15m, price_change_30m
- **Volume Changes** (4): volume_change_1m, volume_change_5m, volume_change_15m, volume_change_30m
- **Spread Changes** (4): spread_change_1m, spread_change_5m, spread_change_15m, spread_change_30m
- **Depth Changes** (4): depth_change_1m, depth_change_5m, depth_change_15m, depth_change_30m

#### 6. **Cechy Zaawansowane (20+ cech)**
- **Volatility** (4): volatility_1m, volatility_5m, volatility_15m, volatility_30m
- **Volatility Regime** (6): volatility_regime, volatility_percentile, volatility_persistence, volatility_momentum, volatility_of_volatility, volatility_term_structure
- **Market Regime** (5): market_regime, market_trend_strength, market_trend_direction, market_choppiness, market_momentum, market_efficiency
- **Orderbook Regime** (5): orderbook_regime, orderbook_trend_strength, orderbook_trend_direction, orderbook_choppiness, orderbook_momentum, orderbook_efficiency
- **Volume Regime** (5): volume_regime, volume_trend_strength, volume_trend_direction, volume_choppiness, volume_momentum, volume_efficiency

### Konfiguracja w `config.py`:
```python
# Okna czasowe
MA_WINDOWS = [60, 240, 1440]  # 1h, 4h, 1d
ORDERBOOK_HISTORY_WINDOW = 120  # 2h
WARMUP_PERIOD_MINUTES = 1440  # 1 dzień

# Parametry techniczne
BOLLINGER_PERIOD = 20
RSI_PERIOD = 14
MACD_SHORT = 12
MACD_LONG = 26
ADX_PERIOD = 14

# Poziomy orderbook
BID_LEVELS = [1, 2, 3, 4, 5]
ASK_LEVELS = [-1, -2, -3, -4, -5]
ORDERBOOK_LEVELS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
```

### Funkcje Inteligentne:
1. **Automatyczne obliczenia**: Wszystkie cechy obliczane automatycznie
2. **Error handling**: Graceful handling błędów z kontynuacją
3. **NaN handling**: Automatyczne zastępowanie wartości NaN
4. **Memory optimization**: Optymalizowane dla dużych zbiorów danych
5. **Progress tracking**: Szczegółowe logi i progress bars
6. **Batch processing**: Obsługuje wszystkie pary jednocześnie

## Etap 7: Etykietowanie Danych

### Plik: `labeler4/main.py`
- **Cel**: Tworzy 3-klasowe etykiety (LONG, SHORT, NEUTRAL) na podstawie przyszłych ruchów ceny
- **Technologia**: System competitive labeling z 15 poziomami TP/SL
- **Format wejściowy**: Pliki feather z cechami z `feature_calculator_download2`
- **Format wyjściowy**: Pliki feather z etykietami

### Kluczowe Funkcje:
- **15 poziomów TP/SL**: Od TP=0.6%, SL=0.2% do TP=1.4%, SL=0.7%
- **Future window**: 120 minut w przyszłość
- **3-klasowe etykiety**: 0=LONG, 1=SHORT, 2=NEUTRAL
- **Batch processing**: Obsługuje pojedynczą parę i wszystkie pary na raz
- **Inteligentne obliczenia**: Automatyczne handling edge cases

### Uruchomienie:
```bash
cd crypto/labeler4
python main.py --symbol ETHUSDT
python main.py --all-pairs
```

### Parametry (opcjonalne):
```bash
python main.py --input-dir "path/to/features" --output-dir "path/to/labels"
```

### Poziomy TP/SL:
```python
TP_SL_LEVELS = [
    {"tp": 0.6, "sl": 0.2},  # label_tp0p6_sl0p2
    {"tp": 0.6, "sl": 0.3},  # label_tp0p6_sl0p3
    {"tp": 0.8, "sl": 0.2},  # label_tp0p8_sl0p2
    {"tp": 0.8, "sl": 0.3},  # label_tp0p8_sl0p3
    {"tp": 0.8, "sl": 0.4},  # label_tp0p8_sl0p4
    {"tp": 1.0, "sl": 0.3},  # label_tp1_sl0p3
    {"tp": 1.0, "sl": 0.4},  # label_tp1_sl0p4
    {"tp": 1.0, "sl": 0.5},  # label_tp1_sl0p5
    {"tp": 1.2, "sl": 0.4},  # label_tp1p2_sl0p4
    {"tp": 1.2, "sl": 0.5},  # label_tp1p2_sl0p5
    {"tp": 1.2, "sl": 0.6},  # label_tp1p2_sl0p6
    {"tp": 1.4, "sl": 0.4},  # label_tp1p4_sl0p4
    {"tp": 1.4, "sl": 0.5},  # label_tp1p4_sl0p5
    {"tp": 1.4, "sl": 0.6},  # label_tp1p4_sl0p6
    {"tp": 1.4, "sl": 0.7}   # label_tp1p4_sl0p7
]
```

### Funkcje Inteligentne:
1. **Competitive labeling**: Porównuje LONG vs SHORT pozycje
2. **Automatyczne obliczenia**: Wszystkie 15 poziomów obliczane automatycznie
3. **Progress tracking**: Szczegółowe logi i progress bars
4. **Batch processing**: Obsługuje wszystkie pary jednocześnie
5. **Error handling**: Graceful handling błędów z kontynuacją

## Etap 8: Trening Modeli

### Plik: `training4/main.py`
- **Cel**: Trenuje modele Multi-Output XGBoost dla wszystkich poziomów TP/SL
- **Technologia**: Native XGBoost API z early stopping
- **Format wejściowy**: Pliki feather z etykietami z `labeler4`
- **Format wyjściowy**: Modele JSON + raporty + wykresy

### Kluczowe Funkcje:
- **Multi-Output**: 15 osobnych modeli XGBoost (jeden na poziom TP/SL)
- **Indywidualne scalery**: Każda para ma własny scaler
- **Batch processing**: Obsługuje pojedynczą parę i wszystkie pary na raz
- **Early stopping**: Zapobieganie overfitting
- **Feature importance**: Analiza ważności cech
- **Confusion matrices**: Szczegółowa analiza wyników

### Uruchomienie:
```bash
cd crypto/training4
python main.py --symbol ETHUSDT
python main.py --all-pairs
```

### Parametry Modelu:
```python
XGB_N_ESTIMATORS = 400          # Maksymalna liczba drzew
XGB_LEARNING_RATE = 0.05        # Współczynnik uczenia
XGB_MAX_DEPTH = 6               # Maksymalna głębokość drzewa
XGB_SUBSAMPLE = 0.8             # Procent próbek użytych do budowy każdego drzewa
XGB_COLSAMPLE_BYTREE = 0.7      # Procent cech użytych do budowy każdego drzewa
XGB_GAMMA = 0.1                 # Minimalna redukcja straty wymagana do podziału
XGB_EARLY_STOPPING_ROUNDS = 20  # Zatrzymaj trening, jeśli metryka nie poprawi się
```

### Struktura Wyjściowa:
```
training4/output/
├── models/
│   ├── ETHUSDT/
│   │   ├── model_tp0p6_sl0p2.json
│   │   ├── model_tp0p6_sl0p3.json
│   │   ├── ...
│   │   ├── scaler.pkl
│   │   ├── metadata.json
│   │   └── models_index.json
│   ├── BCHUSDT/
│   │   └── ...
│   └── ...
├── reports/
│   ├── ETHUSDT/
│   │   ├── results_ETHUSDT_2025-07-31_22-30-00.md
│   │   ├── feature_importance.png
│   │   └── confusion_matrices.png
│   └── ...
└── logs/
    ├── training_all_pairs.log
    └── training_ETHUSDT.log
```

### Funkcje Inteligentne:
1. **Automatyczne wykrywanie cech**: Wykrywa dostępne cechy w danych
2. **Chronologiczny podział**: 70% train, 15% val, 15% test
3. **RobustScaler**: Skalowanie cech odpornne na outliers
4. **Class weights**: Obsługa nierównowagi klas
5. **Progress tracking**: Szczegółowe logi i progress bars
6. **Batch processing**: Obsługuje wszystkie pary jednocześnie

## Kompletna Sekwencja Uruchamiania

### 1. Pobieranie OHLC (Równolegle z Orderbook)
```bash
cd crypto/download2/OHLC
python fast_ohlc_downloader.py
```

### 2. Pobieranie Orderbook
```bash
cd crypto/download2/orderbook
python download_orderbook.py
```

### 3. Łączenie Orderbook
```bash
cd crypto/download2/orderbook
python merge_orderbook_to_feather.py
```

### 4. Uzupełnianie Luk
```bash
cd crypto/download2/orderbook
python fill_orderbook_gaps.py
```

### 5. Łączenie z OHLC
```bash
cd crypto/download2/merge
python merge_ohlc_orderbook.py
```

### 6. Obliczanie Cech
```bash
cd crypto/feature_calculator_download2
python main.py --all-pairs
```

### 7. Etykietowanie Danych
```bash
cd crypto/labeler4
python main.py --all-pairs
```

### 8. Trening Modeli
```bash
cd crypto/training4
python main.py --all-pairs
```

## Struktura Katalogów

```
crypto/
├── download2/
│   ├── OHLC/
│   │   ├── ohlc_raw/              # Surowe pliki CSV OHLC
│   │   ├── fast_ohlc_downloader.py
│   │   └── config.py
│   ├── orderbook/
│   │   ├── orderbook_raw/         # Surowe pliki CSV z API
│   │   ├── merged_raw/            # Połączone dane feather
│   │   ├── orderbook_completed/   # Dane z uzupełnionymi lukami
│   │   └── config.py              # Konfiguracja par
│   └── merge/
│       ├── merged_data/           # Dane połączone OHLC + Orderbook
│       └── merge_ohlc_orderbook.py
├── feature_calculator_download2/
│   ├── main.py                    # Główny skrypt obliczania cech
│   ├── config.py                  # Konfiguracja cech
│   └── output/                    # Wyniki obliczeń
├── labeler4/
│   ├── main.py                    # Główny skrypt etykietowania
│   ├── config.py                  # Konfiguracja etykietowania
│   └── output/                    # Wyniki etykietowania
└── training4/
    ├── main.py                    # Główny skrypt treningu
    ├── config.py                  # Konfiguracja treningu
    ├── data_loader.py             # Wczytywanie danych
    ├── model_builder.py           # Budowanie modeli
    ├── utils.py                   # Funkcje pomocnicze
    └── output/                    # Modele i raporty
```

## Monitoring i Logi

Każdy skrypt generuje:
- **Logi**: `*.log` w katalogu roboczym lub `output/logs/`
- **Metadane**: `*_metadata.json` z podsumowaniem
- **Progress bars**: Informacje o postępie w konsoli
- **Raporty**: Szczegółowe raporty w formacie markdown

## Parametry Konfiguracyjne

### W `download2/OHLC/config.py`:
- `PAIRS`: Lista par do przetwarzania
- `DOWNLOAD_CONFIG`: Parametry pobierania (interval, chunk_size, timeout)
- `FILE_CONFIG`: Ścieżki katalogów
- `LOGGING_CONFIG`: Ustawienia logowania

### W `download2/orderbook/config.py`:
- `PAIRS`: Lista par do przetwarzania
- `FILE_CONFIG`: Ścieżki katalogów
- `LOGGING_CONFIG`: Ustawienia logowania

### W `fill_orderbook_gaps.py`:
- `max_small_gap_minutes`: Maksymalna luka dla interpolacji (domyślnie 5)
- `max_medium_gap_minutes`: Maksymalna luka dla rolling average (domyślnie 60)
- `price_change_threshold`: Próg zmiany ceny w % (domyślnie 2.0)

### W `feature_calculator_download2/config.py`:
- `MA_WINDOWS`: Okna średnich kroczących
- `ORDERBOOK_HISTORY_WINDOW`: Okno historyczne orderbook
- `WARMUP_PERIOD_MINUTES`: Okres rozgrzewania
- `BOLLINGER_PERIOD`, `RSI_PERIOD`, `MACD_*`: Parametry wskaźników technicznych

### W `labeler4/config.py`:
- `PAIRS`: Lista par do przetwarzania
- `FUTURE_WINDOW_MINUTES`: Okno przyszłości (120 minut)
- `TP_SL_LEVELS`: Poziomy Take Profit i Stop Loss
- `LABEL_MAPPING`: Mapowanie etykiet (0=LONG, 1=SHORT, 2=NEUTRAL)

### W `training4/config.py`:
- `PAIRS`: Lista par do przetwarzania
- `XGB_*`: Parametry modelu XGBoost
- `CLASS_WEIGHTS`: Wagi klas dla nierównowagi
- `ENABLE_CLASS_BALANCING`: Włączanie/wyłączanie balansowania klas

## Uwagi Techniczne

1. **Wydajność**: Skrypty używają ThreadPoolExecutor dla równoległego wczytywania
2. **Pamięć**: Optymalizowane dla dużych zbiorów danych
3. **Błędy**: Graceful handling błędów z kontynuacją przetwarzania
4. **Walidacja**: Sprawdzanie integralności danych na każdym etapie
5. **CCXT**: Szybka biblioteka do komunikacji z giełdami
6. **Bamboo_TA**: Zaawansowana biblioteka do analizy technicznej
7. **XGBoost**: Szybka i skuteczna biblioteka do machine learning
8. **Batch Processing**: Obsługa wszystkich par jednocześnie

## Troubleshooting

### Problem: Brak plików CSV OHLC
- Sprawdź czy `fast_ohlc_downloader.py` został uruchomiony
- Sprawdź logi w `fast_ohlc_downloader.log`
- Sprawdź połączenie internetowe i rate limits

### Problem: Brak plików CSV Orderbook
- Sprawdź czy `download_orderbook.py` został uruchomiony
- Sprawdź logi w `download_orderbook.log`

### Problem: Błędy w merge
- Sprawdź czy wszystkie pliki CSV są kompletne
- Sprawdź logi w `merge_orderbook.log`

### Problem: Duże luki w danych
- Dostosuj parametry w `fill_orderbook_gaps.py`
- Sprawdź czy dane źródłowe nie mają długich przerw

### Problem: Rate limiting
- Zwiększ `retry_delay` w konfiguracji
- Sprawdź czy nie przekraczasz limitów API

### Problem: Błędy w obliczaniu cech
- Sprawdź czy dane wejściowe są kompletne
- Sprawdź logi w `feature_calculator.log`
- Sprawdź czy wszystkie wymagane kolumny są obecne

### Problem: Brak pamięci przy obliczaniu cech
- Zmniejsz rozmiar danych wejściowych
- Użyj filtrowania dat (`--start-date`, `--end-date`)
- Sprawdź czy masz wystarczającą ilość RAM

### Problem: Błędy w etykietowaniu
- Sprawdź czy dane wejściowe mają wymagane kolumny OHLC
- Sprawdź logi w `labeler4/output/logs/`
- Sprawdź czy future window nie przekracza dostępnych danych

### Problem: Błędy w treningu
- Sprawdź czy dane wejściowe są kompletne
- Sprawdź logi w `training4/output/logs/`
- Sprawdź czy wszystkie wymagane cechy są obecne
- Sprawdź czy etykiety są poprawnie sformatowane

### Problem: Unicode errors w logach
- To tylko problem wyświetlania w konsoli Windows
- Dane są przetwarzane poprawnie
- Sprawdź pliki logów w katalogach `output/logs/`

## Następne Kroki

Po ukończeniu kompletnego pipeline:
1. **Ewaluacja modeli**: Analiza wyników treningu
2. **Backtesting**: Testowanie na danych historycznych
3. **Live trading**: Implementacja w środowisku produkcyjnym
4. **Monitoring**: Śledzenie wydajności modeli
5. **Retraining**: Regularne aktualizacje modeli

## Podsumowanie Pipeline

Pipeline składa się z 8 głównych etapów:
1. **Pobieranie OHLC** → `download2/OHLC/`
2. **Pobieranie Orderbook** → `download2/orderbook/`
3. **Łączenie Orderbook** → `download2/orderbook/`
4. **Uzupełnianie luk** → `download2/orderbook/`
5. **Łączenie z OHLC** → `download2/merge/`
6. **Obliczanie cech** → `feature_calculator_download2/`
7. **Etykietowanie** → `labeler4/`
8. **Trening modeli** → `training4/`

Każdy etap generuje dane wejściowe dla następnego etapu, tworząc kompletny pipeline od surowych danych do wytrenowanych modeli gotowych do użycia w tradingu. 