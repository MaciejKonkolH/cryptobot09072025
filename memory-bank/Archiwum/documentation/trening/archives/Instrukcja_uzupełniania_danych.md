# Instrukcja Uzupełniania Danych - Moduł Ciągłości Danych

## Przegląd Systemu

Moduł sprawdzający ciągłość danych i uzupełniający luki składa się z dwóch głównych komponentów:

1. **`DataContinuityChecker`** - główny moduł sprawdzający ciągłość
2. **`preprocess_data.py`** - skrypt uruchamiający preprocessing

## Struktura Plików Modułu

### Pliki Główne
```
scripts/
└── preprocess_data.py              # Główny skrypt uruchamiający (293 linie)

core/utils/
└── data_continuity_checker.py      # Moduł DataContinuityChecker (551 linii)
```

### Struktura Katalogów Danych
```
data/
├── raw/                            # Surowe dane wejściowe
│   ├── BTCUSDT_1m.feather         # Przykład: 47KB, 1,368 świec
│   └── BTCUSDT_1m_large_gap.feather # Test z dużymi lukami
├── processed/                      # Dane po preprocessing
│   ├── BTCUSDT_1m_clean.feather   # Uzupełnione dane
│   └── BTCUSDT_1m_validated.feather
└── logs/                          # Raporty jakości
    ├── BTCUSDT_quality_report.json # Szczegółowy raport
    └── detailed_continuity_report.json
```

### Prawdziwe Dane Freqtrade
```
ft_bot_docker_compose/user_data/data/binanceusdm/futures/
└── BTC_USDT-1m-futures.feather    # 76MB, ~2.8M świec (5.3 lat danych!)
```

## Uruchamianie Modułu

### Plik Uruchamiający
**Główny skrypt:** `scripts/preprocess_data.py`

### Metody Uruchamiania

#### 1. Automatyczne Ścieżki (Rekomendowane)
```bash
# Podstawowe użycie
python scripts/preprocess_data.py --pair BTCUSDT

# Z niższym progiem jakości
python scripts/preprocess_data.py --pair BTCUSDT --min-quality 70
```

#### 2. Ręczne Ścieżki
```bash
# Pełna kontrola nad ścieżkami
python scripts/preprocess_data.py \
  --input data/raw/ETHUSDT_1m.feather \
  --output data/processed/ETHUSDT_1m_clean.feather \
  --report data/logs/ETHUSDT_quality_report.json
```

#### 3. Przetwarzanie Prawdziwych Danych Freqtrade
```bash
# Przetwarzanie pełnego 5-letniego datasetu
python scripts/preprocess_data.py \
  --input ft_bot_docker_compose/user_data/data/binanceusdm/futures/BTC_USDT-1m-futures.feather \
  --output data/processed/BTC_USDT_5years_clean.feather \
  --report data/logs/BTC_USDT_5years_report.json
```

## Parametry i Opcje

### Parametry Obowiązkowe
- **`--pair PARA`** LUB **`--input ŚCIEŻKA`** - źródło danych
- **`--output ŚCIEŻKA`** - wymagane z `--input`

### Parametry Opcjonalne
- **`--report ŚCIEŻKA`** - ścieżka raportu JSON (auto-generowana jeśli brak)
- **`--min-quality PROCENT`** - próg jakości danych (domyślnie: 80.0%)

### Obsługiwane Formaty
- **Wejście:** `.feather`, `.csv`, `.parquet`
- **Wyjście:** `.feather` (domyślnie), `.csv`, `.parquet`

## Presety i Konfiguracja

### ❌ Brak Presetów
Moduł `preprocess_data.py` **NIE MA** systemu presetów jak `train_dual_window_model.py`.

### Konfiguracja DataContinuityChecker
```python
# Domyślne ustawienia w kodzie
timeframe = '1m'           # Interwał czasowy
tolerance_seconds = 60     # Tolerancja wykrywania luk
min_quality = 80.0         # Próg jakości danych
```

## Lokalizacja Danych

### Dane Wejściowe (Surowe)
1. **Testowe dane:** `data/raw/`
   - `BTCUSDT_1m.feather` (47KB, 1 dzień)
   - `BTCUSDT_1m_large_gap.feather` (14KB, test luk)

2. **Prawdziwe dane Freqtrade:** `ft_bot_docker_compose/user_data/data/binanceusdm/futures/`
   - `BTC_USDT-1m-futures.feather` (76MB, 5.3 lat!)

### Dane Wyjściowe (Przetworzone)
- **Lokalizacja:** `data/processed/`
- **Format:** `.feather` (domyślnie)
- **Zawartość:** Uzupełnione luki, zachowana ciągłość cenowa

### Raporty Jakości
- **Lokalizacja:** `data/logs/`
- **Format:** JSON
- **Zawartość:** Szczegółowe statystyki, analiza luk, jakość danych

## Strategia Uzupełniania - BRIDGE

### Algorytm BRIDGE (Uniwersalny)
- **Interpolacja cenowa:** Płynne przejście `before_candle.close → after_candle.open`
- **Realistyczny szum:** Proporcjonalny do volatility i trendu
- **Ciągłość OHLC:** Zachowanie relacji między świecami
- **Uniwersalność:** Jedna strategia dla wszystkich rozmiarów luk

### Klasyfikacja Luk
- **Małe:** < 15 minut
- **Średnie:** 15-60 minut  
- **Duże:** > 60 minut (specjalne logowanie, ale nadal uzupełniane)

## Przykład Raportu Jakości

```json
{
  "original_candles": 1368,
  "final_candles": 1440,
  "candles_added": 72,
  "gaps_detected": 5,
  "quality_score": 80.5,
  "quality_status": "ACCEPTABLE",
  "bridge_statistics": {
    "strategy_used": "BRIDGE",
    "all_gaps_filled": true,
    "original_data_ratio": 95.0,
    "synthetic_data_ratio": 5.0
  }
}
```

## Workflow Produkcyjny

### 1. Przygotowanie Danych
```bash
# Przetwórz prawdziwe dane Freqtrade (5 lat)
python scripts/preprocess_data.py \
  --input ft_bot_docker_compose/user_data/data/binanceusdm/futures/BTC_USDT-1m-futures.feather \
  --output data/processed/BTC_USDT_full_clean.feather \
  --min-quality 75
```

### 2. Weryfikacja Jakości
```bash
# Sprawdź raport w data/logs/
cat data/logs/BTC_USDT_full_quality_report.json
```

### 3. Użycie w Treningu
```bash
# Użyj przetworzone dane w treningu ML
docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_dual_window_model.py \
  --preset production \
  --input /freqtrade/user_data/training/data/processed/BTC_USDT_full_clean.feather
```

## Kody Wyjścia

- **0** - Sukces, jakość powyżej progu
- **1** - Błąd krytyczny (brak pliku, błąd przetwarzania)
- **2** - Ostrzeżenie, jakość poniżej progu (dane zapisane)

## Uwagi Techniczne

### Reprodukowalność
- Stały seed: `np.random.seed(42)`
- Deterministyczne uzupełnianie luk

### Wydajność
- Obsługa dużych plików (76MB+)
- Efektywne przetwarzanie w pandas
- Minimalne zużycie pamięci

### Kompatybilność
- Python 3.8+
- pandas, numpy
- Formaty: feather, CSV, parquet
