# 📋 Instrukcja Trenowania Modeli ML (Dual-Window System)

## 🚀 Podstawowe Uruchomienie

### 💻 Bezpośrednio na Host (CPU/GPU)
```bash
# Trening z domyślnymi parametrami
python user_data/training/scripts/train_dual_window_model.py

# Trening z presetem
python user_data/training/scripts/train_dual_window_model.py --preset quick

# Trening z własnymi parametrami
python user_data/training/scripts/train_dual_window_model.py --pair ETH_USDT --epochs 100 --take-profit 2.0
```

### 🐳 Przez Docker Wrapper (GPU Priority)
```bash
# Identyczna składnia, ale w Docker z GPU
python train_gpu.py --preset quick

# Wszystkie parametry działają tak samo
python train_gpu.py --pair ETH_USDT --epochs 100 --take-profit 2.0

# Windows batch launcher
train_gpu.bat --preset production --pair BTC_USDT
```

## 🔄 Docker vs Host - Który Wybrać?

| Cecha | 🐳 Docker Wrapper | 💻 Host Direct |
|-------|-------------------|-----------------|
| **GPU Performance** | ✅ Optymalne | ⚠️ Zależy od systemu |
| **Izolacja** | ✅ Czyste środowisko | ❌ Host dependencies |
| **Setup** | ⚠️ Wymaga Docker | ✅ Prostsze |
| **Parametry** | ✅ Identyczne | ✅ Identyczne |
| **Wyniki** | ✅ Automatycznie na host | ✅ Lokalne |
| **Data Leakage** | ✅ Eliminowany (Dual-Window) | ✅ Eliminowany (Dual-Window) |

**Zalecenie**: Użyj `train_gpu.py` dla produkcji i bezpośrednio skryptu dla rozwoju.

## 📊 Parametry Treningu

### 🎯 Podstawowe Parametry

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--pair` | BTC_USDT | Para kryptowalut do treningu |
| `--epochs` | 50 | Liczba epok treningu |
| `--window-past` | 60 | Ile świec wstecz analizuje model |
| `--window-future` | 60 | Ile świec w przód przewiduje |

### 📅 Parametry Czasowe

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--date-from` | 2024-01-01 | Data początkowa danych |
| `--date-to` | 2024-12-31 | Data końcowa danych |

### 💰 Risk Management

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--take-profit` | 1.0 | Take Profit w procentach |
| `--stop-loss` | 0.5 | Stop Loss w procentach |

### ⚙️ Parametry Modelu

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--batch-size` | 32 | Rozmiar batcha |
| `--learning-rate` | 0.001 | Współczynnik uczenia |
| `--validation-split` | 0.2 | Podział na walidację (20%) |
| `--early-stopping` | 3 | Po ilu epokach bez poprawy zatrzymać |

### 📁 Parametry Wyjścia

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--output-dir` | auto | Katalog wyników (automatyczny) |
| `--config` | auto | Plik konfiguracyjny TrainingConfig |

### 🔍 Parametry Walidacji

| Parametr | Opis |
|----------|------|
| `--validate-data` | Tylko sprawdź dostępność danych (nie trenuj) |
| `--dry-run` | Dry run - sprawdź konfigurację (nie trenuj) |

## 🎨 Gotowe Presety

### `quick` - Szybki Test
```bash
python train_gpu.py --preset quick
```
- **Czas**: ~2-5 min
- **Dane**: Ostatnie 30 dni
- **Epoki**: 5
- **Zastosowanie**: Szybkie testy, rozwój

### `standard` - Standardowy Trening
```bash
python train_gpu.py --preset standard
```
- **Czas**: ~20-30 min
- **Dane**: Cały 2024 rok
- **Epoki**: 50
- **Zastosowanie**: Normalne trenowanie

### `production` - Produkcyjny Model
```bash
python train_gpu.py --preset production
```
- **Czas**: ~1-2 godziny
- **Dane**: Od 2020 do dziś
- **Epoki**: 100
- **Zastosowanie**: Finalne modele

### `test` - Minimalny Test
```bash
python train_gpu.py --preset test
```
- **Czas**: ~30 sekund
- **Dane**: Ostatnie 7 dni
- **Epoki**: 2
- **Zastosowanie**: Debugging, testy kodu

## 💡 Przykłady Użycia

### Podstawowe Scenariusze

```bash
# Szybki test nowej pary
python train_gpu.py --preset quick --pair DOGE_USDT

# Trening z większym TP/SL
python train_gpu.py --take-profit 2.5 --stop-loss 1.0

# Długie okna czasowe
python train_gpu.py --window-past 120 --window-future 30

# Konkretny okres
python train_gpu.py --date-from 2024-06-01 --date-to 2024-12-01

# Szybsze uczenie z większym batch
python train_gpu.py --batch-size 64 --learning-rate 0.002
```

### Kombinacje Presetów

```bash
# Preset + nadpisanie parametrów
python train_gpu.py --preset production --pair ETH_USDT --epochs 150

# Szybki test z własnymi parametrami TP/SL
python train_gpu.py --preset quick --take-profit 3.0 --stop-loss 1.5

# Test z custom oknem
python train_gpu.py --preset test --window-past 30 --window-future 15
```

### Walidacja i Debugging

```bash
# Sprawdź tylko dostępność danych
python train_gpu.py --pair ETH_USDT --validate-data

# Dry run - sprawdź konfigurację
python train_gpu.py --preset production --dry-run

# Z custom config file
python train_gpu.py --config custom_training_config.json
```

### Windows Batch Shortcuts

```cmd
REM Szybkie testy
train_gpu.bat --preset test
train_gpu.bat --preset quick --pair ETH_USDT

REM Produkcyjne treningi
train_gpu.bat --preset production
train_gpu.bat --preset standard --epochs 200 --take-profit 2.0
```

### Host Direct (bez Docker)

```bash
# Z głównego katalogu Freqtrade
python user_data/training/scripts/train_dual_window_model.py --preset quick

# Z katalogu scripts
cd user_data/training/scripts
python train_dual_window_model.py --pair BTC_USDT --epochs 100
```

## 📁 Struktura Wyników

Po treningu system tworzy katalog w `ml_artifacts/` z nazwą:
```
PARA_DATA-OD_DATA-DO_EPOKIepochs_wOKNO-PRZESZŁE-PRZYSZŁE_tpTP_slSL_TIMESTAMP/
```

### Zawartość Katalogu

- `best_model_PARA.keras` - Najlepszy model z treningu
- `config.json` - Konfiguracja TrainingConfig użyta w treningu
- `training_history.json` - Historia treningu (loss, accuracy)
- `evaluation_report.json` - Szczegółowe metryki ewaluacji
- `training_log.txt` - Pełny log treningu
- `README.md` - Podsumowanie treningu

## 🔧 Rozwiązywanie Problemów

### Częste Błędy

**Błąd daty**: Sprawdź format YYYY-MM-DD
```bash
# Źle
--date-from 01-01-2024

# Dobrze  
--date-from 2024-01-01
```

**Za mało danych**: System automatycznie sprawdza dostępność
```bash
# Sprawdź przed treningiem
python train_gpu.py --pair NEW_PAIR --validate-data
```

**Brak danych dla pary**: Upewnij się że dane są w user_data
```bash
❌ Brakuje danych dla pary XXX_USDT
💡 Sprawdź czy dane są w user_data/data/
```

**GPU Issues**: System automatycznie przełączy na CPU

**Docker Issues**: 
```bash
# Brak Docker
❌ Docker nie znaleziony
💡 Zainstaluj Docker Desktop

# Brak GPU w Docker  
❌ GPU nie wykryte, będzie używane CPU
💡 Zainstaluj NVIDIA Container Toolkit

# Brak obrazu freqtrade:gpu
❌ Unable to find image 'freqtrade:gpu'
💡 Dostosuj nazwę obrazu w train_gpu.py
```

### Optymalizacja Wydajności

- **GPU**: Automatyczna detekcja i użycie
- **Batch Size**: Większy = szybszy (ale więcej RAM)
- **Early Stopping**: Zatrzyma gdy model przestanie się poprawiać
- **Validation Split**: 0.2 = optymalne dla większości przypadków
- **Dual-Window**: Eliminuje data leakage automatycznie

## 📈 Monitorowanie Treningu

System pokazuje w czasie rzeczywistym:
- Postęp epok z progress bar
- Accuracy i loss (train/validation)  
- Czasochłonność epok
- Informacje o GPU/CPU
- Metryki dual-window approach

## 🎯 Najlepsze Praktyki

1. **Zawsze zacznij od presetu `quick`** dla nowych par/parametrów
2. **Użyj `standard`** dla normalnego rozwoju  
3. **`production`** tylko dla finalnych modeli
4. **Zapisuj konfiguracje** które działają dobrze w config files
5. **Monitoruj overfit** - validation loss vs training loss
6. **Docker dla produkcji** - `train_gpu.py` dla długich treningów
7. **Host dla testów** - bezpośrednio skrypt dla szybkich eksperymentów
8. **Sprawdzaj dane** - używaj `--validate-data` przed długimi treningami

## 🐳 Docker Wrapper - Szczegóły

### 🛠️ Wymagania Docker
- Docker Desktop z GPU support
- NVIDIA Container Toolkit (dla GPU)
- Obraz `freqtrade:gpu` (lub dostosuj nazwę w train_gpu.py)

### 🚀 Uruchomienie Wrapper

```bash
# Podstawowe (automatyczne GPU jeśli dostępne)
python train_gpu.py

# Wszystkie opcje działają identycznie
python train_gpu.py --preset quick --pair DOGE_USDT
python train_gpu.py --epochs 200 --take-profit 3.0 --window-past 120

# Windows
train_gpu.bat --preset production
```

### 📁 Montowanie Katalogów

Wrapper automatycznie montuje:
- `./ml_artifacts` → `/workspace/ml_artifacts` (wyniki)
- `./user_data` → `/workspace/user_data` (dane)
- `./` → `/workspace/host` (kod źródłowy)

### 🔧 Automatyczne Funkcje

- **GPU Detection**: Automatyczne `--gpus all` jeśli GPU dostępne
- **CPU Fallback**: Przełączenie na CPU gdy brak GPU
- **Directory Creation**: Tworzenie katalogów `ml_artifacts/`, `user_data/`
- **Parameter Pass-Through**: Wszystkie parametry przekazane 1:1
- **Error Handling**: Propagacja kodów błędów z kontenera
- **Dual-Window Pipeline**: Używa zaawansowanego systemu eliminującego data leakage

### 💡 Przykłady Docker Wrapper

```bash
# Szybki test w Docker
train_gpu.bat --preset test

# Produkcyjny model z GPU
python train_gpu.py --preset production --pair ETH_USDT --epochs 150

# Eksperyment z parametrami
python train_gpu.py --window-past 120 --window-future 30 --learning-rate 0.0005

# Długi trening overnight
python train_gpu.py --preset production --epochs 500 --early-stopping 10

# Z walidacją danych
python train_gpu.py --pair NEW_PAIR --validate-data
```

## ⚡ Różnice vs Poprzednia Wersja

| Aspekt | Poprzednia (Universal) | Obecna (Dual-Window) |
|--------|----------------------|---------------------|
| **Dane** | Mock/Symulowane | Prawdziwe z Freqtrade |
| **Data Leakage** | Możliwy | Eliminowany |
| **Architektura** | Prosta LSTM | Zaawansowana Dual-Window |
| **Ewaluacja** | Podstawowa | Comprehensive |
| **Konfiguracja** | CLI only | CLI + TrainingConfig |
| **Presety** | Proste | Zaawansowane (TrainingConfig) |
