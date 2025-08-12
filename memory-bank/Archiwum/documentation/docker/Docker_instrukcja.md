# Docker Instrukcja - Projekt Freqtrade ML

**Data utworzenia:** 27 maja 2025  
**Kontekst:** Rozwiązanie problemów Docker z treningiem ML  
**Status:** ZWERYFIKOWANE - DZIAŁAJĄCE  
**Ostatnia aktualizacja:** 27 maja 2025 - Czyszczenie zbędnych obrazów

## 🎯 **PRZEGLĄD SYSTEMU DOCKER**

### **Struktura obrazów w projekcie:**
```
ft_bot_docker_compose/
├── docker-compose.yml              # ✅ GŁÓWNY - używany przez wrapper
├── docker-compose.gpu.yml          # 🔧 GPU - alternatywny
├── Dockerfile.custom               # ✅ GŁÓWNY - używany przez wrapper
├── Dockerfile.gpu                  # 🔧 GPU - alternatywny
├── backup_cpu_image_20250524_152051.tar  # 💾 BACKUP (6.0GB)
└── backup_gpu_failed_cudnn.tar     # ❌ USZKODZONY BACKUP
```

## 📋 **DOSTĘPNE OBRAZY DOCKER**

### **✅ DZIAŁAJĄCE OBRAZY (zweryfikowane 27.05.2025):**

#### **1. `ft_bot_docker_compose-freqtrade:latest` - GŁÓWNY**
- **Rozmiar:** 19.2GB
- **Status:** ✅ DZIAŁAJĄCY
- **Użycie:** Wrapper `train_gpu.py`
- **GPU:** ✅ CUDA 11.8.0 + cuDNN 8906
- **Dockerfile:** `Dockerfile.custom`
- **Compose:** `docker-compose.yml`

#### **2. `freqtrade-cpu-backup:20250524_152008` - BACKUP**
- **Rozmiar:** 19.2GB  
- **Status:** ✅ DZIAŁAJĄCY
- **Użycie:** Backup głównego obrazu
- **GPU:** ✅ CUDA 11.8.0

#### **3. `ft_bot_docker_compose-freqtrade-gpu:latest` - GPU**
- **Rozmiar:** 19.2GB
- **Status:** ✅ DZIAŁAJĄCY
- **Użycie:** Alternatywny GPU
- **Dockerfile:** `Dockerfile.gpu`
- **Compose:** `docker-compose.gpu.yml`

### **🔧 POMOCNICZE OBRAZY:**

#### **4. `freqtradeorg/freqtrade:stable`**
- **Rozmiar:** 1.37GB
- **Status:** ✅ OFICJALNY
- **Użycie:** Bazowy obraz Freqtrade

#### **5. `nvidia/cuda:11.8.0-base-ubuntu20.04`**
- **Rozmiar:** 345MB
- **Status:** ✅ BAZOWY
- **Użycie:** Obraz bazowy CUDA (może być używany)

### **🗑️ USUNIĘTE OBRAZY (27.05.2025):**

#### **Zbędne obrazy Freqtrade:**
- ❌ `freqtrade-freqtrade:latest` (9.45GB) - status nieznany
- ❌ `freqtrade-gpu:latest` (4.53GB) - status nieznany
- ❌ `binancebot-freqtrade:latest` (4.55GB) - nie wymieniony w dokumentacji

#### **Zbędne obrazy CUDA:**
- ❌ `nvidia/cuda:11.6.2-base-ubuntu20.04` (224MB) - nieużywany
- ❌ `nvidia/cuda:12.0.0-base-ubuntu22.04` (338MB) - nieużywany

#### **Inne:**
- ❌ `hello-world:latest` (20.4kB) - testowy
- ❌ **Build Cache** (30.67GB) - wyczyszczony

### **💾 OSZCZĘDNOŚCI MIEJSCA:**
- **Usunięte obrazy:** ~49.5GB
- **Całkowite oszczędności:** **~52.8GB**
- **Aktualny stan:** 19.91GB (tylko ważne obrazy)

## 🚀 **WRAPPER TRAIN_GPU.PY**

### **Jak działa wrapper:**
```python
# Wrapper używa ZAWSZE docker-compose.yml (nie GPU!)
docker-compose run --rm freqtrade python3 /freqtrade/user_data/training/scripts/train_dual_window_model.py [parametry]
```

### **Konfiguracja wrapper:**
- **Plik:** `docker-compose.yml`
- **Serwis:** `freqtrade`
- **Obraz:** `ft_bot_docker_compose-freqtrade:latest`
- **Dockerfile:** `Dockerfile.custom`
- **Komenda:** `["sleep", "infinity"]` (umożliwia uruchomienie skryptów)

## ⚠️ **TYPOWE PROBLEMY I ROZWIĄZANIA**

### **🚨 Problem 1: "Docker Engine nie odpowiada"**
**Objawy:**
```
request returned 500 Internal Server Error for API route and version http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping
```

**Rozwiązanie:**
1. **Zamknij Docker Desktop** (Quit z systemtray)
2. **Poczekaj 30 sekund**
3. **Uruchom Docker Desktop ponownie**
4. **Poczekaj 2-3 minuty** na pełne uruchomienie
5. **Test:** `docker --version`

### **🚨 Problem 2: "Wrapper się zawiesza"**
**Objawy:**
- Wrapper uruchamia się ale zatrzymuje po "Uruchamianie Docker Compose"
- Tworzą się nowe kontenery ale nic się nie dzieje

**Przyczyny:**
1. **Obraz nie istnieje** - Docker próbuje zbudować obraz
2. **Błędna komenda** - kontener ma `freqtrade trade` zamiast `sleep infinity`
3. **Brak skryptów** - ścieżka do `train_dual_window_model.py` nie istnieje

**Rozwiązanie:**
```bash
# 1. Sprawdź czy obraz istnieje
docker images | findstr ft_bot_docker_compose-freqtrade

# 2. Sprawdź konfigurację
docker-compose config

# 3. Sprawdź skrypty
dir user_data\training\scripts\

# 4. Test obrazu
docker run --rm ft_bot_docker_compose-freqtrade:latest python3 --version
```

### **🚨 Problem 3: "Skrypty nie znalezione"**
**Objawy:**
```
ls: cannot access '/freqtrade/user_data/training/scripts/': No such file or directory
```

**Rozwiązanie:**
```bash
# Sprawdź na hoście
dir user_data\training\scripts\train_dual_window_model.py

# Sprawdź montowanie volumów w docker-compose.yml
volumes:
  - "./user_data:/freqtrade/user_data"
```

### **🚨 Problem 4: "Wiele kontenerów się tworzy"**
**Objawy:**
- Po każdym uruchomieniu wrapper tworzy nowy kontener
- Kontenery `freqtrade-run-XXXXX` się gromadzą

**Przyczyna:**
- `docker-compose run` tworzy nowy kontener za każdym razem
- To normalne zachowanie

**Czyszczenie:**
```bash
# Usuń zatrzymane kontenery
docker container prune -f

# Usuń nieużywane obrazy
docker image prune -f
```

## 💾 **BACKUP I RESTORE**

### **Tworzenie backup obrazu:**
```bash
# Backup głównego obrazu
docker save ft_bot_docker_compose-freqtrade:latest -o backup_main_image_$(date +%Y%m%d_%H%M%S).tar

# Sprawdź rozmiar
ls -lh backup_*.tar
```

### **Przywracanie backup:**
```bash
# Przywróć obraz z backup
docker load -i backup_cpu_image_20250524_152051.tar

# Sprawdź czy się przywrócił
docker images | findstr freqtrade
```

### **Dostępne backupy:**
- ✅ `backup_cpu_image_20250524_152051.tar` (6.0GB) - DZIAŁAJĄCY
- ❌ `backup_gpu_failed_cudnn.tar` (5.3GB) - USZKODZONY

## 🔧 **DIAGNOSTYKA KROK PO KROKU**

### **1. Sprawdź Docker Engine:**
```bash
docker --version
docker-compose --version
```

### **2. Sprawdź dostępne obrazy:**
```bash
docker images
```

### **3. Sprawdź konfigurację wrapper:**
```bash
docker-compose config
```

### **4. Test obrazu:**
```bash
docker run --rm ft_bot_docker_compose-freqtrade:latest python3 --version
```

### **5. Test GPU w obrazie:**
```bash
docker run --rm --gpus all ft_bot_docker_compose-freqtrade:latest nvidia-smi
```

### **6. Test skryptów:**
```bash
docker run --rm -v "./user_data:/freqtrade/user_data" ft_bot_docker_compose-freqtrade:latest ls -la /freqtrade/user_data/training/scripts/
```

### **7. Uruchom wrapper:**
```bash
python train_gpu.py --pair BTC_USDT --epochs 5 --start-date 2024-04-01 --end-date 2024-04-07 --window-size 60 --tp-pct 0.8 --sl-pct 0.4
```

## 📊 **WERYFIKACJA DZIAŁANIA**

### **✅ Oznaki poprawnego działania:**
```
🚀 GPU TRAINING DOCKER WRAPPER (DUAL-WINDOW)
✅ Docker Compose dostępny: Docker Compose version v2.35.1-desktop.1
✅ Serwis freqtrade dostępny
🐳 Uruchamianie Docker Compose:

==========
== CUDA ==
==========
CUDA Version 11.8.0

🎯 UNIVERSAL DUAL-WINDOW ML TRAINING
```

### **❌ Oznaki problemów:**
```
request returned 500 Internal Server Error
❌ Docker Compose nie znaleziony
❌ Serwis freqtrade nie znaleziony
ls: cannot access '/freqtrade/user_data/training/scripts/'
```

## 🎯 **NAJLEPSZE PRAKTYKI**

### **1. Przed każdym treningiem:**
```bash
# Sprawdź Docker
docker --version

# Sprawdź obrazy
docker images | findstr ft_bot_docker_compose-freqtrade

# Wyczyść stare kontenery
docker container prune -f
```

### **2. W przypadku problemów:**
1. **Restart Docker Desktop**
2. **Sprawdź konfigurację** (`docker-compose config`)
3. **Test obrazu** (`docker run --rm [obraz] python3 --version`)
4. **Przywróć backup** jeśli potrzeba

### **3. Regularne backupy:**
```bash
# Co tydzień
docker save ft_bot_docker_compose-freqtrade:latest -o backup_weekly_$(date +%Y%m%d).tar
```

### **4. Monitoring i czyszczenie (NOWE):**
```bash
# Sprawdź wykorzystanie miejsca
docker system df

# Wyczyść build cache (jeśli zajmuje dużo miejsca)
docker builder prune -f

# Wyczyść nieużywane obrazy
docker image prune -f

# Wyczyść wszystko nieużywane (OSTROŻNIE!)
docker system prune -f

# Lista wszystkich obrazów z rozmiarami
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

## 🏆 **PODSUMOWANIE**

### **✅ DZIAŁAJĄCA KONFIGURACJA (27.05.2025):**
- **Wrapper:** `train_gpu.py`
- **Compose:** `docker-compose.yml`
- **Obraz:** `ft_bot_docker_compose-freqtrade:latest`
- **GPU:** CUDA 11.8.0 + cuDNN 8906
- **Backup:** `backup_cpu_image_20250524_152051.tar`

### **📊 STAN OBRAZÓW PO CZYSZCZENIU (27.05.2025):**
- **Aktywne obrazy:** 5 (wszystkie ważne)
- **Całkowity rozmiar:** 19.91GB
- **Usunięte obrazy:** 6 + cache (52.8GB oszczędności)
- **Reclaimable space:** 13.49GB (warstwy współdzielone)

### **🎯 KLUCZOWE PUNKTY:**
1. **Wrapper używa `docker-compose.yml`** (nie GPU!)
2. **Obraz musi mieć `sleep infinity`** jako komendę
3. **Skrypty muszą być w `user_data/training/scripts/`**
4. **Docker Desktop musi być w pełni uruchomiony**
5. **Backup jest dostępny** w przypadku problemów
6. **Zbędne obrazy zostały usunięte** (52.8GB oszczędności)

### **🔧 KONSERWACJA:**
- **Regularne czyszczenie:** `docker container prune -f`
- **Backup co tydzień:** `docker save ft_bot_docker_compose-freqtrade:latest`
- **Monitoring miejsca:** `docker system df`

---

*Instrukcja utworzona na podstawie rzeczywistych problemów i rozwiązań z 27 maja 2025*
*Ostatnia aktualizacja: 27 maja 2025 - Czyszczenie zbędnych obrazów Docker*
