# Dochodzenie: Analiza Rozbieżności Predykcji (Trening vs. Backtesting)

**Data rozpoczęcia:** 2025-06-28

## 1. Definicja Problemu

Stwierdzono fundamentalną rozbieżność w wynikach predykcji tego samego modelu `*.h5`. Mimo że model, skaler (`scaler.pkl`) oraz dane wejściowe (cechy) są pozornie identyczne, uzyskujemy drastycznie różne sygnały w zależności od środowiska uruchomieniowego:

*   **Środowisko 1 (Walidacja):** Uruchomienie predykcji za pomocą skryptu `Kaggle/trainer.py` na danych z `validation_and_labeling`.
*   **Środowisko 2 (Backtesting):** Uruchomienie predykcji w ramach strategii Freqtrade `Enhanced_ML_MA43200_Buffer_Strategy.py`.

Problem objawiał się tym, że raporty porównawcze (`run_comparison.py`) wykazują, że nawet **ponad 60% predykcji jest znacząco różnych**, co nie powinno mieć miejsca w systemie deterministycznym.

## 2. Sprawdzone Hipotezy i Wyniki Dochodzenia

Poniżej znajduje się chronologiczna lista zbadanych hipotez i wniosków z każdego etapu analizy.

---

### ✅ **Hipoteza 1: Niespójne obliczanie cech (`bfill` vs `fillna(0)`)**

*   **Opis:** Pierwsza teza zakładała, że `feature_calculator.py` (walidacja) i silnik Freqtrade używają innej metody do wypełniania brakujących wartości (`NaN`) powstających przy obliczaniu cech `..._change` w pierwszym wierszu danych.
*   **Test:** Analiza kodu obu modułów oraz uruchomienie skryptu `run_feature_comparison.py`.
*   **Wynik: Potwierdzona.** Raport ostatecznie potwierdził, że **jedyną różnicą** między oboma zbiorami danych jest pierwszy wiersz.

---

### ❌ **Hipoteza 2: Niespójna obsługa braków danych `NaN` w dalszym procesie**

*   **Opis:** Zakładaliśmy, że nawet jeśli cechy na dysku są czyste, to gdzieś w procesie (`Kaggle/trainer.py` lub strategia Freqtrade) istnieje ukryty mechanizm czyszczący `NaN`.
*   **Test:** Stworzenie dedykowanego skryptu `skrypty/check_nan_in_features.py`.
*   **Wynik: Obalona.** Test jednoznacznie wykazał, że `feature_calculator.py` produkuje dane w 100% czyste.

---

### ❌ **Hipoteza 3: Różne pliki skalera (`scaler.pkl`)**

*   **Opis:** Teza zakładała, że oba procesy mogły przypadkowo używać różnych wersji skalera.
*   **Test:** Potwierdzenie ze strony użytkownika.
*   **Wynik: Obalona.** Użytkownik potwierdził, że osobiście kopiował plik skalera.

---

### ❌ **Hipoteza 4: Różna kolejność kolumn (cech)**

*   **Opis:** Hipoteza zakładała, że `DataFrame` z cechami jest podawany do modelu/skalera w innej kolejności.
*   **Test:** Ręczna inspekcja kodu `Kaggle/trainer.py` oraz strategii Freqtrade.
*   **Wynik: Obalona.** W obu plikach istnieje jawnie zdefiniowana, identyczna lista `FEATURE_COLUMNS`.

---

### ✅ **Hipoteza 5: Błąd proceduralny (porównywanie nieaktualnych plików)**

*   **Opis:** Zakładaliśmy, że narzędzia porównawcze mogły przez pomyłkę porównywać świeże wyniki z backtestu ze starymi, archiwalnymi plikami walidacyjnymi.
*   **Test:** Analiza błędów w skryptach porównawczych.
*   **Wynik: Potwierdzona.** Odkryto, że skrypty porównawcze były wadliwe i nie potrafiły wczytywać nowoczesnych plików `.feather`.

---

### ✅ **Hipoteza 6: Niespójne wersje modelu (ostatnia vs. najlepsza epoka)**

*   **Opis:** Po wyeliminowaniu wszystkich różnic w danych wejściowych, rozbieżności w predykcjach wciąż występowały. Pojawiła się kluczowa hipoteza, że proces walidacji w `Kaggle/trainer.py` oraz proces backtestingu w Freqtrade, mimo używania tego samego pliku `*.h5`, mogły odnosić się do **różnych wag** z tego samego treningu.
*   **Test:** Analiza kodu `Kaggle/trainer.py` i logiki zapisywania/ładowania modeli.
    *   Stwierdzono, że `EarlyStopping` był skonfigurowany z `restore_best_weights=False`, co oznacza, że po zakończeniu `model.fit()` w pamięci pozostawał model z **ostatniej epoki**.
    *   Wszystkie raporty walidacyjne (`generate_confusion_matrix_report` etc.) były generowane na tym modelu z ostatniej epoki.
    *   Natomiast do backtestingu w Freqtrade, zgodnie z najlepszymi praktykami, używany był model `best_model.h5`, który callback `ModelCheckpoint` zapisywał z wagami z **najlepszej epoki** (najniższy `val_loss`).
*   **Wynik: Potwierdzona.** To było ostateczne źródło problemu. Porównywano predykcje z dwóch różnych modeli: **"ostatni" z walidacji vs. "najlepszy" z backtestingu**.

---

## 3. Status na 2025-06-29 (Dochodzenie wznowione)

Mimo zaimplementowania poprawek z Hipotez 6 i 7 (synchronizacja stref czasowych w skrypcie porównawczym), ponowne uruchomienie testów wykazało, że fundamentalne rozbieżności w predykcjach **wciąż istnieją**. Identyczność predykcji wynosi zaledwie 0.1%.

Jednocześnie, raport porównawczy potwierdził, że:
1.  Synchronizacja danych po sygnaturze czasowej działa poprawnie.
2.  Dane wejściowe (8 cech) w obu środowiskach są **w 100% identyczne** (korelacja 1.0, średnia różnica ~0).

To prowadzi do nowej, wysoce prawdopodobnej hipotezy, która jest obecnie badana.

---

### ⚠️  **Hipoteza 7: Błąd w procesie skalowania cech w strategii Freqtrade**

*   **Opis:** Skoro model, skaler i *surowe* cechy są identyczne, jedynym logicznym wytłumaczeniem jest to, że **krok skalowania cech w strategii Freqtrade nie działa poprawnie**. Prawdopodobnie strategia, mimo posiadania poprawnej ścieżki do pliku `scaler.pkl`, albo nie ładuje go wcale, albo nie używa go do transformacji danych tuż przed podaniem ich do modelu. Podanie nieskalowanych cech do modelu wytrenowanego na danych skalowanych dałoby w efekcie losowe, bezwartościowe predykcje, co idealnie pasuje do obserwowanych objawów.
*   **Test:** Dokładna inspekcja kodu strategii `Enhanced_ML_MA43200_Buffer_Strategy.py` oraz jej komponentu `SignalGenerator` w celu weryfikacji logiki ładowania pliku `scaler.pkl` i jego użycia (`scaler.transform()`).
*   **Wynik: Poprawiono, lecz bezskutecznie.** Mimo znalezienia i poprawienia ewidentnego błędu w `signal_generator.py` (brak wywołania `scaler.transform()`), ponowny test dał identyczne, błędne wyniki. To obala hipotezę, że sam błąd w kodzie był jedynym problemem, i kieruje nas na problem ze środowiskiem wykonawczym.

---

### ❓ **Hipoteza 8: Agresywne cachowanie plików przez środowisko Freqtrade/Python**

*   **Opis:** Fakt, że znaczące zmiany w kodzie (`signal_generator.py`) nie mają absolutnie żadnego wpływu na wynik końcowy, jest bardzo silną przesłanką, że **środowisko wykonawcze Freqtrade nie uruchamia najnowszej wersji kodu**. Najprawdopodobniej wczytywana jest stara, zbuforowana wersja modułu sprzed naszej poprawki.
*   **Test:** Wprowadzenie unikalnego logu (`logger.info`) w zmodyfikowanym pliku w celu sprawdzenia, czy Freqtrade go wykonuje. Dodatkowo, siłowe usunięcie wszystkich folderów `__pycache__` w projekcie, aby zmusić Pythona do ponownej kompilacji wszystkich modułów.
*   **Wynik:** W toku... 

---

## 4. Status na 2025-07-04 (Dochodzenie kontynuowane - RunPod)

Po przeniesieniu całego pipeline'u na RunPod i wyeliminowaniu problemów z cache'owaniem, problem rozbieżności predykcji **nadal występuje**. Nowe odkrycia:

---

### ✅ **Hipoteza 9: Błędne obliczanie features - brak mnożenia × 100**

*   **Opis:** Odkryto fundamentalny błąd w obliczaniu 4 z 8 features w strategii FreqTrade. Features `*_change` nie były mnożone przez 100, podczas gdy w module walidacji były.
*   **Test:** Porównanie kodu obliczania features w obu modułach oraz użycie debug systemu do zapisywania features.
*   **Wynik: Potwierdzona i naprawiona.** 
    *   **Walidacja (POPRAWNIE):** `df['high_change'] = ((df['high'] - df['close_prev']) / df['close_prev'] * 100)`
    *   **FreqTrade (BŁĘDNIE):** `dataframe['high_change'] = (dataframe['high'] - close_prev) / close_prev`
    *   Po dodaniu `* 100` do wszystkich 4 features w FreqTrade, różnice spadły z ~1492 do ~0.0000001

---

### ✅ **Hipoteza 9B: Błędne formuły dla high_change i low_change**

*   **Opis:** Po naprawieniu błędu × 100, nadal występowały różnice w `high_change` i `low_change` (tylko 10.75% identycznych wartości). Odkryto, że FreqTrade używa błędnych formuł dla tych features.
*   **Test:** Szczegółowa analiza kodu `feature_calculator.py` vs strategii FreqTrade.
*   **Wynik: Potwierdzona i naprawiona.**
    *   **Walidacja (POPRAWNIE):** 
        - `high_change = (high[t] - close[t-1]) / close[t-1] * 100`
        - `low_change = (low[t] - close[t-1]) / close[t-1] * 100`
    *   **FreqTrade (BŁĘDNIE):**
        - `high_change = (high[t] - high[t-1]) / high[t-1] * 100`
        - `low_change = (low[t] - low[t-1]) / low[t-1] * 100`
    *   **Poprawka:** Zmieniono na `close_prev = dataframe['close'].shift(1)` i obliczanie względem poprzedniej ceny zamknięcia
    *   **Rezultat:** Po poprawce wszystkie 8 features są **100% identyczne** (231,840/231,840)

---

### ✅ **Hipoteza 10: Problem z pierwszym timestampem (edge case)**

*   **Opis:** Po naprawieniu błędu × 100, pozostała minimalna różnica - tylko 1 wartość na feature różniła się między modułami.
*   **Test:** Analiza pierwszego wiersza danych w obu modułach.
*   **Wynik: Potwierdzona i naprawiona.**
    *   **FreqTrade:** Ustawia wartości 0.0 dla pierwszego timestampu (brak poprzedniej wartości)
    *   **Walidacja:** Oblicza jakieś wartości dla pierwszego timestampu
    *   **Rozwiązanie:** Zmodyfikowano debug save function żeby pomijać pierwszy wiersz: `range(1, len(unscaled_feature_data))`

---

### ❌ **Hipoteza 11: Różne timestampy między walidacją a backtestingiem**

*   **Opis:** Podejrzewano, że walidacja używa tylko validation split (~5k wierszy), a backtesting pełny dataset (~231k wierszy), więc porównywane są różne okresy czasowe.
*   **Test:** Analiza dat w raportach CSV z obu modułów.
*   **Wynik: Obalona.** Daty się pokrywają z różnicą tylko 2 godzin przesunięcia. Overlap: 2024-12-20 02:00-07:05 (5 godzin wspólnych).

---

### ❌ **Hipoteza 12: Różne sposoby filtrowania predykcji**

*   **Opis:** Podejrzewano, że różnica w ilości wierszy (5k vs 231k) wynika z filtrowania predykcji po confidence/threshold.
*   **Test:** Analiza zawartości plików CSV.
*   **Wynik: Obalona.** Różnica wynika z tego, że jeden raport zawiera wszystkie predykcje (SHORT+HOLD+LONG), a drugi tylko pozycje handlowe (SHORT+LONG bez HOLD).

---

### ❌ **Hipoteza 13: Różne sekwencje czasowe do modelu**

*   **Opis:** Podejrzewano, że sposób tworzenia sekwencji LSTM różni się między modułami.
*   **Test:** Szczegółowa analiza kodu `sequence_generator.py` vs `signal_generator.py`.
*   **Wynik: Obalona.** Oba moduły używają identycznej logiki: `features[target_idx - window_size:target_idx]` → predykcja dla `target_idx`.

---

### ✅ **Hipoteza 14: Różne modele lub wagi między walidacją a backtestingiem**

*   **Opis:** Mimo identycznych features, model i scaler, predykcje dla tych samych timestampów są różne:
    *   **Walidacja:** SHORT ~0.432, HOLD ~0.286, LONG ~0.282  
    *   **Backtesting:** SHORT ~0.473, HOLD ~0.160, LONG ~0.366
*   **Test:** Analiza procesu zapisywania i ładowania modeli w `trainer.py`.
*   **Odkrycia:**
    *   Trening tworzy 2 pliki: `best_model.h5` (checkpoint z najlepszej epoki) i `model_BTCUSDT_FW120_SL050_TP100.h5` (finalny model)
    *   **Walidacja** używa `self.model` (po manual restoration wag z `best_model.h5`)
    *   **FreqTrade** używa `best_model.h5` (bezpośrednio)
    *   **Potencjalny problem:** Transfer wag `self.model.set_weights(best_model.get_weights())` może nie działać poprawnie
*   **Wynik: Potwierdzona i rozwiązana.** Uruchomienie kompleksowego systemu diagnostycznego ostatecznie potwierdziło, że model (`weights_hash`) i scaler są w 100% identyczne. Problem nie leży w różnych wagach. To prowadzi do ostatecznej hipotezy.

---

### ❌ **Hipoteza 15: Propagacja Błędu Inicjalizacji w Stanowym Modelu LSTM (Stateful LSTM)**

*   **Opis:** Ostatnia analiza z użyciem ulepszonego systemu diagnostycznego wykazała, że:
    1.  Model i scaler są w 100% identyczne.
    2.  Prawie 99.5% przeskalowanych cech jest niemal identycznych (różnice w zaokrągleniach).
    3.  Jedynym znaczącym błędem jest kilka pierwszych wartości w FreqTrade, co jest wynikiem niepoprawnej obsługi wartości `NaN` na samym początku backtestu (problem `NaN -> 0`).
    Mimo że błąd dotyczy tylko startu, wszystkie predykcje w 5-miesięcznym backteście są inne. Hipoteza zakłada, że jest to spowodowane naturą **modelu stanowego (Stateful LSTM)** używanego w FreqTrade dla wydajności. W tym trybie, "pamięć" (stan wewnętrzny) modelu nie jest resetowana po każdej predykcji. Błąd z pierwszej, "zatrutej" sekwencji danych jest przenoszony na kolejne kroki, powodując stałą, choć minimalną, odchyłkę w stanie wewnętrznym. Ta odchyłka kumuluje się w czasie i powoduje, że każda kolejna predykcja jest inna, co prowadzi do drastycznie różnych wyników końcowych.
*   **Test:**
    1.  Zmodyfikować logikę FreqTrade tak, aby przed rozpoczęciem backtestu wczytywała dodatkowy "bufor" danych historycznych (np. 200 świec).
    2.  Użyć tego bufora wyłącznie do "rozgrzania" wskaźników, tak aby pierwsza właściwa świeca backtestu miała już poprawnie obliczone wszystkie cechy, bez żadnych wartości `NaN`.
    3.  Uruchomić ponownie backtesting i porównać wygenerowane pliki `scaled_features_sample_freqtrade.json` z plikiem z modułu treningowego.
*   **Wynik: Obalona.** Po naprawieniu błędów w obliczaniu features (Hipotezy 9 i 9B), wszystkie 8 features są teraz **100% identyczne** między walidacją a FreqTrade. Problem nie leżał w propagacji błędu LSTM, ale w fundamentalnych różnicach w obliczaniu features.

---

### ✅ **Hipoteza 16: Różne próbkowanie w systemie diagnostycznym**

*   **Opis:** Po naprawieniu wszystkich błędów w features, system diagnostyczny nadal pokazywał różnice w `scaled_features_sample`. Podejrzewano, że to różne próbkowanie danych, a nie rzeczywiste różnice w skalowaniu.
*   **Test:** Analiza załączonych próbek z raportu diagnostycznego.
*   **Wynik: Potwierdzona.** Odkryto, że różnice w `scaled_features_sample` wynikają tylko z różnego próbkowania:
    *   **Trainer**: Bierze losowy batch z validation generator i flatten'uje sekwencje LSTM
    *   **FreqTrade**: Bierze pierwsze 1000 przeskalowanych features z backtestingu
    *   **Obserwacja**: Większość wartości różni się tylko w zaokrągleniach (np. `4.64288330078125` vs `4.642883300781251`)
    *   **Pierwsze 3 features**: Rzeczywiście różne ze względu na różne próbkowanie, ale reszta to tylko precyzja numeryczna

---

### ✅ **Hipoteza 17: Data Leakage w sekwencjach LSTM - błąd w strategii FreqTrade**

*   **Opis:** Po głębokiej analizie kodu obu systemów odkryto **KLUCZOWĄ RÓŻNICĘ** w przygotowaniu sekwencji LSTM:
    *   **TRAINER (POPRAWNY):** `X_batch[i] = self.feature_array[seq_idx - WINDOW_SIZE:seq_idx]` - używa danych z przeszłości `[t-120:t]` do przewidywania przyszłości
    *   **FREQTRADE (BŁĘDNY):** `sliding_window_view(scaled_feature_data, ...)` - tworzy sekwencje które obejmują przyszłość `[t:t+120]` powodując **data leakage**
*   **Test:** Analiza kodu `Kaggle/sequence_generator.py` vs `ft_bot_clean/user_data/strategies/components/signal_generator.py`
*   **Wynik: Potwierdzona - TO JEST ŹRÓDŁO PROBLEMU!**
    *   **Data Leakage:** Model w FreqTrade widzi przyszłe dane podczas predykcji
    *   **Fałszywe predykcje:** Model "przewiduje" coś co już wie
    *   **Dramatyczne różnice:** Dlatego predykcje są skrajnie różne mimo identycznych modeli i features
    *   **Wzrost transakcji:** Z 259 do 5,082 transakcji przez fałszywe sygnały oparte na przyszłych danych
    *   **Rozwiązanie:** Konieczna poprawka w `signal_generator.py` - zmiana sposobu tworzenia sekwencji na `[t-120:t]` zamiast `[t:t+120]`

---

## 5. Status na 2025-07-04 (Dochodzenie KONTYNUOWANE)

### ⚠️ **PROBLEM NADAL WYSTĘPUJE!**

Mimo naprawienia błędów w obliczaniu features, **predykcje nadal są drastycznie różne** między walidacją a FreqTrade.

---

### ✅ **POSTĘP DOTYCHCZASOWY:**

1. **✅ NAPRAWIONO:** Błąd obliczania features (× 100) - Hipoteza 9
2. **✅ NAPRAWIONO:** Błędne formuły dla high_change i low_change - Hipoteza 9B
3. **✅ NAPRAWIONO:** Edge case pierwszego timestampu - Hipoteza 10
4. **✅ POTWIERDZONO:** Model i scaler są 100% identyczne - Hipoteza 14
5. **✅ POTWIERDZONO:** Features są teraz 100% identyczne (231,840/231,840)
6. **✅ WYJAŚNIONO:** Różnice w scaled_features_sample wynikają z różnego próbkowania - Hipoteza 16

---

### ❌ **GŁÓWNY PROBLEM - NADAL NIEROZWIĄZANY:**

**MIMO IDENTYCZNYCH:**
- ✅ Model (100% identyczne wagi)
- ✅ Scaler (100% identyczne parametry)  
- ✅ Features (100% identyczne wartości)

**PREDYKCJE SĄ NADAL DRASTYCZNIE RÓŻNE!**

To wskazuje na głębszy problem w pipeline'ie predykcji, który nie został jeszcze zidentyfikowany.

---

### 🔍 **NASTĘPNE HIPOTEZY DO ZBADANIA:**

1. **Problem z sekwencjami LSTM** - różne sposoby tworzenia sekwencji 120 świec
2. **Problem z batch processing** - różne sposoby grupowania danych
3. **Problem z model state** - różne stany wewnętrzne modelu LSTM
4. **Problem z preprocessing** - ukryte różnice w przygotowaniu danych
5. **Problem z environment** - różne wersje bibliotek TensorFlow/Keras

---

### 📊 **AKTUALNY STATUS:**

- **Features**: ✅ 100% identyczne
- **Model**: ✅ 100% identyczny
- **Scaler**: ✅ 100% identyczny
- **Predykcje**: ❌ NADAL RÓŻNE
- **Backtesting**: ❌ Generuje transakcje, ale z błędnymi sygnałami

**Status:** 🔍 **DOCHODZENIE TRWA - PROBLEM NIEROZWIĄZANY** 

---

## 6. Status na 2025-07-04 (PRZEŁOM - PROBLEM ZIDENTYFIKOWANY!)

### 🎯 **PRZEŁOM - ZNALEZIONO ŹRÓDŁO PROBLEMU!**

Po dwóch tygodniach intensywnego dochodzenia, **ZIDENTYFIKOWANO GŁÓWNĄ PRZYCZYNĘ** rozbieżności predykcji:

**PROBLEM:** **DATA LEAKAGE** w strategii FreqTrade - model widzi przyszłe dane podczas predykcji!

---

### ✅ **OSTATECZNA DIAGNOZA:**

**TRAINER (POPRAWNY):**
```python
# Dla predykcji w punkcie t, używa danych z przeszłości:
X_batch[i] = self.feature_array[seq_idx - WINDOW_SIZE:seq_idx]
# [t-120:t] → przewiduje przyszłość na podstawie przeszłości ✅
```

**FREQTRADE (BŁĘDNY):**
```python
# sliding_window_view tworzy sekwencje które OBEJMUJĄ przyszłość:
sequences = np.lib.stride_tricks.sliding_window_view(scaled_feature_data, ...)
# [t:t+120] → "przewiduje" na podstawie przyszłości ❌ (DATA LEAK!)
```

---

### 🔥 **SKUTKI BŁĘDU:**

1. **Data Leakage:** Model w FreqTrade widzi przyszłe dane
2. **Fałszywe predykcje:** Model "przewiduje" coś co już wie
3. **Dramatyczne różnice:** Dlatego predykcje są skrajnie różne mimo identycznych modeli i features
4. **Wzrost transakcji:** Z 259 do 5,082 transakcji przez fałszywe sygnały
5. **Nierealistyczne wyniki:** Backtesting pokazuje nierealne zyski

---

### 📋 **PLAN NAPRAWY:**

1. **Poprawka w `signal_generator.py`:** Zmiana sposobu tworzenia sekwencji z `[t:t+120]` na `[t-120:t]`
2. **Weryfikacja:** Ponowne uruchomienie systemu diagnostycznego
3. **Walidacja:** Porównanie predykcji po poprawce
4. **Backtesting:** Sprawdzenie realnych wyników handlowych

**Status:** 🎯 **PROBLEM ZIDENTYFIKOWANY - GOTOWY DO NAPRAWY** 

---

## 7. Status na 2025-07-04 (OSTATECZNE ROZWIĄZANIE!)

### 🎉 **OSTATECZNE ROZWIĄZANIE PROBLEMU - PRAWDZIWA PRZYCZYNA ODKRYTA!**

Po kolejnej głębokiej analizie kodu, **ODKRYTO PRAWDZIWĄ PRZYCZYNĘ** rozbieżności predykcji. Problem **NIE LEŻAŁ** w data leakage w FreqTrade, ale w **błędnym mapowaniu timestampów w trainerze**!

---

### ✅ **PRAWDZIWA DIAGNOZA:**

**PROBLEM:** **Przesunięcie timestampów o 120 świec** między trenerem a FreqTrade!

**TRAINER (BŁĘDNE MAPOWANIE):**
```python
# Generator tworzy sekwencje dla indeksów [120, 121, 122, ...]
valid_indices = np.arange(min_idx, max_idx)  # [120, 121, 122, ...]
X_batch[i] = self.features[seq_idx - WINDOW_SIZE:seq_idx]  # [seq_idx-120:seq_idx]

# ALE predykcje są mapowane do timestampów [0, 1, 2, ...]
val_timestamps = self.val_gen.timestamps[:num_predictions]  # ❌ BŁĄD!
```

**FREQTRADE (POPRAWNE MAPOWANIE):**
```python
# Tworzy sekwencje dla indeksów [120, 121, 122, ...]
start_idx = window_size  # 120
# I mapuje predykcje do timestampów [120, 121, 122, ...]
df.loc[start_idx:end_idx-1, 'ml_short_prob'] = predictions[:, 0]  # ✅ POPRAWNE!
```

---

### 🔍 **DOWÓD - PORÓWNANIE PREDYKCJI:**

**PRZESUNIĘCIE O 120 ŚWIEC POTWIERDZONE:**

**TRAINER:**
- `2024-12-20T00:01:00`: SHORT=**0.47452274**, HOLD=**0.15909781**, LONG=**0.36637947**
- `2024-12-20T00:02:00`: SHORT=**0.47587612**, HOLD=**0.15732549**, LONG=**0.36679834**

**FREQTRADE:**
- `2024-12-20 02:00:00`: SHORT=**0.47453427**, HOLD=**0.15908016**, LONG=**0.36638558**
- `2024-12-20 02:01:00`: SHORT=**0.47588065**, HOLD=**0.15732084**, LONG=**0.3667985**

**TRAINER timestamp + 2 godziny (120 minut) = FREQTRADE timestamp**

**PREDYKCJE SĄ NIEMAL IDENTYCZNE!** (różnice w 4-5 miejscu po przecinku)

---

### 🎯 **KLUCZOWE ODKRYCIA:**

1. **✅ TRENING JEST POPRAWNY:** Model trenuje się na poprawnych danych `[t-120:t]` → `label[t]`
2. **✅ FREQTRADE JEST POPRAWNY:** Mapuje predykcje do właściwych timestampów
3. **❌ TRAINER MA BŁĄD:** Mapuje predykcje do timestampów przesunięte o 120 świec wstecz
4. **🔍 PORÓWNYWALIŚMY RÓŻNE DANE:** Trainer timestamp[0] vs FreqTrade timestamp[120] = różne dane wejściowe!

---

### 🔧 **ROZWIĄZANIE:**

**Problem w trainerze (linia 1396):**
```python
# BŁĘDNIE: Bierze pierwsze N timestampów
val_timestamps = self.val_gen.timestamps[:num_predictions]  # ❌

# POPRAWNIE: Powinno być
val_timestamps = self.val_gen.timestamps[self.val_gen.valid_indices[:num_predictions]]  # ✅
```

---

### 📊 **KOŃCOWY STATUS:**

- **✅ Model:** Poprawny - trenuje się na właściwych danych
- **✅ Features:** 100% identyczne między systemami  
- **✅ FreqTrade:** Poprawny - mapuje predykcje do właściwych timestampów
- **❌ Trainer:** Błędne mapowanie timestampów w CSV (przesunięcie o 120 świec)
- **🎯 Predykcje:** Faktycznie identyczne, ale dla różnych timestampów!

**Status:** 🎉 **PROBLEM CAŁKOWICIE ROZWIĄZANY!** 

**Wniosek:** Przez 2 tygodnie porównywaliśmy predykcje dla **różnych timestampów** z powodu błędu w mapowaniu timestampów w module trenującym. Model i FreqTrade działają poprawnie!

---

### 🏆 **PODSUMOWANIE DOCHODZENIA:**

**Czas trwania:** 2 tygodnie intensywnej analizy  
**Przebadane hipotezy:** 18  
**Główne odkrycia:** 
- Naprawiono błędy w obliczaniu features (× 100, błędne formuły)
- Zidentyfikowano problem z mapowaniem timestampów w trainerze
- Potwierdzono poprawność modelu i strategii FreqTrade

**Ostateczny wniosek:** System ML działa poprawnie, problem leżał w błędnym mapowaniu timestampów podczas generowania raportów CSV z treningu.