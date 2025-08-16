### Cel dokumentu
Zebrać ustalenia i hipotezy wyjaśniające, dlaczego wyniki treningu `training3` są istotnie lepsze niż `training5` (na nowym pipeline). Dokument ma służyć jako mapa śledcza: jakie dane mamy, co już zweryfikowaliśmy, co jeszcze sprawdzić, oraz jak z tego wyprowadzić decyzje o najlepszym zestawie cech do treningu.

### Moduły i pipeline (porównanie)
- **training3**:
  - Wejście: `labeler3/output/ohlc_orderbook_labeled_3class_fw120m_15levels.feather`.
  - Cechy: 37 kuratorowanych (rdzeń historycznie sprawdzony).
  - XGBoost Multi‑Output; early stopping; wagi klas włączone.

- **training5** (obecny pipeline):
  - Wejście: `labeler5/output/labeled_{symbol}.feather` (cechy z `feature_calculator_4`).
  - Cechy: ~67 nowych (kanały 240/180/120, OB ±1/±2, interakcje kanał×OB, OHLC/TA, MFI/OBV, itp.) + OHLCV do labelingu.
  - XGBoost Multi‑Output; early stopping; wagi klas włączone.

### Obserwacje wyników
- `training3`: spadki validation-mlogloss do ~0.43–0.70 (zależnie od poziomu). Przykłady z logów:
  - TP 1.4 / SL 0.4: best ~0.4301
  - TP 1.2 / SL 0.5: best ~0.5556
- `training5`: wyższe validation-mlogloss, zatrzymania ~0.64–1.05 (zależnie od poziomu) przy podobnych parametrach early stopping.

### Porównanie etykiet (labeler3 vs labeler5)
- Porównanie wykonane skryptem: `analysis/compare_labelers.py` (normalizacja timestamp do minut, próby offsetu ±1m).
- Zakresy:
  - L3: 2023-01-31 00:06 → 2025-06-30 23:59 (1,270,074 wierszy)
  - L5: 2023-01-01 04:00 → 2025-08-08 21:59 (1,369,080 wierszy)
  - Część wspólna: 1,270,074 minut
- Wyniki zgodności (wycinek z `analysis/labeler3_vs_labeler5.txt`):
  - Zgodność 99.95–99.99% dla wszystkich porównanych poziomów (12 wspólnych kolumn `label_tp*_sl*`).
  - Niezgodności: rząd setek przypadków na >1.27 mln próbek (najczęściej LONG/SHORT → NEUTRAL lub odwrotnie). 
- Wniosek: różnice w labelingach istnieją, ale są marginalne i najprawdopodobniej nie tłumaczą dużych rozjazdów metryk.

### Różnice w cechach (kluczowa hipoteza)
- `training3` (37 cech): zestaw “destylowany”, sprawdzony w czasie.
- `training5` (~67 cech): szerszy zestaw, zawiera:
  - Kanały: `pos_in_channel_{240,180,120}`, `width_over_ATR_{…}`, `slope_over_ATR_window_{…}`, `channel_fit_score_{…}`.
  - Interakcje: `*_x_imbalance_1pct` (kanał × OB).
  - Orderbook ±1%/±2% (notional): `imbalance_1pct_notional`, `log_ratio_ask_2_over_1`, `log_ratio_bid_2_over_1`, `ask_near_ratio`, `bid_near_ratio`, `concentration_near_mkt`, `ask_com`, `bid_com`, `com_diff`, `pressure_12`, `pressure_12_norm`, `side_skew`, `dA1`, `dB1`, `dImb1`, `ema_dImb1_{5,10}`, `persistence_imbalance_1pct_ema{5,10}`.
  - OHLC/TA: `body_ratio`, `wick_up_ratio`, `wick_down_ratio`, `r_1`, `r_5`, `r_15`, `slope_return_120`, `vol_regime_120`, `vol_of_vol_120`, `r2_trend_120`, `RSI_14`, `RSI_30`, `StochK_14_3`, `StochD_14_3`, `MACD_hist_over_ATR`, `ADX_14`, `di_diff_14`, `CCI_20_over_ATR`, `bb_pos_20`, `bb_width_over_ATR_20`, `donch_pos_60`, `donch_width_over_ATR_60`, `close_vs_ema_60`, `close_vs_ema_120`, `slope_ema_60_over_ATR`, `MFI_14`, `OBV_slope_over_ATR`.
- Dodatkowa uwaga: A1/A2/B1/B2 w `feature_calculator_4` mogą być aproksymowane (sumy `notional_1_*` i `notional_2_*`), co pogarsza wierność koszyków ±1/±2% vs dane źródłowe. To potencjalne źródło szumu.

### Różnice danych i zakresu
- `training3` bazuje na jednym gotowym pliku `labeler3` (wycięty i spójny okres).
- `training5` używa świeżo policzonych cech `feature_calculator_4` + `labeler5`. Zakres jest szerszy i obejmuje dodatkowe okresy (inne reżimy rynku).
- Wpływ: inny reżim zmienności i proporcji klas może utrudniać separację w training5.

### Parametry i procedura treningu
- Oba: XGBoost (native), early stopping (20 rund), learning_rate 0.05, max_depth 6, subsample/colsample/gamma podobne, wagi klas włączone. 
- To wskazuje, że główny efekt różnic pochodzi z danych/cech, a nie z samej konfiguracji modelu.

### Hipotezy (priorytet)
1. **Zestaw cech**: szerszy, mniej kuratorowany zestaw w training5 (w tym aproksymacje OB i interakcje) dodaje korelowany szum → gorsza generalizacja i wyższy mlogloss.
2. **Zakres czasowy i reżimy**: training5 obejmuje trudniejsze odcinki rynku → wyższa trudność zadania.
3. **Labeling**: różnice są minimalne; wpływ raczej drugorzędny (potwierdzony raportem zgodności).

### Plan weryfikacji (bez zmian logiki modelu)
- A. Kontrolny eksperyment CECH:
  - Uruchomić `training5` z etykietami `labeler5`, ale tylko z 37 cechami rdzenia `training3` (ta sama lista). 
  - Oczekiwane: jeśli mlogloss istotnie spadnie, problemem jest dobór cech/ich jakość (a nie labeling).

- B. Kontrolny eksperyment LABELI:
  - Uruchomić `training5` z cechami `feature_calculator_4`, ale z etykietami `labeler3` (te same zakresy czasu). 
  - Oczekiwane: jeśli mlogloss pozostaje znacząco wyższy niż w `training3`, problemem są cechy; jeśli zbliża się do `training3`, etykiety/zakres miały wyraźny wpływ.

- C. Redukcja szumu w cechach OB:
  - Wyłączyć (czasowo) cechy oparte na aproksymowanych A1/A2/B1/B2 i interakcje kanał×OB (lub zastąpić wersjami normalizowanymi/rdzeniem ±1% tylko), i sprawdzić wpływ na mlogloss.

- D. Pruning wg ważności:
  - Uruchomić `training5` i zebrać gain/importance; zachować top‑K (np. 20–30), retrain → sprawdzić, czy mlogloss spada.

- E. Wyrównanie zakresu czasu:
  - Przeciąć `training5` do range wspólnego z `labeler3` i porównać mlogloss poziom‑poziom.

### Dane pomocnicze/artefakty
- Raport zgodności etykiet: `analysis/labeler3_vs_labeler5.txt` (zawiera % zgodności, rozkłady klas, macierze 3×3 dla 12 poziomów).
- Lista cech użytych przez `training5`: pliki `training5/output/reports/features_used_{symbol}_{ts}.txt` (logowane przed treningiem).
- Weryfikacja NaN (wejścia treningów):
  - Sprawdzone skryptem `analysis/check_nans.py` dla:
    - `labeler3/output/ohlc_orderbook_labeled_3class_fw120m_15levels.feather`
    - `labeler5/output/labeled_BTCUSDT.feather`
  - Wynik: 0 wierszy z NaN, 0 kolumn z NaN – NaN nie są przyczyną różnic.

### Kryteria decyzji “które cechy są najlepsze”
- Metryki modelu: mlogloss/accuracy na walidacji/test.
- Stabilność: powtarzalność ważności cech i wyników w różnych podzakresach czasu.
- Prostota: mniejszy, silniejszy rdzeń cech > szeroki zestaw o niskiej średniej ważności.
- Zgodność z handlem: cechy kanałowe + rdzeń OB ±1% powinny być interpretowalne i odporne na reżimy.

### Następne kroki (operacyjne)
- Przygotować wariant `training5` z listą 37 cech `training3` (A) i uruchomić.
- Przygotować wariant `training5` z etykietami `labeler3` (B) na tym samym zakresie.
- Zebrać ważności i wykonać pruning (D), a następnie porównać.
- Zredukować/wyczyścić cechy OB aproksymowane (C) i sprawdzić wpływ.


### Wyniki porównania (logloss) – aktualizacja
- training5 (uruchomiony na danych `labeler3`, whitelist 37 cech jak w `training3`): najlepsze validation-mlogloss per poziom TP/SL
  -1-   tp0.6/sl0.2: 1.03436 (iter 93)
  -2-   tp0.6/sl0.3: 1.04365 (iter 100)
  -3-   tp0.8/sl0.2: 0.89532 (iter 111)
  -4-   tp0.8/sl0.3: 0.94321 (iter 100)
  -5-   tp0.8/sl0.4: 0.95687 (iter 100)
  -6-   tp1.0/sl0.3: 0.80083 (iter 100)
  -7-   tp1.0/sl0.4: 0.82760 (iter 100)
  -8-   tp1.0/sl0.5: 0.84077 (iter 100–128)
  -9-   tp1.2/sl0.4: 0.69499 (iter 100)
  -10-  tp1.2/sl0.5: 0.71286 (iter 100)
  -11-  tp1.2/sl0.6: 0.72217 (iter 100–122)
  -12-  tp1.4/sl0.4: 0.58370 (iter 100)
  -13-  tp1.4/sl0.5: 0.60427 (iter 100)
  -14-  tp1.4/sl0.6: 0.61605 (iter 139)
  -15-  tp1.4/sl0.7: 0.62323 (iter 130)

- training3: w toku ponownego uruchomienia z ograniczonym logowaniem (co 50 rund). Końcowe logloss per poziom zapisują się do:
  - `training3/output/reports/logloss_summary_latest.csv`
  - po zakończeniu zostaną dopisane do tego dokumentu.


### training3 – końcowe validation-mlogloss (z logów)
-1-   tp0.6/sl0.2: 0.94777 (iter 129)
-2-   tp0.6/sl0.3: 1.00558 (iter 83)
-3-   tp0.8/sl0.2: 0.74047 (iter 158)
-4-   tp0.8/sl0.3: 0.81623 (iter 149)
-5-   tp0.8/sl0.4: 0.84885 (iter 136)
-6-   tp1.0/sl0.3: 0.64180 (iter 147)
-7-   tp1.0/sl0.4: 0.67734 (iter 142)
-8-   tp1.0/sl0.5: 0.69683 (iter 137)
-9-   tp1.2/sl0.4: 0.53595 (iter 134)
-10-  tp1.2/sl0.5: 0.55560 (iter 149)
-11-  tp1.2/sl0.6: 0.56688 (iter 141)
-12-  tp1.4/sl0.4: 0.43014 (iter 137)
-13-  tp1.4/sl0.5: 0.44919 (iter 141)
-14-  tp1.4/sl0.6: 0.46096 (iter 137)
-15-  tp1.4/sl0.7: 0.46819 (iter 135)

Porównując do training5 (na tych samych danych i 37 cechach), training3 pozostaje istotnie lepszy (np. tp1.4/sl0.4: 0.430 vs 0.584).

### Analiza różnic (bez zmian kodu – hipotezy do weryfikacji)
- Przetwarzanie braków danych [ZWERYFIKOWANE – ODRZUCONE]:
  - Wejścia obu treningów nie zawierają NaN (sprawdzone `analysis/check_nans.py`).
  - Wniosek: ścieżka imputacji w `training5` prawdopodobnie nie była aktywna (lub bez znaczenia); NaN nie tłumaczą różnic.
- Wagi na walidacji:
  - training5 przekazuje sample_weight także do zbioru walidacyjnego (DMatrix dla val), co wpływa na metrykę early stopping; training3 stosuje wagi na train, ale walidację liczy bez wag. Różnica może zmieniać trajektorie i iterację zatrzymania.
  - Prościej: przy walidacji bez wag każdy przykład liczy się „po równo”. Przy walidacji z wagami błędy np. LONG/SHORT (wyższa waga) liczą się bardziej niż NEUTRAL. Monitorowana strata (mlogloss) jest więc liczona „inaczej” w obu podejściach, więc model może zatrzymać się wcześniej/później i wybrać inną najlepszą iterację.
- Różnice drobne w parametrach/ustawieniach domyślnych:
  - training5 wymusza `tree_method='hist'`; training3 pozostawia `tree_method` domyślne (może też być 'hist', ale zależnie od środowiska). Subtelne efekty implementacyjne mogą wpływać na mlogloss.
- Kolejność i zakres danych:
  - Oba używają chronologicznego podziału 70/15/15 na tym samym pliku. Warto jawnie zapisać zakresy czasu train/val/test w training5 i porównać do training3 (training3 je loguje). Ewentualne różnice kilku minut (np. po czyszczeniu NaN) mogą mieć wpływ.
- Skaler i kolejność operacji:
  - Oba używają RobustScaler, ale training5 skaluje po imputacji; training3 skaluje po odrzuceniu wierszy. To spójne z pierwszym punktem i może być głównym czynnikiem.

### Zalecane testy weryfikacyjne (bez refaktoru modeli)
1) training5 “bez imputacji”: (opcjonalnie) wymuś tryb drop‑NaN przed skalowaniem – spodziewany brak wpływu (bo wejścia nie mają NaN) – test sanity.
2) training5 “bez wag na walidacji”: twórz `dval` bez parametru weight; porównaj mlogloss i iterację early stopping.
3) training5 “zakresy czasu”: zaloguj zakresy Train/Val/Test (timestamp min/max) i porównaj z training3; upewnij się, że różnice po czyszczeniu NaN nie przesuwają okna.
4) training5 “tree_method”: usuń wymuszenie `'hist'` i zostaw auto (na próbę); sprawdź wpływ.
5) Replikacja skaler/logiki: zastosuj dokładnie tę samą kolejność co training3 (drop NaN → split → scale) i zweryfikuj, czy mlogloss zbiega.

Powyższe testy mają na celu precyzyjnie wskazać dominujący czynnik rozbieżności (najpewniej czyszczenie vs imputacja i/lub wagi na walidacji).


### Wyniki porównania – po korekcie walidacji (training5)
- training5 (po korekcie: wagi TYLKO na train, walidacja bez wag; dane `labeler3`, whitelist 37 cech) – najlepsze validation-mlogloss per poziom TP/SL:
  -1-   tp0.6/sl0.2: 0.94789 (iter 143)
  -2-   tp0.6/sl0.3: 1.00561 (iter 100)
  -3-   tp0.8/sl0.2: 0.74062 (iter 150)
  -4-   tp0.8/sl0.3: 0.81634 (iter 150)
  -5-   tp0.8/sl0.4: 0.84890 (iter 150)
  -6-   tp1.0/sl0.3: 0.64182 (iter 150)
  -7-   tp1.0/sl0.4: 0.67739 (iter 150)
  -8-   tp1.0/sl0.5: 0.69712 (iter 157)
  -9-   tp1.2/sl0.4: 0.53596 (iter 150)
  -10-  tp1.2/sl0.5: 0.55573 (iter 150)
  -11-  tp1.2/sl0.6: 0.56697 (iter 160)
  -12-  tp1.4/sl0.4: 0.43051 (iter 150)
  -13-  tp1.4/sl0.5: 0.44964 (iter 150)
  -14-  tp1.4/sl0.6: 0.46106 (iter 150)
  -15-  tp1.4/sl0.7: 0.46844 (iter 150)

Wartości te praktycznie pokrywają się z wynikami `training3` dla tych samych danych i listy 37 cech.

### Nowe ustalenia (kluczowe)
- Główna przyczyna rozbieżności: walidacja z wagami w `training5` (w przeszłości) vs walidacja bez wag w `training3`.
- Ujednolicenie podejścia (wagi tylko na zbiorze treningowym; walidacja bez wag) sprawiło, że `training5` osiąga mlogloss zgodny z `training3` przy tych samych danych i cechach.
- Raporty przy progach pewności (np. 0.5) wskazują poprawę jakości sygnałów LONG/SHORT po tej korekcie, co zwiększa szanse na dodatni wynik przy backtestach.

### training5 – wyniki na `labeler5` (pełny zestaw cech, ~111) – 2025-08-15 16:01
- Dane: `labeler5/output/labeled_BTCUSDT.feather`
- Cecha: pełny zestaw (wykryto 111 kolumn X, whitelist wyłączony)
- Najlepsze validation-mlogloss per poziom (iter → najniższa wartość z logu):
  -1-   tp0.6/sl0.2: 0.97527 (iter 100)
  -2-   tp0.6/sl0.3: 1.03339 (iter 50)
  -3-   tp0.8/sl0.2: 0.78784 (iter 100)
  -4-   tp0.8/sl0.3: 0.86177 (iter 142)
  -5-   tp0.8/sl0.4: 0.90696 (iter 76)
  -6-   tp1.0/sl0.3: 0.69979 (iter 100)
  -7-   tp1.0/sl0.4: 0.75089 (iter 86)
  -8-   tp1.0/sl0.5: 0.76032 (iter 110)
  -9-   tp1.2/sl0.4: 0.62223 (iter 100)
  -10-  tp1.2/sl0.5: 0.62984 (iter 165)
  -11-  tp1.2/sl0.6: 0.63430 (iter 100)
  -12-  tp1.4/sl0.4: 0.52587 (iter 100)
  -13-  tp1.4/sl0.5: 0.54234 (iter 105)
  -14-  tp1.4/sl0.6: 0.53614 (iter 97)
  -15-  tp1.4/sl0.7: 0.54021 (iter 103)

Komentarz: Wyniki wyraźnie lepsze niż pierwsze uruchomienia `training5` (sprzed korekty walidacji), ale nadal słabsze niż `training3` na 37‑cechowym rdzeniu (np. dla tp1.4/sl0.4: 0.5259 vs 0.4301).

### training5 – wyniki na `labeler5` z 37 cechami `training3` (tryb t3_37) – 2025-08-15 18:43
- Dane: `labeler5/output/labeled_BTCUSDT.feather`
- Cecha: whitelist 37 (częściowo dostępne 25; brakujące 12: `spread_tightness`, `depth_ratio_s1`, `depth_ratio_s2`, `depth_momentum`, `volume_imbalance`, `weighted_volume_imbalance`, `volume_imbalance_trend`, `price_pressure`, `weighted_price_pressure`, `price_pressure_momentum`, …)
- Najlepsze validation-mlogloss per poziom (z logów):
  -1-   tp0.6/sl0.2: 0.96386
  -2-   tp0.6/sl0.3: 1.01949
  -3-   tp0.8/sl0.2: 0.76255–0.76265
  -4-   tp0.8/sl0.3: 0.84096–0.84099
  -5-   tp0.8/sl0.4: 0.87337
  -6-   tp1.0/sl0.3: 0.67474
  -7-   tp1.0/sl0.4: 0.71279–0.71284
  -8-   tp1.0/sl0.5: 0.73320–0.73325
  -9-   tp1.2/sl0.4: 0.57882
  -10-  tp1.2/sl0.5: 0.60073
  -11-  tp1.2/sl0.6: 0.61344–0.61346
  -12-  tp1.4/sl0.4: 0.47332
  -13-  tp1.4/sl0.5: 0.49505
  -14-  tp1.4/sl0.6: 0.50822
  -15-  tp1.4/sl0.7: 0.51691

Wniosek: nawet przy próbie ograniczenia do 37 cech, `training5` użył tylko 25 cech – 12 cech z rdzenia `training3` nie było dostępnych w danych (`labeler5`), co samo w sobie tłumaczy rozjazd względem `training3`.

### Analiza przyczyny bieżącej rozbieżności
- Braki cech rdzenia `training3` w pliku wejściowym `labeler5`:
  - brak m.in. cech OB i presji: `spread_tightness`, `depth_ratio_s1`, `depth_ratio_s2`, `depth_momentum`, `volume_imbalance`, `weighted_volume_imbalance`, `volume_imbalance_trend`, `price_pressure`, `weighted_price_pressure`, `price_pressure_momentum`.
  - te cechy w `training3` pochodzą z modułu `feature_calculator_ohlc_snapshot` (snapshotowe wolumeny/spread lub ich odpowiedniki).
- Nasz nowy `feature_calculator_4` dodaje rdzeń 37 cech, ale dla części OB używa fallbacków opartych o notional A1/B1 i/lub wymaga kolumn jak `snapshot1_bid_volume`/`snapshot1_ask_volume`/`spread`, których może brakować w zmergowanym pliku `download3` (stąd nie zostały wygenerowane → nie trafiają do `labeler5`).
- Efekt: przy trybie `t3_37` model realnie widzi tylko 25/37 cech.

Rekomendacja techniczna:
1) Upewnić się, że `feature_calculator_4` rzeczywiście generuje brakujące 12 cech w oparciu o dostępne kolumny z `download3` (np. rozpoznać nazwy kolumn spreadu i bliskiego rynku; zmapować odpowiedniki bid/ask volume albo konsekwentnie użyć A1/B1 jako substytutów) i potwierdzić ich obecność w `feature_calculator_4/output/features_BTCUSDT.feather`.
2) Jeśli nie da się odtworzyć dokładnej definicji (brak surowych wolumenów bid/ask), należy:
   - zdefiniować zgodne substytuty (udokumentowane) i używać ich spójnie,
   - albo tymczasowo wyłączyć brakujące cechy z whitelisty (i komunikować „25‑cechowy rdzeń”) do czasu zapewnienia źródłowych kolumn.
3) Alternatywnie, na potrzeby ścisłego porównania, uruchomić `training5` na dokładnie tym samym pliku danych co `training3` (z `labeler3`) – co wcześniej dało zbieżne wyniki.