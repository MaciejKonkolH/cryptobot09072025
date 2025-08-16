## Lista cech do treningu (kanały 240/180/120 min) — normalizacja do ATR

Poniżej spisujemy uzgodnione cechy bazujące na równoległym kanale cenowym, liczone dla trzech okien: 240, 180, 120 minut.
Tam, gdzie ma to sens, wartości są normalizowane do ATR z tego samego okna, aby były porównywalne między rynkami i poziomami cen.

Konwencje i definicje wspólne:
- `window` ∈ {240, 180, 120} minut.
- Kanał: dopasowanie linii środkowej (LS) do `close` oraz wyznaczenie `support`/`resistance` przez minimalne/maksymalne rezydua.
- `ATR(window)`: średni True Range z okna `window` (może być SMA TR lub klasyczny ATR Wildera – ważne, by spójnie użyć jednej definicji w całym projekcie).
- Normalizacja: dzielenie przez `ATR(window)`. Dla uniknięcia dzielenia przez 0: `denominator = max(ATR(window), 1e-12)`.
- `pos_in_channel` jest z natury znormalizowane do [0, 1], więc nie wymaga ATR.
- `fit_score` jest bezwymiarowy, w [0, 1].

Nazewnictwo: `feature_name_{window}` np. `pos_in_channel_240`.

1. pos_in_channel_{window}
   - Opis: względna pozycja ostatniego zamknięcia w kanale, 0 = wsparcie, 1 = opór.
   - Wzór: `(close_last - support_last) / (resistance_last - support_last)`, z clip do [0,1].
   - Normalizacja: nie dotyczy (już bezwymiarowe).
   - Zastosowanie: mean-reversion vs breakout, pozycja względem granic.

2. width_over_ATR_{window}
   - Opis: szerokość kanału w jednostkach ATR (im większa, tym „szerszy” kanał względem typowego ruchu).
   - Wzór: `(resistance_last - support_last) / ATR(window)`.
   - Normalizacja: przez ATR(window).
   - Zastosowanie: porównywalna miara „oddychania”/zmienności kanału.

3. slope_over_ATR_window_{window}
   - Opis: nachylenie linii środkowej w oknie, wyrażone w jednostkach ATR (z zachowaniem znaku trendu).
   - Wzór: `(center_last - center_first) / ATR(window)`.
   - Uwaga: zachowujemy znak (dodatni = trend rosnący, ujemny = malejący). Alternatywnie można użyć wartości bezwzględnej i dodać osobną cechę znaku.
   - Normalizacja: przez ATR(window).
   - Zastosowanie: siła trendu niezależna od poziomu ceny.

4. channel_fit_score_{window}
   - Opis: prosty wskaźnik dopasowania kanału – im bliżej 1, tym „czyściej” cena porusza się wewnątrz kanału.
   - Wzór: `1 - IQR(residuals) / channel_width`, przy czym `channel_width = max(residuals) - min(residuals)` i wynik clip do [0,1].
   - Normalizacja: nie dotyczy (bezwymiarowe).
   - Zastosowanie: filtr jakości – odrzuca „sztuczne” kanały o rozrzucie rezyduów.

Wersje okienne (łącznie 12 cech):
- pos_in_channel_240, width_over_ATR_240, slope_over_ATR_window_240, channel_fit_score_240
- pos_in_channel_180, width_over_ATR_180, slope_over_ATR_window_180, channel_fit_score_180
- pos_in_channel_120, width_over_ATR_120, slope_over_ATR_window_120, channel_fit_score_120

Uwagi implementacyjne:
- Dla `ATR(window)` należy użyć tej samej definicji, co w module cech – rekomendowany klasyczny ATR (Wilder) lub SMA TR, ale konsekwentnie w całym pipeline.
- Przy bardzo małym ATR (ryzyko dzielenia przez ~0), stosować zabezpieczenie `max(ATR, 1e-12)` i opcjonalne winsoryzacje/capping.
- `pos_in_channel` zawsze clip do [0,1].
- Warto rozważyć wersje wygładzone (EMA) lub medianowe dla stabilności – możemy dopisać je jako alternatywy w kolejnych iteracjach.



## Cechy z orderbook (±1% i ±2%) — liczone per model (TP,SL)

Założenia wspólne:
- Definicje: A1 = notional po stronie ask w +1%, A2 = +2%; B1 = notional po stronie bid w −1%, B2 = −2%.
- Parametry modelu: `tp`, `sl` (w %, np. tp=1.2 → 1.2%).
- Wygodne wagi: `w1 = min(tp, 1.0)`, `w2 = max(tp - 1.0, 0.0)`; `s1 = min(sl, 1.0)`, `s2 = max(sl - 1.0, 0.0)`.
- Stała zabezpieczająca: `ε = 1e-9`.
- Normalizacja: rekomendujemy warianty dzielone przez ATR_120 lub przez sumę notional w ±{1,2,3,4,5}% na danej stronie (dla porównywalności w różnych reżimach).

5. reach_TP_notional
   - Opis: łączna „ściana” podaży do poziomu TP.
   - Wzór: `reach_TP = w1·A1 + w2·A2`.
   - Normalizacja (opcjonalnie): `reach_TP_over_ATR = reach_TP / ATR_120`.

6. reach_SL_notional
   - Opis: łączna „ściana” popytu do poziomu SL.
   - Wzór: `reach_SL = s1·B1 + s2·B2`.
   - Normalizacja (opcjonalnie): `reach_SL_over_ATR = reach_SL / ATR_120`.

7. resistance_vs_support
   - Opis: względna trudność dojścia do TP względem SL.
   - Wzór: `resistance_vs_support = reach_TP / (reach_SL + ε)`.
   - Normalizacja: bezwymiarowe; można logować `log(resistance_vs_support + ε)` dla stabilności.

8. imbalance_1pct_notional
   - Opis: lokalna nierównowaga płynności blisko rynku (±1%).
   - Wzór: `(B1 - A1) / (B1 + A1 + ε)`.
   - Normalizacja: bezwymiarowe; sugerowany też wariant wygładzony (pkt 12).

9. log_ratio_ask_2_over_1
   - Opis: jak szybko rośnie podaż w głąb po stronie ask.
   - Wzór: `log_ratio_ask = ln((A2 + ε) / (A1 + ε))`.
   - Normalizacja: bezwymiarowe.

10. log_ratio_bid_2_over_1
    - Opis: jak szybko rośnie popyt w głąb po stronie bid.
    - Wzór: `log_ratio_bid = ln((B2 + ε) / (B1 + ε))`.
    - Normalizacja: bezwymiarowe.

11. concentration_near_mkt
    - Opis: koncentracja płynności najbliżej rynku vs bliżej + dalej.
    - Wzór: `(A1 + B1) / (A1 + A2 + B1 + B2 + ε)`.
    - Normalizacja: bezwymiarowe.

12. persistence_imbalance_1pct_ema{L}
    - Opis: wygładzona trwałość przewagi (np. EMA z L=5 i L=10 snapshotów).
    - Wzór: `EMA(imbalance_1pct_notional, L)`.
    - Normalizacja: bezwymiarowe; zalecamy dwie wersje: `ema5`, `ema10`.

Uwagi praktyczne (OB):
- Wystarczy przechowywać A1, A2, B1, B2; cechy 5–7 liczyć parametrycznie per model (znając jego `tp`, `sl`).
- Dla stabilności: można stosować winsoryzację/log(1+x)/clip; zawsze dodawaj `ε` przy dzieleniu i logowaniu.
- Jeśli dostępne są także depth (ilość) poza notional (wartość), można zdublować wybrane cechy w wariantach „depth‑based”.

## Dodatkowe cechy — do rozważenia (OB, OHLC, interakcje)

13. ask_com, bid_com, com_diff
    - Opis: „środek ciężkości” płynności po stronach oraz ich różnica.
    - Wzór: `ask_com = (1·A1 + 2·A2) / (A1 + A2 + ε)`, `bid_com = (1·B1 + 2·B2) / (B1 + B2 + ε)`, `com_diff = bid_com - ask_com`.
    - Normalizacja: bezwymiarowe w jednostkach koszyków.

14. ask_near_ratio, bid_near_ratio
    - Opis: koncentracja najbliżej rynku.
    - Wzór: `ask_near_ratio = A1 / (A1 + A2 + ε)`, `bid_near_ratio = B1 / (B1 + B2 + ε)`.
    - Normalizacja: bezwymiarowe.

15. pressure_12, pressure_12_norm
    - Opis: presja netto popyt−podaż w okolicy ±1..2% i jej wersja znormalizowana.
    - Wzór: `pressure_12 = (B1 + 0.5·B2) − (A1 + 0.5·A2)`; `pressure_12_norm = pressure_12 / (A1 + A2 + B1 + B2 + ε)`.
    - Normalizacja: `pressure_12_norm` zalecane.

16. dA1, dB1, dImb1, ema_dImb1_{L}
    - Opis: impulsy zmian najbliższej płynności i nierównowagi oraz ich wygładzenie.
    - Wzór: `dA1 = A1_t − A1_{t−1}`, `dB1 = B1_t − B1_{t−1}`, `dImb1 = ((B1−A1)/(B1+A1+ε))_t − (...)_{t−1}`, `ema_dImb1_{L} = EMA(dImb1, L)`.
    - Normalizacja: rozważ `log(1+x)`, winsoryzację; opcjonalnie dziel przez `(A1+A2+B1+B2)`.

17. side_skew
    - Opis: skumulowana asymetria po obu stronach.
    - Wzór: `side_skew = (B1 + B2 − A1 − A2) / (A1 + A2 + B1 + B2 + ε)`.
    - Normalizacja: bezwymiarowe.

18. body_ratio, wick_up_ratio, wick_down_ratio (OHLC)
    - Opis: kształt świecy w relacji do zasięgu.
    - Wzór: `body_ratio = |close−open|/(high−low+ε)`, `wick_up_ratio = (high−max(open,close))/(high−low+ε)`, `wick_down_ratio = (min(open,close)−low)/(high−low+ε)`.
    - Normalizacja: bezwymiarowe.

19. returns i trend procentowy (OHLC)
    - Opis: krótkie stopy zwrotu i trend procentowy w oknie.
    - Wzór: `r_1 = close/close_{-1}−1`, `r_5 = close/close_{-5}−1`, `r_15 = close/close_{-15}−1`, `slope_return_120 = (close−close_{-120})/close_{-120}`.
    - Normalizacja: bezwymiarowe; można dzielić przez ATR dla spójności z innymi.

20. vol_regime i vol_of_vol (OHLC)
    - Opis: reżim zmienności i zmienność zmienności.
    - Wzór: `vol_regime_120 = std(returns_{1m} w oknie 120)`, `vol_of_vol_120 = std(rolling_std_returns w oknie 120)`.
    - Normalizacja: można przeskalować przez medianę/ATR.

21. r2_trend_120 (OHLC)
    - Theil/R² dopasowania trendu liniowego `close` w oknie 120.
    - Wzór: R² regresji liniowej `close ~ t` w oknie; 0–1.
    - Normalizacja: bezwymiarowe.

22. Interakcje kanał × OB (przykładowe)
    - Opis: połączenia sygnałów strukturalnych z presją płynności.
    - Wzór: `pos_in_channel_{window} × imbalance_1pct_notional`, `slope_over_ATR_window_{window} × imbalance_1pct_notional`, `width_over_ATR_{window} × imbalance_1pct_notional`.
    - Normalizacja: komponenty już bezwymiarowe lub w ATR.

## Podstawowe wskaźniki techniczne (OHLC) — z normalizacją do ATR tam, gdzie zasadne

23. RSI_14, RSI_30
    - Opis: siła relatywna (wykupienie/wyprzedanie) w krótkim i średnim horyzoncie.
    - Wzór: klasyczny RSI z oknami 14 i 30.
    - Normalizacja: bezwymiarowe (0–100). Opcjonalnie percentyl RSI w oknie.

24. Stoch_14_3 (K i D)
    - Opis: lokalne wykupienie/wyprzedanie względem zakresu cen.
    - Wzór: `%K = 100·(close−low_{14})/(high_{14}−low_{14}+ε)`, `%D = SMA(%K,3)`.
    - Normalizacja: bezwymiarowe (0–100).

25. MACD_12_26_9_hist_over_ATR
    - Opis: momentum trendu; używamy histogramu znormalizowanego do ATR.
    - Wzór: `MACD_hist = (EMA12−EMA26) − EMA9(MACD)`, `MACD_hist_over_ATR = MACD_hist / ATR_120`.
    - Normalizacja: przez ATR (np. ATR_120).

26. ADX_14, di_diff_14
    - Opis: siła trendu; różnica kierunku (+DI vs −DI).
    - Wzór: klasyczny ADX(14); `di_diff_14 = +DI_14 − (−DI_14)`.
    - Normalizacja: ADX bezwymiarowe; `di_diff_14` bezwymiarowe (−100..100).

27. CCI_20_over_ATR
    - Opis: odchylenie ceny od typowego poziomu; skala unifikowana przez ATR.
    - Wzór: `CCI_20_over_ATR = CCI_20 / ATR_120`.
    - Normalizacja: przez ATR (dla porównywalności reżimów).

28. BB_20: bb_pos_20, bb_width_over_ATR_20
    - Opis: pozycja ceny w pasmach i szerokość pasma (zmienność) w ATR.
    - Wzór: `bb_pos = (close−BB_mid)/(BB_up−BB_low+ε)`, `bb_width_over_ATR = (BB_up−BB_low)/ATR_120`.
    - Normalizacja: `bb_pos` bezwymiarowe (−0.5..0.5 lub 0..1 w zależności od definicji), szerokość przez ATR.

29. Donchian_60: donch_pos_60, donch_width_over_ATR_60
    - Opis: pozycja względem kanału najwyższe−najniższe; szerokość kanału w ATR.
    - Wzór: `donch_pos_60 = (close−low_{60})/(high_{60}−low_{60}+ε)`, `donch_width_over_ATR_60 = (high_{60}−low_{60})/ATR_120`.
    - Normalizacja: `donch_pos` bezwymiarowe; szerokość przez ATR.

30. Price_vs_EMA: close_vs_ema_60, close_vs_ema_120
    - Opis: relacja ceny do wygładzonego poziomu (trend mean-reversion).
    - Wzór: `close_vs_ema_n = (close−EMA_n)/EMA_n` dla n∈{60,120}.
    - Normalizacja: bezwymiarowe; alternatywnie przez ATR/close.

31. slope_ema_60_over_ATR
    - Opis: tempo zmiany trendu wygładzonego, w jednostkach ATR.
    - Wzór: `slope_ema_60_over_ATR = (EMA_60 − EMA_60.shift(k)) / ATR_120`, np. k=10.
    - Normalizacja: przez ATR.

32. MFI_14 oraz/lub OBV_slope_over_ATR
    - Opis: przepływ kapitału (MFI) i kierunek wolumenowy (OBV).
    - Wzór: `MFI_14` klasyczne; `OBV_slope_over_ATR = (OBV − OBV.shift(k))/ATR_120`.
    - Normalizacja: MFI bezwymiarowe (0–100); OBV nachylenie przez ATR.
