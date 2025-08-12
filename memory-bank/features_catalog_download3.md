## Katalog cech – download3 (wide orderbook + OHLC)

### Cel
- Spójna, czytelna lista proponowanych cech do treningu na danych z modułu `download3` (finalny plik: `merge/merged_data/merged_{SYMBOL}.parquet`).
- Format wide: 2 snapshoty/minutę, kolumny `depth_{1|2}_{m1..m5|p1..p5}`, `notional_{1|2}_{m1..m5|p1..p5}`, `ts_1`, `ts_2` oraz OHLC (`open, high, low, close, volume`).

### Konwencje i skróty
- mK = strona bid (negatywne poziomy), pK = strona ask (pozytywne poziomy), K ∈ {1..5} (1 = najbliżej midprice, 5 = dalej).
- rank 1/2: `1` odpowiada `ts_1` (pierwszy snapshot w minucie), `2` → `ts_2` (ostatni snapshot).
- Suma po poziomach: Σ_bid(depth_s) = Σ_{k=1..5} depth_{s}_mK; analogicznie Σ_ask(depth_s) = Σ_{k=1..5} depth_{s}_pK. Tak samo dla `notional`.
- Nazewnictwo feature: `*_s1` → liczone z migawki 1; `*_s2` → z migawki 2; `delta_*` lub `*_shift` → różnica s2 − s1.
- Wszystkie cechy są względne/bezwymiarowe: pracujemy na udziałach, ratio, zwrotach lub wartościach znormalizowanych (np. przez sumy/średnie), unikając wielkości bezwzględnych.

---

### A) Cechy rdzeniowe Orderbook (intraminutowe, 2 snapshoty)
1) Imbalance (depth)
- imb_s1 = (Σ_bid(depth_1) − Σ_ask(depth_1)) / (Σ_bid(depth_1) + Σ_ask(depth_1) + 1e-9)
- imb_s2 = (Σ_bid(depth_2) − Σ_ask(depth_2)) / (Σ_bid(depth_2) + Σ_ask(depth_2) + 1e-9)
- imb_delta = imb_s2 − imb_s1

2) Near-touch (poziom ±1; w udziałach)
- near_bid_share_s{1,2} = depth_{ {1,2} }_m1 / (Σ depth_{ {1,2} }_{m1..m5} + 1e-9)
- near_ask_share_s{1,2} = depth_{ {1,2} }_p1 / (Σ depth_{ {1,2} }_{p1..p5} + 1e-9)
- near_pressure_ratio_s{1,2} = near_ask_share_s{1,2} / (near_bid_share_s{1,2} + 1e-9)
- near_bid_notional_share_s{1,2} = notional_{ {1,2} }_m1 / (Σ notional_{ {1,2} }_{m1..m5} + 1e-9)
- near_ask_notional_share_s{1,2} = notional_{ {1,2} }_p1 / (Σ notional_{ {1,2} }_{p1..p5} + 1e-9)
- near_notional_ratio_s{1,2} = near_ask_notional_share_s{1,2} / (near_bid_notional_share_s{1,2} + 1e-9)

3) Krzywizna/pochylenie książki (na udziałach)
- Dla bids (m1..m5) i asks (p1..p5) dopasuj regresję po poziomie k=1..5 do udziałów:
  - slope_bid_share_s{1,2}, slope_ask_share_s{1,2} (regresja liniowa share vs k)
  - curv_bid_share_s{1,2}, curv_ask_share_s{1,2} (współczynnik kwadratowy, opcjonalne)
  - slope_not_bid_share_s{1,2}, slope_not_ask_share_s{1,2} (analogicznie dla notional shares)

4) Weighted average distance of liquidity (WADL; na udziałach)
- wadl_bid_share_s{1,2} = Σ(k · share(depth_{ {1,2} }_mK)) / (Σ share(depth_{ {1,2} }_mK) + 1e-9) = Σ(k · share)
- wadl_ask_share_s{1,2} = Σ(k · share(depth_{ {1,2} }_pK)) / (Σ share(depth_{ {1,2} }_pK) + 1e-9)
- wadl_diff_share_s{1,2} = wadl_ask_share_s{1,2} − wadl_bid_share_s{1,2}

5) Zmiany intraminutowe (s2 vs s1; relatywne)
- delta_depth_bid_rel = (Σ depth_2_mK − Σ depth_1_mK) / (Σ depth_1_mK + 1e-9)
- delta_depth_ask_rel = (Σ depth_2_pK − Σ depth_1_pK) / (Σ depth_1_pK + 1e-9)
- delta_not_bid_rel = (Σ notional_2_mK − Σ notional_1_mK) / (Σ notional_1_mK + 1e-9)
- delta_not_ask_rel = (Σ notional_2_pK − Σ notional_1_pK) / (Σ notional_1_pK + 1e-9)
- delta_depth_ratio = (delta_depth_ask_rel − delta_depth_bid_rel) / (|delta_depth_ask_rel| + |delta_depth_bid_rel| + 1e-9)

6) Stabilność rozkładu po poziomach (s)
- entropy_bid_s{1,2} = entropia(softmax(depth_{s}_m1..m5))
- entropy_ask_s{1,2} = entropia(softmax(depth_{s}_p1..p5))
- cv_bid_s{1,2} = std(depth_{s}_m1..m5) / (mean(depth_{s}_m1..m5) + 1e-9)
- cv_ask_s{1,2} = std(depth_{s}_p1..p5) / (mean(depth_{s}_p1..p5) + 1e-9)

7) Asymetrie płynności
- ask_over_bid_s{1,2} = Σ_ask(depth_s) / (Σ_bid(depth_s) + 1e-9)
- not_ask_over_bid_s{1,2} = Σ_ask(notional_s) / (Σ_bid(notional_s) + 1e-9)

8) Proxy mikro-ceny (z notional na ±1)
- microprice_proxy_s{1,2} = notional_{ {1,2} }_p1 / (notional_{ {1,2} }_p1 + notional_{ {1,2} }_m1 + 1e-9)
- microprice_shift = microprice_proxy_s2 − microprice_proxy_s1

9) Tightness i steepness (na udziałach)
- tightness_rel = near_ask_share_s1 + near_bid_share_s1
- steep_bid_share_s{1,2} = share(depth_{ {1,2} }_m1) / (Σ share(depth_{ {1,2} }_{m2..m5}) + 1e-9)
- steep_ask_share_s{1,2} = share(depth_{ {1,2} }_p1) / (Σ share(depth_{ {1,2} }_{p2..p5}) + 1e-9)

---

### B) Interakcje z ceną i wolumenem (OHLC)
- ret_1m = close_t / close_{t-1} − 1 (lub log-zwrot)
- ret_5m = close_t / close_{t-5} − 1
- rv_5m, rv_15m = zrealizowana zmienność (std log-zwrotów) w oknach 5 i 15 min
- amihud_proxy = |ret_1m| / (volume_ma_60 + 1e-9)  (wolumen znormalizowany)
- price_vs_ma_60 = close / ma_60; price_vs_ma_240 = close / ma_240
- price_vs_ma_1440 = close / ma_1440; price_vs_ma_10080 = close / ma_10080
- slope_price_5m_rel = nachylenie OLS dla serii znormalizowanej (np. close/ma_60 lub log-close) w oknie 5 min
- ob_alpha_5m = EMA_5(imb_s1); ob_alpha_delta = EMA_5(imb_delta)

---

### C) Cechy czasowe i techniczne
- sin_hour = sin(2π · hour/24), cos_hour = cos(2π · hour/24)
- sin_dow = sin(2π · dow/7), cos_dow = cos(2π · dow/7)
- ts_gap_frac = (ts_2 − ts_1) / 60 (w jednostkach minut; opcjonalne QC)

---

### D) Lags i agregacje rolling (dla modeli tabularnych)
Zalecane bazowe miary do lagów i rolling (wszystkie względne):
- imb_s1, imb_s2, imb_delta
- near_bid_share_s1, near_ask_share_s1, wadl_bid_share_s1, wadl_ask_share_s1
- delta_depth_bid_rel, delta_depth_ask_rel
- rv_5m, ret_1m

Lagi (spłaszczone kolumny):
- lag1_*, lag2_*, lag5_* (t−1, t−2, t−5 min)

Agregacje rolling:
- ema3_imb_s1, ema10_imb_s1; std5_imb_s1, max5_imb_s1, min5_imb_s1
- ema5_delta_depth_bid, ema5_delta_depth_ask
- rollcorr10_imb_s1_ret1m (korelacja krocząca 10-min)

Konwencje nazw:
- `lagK_{feature}` (np. `lag2_imb_s1`), `emaN_{feature}`, `stdN_{feature}`, `maxN_{feature}`, `minN_{feature}`

---

### E) Wariant sekwencyjny (dla LSTM/Transformer)
Zamiast wielu lagów, podaj sekwencję (czas × cecha) z okna 30–60 min (np. 30/60 kroków) dla 12–16 kluczowych miar:
- imb_s1, imb_s2, imb_delta
- near_bid_depth_s1, near_ask_depth_s1, near_notional_ratio_s1
- slope_bid_s1, slope_ask_s1, wadl_bid_s1, wadl_ask_s1
- delta_depth_bid, delta_depth_ask
- ret_1m, rv_5m, amihud_proxy

Uwaga: sekwencje przekazuj jako 2D (window_size × num_features) bez spłaszczania.

---

### F) Minimalny praktyczny zestaw (start, ok. 40–60 cech)
- imb_s1, imb_s2, imb_delta
- near_bid_share_s1, near_ask_share_s1, near_bid_share_s2, near_ask_share_s2
- near_pressure_ratio_s1, slope_bid_share_s1, slope_ask_share_s1, slope_not_bid_share_s1, slope_not_ask_share_s1
- wadl_bid_share_s1, wadl_ask_share_s1, wadl_diff_share_s1
- delta_depth_bid_rel, delta_depth_ask_rel, delta_depth_ratio
- entropy_bid_s1, entropy_ask_s1
- ask_over_bid_s1, not_ask_over_bid_s1
- microprice_proxy_s1, microprice_shift
- tightness_rel, steep_bid_share_s1, steep_ask_share_s1
- ret_1m, rv_5m, amihud_proxy, price_vs_ma_60, price_vs_ma_240, price_vs_ma_1440, price_vs_ma_10080, slope_price_5m_rel
- sin_hour, cos_hour, sin_dow, cos_dow, ts_gap_frac
- LAGI: (lag1, lag2, lag5) dla: imb_s1, imb_delta, wadl_bid_share_s1, wadl_ask_share_s1, delta_depth_bid_rel, delta_depth_ask_rel, rv_5m, ret_1m
- ROLLING: ema3/ema10(imb_s1), std5(imb_s1), ema5(delta_depth_bid_rel), ema5(delta_depth_ask_rel)

---

### J) Zestaw v1 z numeracją (69 cech)
Bazowe (40):
1) imb_s1
2) imb_s2
3) imb_delta
4) near_bid_share_s1
5) near_ask_share_s1
6) near_bid_share_s2
7) near_ask_share_s2
8) near_pressure_ratio_s1
9) slope_bid_share_s1
10) slope_ask_share_s1
11) slope_not_bid_share_s1
12) slope_not_ask_share_s1
13) wadl_bid_share_s1
14) wadl_ask_share_s1
15) wadl_diff_share_s1
16) delta_depth_bid_rel
17) delta_depth_ask_rel
18) delta_depth_ratio
19) entropy_bid_s1
20) entropy_ask_s1
21) ask_over_bid_s1
22) not_ask_over_bid_s1
23) microprice_proxy_s1
24) microprice_shift
25) tightness_rel
26) steep_bid_share_s1
27) steep_ask_share_s1
28) ret_1m
29) rv_5m
30) amihud_proxy
31) slope_price_5m_rel
32) price_vs_ma_60
33) price_vs_ma_240
34) price_vs_ma_1440
35) price_vs_ma_10080
36) sin_hour
37) cos_hour
38) sin_dow
39) cos_dow
40) ts_gap_frac

Lagi/rolling (29):
41) lag1_imb_s1
42) lag2_imb_s1
43) lag5_imb_s1
44) lag1_imb_delta
45) lag2_imb_delta
46) lag5_imb_delta
47) lag1_wadl_bid_share_s1
48) lag2_wadl_bid_share_s1
49) lag5_wadl_bid_share_s1
50) lag1_wadl_ask_share_s1
51) lag2_wadl_ask_share_s1
52) lag5_wadl_ask_share_s1
53) lag1_delta_depth_bid_rel
54) lag2_delta_depth_bid_rel
55) lag5_delta_depth_bid_rel
56) lag1_delta_depth_ask_rel
57) lag2_delta_depth_ask_rel
58) lag5_delta_depth_ask_rel
59) lag1_rv_5m
60) lag2_rv_5m
61) lag5_rv_5m
62) lag1_ret_1m
63) lag2_ret_1m
64) lag5_ret_1m
65) ema3_imb_s1
66) ema10_imb_s1
67) std5_imb_s1
68) ema5_delta_depth_bid_rel
69) ema5_delta_depth_ask_rel

---

### K) Zestaw sekwencyjny (flatten dla XGBoost, okno historii W=30 min)
Konfiguracja sekwencyjna bez LSTM: spłaszczone lagi i rolling, w pełni względne.
- LAGS: [1, 2, 3, 5, 10, 15, 30] minut
- Baza (12 cech): imb_s1, imb_s2, imb_delta, near_pressure_ratio_s1, wadl_bid_share_s1, wadl_ask_share_s1, delta_depth_bid_rel, delta_depth_ask_rel, rv_5m, ret_1m, price_vs_ma_240, price_vs_ma_1440
- Rolling (dodatkowe, bez duplikowania pozycji z sekcji J): EMA i STD dla wybranych kluczowych cech

Sekwencyjne lagi (84):
70) lag1_imb_s1
71) lag2_imb_s1
72) lag3_imb_s1
73) lag5_imb_s1
74) lag10_imb_s1
75) lag15_imb_s1
76) lag30_imb_s1
77) lag1_imb_s2
78) lag2_imb_s2
79) lag3_imb_s2
80) lag5_imb_s2
81) lag10_imb_s2
82) lag15_imb_s2
83) lag30_imb_s2
84) lag1_imb_delta
85) lag2_imb_delta
86) lag3_imb_delta
87) lag5_imb_delta
88) lag10_imb_delta
89) lag15_imb_delta
90) lag30_imb_delta
91) lag1_near_pressure_ratio_s1
92) lag2_near_pressure_ratio_s1
93) lag3_near_pressure_ratio_s1
94) lag5_near_pressure_ratio_s1
95) lag10_near_pressure_ratio_s1
96) lag15_near_pressure_ratio_s1
97) lag30_near_pressure_ratio_s1
98) lag1_wadl_bid_share_s1
99) lag2_wadl_bid_share_s1
100) lag3_wadl_bid_share_s1
101) lag5_wadl_bid_share_s1
102) lag10_wadl_bid_share_s1
103) lag15_wadl_bid_share_s1
104) lag30_wadl_bid_share_s1
105) lag1_wadl_ask_share_s1
106) lag2_wadl_ask_share_s1
107) lag3_wadl_ask_share_s1
108) lag5_wadl_ask_share_s1
109) lag10_wadl_ask_share_s1
110) lag15_wadl_ask_share_s1
111) lag30_wadl_ask_share_s1
112) lag1_delta_depth_bid_rel
113) lag2_delta_depth_bid_rel
114) lag3_delta_depth_bid_rel
115) lag5_delta_depth_bid_rel
116) lag10_delta_depth_bid_rel
117) lag15_delta_depth_bid_rel
118) lag30_delta_depth_bid_rel
119) lag1_delta_depth_ask_rel
120) lag2_delta_depth_ask_rel
121) lag3_delta_depth_ask_rel
122) lag5_delta_depth_ask_rel
123) lag10_delta_depth_ask_rel
124) lag15_delta_depth_ask_rel
125) lag30_delta_depth_ask_rel
126) lag1_rv_5m
127) lag2_rv_5m
128) lag3_rv_5m
129) lag5_rv_5m
130) lag10_rv_5m
131) lag15_rv_5m
132) lag30_rv_5m
133) lag1_ret_1m
134) lag2_ret_1m
135) lag3_ret_1m
136) lag5_ret_1m
137) lag10_ret_1m
138) lag15_ret_1m
139) lag30_ret_1m
140) lag1_price_vs_ma_240
141) lag2_price_vs_ma_240
142) lag3_price_vs_ma_240
143) lag5_price_vs_ma_240
144) lag10_price_vs_ma_240
145) lag15_price_vs_ma_240
146) lag30_price_vs_ma_240
147) lag1_price_vs_ma_1440
148) lag2_price_vs_ma_1440
149) lag3_price_vs_ma_1440
150) lag5_price_vs_ma_1440
151) lag10_price_vs_ma_1440
152) lag15_price_vs_ma_1440
153) lag30_price_vs_ma_1440

Dodatkowe rolling (24):
154) ema30_imb_s1
155) std10_imb_s1
156) std30_imb_s1
157) ema10_imb_delta
158) ema30_imb_delta
159) std5_imb_delta
160) std10_imb_delta
161) std30_imb_delta
162) ema10_delta_depth_bid_rel
163) ema30_delta_depth_bid_rel
164) std5_delta_depth_bid_rel
165) std10_delta_depth_bid_rel
166) std30_delta_depth_bid_rel
167) ema10_delta_depth_ask_rel
168) ema30_delta_depth_ask_rel
169) std5_delta_depth_ask_rel
170) std10_delta_depth_ask_rel
171) std30_delta_depth_ask_rel
172) ema3_ret_1m
173) ema10_ret_1m
174) ema30_ret_1m
175) std5_ret_1m
176) std10_ret_1m
177) std30_ret_1m

Łącznie (sekcja K): 84 (lagi) + 24 (rolling) = 108 nowych cech.

---

### L) Binning czasu (3 wiadra) i rekomendacja hybrydowa
Cel: zmniejszyć wymiar sekwencji i ustabilizować sygnał bez utraty informacji o krótkim, średnim i długim horyzoncie.

- Definicja wiader czasu (minuty wstecz):
  - bin13: [1–3]
  - bin410: [4–10]
  - bin1130: [11–30]

- Statystyki per wiadro (bezwymiarowe): mean, std (opcjonalnie: min/max, ema)

- Konwencje nazw:
  - bin13_mean_{feature}, bin13_std_{feature}
  - bin410_mean_{feature}, bin410_std_{feature}
  - bin1130_mean_{feature}, bin1130_std_{feature}

- Baza cech do binningu (12):
  - imb_s1, imb_s2, imb_delta, near_pressure_ratio_s1,
  - wadl_bid_share_s1, wadl_ask_share_s1,
  - delta_depth_bid_rel, delta_depth_ask_rel,
  - rv_5m, ret_1m, price_vs_ma_240, price_vs_ma_1440

- Liczność (pełny binning): 12 cech × 3 wiadra × 2 statystyki = 72 kolumn

- Rekomendacja hybrydowa:
  - Zostaw krótkie lagi (lag1, lag2, lag5) TYLKO dla kluczowych 6 cech: imb_s1, imb_delta, delta_depth_bid_rel, delta_depth_ask_rel, rv_5m, ret_1m → 6 × 3 = 18 kolumn
  - Dłuższy kontekst zastąp binningiem (72 kolumn)
  - Razem sekwencja (hybryda): ~90 kolumn (vs 108 w pełnych lagach z sekcji K)

- Zasady: wyłącznie przeszłość (brak leakage), po generacji utnij warm‑up; wszystkie cechy pozostają względne.

---

### Podsumowanie liczby cech
- Sekcja J (v1): 69 cech
- Sekcja K (sekwencja dla XGBoost): 108 cech
- Razem: 177 cech

Alternatywa (wariant hybrydowy z sekcji L):
- Sekcja J (v1): 69 cech
- Sekwencja (hybryda: 72 bin + 18 krótkich lagów): ~90 cech
- Razem: ~159 cech

Uwaga: wszystkie cechy pozostają względne/bezwymiarowe; lagi i rolling liczone wyłącznie wstecz (brak leakage), po wygenerowaniu usuwamy wiersze z warm-up.

### G) Higiena danych i uwagi implementacyjne
- Brak dziel. przez zero: dodawaj 1e-9 w mianowniku. Zastępuj ±inf → wartości neutralne (np. 0 lub 1 w ratio), NaN → 0 (chyba że inaczej uzasadnione).
- Normalizacja: preferuj cechy względne/udziały/ratio; pochylenia/krzywizny stabilizuj skalując k ∈ [1..5].
- Okna i warmup: rolling/EMA wymagają warmup; przycinaj początek (zgodnie z polityką modułu cech).
- Timestampy: korzystaj z `ts_1`, `ts_2` (zachowuj unikalność w minucie). Feature intraminutowe opieraj na s1/s2.
- Spójność nazw: trzymaj prefiksy `lagK_`, `emaN_`, `stdN_`, `slope_`, `curv_`, `near_`, `delta_`.

---

### H) Mapowanie do kolumn wejściowych (merged parquet)
- OHLC: `open, high, low, close, volume`.
- Orderbook: `ts_1, ts_2`, `depth_{1|2}_{m1..m5|p1..p5}`, `notional_{1|2}_{m1..m5|p1..p5}`.
- Indeks czasu: kolumna `timestamp` (początek minuty, UTC, ciągły bez braków).

---

### I) Dalsze rozszerzenia (opcjonalne)
- Zamiast depth/notional w wartościach bezwzględnych – standaryzacje dzienne lub per-smugi zmienności.
- Wersje winsoryzowane dla entropii i CV (odporność na outliery).
- HHI/Gini dla koncentracji płynności po poziomach (bids/asks osobno).

---

### Autor: download3 | Wersja: wstępna (do iteracji)

