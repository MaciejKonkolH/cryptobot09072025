## Podział cech: OHLC vs Orderbook (feature_calculator3)

Źródło: `feature_calculator3/feature_builder.py` oraz `feature_calculator3/config.py` (aktualny stan). Lista obejmuje cechy generowane do pliku `features_{symbol}.feather` przed etykietowaniem i treningiem.

### OHLC-derived (pochodne z OHLC)

- Bazowe/zmienność krótkoterminowa
  - `ret_1m`
  - `rv_5m`, `rv_30`, `rv_60`, `rv_120`
- ATR i pochodne
  - `atr_14`, `atr_30`, `atr_60`, `atr_pct_14`
- Parkinson / Bollinger
  - `parkinson_60`, `parkinson_120`
  - `bb_width_60`, `bb_pos_60`
- Donchian i dystanse/wybicia
  - `donchian_high_60`, `donchian_low_60`, `donchian_high_120`, `donchian_low_120`
  - `dist_to_high_60`, `dist_to_low_60`
  - `dist_to_high_60_atr`, `range_60_atr`, `range_120_atr`
  - `breakout_score_60`, `breakdown_score_60`
  - `since_high_break_60`, `since_low_break_60`
- True Range (sumy)
  - `tr_sum_30`, `tr_sum_60`, `tr_sum_120`
- Cena względem średnich (SMA z configu)
  - `price_vs_ma_60`, `price_vs_ma_240`, `price_vs_ma_360`, `price_vs_ma_720`, `price_vs_ma_1440`, `price_vs_ma_2880`, `price_vs_ma_4320`, `price_vs_ma_10080`
- Reachability (na bazie sigma/ATR)
  - `sigma_1m_240`, `expected_sigma_120`
  - `tp_sigma_ratio_0p6`, `tp_sigma_ratio_0p8`, `tp_sigma_ratio_1p0`, `tp_sigma_ratio_1p2`, `tp_sigma_ratio_1p4`
  - `sl_sigma_ratio_0p3`, `sl_sigma_ratio_0p4`, `sl_sigma_ratio_0p5`, `sl_sigma_ratio_0p6`, `sl_sigma_ratio_0p7`
  - `tp_atr_ratio_0p6`, `tp_atr_ratio_0p8`, `tp_atr_ratio_1p0`, `sl_atr_ratio_0p3`
- Lags (krótkie, tylko dla wybranych kolumn)
  - `lag{1,2,5}_rv_5m`, `lag{1,2,5}_ret_1m`
- Binning czasowy (średnia i std dla przesunięć; patrz `config.BIN_FEATURES`)
  - Dla kolumn OHLC-pochodnych w binningu: `rv_5m`, `ret_1m`, `price_vs_ma_60`, `price_vs_ma_240`, `price_vs_ma_360`, `price_vs_ma_1440`
  - Wzorce nazw: `bin13_mean_{col}`, `bin13_std_{col}`, `bin410_mean_{col}`, `bin410_std_{col}`, `bin1130_*_{col}`, `bin3160_*_{col}`

### Orderbook-derived (pochodne z orderbooka)

- Sumy/główne miary na snapshotach (S1/S2) i ich relacje
  - `imb_s1`, `imb_s2`, `imb_delta`
  - `delta_depth_bid_rel`, `delta_depth_ask_rel`, `delta_depth_ratio`
  - `ask_over_bid_s1`
- Udziały „near touch” i presja
  - `near_bid_share_s1`, `near_ask_share_s1`, `near_pressure_ratio_s1`
- WADL (Weighted Avg Depth Level)
  - `wadl_bid_share_s1`, `wadl_ask_share_s1`, `wadl_diff_share_s1`
- Mikro-cena i kształt książki
  - `microprice_proxy_s1`
  - `tightness_rel`, `steep_bid_share_s1`, `steep_ask_share_s1`
- Agregaty wolne (rolling na miarach OB)
  - `imb_mean_15`, `imb_mean_30`, `imb_mean_60`
  - `imb_persistence_30`, `imb_sign_consistency_30`
  - `microprice_trend_30`, `microprice_std_30`
- Lags (krótkie, tylko dla wybranych kolumn)
  - `lag{1,2,5}_imb_s1`, `lag{1,2,5}_imb_delta`, `lag{1,2,5}_delta_depth_bid_rel`, `lag{1,2,5}_delta_depth_ask_rel`
- Binning czasowy (średnia i std dla przesunięć; patrz `config.BIN_FEATURES`)
  - Dla kolumn OB w binningu: `imb_s1`, `imb_s2`, `imb_delta`, `near_pressure_ratio_s1`, `wadl_bid_share_s1`, `wadl_ask_share_s1`, `delta_depth_bid_rel`, `delta_depth_ask_rel`
  - Wzorce nazw: `bin13_mean_{col}`, `bin13_std_{col}`, `bin410_mean_{col}`, `bin410_std_{col}`, `bin1130_*_{col}`, `bin3160_*_{col}`

### Uwagi

- Surowe kolumny `open`, `high`, `low`, `close`, `volume` są przenoszone do wyjścia, ale nie są używane jako cechy wejściowe do modelu (są zdejmowane w `training5/data_loader.py`).
- Wzorce nazw w sekcjach „Lags” i „Binning” generują wiele kolumn; ich dokładna lista zależy od zestawu bazowych kolumn w `config.KEY_LAG_FEATURES` i `config.BIN_FEATURES`.
- Pliki źródłowe: `feature_calculator3/feature_builder.py`, `feature_calculator3/config.py`.

