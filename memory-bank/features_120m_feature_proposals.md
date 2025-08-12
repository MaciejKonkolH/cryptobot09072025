# Propozycje dodatkowych cech dla horyzontu 120 min

Cel: poprawa predykcji etykiet "TP przed SL w 120 min" bez przecieku danych. Poniżej wyraźny podział i numeracja: najpierw cechy JUŻ używane w treningu (pełna, dokładna lista), a następnie NOWE propozycje (ponumerowane), aby mieć wgląd w liczność.

## Zasady
- Używamy tylko informacji dostępnej do końca minuty t (brak podglądu w przyszłość).
- Nazewnictwo: snake_case, sufiksy z oknami w minutach.
- Okna: krótkie (5–30), średnie (60–240), długie (360–1440+).

## Obecne cechy (113)
1. imb_s1
2. imb_s2
3. imb_delta
4. near_bid_share_s1
5. near_ask_share_s1
6. near_pressure_ratio_s1
7. wadl_bid_share_s1
8. wadl_ask_share_s1
9. wadl_diff_share_s1
10. delta_depth_bid_rel
11. delta_depth_ask_rel
12. delta_depth_ratio
13. ask_over_bid_s1
14. microprice_proxy_s1
15. tightness_rel
16. steep_bid_share_s1
17. steep_ask_share_s1
18. ret_1m
19. rv_5m
20. price_vs_ma_60
21. price_vs_ma_240
22. price_vs_ma_1440
23. price_vs_ma_10080
24. lag1_imb_s1
25. lag2_imb_s1
26. lag5_imb_s1
27. lag1_imb_delta
28. lag2_imb_delta
29. lag5_imb_delta
30. lag1_delta_depth_bid_rel
31. lag2_delta_depth_bid_rel
32. lag5_delta_depth_bid_rel
33. lag1_delta_depth_ask_rel
34. lag2_delta_depth_ask_rel
35. lag5_delta_depth_ask_rel
36. lag1_rv_5m
37. lag2_rv_5m
38. lag5_rv_5m
39. lag1_ret_1m
40. lag2_ret_1m
41. lag5_ret_1m
42. bin13_mean_imb_s1
43. bin13_std_imb_s1
44. bin410_mean_imb_s1
45. bin410_std_imb_s1
46. bin1130_mean_imb_s1
47. bin1130_std_imb_s1
48. bin13_mean_imb_s2
49. bin13_std_imb_s2
50. bin410_mean_imb_s2
51. bin410_std_imb_s2
52. bin1130_mean_imb_s2
53. bin1130_std_imb_s2
54. bin13_mean_imb_delta
55. bin13_std_imb_delta
56. bin410_mean_imb_delta
57. bin410_std_imb_delta
58. bin1130_mean_imb_delta
59. bin1130_std_imb_delta
60. bin13_mean_near_pressure_ratio_s1
61. bin13_std_near_pressure_ratio_s1
62. bin410_mean_near_pressure_ratio_s1
63. bin410_std_near_pressure_ratio_s1
64. bin1130_mean_near_pressure_ratio_s1
65. bin1130_std_near_pressure_ratio_s1
66. bin13_mean_wadl_bid_share_s1
67. bin13_std_wadl_bid_share_s1
68. bin410_mean_wadl_bid_share_s1
69. bin410_std_wadl_bid_share_s1
70. bin1130_mean_wadl_bid_share_s1
71. bin1130_std_wadl_bid_share_s1
72. bin13_mean_wadl_ask_share_s1
73. bin13_std_wadl_ask_share_s1
74. bin410_mean_wadl_ask_share_s1
75. bin410_std_wadl_ask_share_s1
76. bin1130_mean_wadl_ask_share_s1
77. bin1130_std_wadl_ask_share_s1
78. bin13_mean_delta_depth_bid_rel
79. bin13_std_delta_depth_bid_rel
80. bin410_mean_delta_depth_bid_rel
81. bin410_std_delta_depth_bid_rel
82. bin1130_mean_delta_depth_bid_rel
83. bin1130_std_delta_depth_bid_rel
84. bin13_mean_delta_depth_ask_rel
85. bin13_std_delta_depth_ask_rel
86. bin410_mean_delta_depth_ask_rel
87. bin410_std_delta_depth_ask_rel
88. bin1130_mean_delta_depth_ask_rel
89. bin1130_std_delta_depth_ask_rel
90. bin13_mean_rv_5m
91. bin13_std_rv_5m
92. bin410_mean_rv_5m
93. bin410_std_rv_5m
94. bin1130_mean_rv_5m
95. bin1130_std_rv_5m
96. bin13_mean_ret_1m
97. bin13_std_ret_1m
98. bin410_mean_ret_1m
99. bin410_std_ret_1m
100. bin1130_mean_ret_1m
101. bin1130_std_ret_1m
102. bin13_mean_price_vs_ma_240
103. bin13_std_price_vs_ma_240
104. bin410_mean_price_vs_ma_240
105. bin410_std_price_vs_ma_240
106. bin1130_mean_price_vs_ma_240
107. bin1130_std_price_vs_ma_240
108. bin13_mean_price_vs_ma_1440
109. bin13_std_price_vs_ma_1440
110. bin410_mean_price_vs_ma_1440
111. bin410_std_price_vs_ma_1440
112. bin1130_mean_price_vs_ma_1440
113. bin1130_std_price_vs_ma_1440

(Uwaga: do X nie trafiają etykiety ani kolumny: open, high, low, close, volume.)

## Nowe propozycje cech (103)

### 1) Trend i momentum wielohoryzontowe (25)
1. price_vs_ma_360
2. price_vs_ma_720
3. price_vs_ma_2880
4. price_vs_ma_4320
5. ma_slope_60
6. ma_slope_240
7. ma_slope_720
8. price_vs_ma_slope_60
9. price_vs_ma_slope_240
10. ema_ratio_60_240
11. ema_ratio_240_720
12. macd_12_26_9
13. macd_signal_12_26_9
14. macd_hist_12_26_9
15. rsi_14
16. rsi_30
17. rsi_60
18. momentum_10
19. momentum_30
20. momentum_60
21. pos_ret_share_30
22. pos_ret_share_60
23. pos_ret_share_120
24. runlen_up_30
25. runlen_down_30

### 2) Zmienność i niepewność (16)
26. atr_14
27. atr_30
28. atr_60
29. atr_pct_14
30. rv_30
31. rv_60
32. rv_120
33. vol_of_vol_60
34. vol_of_vol_120
35. parkinson_60
36. parkinson_120
37. bb_width_60
38. bb_pos_60
39. tr_sum_30
40. tr_sum_60
41. tr_sum_120

### 3) Zakres, poziomy i wybicia (13)
42. donchian_high_60
43. donchian_low_60
44. donchian_high_120
45. donchian_low_120
46. dist_to_high_60
47. dist_to_low_60
48. dist_to_high_60_atr
49. range_60_atr
50. range_120_atr
51. breakout_score_60
52. breakdown_score_60
53. since_high_break_60
54. since_low_break_60

### 4) Reżimy rynku (8)
55. trend_regime_240
56. trend_regime_1440
57. vol_regime_60
58. regime_combo
59. time_of_day_sin
60. time_of_day_cos
61. day_of_week_sin
62. day_of_week_cos

### 5) Struktura świec i price-action (9)
63. candle_body_1m
64. body_share_1m
65. upper_wick_share_1m
66. lower_wick_share_1m
67. body_dir_1m
68. body_dir_share_30
69. engulfing_5
70. pinbar_5
71. pattern_counts_30

### 6) Agregacje czasowe z orderbooka (wolne) (12)
72. imb_mean_15
73. imb_mean_30
74. imb_mean_60
75. imb_persistence_30
76. imb_sign_consistency_30
77. microprice_trend_30
78. delta_depth_bid_rel_sum_30
79. delta_depth_ask_rel_sum_30
80. near_pressure_mean_30
81. wadl_diff_mean_30
82. tightness_mean_30
83. tightness_std_30

### 7) Reachability progu (14)
84. sigma_1m
85. expected_sigma_120
86. tp_sigma_ratio_0p6
87. tp_sigma_ratio_0p8
88. tp_sigma_ratio_1p0
89. tp_sigma_ratio_1p2
90. tp_sigma_ratio_1p4
91. sl_sigma_ratio_0p3
92. sl_sigma_ratio_0p4
93. sl_sigma_ratio_0p5
94. sl_sigma_ratio_0p6
95. sl_sigma_ratio_0p7
96. tp_atr_ratio_0p8
97. sl_atr_ratio_0p3

### 8) Trwałość kierunku i burstiness (4)
98. directional_persistence_30
99. burst_up_max_30
100. burst_down_max_30
101. mean_reversion_score_30

### 9) Interakcje RR ze zmiennością (2)
102. rr_vs_regime_1
103. rr_vs_regime_2

---
- Obecne cechy: 113.
- Nowe propozycje: 103.
- MVP do wdrożenia najpierw: (1) ATR/vol/range, (2) Trend dłuższy/slope/RSI/momentum, (3) Donchian + dystanse/ATR, (4) OB‑slow, (5) Reachability.