# Blueprint cech (120 min) – zaawansowana propozycja

Cel: zbudować możliwie silny zestaw cech pod etykietę „TP przed SL w 120 min” na danych 1m (OHLC + 2 snapshoty orderbook/min), bez przecieku. Poniżej NUMEROWANA lista propozycji, z podziałem na grupy i krótkim uzasadnieniem. Na końcu sumy per grupę i łączna liczba.

Zasady: wyłącznie informacja do końca minuty t; rolling/lag bez lookahead; nazewnictwo snake_case.

## A) Trend i momentum ceny (34)
1. price_vs_ma_360  
2. price_vs_ma_720  
3. price_vs_ma_2880  
4. price_vs_ma_4320  
5. ma_slope_60 (nachylenie SMA60)  
6. ma_slope_240  
7. ma_slope_720  
8. price_vs_ma_slope_60  
9. price_vs_ma_slope_240  
10. ema_ratio_60_240 (EMA60/EMA240)  
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
21. pos_ret_share_30 (udział wzrostowych świec)  
22. pos_ret_share_60  
23. pos_ret_share_120  
24. runlen_up_30 (najdłuższa seria wzrostów)  
25. runlen_down_30  
26. ret_cum_15 (skumulowany logret 15m)  
27. ret_cum_30  
28. ret_cum_60  
29. ret_max_up_60 (maksymalny dodatni swing 60m)  
30. ret_max_down_60  
31. zscore_close_60  
32. zscore_close_240  
33. vr_ratio_30_120 (variance ratio)  
34. acf1_ret_30 (autokorelacja 1-lag w 30m)

## B) Zmienność i rozkład (28)
35. atr_14  
36. atr_30  
37. atr_60  
38. atr_pct_14 (ATR/close)  
39. rv_30 (std logret)  
40. rv_60  
41. rv_120  
42. vol_of_vol_60 (std z rv_*)  
43. vol_of_vol_120  
44. parkinson_60  
45. parkinson_120  
46. garman_klass_60  
47. rogers_satchell_60  
48. bb_width_60 (Bollinger width)  
49. bb_pos_60 (pozycja w pasmach)  
50. tr_sum_30 (suma True Range)  
51. tr_sum_60  
52. tr_sum_120  
53. realized_skew_60  
54. realized_skew_120  
55. realized_kurt_60  
56. realized_kurt_120  
57. quarticity_60  
58. quarticity_120  
59. bipower_var_60  
60. bipower_var_120  
61. noise_to_signal_60 (rv_60/|ret_cum_60|)  
62. noise_to_signal_120

## C) Zakres, poziomy, wybicia (18)
63. donchian_high_60  
64. donchian_low_60  
65. donchian_high_120  
66. donchian_low_120  
67. dist_to_high_60 (w %)  
68. dist_to_low_60  
69. dist_to_high_60_atr (w ATR)  
70. range_60_atr ((H-L)/ATR)  
71. range_120_atr  
72. breakout_score_60 ((close-high_60)/ATR)  
73. breakdown_score_60  
74. since_high_break_60 (minuty od wybicia)  
75. since_low_break_60  
76. keltner_width_60  
77. keltner_pos_60  
78. rolling_hilo_spread_60 (avg(H-L))  
79. rolling_hilo_spread_120  
80. close_in_range_60 ((close-L)/(H-L))

## D) Reżimy i sezonowość (16)
81. trend_regime_240 (1[price_vs_ma_240>1])  
82. trend_regime_1440  
83. vol_regime_60 (kwantyl rv_60 vs 1440m)  
84. regime_combo (trend×vol)  
85. time_of_day_sin  
86. time_of_day_cos  
87. day_of_week_sin  
88. day_of_week_cos  
89. us_session_open_proximity (minuty do 13:30 UTC)  
90. us_session_close_proximity  
91. asia_session_flag  
92. eu_session_flag  
93. holiday_dummy (jeśli dostępne)  
94. month_sin  
95. month_cos  
96. weekend_flag

## E) Struktura świec / price‑action (12)
97. candle_body_1m (|C-O|)  
98. body_share_1m (body/(H-L))  
99. upper_wick_share_1m  
100. lower_wick_share_1m  
101. body_dir_1m (sign(C-O))  
102. body_dir_share_30 (udział dodatnich korpusów)  
103. engulfing_5 (flaga)  
104. pinbar_5  
105. three_soldiers_5  
106. doji_rate_30  
107. hammer_rate_30  
108. marubozu_rate_30

## F) Orderbook – mikrostruktura rozszerzona (32)
109. q_imbalance_s1 ((B1-A1)/(B1+A1), gdzie B1,A1 to ilości na lvl1)  
110. q_imbalance_s2 (analogicznie dla snapshot2)  
111. q_imbalance_delta (s2-s1)  
112. depth_slope_bid_s1 (regresja indeksu poziomów vs wolumen)  
113. depth_slope_ask_s1  
114. depth_slope_bid_s2  
115. depth_slope_ask_s2  
116. liquidity_asymmetry_s1 (sum_bid/sum_ask)  
117. liquidity_asymmetry_s2  
118. microprice (notional‑based) s1  
119. microprice_delta (s2-s1)  
120. spread_proxy (1/micro_liquidity)  
121. pressure_near_s1 (depth_1_m1 - depth_1_p1)  
122. pressure_near_s2  
123. wadl_skew_s1 (waga poziomów bid vs ask)  
124. wadl_skew_s2  
125. delta_depth_bid_rel_pos (max(delta,0))  
126. delta_depth_ask_rel_pos  
127. delta_depth_diff_rel (ask_rel - bid_rel)  
128. notional_imb_s1 ((not_bid1-not_ask1)/(not_bid1+not_ask1))  
129. notional_imb_delta  
130. near_stacked_bid (1[depth_1_m1/sum_bid_1 > t])  
131. near_stacked_ask  
132. far_void_bid (1[depth_1_m5/sum_bid_1 < t2])  
133. far_void_ask  
134. cross_imb_flag (sign changes s1→s2)  
135. book_resilience_5m (|Δimb| rolling std 5m)  
136. book_trend_30m (slope z imb_s1 30m)  
137. book_vol_30m (std imb_s1 30m)  
138. book_skew_30m (skew imb_s1 30m)  
140. book_entropy_30m (entropia histogramu imb_s1)

## G) Agregacje wolne z OB (16)
141. imb_mean_15  
142. imb_mean_30  
143. imb_mean_60  
144. imb_persistence_30 (udział |imb|>0.2)  
145. imb_sign_consistency_30  
146. microprice_trend_30 (slope)  
147. microprice_std_30  
148. delta_depth_bid_rel_sum_30  
149. delta_depth_ask_rel_sum_30  
150. near_pressure_mean_30  
151. near_pressure_std_30  
152. wadl_diff_mean_30  
153. wadl_diff_std_30  
154. tightness_mean_30  
155. tightness_std_30  
156. ob_kurtosis_30

## H) „Reachability” TP/SL – bez przecieku (20)
157. sigma_1m_240 (std 1m z okna 240)  
158. expected_sigma_120 (sqrt(120)*sigma_1m_240)  
159. tp_sigma_ratio_0p6 (0.006/expected_sigma_120)  
160. tp_sigma_ratio_0p8  
161. tp_sigma_ratio_1p0  
162. tp_sigma_ratio_1p2  
163. tp_sigma_ratio_1p4  
164. sl_sigma_ratio_0p3  
165. sl_sigma_ratio_0p4  
166. sl_sigma_ratio_0p5  
167. sl_sigma_ratio_0p6  
168. sl_sigma_ratio_0p7  
169. tp_atr_ratio_0p6 (0.006/(atr_14/close))  
170. tp_atr_ratio_0p8  
171. tp_atr_ratio_1p0  
172. sl_atr_ratio_0p3  
173. sl_atr_ratio_0p4  
174. sl_atr_ratio_0p5  
175. sl_atr_ratio_0p6  
176. sl_atr_ratio_0p7

## I) Spaced lags i okna rozszerzone dla długich MA (18)
177. lag{1}_price_vs_ma_1440  
178. lag{2}_price_vs_ma_1440  
179. lag{5}_price_vs_ma_1440  
180. lag{10}_price_vs_ma_1440  
181. lag{15}_price_vs_ma_1440  
182. lag{30}_price_vs_ma_1440  
183. lag{45}_price_vs_ma_1440  
184. lag{60}_price_vs_ma_1440  
185. lag{1}_price_vs_ma_4320  
186. lag{2}_price_vs_ma_4320  
187. lag{5}_price_vs_ma_4320  
188. lag{10}_price_vs_ma_4320  
189. lag{15}_price_vs_ma_4320  
190. lag{30}_price_vs_ma_4320  
191. lag{45}_price_vs_ma_4320  
192. lag{60}_price_vs_ma_4320  
193. bin3160_mean_price_vs_ma_4320  
194. bin3160_std_price_vs_ma_4320

---
Suma cech w blueprintcie: 194.

Uwagi strategiczne (skrót):
- 120 min wymaga silnych cech trend/vol/regime; mikro‑OB używać głównie w uśrednieniach/slow‑trends.
- Unikać nadmiarowych 60× lagów co 1 min – stosować spaced lags + statystyki okna.
- Dla modelu: mocne wagi klas, optymalizacja progu pod expected profit, walidacja time‑series, filtracja próbek intrabar‑BOTH (opcjonalnie).