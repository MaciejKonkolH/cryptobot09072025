# LISTA WSZYSTKICH CECH Z PROCENTOWYM DOPASOWANIEM

**Data analizy:** 4 sierpnia 2025  
**Liczba analizowanych cech:** 101  
**Liczba porównanych wartości:** 1,270,074  

## 📊 **CECHY POSORTOWANE OD NAJGORSZYCH DO NAJLEPSZYCH**

### 🔴 **CECHY Z NISKĄ IDENTYCZNOŚCIĄ (<50%)**

| # | Cecha | Identyczność | Korelacja | RMSE (%) | Status |
|---|-------|--------------|-----------|----------|---------|
| 1 | **pressure_volume_corr** | **5.13%** | 0.9988 | 13.92% | ❌ **KRYTYCZNY** |
| 2 | **depth_price_corr** | **5.13%** | 0.9987 | 5.91% | ❌ **KRYTYCZNY** |
| 3 | **volume_price_correlation** | **32.16%** | 0.9999 | 6.89% | ❌ **PROBLEMATYCZNY** |
| 4 | **bollinger_band_width** | **32.60%** | 0.9999 | 0.57% | ❌ **PROBLEMATYCZNY** |
| 5 | **bb_width** | **32.63%** | 0.9999 | 0.57% | ❌ **PROBLEMATYCZNY** |
| 6 | **bb_position** | **32.64%** | 0.9999 | 121.92% | ❌ **PROBLEMATYCZNY** |

### 🟡 **CECHY ZE ŚREDNIĄ IDENTYCZNOŚCIĄ (50-90%)**

| # | Cecha | Identyczność | Korelacja | RMSE (%) | Status |
|---|-------|--------------|-----------|----------|---------|
| 7 | volume_volatility_rolling | 72.04% | 0.9999 | 0.82% | ⚠️ **ŚREDNI** |
| 8 | market_trend_direction | 76.74% | 0.9999 | 15.35% | ⚠️ **ŚREDNI** |
| 9 | ma_1440 | 80.20% | 1.0000 | 0.00% | ⚠️ **ŚREDNI** |
| 10 | ma_trend | 83.49% | 0.9999 | 0.34% | ⚠️ **ŚREDNI** |
| 11 | ma_60_to_ma_240 | 86.60% | 0.9999 | 0.00% | ⚠️ **ŚREDNI** |
| 12 | ma_240 | 89.08% | 1.0000 | 0.00% | ⚠️ **ŚREDNI** |
| 13 | price_to_ma_1440 | 82.15% | 0.9999 | 0.00% | ⚠️ **ŚREDNI** |
| 14 | price_to_ma_240 | 90.77% | 0.9999 | 0.00% | ⚠️ **ŚREDNI** |
| 15 | price_to_ma_60 | 92.80% | 0.9999 | 0.00% | ⚠️ **ŚREDNI** |
| 16 | price_vs_ma_240 | 89.08% | 0.9999 | 0.48% | ⚠️ **ŚREDNI** |
| 17 | price_vs_ma_60 | 91.51% | 0.9999 | 18.30% | ⚠️ **ŚREDNI** |
| 18 | volatility_momentum | 92.05% | 0.9999 | 1.24% | ⚠️ **ŚREDNI** |
| 19 | volatility_of_volatility | 92.05% | 0.9999 | 0.14% | ⚠️ **ŚREDNI** |
| 20 | volatility_term_structure | 92.31% | 0.9999 | 2.33% | ⚠️ **ŚREDNI** |

### 🟢 **CECHY Z WYSOKĄ IDENTYCZNOŚCIĄ (90-99%)**

| # | Cecha | Identyczność | Korelacja | RMSE (%) | Status |
|---|-------|--------------|-----------|----------|---------|
| 21 | ma_60 | 91.51% | 1.0000 | 0.00% | ✅ **DOBRY** |
| 22 | adx_14 | 99.69% | 0.9999 | 0.37% | ✅ **DOBRY** |
| 23 | market_trend_strength | 99.69% | 0.9999 | 0.37% | ✅ **DOBRY** |
| 24 | rsi_14 | 99.73% | 0.9999 | 0.23% | ✅ **DOBRY** |
| 25 | buy_sell_ratio_s1 | 99.74% | 0.9986 | 1.00% | ✅ **DOBRY** |
| 26 | buy_sell_ratio_s2 | 99.74% | 0.9986 | 1.02% | ✅ **DOBRY** |
| 27 | depth_ratio_s1 | 99.74% | 0.9986 | 1.00% | ✅ **DOBRY** |
| 28 | depth_ratio_s2 | 99.74% | 0.9986 | 1.02% | ✅ **DOBRY** |
| 29 | imbalance_s1 | 99.74% | 0.9991 | 134.83% | ✅ **DOBRY** |
| 30 | imbalance_s2 | 99.74% | 0.9991 | 187.71% | ✅ **DOBRY** |
| 31 | pressure_change | 99.74% | 0.9993 | 47.23% | ✅ **DOBRY** |
| 32 | snapshot1_depth_-1 | 99.74% | 0.9997 | 0.88% | ✅ **DOBRY** |
| 33 | snapshot1_depth_-2 | 99.74% | 0.9998 | 0.69% | ✅ **DOBRY** |
| 34 | snapshot1_depth_-3 | 99.74% | 0.9998 | 0.74% | ✅ **DOBRY** |
| 35 | snapshot1_depth_-4 | 99.74% | 0.9997 | 0.81% | ✅ **DOBRY** |
| 36 | snapshot1_depth_-5 | 99.74% | 0.9998 | 0.85% | ✅ **DOBRY** |
| 37 | snapshot1_depth_1 | 99.74% | 0.9998 | 0.90% | ✅ **DOBRY** |
| 38 | snapshot1_depth_2 | 99.74% | 0.9999 | 0.53% | ✅ **DOBRY** |
| 39 | snapshot1_depth_3 | 99.74% | 0.9999 | 0.44% | ✅ **DOBRY** |
| 40 | snapshot1_depth_4 | 99.74% | 0.9999 | 0.41% | ✅ **DOBRY** |
| 41 | snapshot1_depth_5 | 99.74% | 0.9999 | 0.42% | ✅ **DOBRY** |
| 42 | snapshot1_notional_-1 | 99.74% | 0.9998 | 0.90% | ✅ **DOBRY** |
| 43 | snapshot1_notional_-2 | 99.74% | 0.9998 | 0.70% | ✅ **DOBRY** |
| 44 | snapshot1_notional_-3 | 99.74% | 0.9997 | 0.70% | ✅ **DOBRY** |
| 45 | snapshot1_notional_-4 | 99.74% | 0.9997 | 0.68% | ✅ **DOBRY** |
| 46 | snapshot1_notional_-5 | 99.74% | 0.9997 | 0.74% | ✅ **DOBRY** |
| 47 | snapshot1_notional_1 | 99.74% | 0.9999 | 1.06% | ✅ **DOBRY** |
| 48 | snapshot1_notional_2 | 99.74% | 0.9999 | 0.66% | ✅ **DOBRY** |
| 49 | snapshot1_notional_3 | 99.74% | 0.9999 | 0.52% | ✅ **DOBRY** |
| 50 | snapshot1_notional_4 | 99.74% | 0.9999 | 0.53% | ✅ **DOBRY** |
| 51 | snapshot1_notional_5 | 99.74% | 0.9999 | 0.58% | ✅ **DOBRY** |
| 52 | snapshot2_depth_-1 | 99.74% | 0.9996 | 0.92% | ✅ **DOBRY** |
| 53 | snapshot2_depth_-2 | 99.74% | 0.9998 | 0.71% | ✅ **DOBRY** |
| 54 | snapshot2_depth_-3 | 99.74% | 0.9998 | 0.75% | ✅ **DOBRY** |
| 55 | snapshot2_depth_-4 | 99.74% | 0.9997 | 0.82% | ✅ **DOBRY** |
| 56 | snapshot2_depth_-5 | 99.74% | 0.9998 | 0.85% | ✅ **DOBRY** |
| 57 | snapshot2_depth_1 | 99.74% | 0.9997 | 0.93% | ✅ **DOBRY** |
| 58 | snapshot2_depth_2 | 99.74% | 0.9999 | 0.54% | ✅ **DOBRY** |
| 59 | snapshot2_depth_3 | 99.74% | 0.9999 | 0.44% | ✅ **DOBRY** |
| 60 | snapshot2_depth_4 | 99.74% | 0.9999 | 0.41% | ✅ **DOBRY** |
| 61 | snapshot2_depth_5 | 99.74% | 0.9999 | 0.42% | ✅ **DOBRY** |
| 62 | snapshot2_notional_-1 | 99.74% | 0.9998 | 0.91% | ✅ **DOBRY** |
| 63 | snapshot2_notional_-2 | 99.74% | 0.9998 | 0.71% | ✅ **DOBRY** |
| 64 | snapshot2_notional_-3 | 99.74% | 0.9997 | 0.70% | ✅ **DOBRY** |
| 65 | snapshot2_notional_-4 | 99.74% | 0.9997 | 0.69% | ✅ **DOBRY** |
| 66 | snapshot2_notional_-5 | 99.74% | 0.9997 | 0.74% | ✅ **DOBRY** |
| 67 | snapshot2_notional_1 | 99.74% | 0.9999 | 1.11% | ✅ **DOBRY** |
| 68 | snapshot2_notional_2 | 99.74% | 0.9999 | 0.68% | ✅ **DOBRY** |
| 69 | snapshot2_notional_3 | 99.74% | 0.9999 | 0.52% | ✅ **DOBRY** |
| 70 | snapshot2_notional_4 | 99.74% | 0.9999 | 0.53% | ✅ **DOBRY** |
| 71 | snapshot2_notional_5 | 99.74% | 0.9999 | 0.58% | ✅ **DOBRY** |
| 72 | spread | 99.74% | 0.9985 | 339.17% | ✅ **DOBRY** |
| 73 | depth_momentum | 99.75% | 0.9986 | 7.50% | ✅ **DOBRY** |
| 74 | spread_tightness | 99.75% | 0.9999 | 80.61% | ✅ **DOBRY** |
| 75 | macd_hist | 99.77% | 0.9999 | 46.55% | ✅ **DOBRY** |
| 76 | volume_momentum | 99.97% | 0.9990 | 53.42% | ✅ **DOBRY** |
| 77 | volatility_percentile | 99.97% | 0.9999 | 6.69% | ✅ **DOBRY** |
| 78 | volatility_persistence | 99.97% | 0.9999 | 0.10% | ✅ **DOBRY** |
| 79 | price_momentum | 99.99% | 0.9999 | 18.32% | ✅ **DOBRY** |
| 80 | price_trend_30m | 99.99% | 0.9999 | 30.49% | ✅ **DOBRY** |
| 81 | volume_change_norm | 99.99% | 0.9973 | 29.82% | ✅ **DOBRY** |
| 82 | price_consistency_score | 99.99% | 0.9999 | 0.92% | ✅ **DOBRY** |
| 83 | volume_intensity | 99.91% | 0.9982 | 0.72% | ✅ **DOBRY** |
| 84 | volume_trend_1h | 99.98% | 0.9990 | 2.62% | ✅ **DOBRY** |
| 85 | price_trend_2h | 99.98% | 0.9999 | 5.86% | ✅ **DOBRY** |
| 86 | price_trend_6h | 99.98% | 0.9999 | 1.22% | ✅ **DOBRY** |
| 87 | price_strength | 99.98% | 0.9999 | 1.23% | ✅ **DOBRY** |
| 88 | price_volatility_rolling | 99.98% | 0.9999 | 0.37% | ✅ **DOBRY** |
| 89 | volatility_regime | 99.99% | 0.9999 | 0.50% | ✅ **DOBRY** |
| 90 | market_regime | 100.00% | 0.9999 | 0.39% | ✅ **DOSKONAŁY** |
| 91 | market_choppiness | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 92 | order_flow_imbalance | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 93 | order_flow_trend | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 94 | price_pressure | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 95 | price_pressure_momentum | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 96 | volume_imbalance | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 97 | volume_imbalance_trend | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 98 | weighted_price_pressure | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |
| 99 | weighted_volume_imbalance | 100.00% | - | 0.00% | ✅ **DOSKONAŁY** |

## 📈 **PODSUMOWANIE STATYSTYK**

### **Rozkład identyczności:**
- **<50%:** 6 cech (5.9%) - **KRYTYCZNE**
- **50-90%:** 14 cech (13.9%) - **ŚREDNIE**
- **90-99%:** 69 cech (68.3%) - **DOBRE**
- **100%:** 12 cech (11.9%) - **DOSKONAŁE**

### **Najgorsze cechy (do naprawy):**
1. **pressure_volume_corr** (5.13%)
2. **depth_price_corr** (5.13%)
3. **volume_price_correlation** (32.16%)
4. **bollinger_band_width** (32.60%)
5. **bb_width** (32.63%)
6. **bb_position** (32.64%)

### **Najlepsze cechy:**
- 12 cech z 100% identycznością
- 69 cech z >90% identycznością

## 🎯 **WNIOSKI**

**Problem dotyczy głównie:**
1. **Korelacji** (pressure_volume_corr, depth_price_corr, volume_price_correlation)
2. **Bollinger Bands** (bb_width, bb_position, bollinger_band_width)

**Pozostałe 95 cech mają bardzo dobrą lub doskonałą zgodność.** 