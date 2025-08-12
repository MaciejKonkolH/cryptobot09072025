# OSTATECZNA LISTA CECH DO OBLICZANIA - FEATURE_CALCULATOR_DOWNLOAD2

## 📊 **PODSUMOWANIE**
- **Łączna liczba cech do obliczania: 71**
- **Surowe dane OHLC i orderbook: ZOSTAJĄ** (nie są obliczane, tylko wczytywane)
- **Cechy bezwzględne: NIE OBLICZANE** (celowo pominięte)

---

## ✅ **CECHY DO OBLICZANIA (71 cech)**

### **1. CECHY Z TRAINING3 (37 cech) - PODSTAWOWE**

#### **Cechy trendu ceny (5 cech) - WZGLĘDNE**
- `price_trend_30m` - trend ceny 30-minutowy
- `price_trend_2h` - trend ceny 2-godzinny  
- `price_trend_6h` - trend ceny 6-godzinny
- `price_strength` - siła trendu ceny
- `price_consistency_score` - spójność trendu ceny

#### **Cechy pozycji ceny (4 cechy) - WZGLĘDNE**
- `price_vs_ma_60` - stosunek ceny do MA60
- `price_vs_ma_240` - stosunek ceny do MA240
- `ma_trend` - trend średnich kroczących
- `price_volatility_rolling` - zmienność ceny (rolling)

#### **Cechy volume (5 cech) - WZGLĘDNE**
- `volume_trend_1h` - trend wolumenu 1-godzinny
- `volume_intensity` - intensywność wolumenu
- `volume_volatility_rolling` - zmienność wolumenu (rolling)
- `volume_price_correlation` - korelacja wolumenu z ceną
- `volume_momentum` - momentum wolumenu

#### **Cechy orderbook (4 cechy) - WZGLĘDNE**
- `spread_tightness` - ciasność spreadu
- `depth_ratio_s1` - stosunek głębokości snapshot1
- `depth_ratio_s2` - stosunek głębokości snapshot2
- `depth_momentum` - momentum głębokości

#### **Market regime (5 cech) - ZAAWANSOWANE**
- `market_trend_strength` - siła trendu rynku
- `market_trend_direction` - kierunek trendu rynku
- `market_choppiness` - chaotyczność rynku
- `bollinger_band_width` - szerokość wstęg Bollingera
- `market_regime` - reżim rynku

#### **Volatility clustering (6 cech) - ZAAWANSOWANE**
- `volatility_regime` - reżim zmienności
- `volatility_percentile` - percentyl zmienności
- `volatility_persistence` - trwałość zmienności
- `volatility_momentum` - momentum zmienności
- `volatility_of_volatility` - zmienność zmienności
- `volatility_term_structure` - struktura terminowa zmienności

#### **Order book imbalance (8 cech) - ZAAWANSOWANE**
- `volume_imbalance` - nierównowaga wolumenu
- `weighted_volume_imbalance` - ważona nierównowaga wolumenu
- `volume_imbalance_trend` - trend nierównowagi wolumenu
- `price_pressure` - presja cenowa
- `weighted_price_pressure` - ważona presja cenowa
- `price_pressure_momentum` - momentum presji cenowej
- `order_flow_imbalance` - nierównowaga przepływu zleceń
- `order_flow_trend` - trend przepływu zleceń

### **2. DODATKOWE CECHY WZGLĘDNE (34 cechy) - DO EKSPERYMENTÓW**

#### **Dodatkowe cechy OHLC (12 cech) - WZGLĘDNE**
- `bb_width` - szerokość wstęg Bollingera (względna)
- `bb_position` - pozycja w wstęgach Bollingera (względna)
- `rsi_14` - RSI 14-okresowy
- `macd_hist` - histogram MACD
- `adx_14` - ADX 14-okresowy
- `price_to_ma_60` - stosunek ceny do MA60
- `price_to_ma_240` - stosunek ceny do MA240
- `ma_60_to_ma_240` - stosunek MA60 do MA240
- `price_to_ma_1440` - stosunek ceny do MA1440
- `volume_change_norm` - znormalizowana zmiana wolumenu
- `upper_wick_ratio_5m` - stosunek górnego knota (5-minutowy)
- `lower_wick_ratio_5m` - stosunek dolnego knota (5-minutowy)

#### **Dodatkowe cechy bamboo_ta (6 cech) - WZGLĘDNE**
- `stoch_k` - Stochastic %K
- `stoch_d` - Stochastic %D
- `cci` - Commodity Channel Index
- `williams_r` - Williams %R
- `mfi` - Money Flow Index
- `trange` - True Range

#### **Dodatkowe cechy orderbook (6 cech) - WZGLĘDNE**
- `buy_sell_ratio_s1` - stosunek kupujących do sprzedających (snapshot1)
- `buy_sell_ratio_s2` - stosunek kupujących do sprzedających (snapshot2)
- `imbalance_s1` - nierównowaga orderbook (snapshot1)
- `imbalance_s2` - nierównowaga orderbook (snapshot2)
- `spread_pct` - spread w procentach
- `price_imbalance` - nierównowaga cenowa

#### **Cechy hybrydowe (10 cech) - WZGLĘDNE**
- `market_microstructure_score` - skorupa mikrostruktury rynku
- `liquidity_score` - skorupa płynności
- `depth_price_corr` - korelacja głębokości z ceną
- `pressure_volume_corr` - korelacja presji z wolumenem
- `hour_of_day` - godzina dnia (0-23)
- `day_of_week` - dzień tygodnia (0-6)
- `price_momentum` - momentum ceny
- `market_efficiency_ratio` - wskaźnik efektywności rynku
- `price_efficiency_ratio` - wskaźnik efektywności ceny
- `volume_efficiency_ratio` - wskaźnik efektywności wolumenu

---

## 📋 **SUROWE DANE (ZOSTAJĄ W DATAFRAME)**

### **Dane OHLC (6 kolumn)**
- `timestamp` - znacznik czasu
- `open` - cena otwarcia
- `high` - cena maksymalna
- `low` - cena minimalna
- `close` - cena zamknięcia
- `volume` - wolumen

### **Dane orderbook (48 kolumn)**
- `snapshot1_timestamp`, `snapshot2_timestamp` - znaczniki czasu
- `snapshot1_depth_-5` do `snapshot1_depth_5` - głębokości snapshot1
- `snapshot1_notional_-5` do `snapshot1_notional_5` - notional snapshot1
- `snapshot2_depth_-5` do `snapshot2_depth_5` - głębokości snapshot2
- `snapshot2_notional_-5` do `snapshot2_notional_5` - notional snapshot2

---

## ❌ **CECHY NIE OBLICZANE (CELOWO POMINIĘTE)**

### **Cechy bamboo_ta bezwzględne**
- `obv` - On Balance Volume (bezwzględny)
- `vwap` - Volume Weighted Average Price (bezwzględny)
- `bbands_upper`, `bbands_middle`, `bbands_lower` - Bollinger Bands (bezwzględne)

### **Średnie kroczące bezwzględne**
- `ma_60`, `ma_240`, `ma_1440` - obliczane jako wartości pośrednie, ale nie zapisywane

### **Cechy orderbook bezwzględne**
- `total_bid_volume`, `total_ask_volume`, `total_volume`
- `bid_price_s*`, `ask_price_s*`
- `bid_volume_s*`, `ask_volume_s*`
- `spread` (bezwzględny)
- `pressure_change`, `depth_velocity`
- `tp_*_depth_s1`, `sl_*_depth_s1`

---

## 🎯 **PODSUMOWANIE FINALNE**

### **Łączna liczba kolumn w DataFrame: ~77**
- **71 cech obliczanych** (wszystkie względne)
- **6 kolumn surowych OHLC** (wczytywane)
- **48 kolumn surowych orderbook** (wczytywane)

### **Rekomendacja:**
- **Do treningu podstawowego:** Używać 37 cech z training3
- **Do eksperymentów:** Dodać dodatkowe 34 cechy względne
- **Surowe dane:** Zostają do analizy i debugowania
- **Bezwzględne cechy:** Nie obliczane (celowo pominięte)

---

## 📋 **LISTA DO SKOPIOWANIA (71 cech)**

```python
TRAINING_FEATURES_EXTENDED = [
    # Cechy z training3 (37)
    'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
    'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
    'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
    'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
    'market_trend_strength', 'market_trend_direction', 'market_choppiness', 'bollinger_band_width', 'market_regime',
    'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
    'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend', 'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend',
    
    # Dodatkowe cechy względne (34)
    'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'adx_14',
    'price_to_ma_60', 'price_to_ma_240', 'ma_60_to_ma_240', 'price_to_ma_1440',
    'volume_change_norm', 'upper_wick_ratio_5m', 'lower_wick_ratio_5m',
    'stoch_k', 'stoch_d', 'cci', 'williams_r', 'mfi', 'trange',
    'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'imbalance_s1', 'imbalance_s2',
    'spread_pct', 'price_imbalance',
    'market_microstructure_score', 'liquidity_score', 'depth_price_corr',
    'pressure_volume_corr', 'hour_of_day', 'day_of_week', 'price_momentum',
    'market_efficiency_ratio', 'price_efficiency_ratio', 'volume_efficiency_ratio'
]
``` 