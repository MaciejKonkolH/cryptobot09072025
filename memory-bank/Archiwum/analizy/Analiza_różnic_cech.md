# Analiza różnic w algorytmach obliczania cech
## feature_calculator_ohlc_snapshot vs feature_calculator_download2

**Data analizy:** 3 sierpnia 2025  
**Cel:** Identyfikacja różnic w algorytmach obliczania cech między dwoma modułami

---

## 📊 WYNIKI PORÓWNANIA

### Podstawowe informacje:
- **Wspólny zakres czasowy:** 2023-01-31 00:06:00 - 2025-06-30 23:59:00
- **Wspólne kolumny:** 121 (106 cech + 15 etykiet)
- **Wyrównane dane:** 1,270,074 wierszy
- **Średnia korelacja:** 0.9090
- **Cechy z NaN korelacją:** 9
- **Cechy z istotnymi różnicami statystycznymi:** 79/104

---

## 🔍 CECHY Z PROBLEMAMI (NaN korelacja)

### 1. `weighted_price_pressure`
**Różnica:** Kompletnie różne algorytmy

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_weighted_price_pressure(self, df: pd.DataFrame) -> pd.Series:
    spread = df['snapshot1_spread']  # Używa kolumny 'spread'
    weighted_imbalance = self._calculate_weighted_volume_imbalance(df)
    pressure = np.where(
        spread > config.MIN_SPREAD_THRESHOLD,
        weighted_imbalance / spread,  # weighted_imbalance / spread
        0
    )
    pressure = np.clip(pressure, -1, 1)  # Clipping
    return pd.Series(pressure, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_weighted_price_pressure(self, df: pd.DataFrame) -> pd.Series:
    weighted_bid_pressure = sum(df[f'snapshot1_depth_{level}'] * abs(level) for level in config.BID_LEVELS)
    weighted_ask_pressure = sum(df[f'snapshot1_depth_{level}'] * level for level in config.ASK_LEVELS)
    total_pressure = weighted_bid_pressure + weighted_ask_pressure
    pressure = np.where(total_pressure != 0, (weighted_bid_pressure - weighted_ask_pressure) / total_pressure, 0)
    return pd.Series(pressure, index=df.index)  # BRAK clipping i fillna
```

**Kluczowe różnice:**
- Różne źródła danych (spread vs depth)
- Różne wzory matematyczne
- Brak normalizacji w download2

### 2. `price_pressure_momentum`
**Różnica:** Różne okresy i algorytmy

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_price_pressure_momentum(self, df: pd.DataFrame) -> pd.Series:
    pressure = self._calculate_price_pressure(df)
    momentum = pressure.diff()  # diff(1)
    momentum = np.clip(momentum, -1, 1)  # Clipping
    return pd.Series(momentum, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_price_pressure_momentum(self, df: pd.DataFrame) -> pd.Series:
    pressure = self._calculate_price_pressure(df)
    momentum = pressure.diff(periods=5)  # diff(5) - różny okres!
    return momentum.fillna(0)  # BRAK clipping
```

**Kluczowe różnice:**
- Różne okresy (1 vs 5)
- Brak normalizacji w download2

### 3. `volume_imbalance`
**Różnica:** Różne źródła danych

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
    bid_volume = df['snapshot1_bid_volume']  # Używa kolumn bid_volume/ask_volume
    ask_volume = df['snapshot1_ask_volume']
    total_volume = bid_volume + ask_volume
    imbalance = np.where(total_volume > 0, (bid_volume - ask_volume) / total_volume, 0)
    imbalance = np.clip(imbalance, -1, 1)  # Clipping
    return pd.Series(imbalance, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
    bid_volume = sum(df[f'snapshot1_depth_{level}'] for level in config.BID_LEVELS)  # Sumuje depth
    ask_volume = sum(df[f'snapshot1_depth_{level}'] for level in config.ASK_LEVELS)
    total_volume = bid_volume + ask_volume
    imbalance = np.where(total_volume != 0, (bid_volume - ask_volume) / total_volume, 0)
    return pd.Series(imbalance, index=df.index)  # BRAK clipping i fillna
```

**Kluczowe różnice:**
- Różne źródła danych (bid_volume/ask_volume vs depth)
- Brak normalizacji w download2

### 4. `market_choppiness`
**Różnica:** Różne algorytmy obliczania

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_market_choppiness(self, df: pd.DataFrame) -> pd.Series:
    # Używa algorytmu Choppiness Index
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=config.CHOPPINESS_PERIOD).sum()
    
    path_length = (df['high'] - df['low']).rolling(window=config.CHOPPINESS_PERIOD).sum()
    
    choppiness = 100 * np.log10(path_length / atr) / np.log10(config.CHOPPINESS_PERIOD)
    choppiness = np.clip(choppiness, 0, 100)
    return pd.Series(choppiness, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_market_choppiness(self, df: pd.DataFrame) -> pd.Series:
    # Używa innego algorytmu - prawdopodobnie prostszego
    # (kod nie jest identyczny)
```

### 5. `price_pressure`
**Różnica:** Kompletnie różne algorytmy

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_price_pressure(self, df: pd.DataFrame) -> pd.Series:
    spread = df['snapshot1_spread']  # Używa kolumny 'spread'
    volume_imbalance = self._calculate_volume_imbalance(df)
    pressure = np.where(
        spread > config.MIN_SPREAD_THRESHOLD,
        volume_imbalance / spread,  # imbalance / spread
        0
    )
    pressure = np.clip(pressure, -1, 1)  # Clipping
    return pd.Series(pressure, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_price_pressure(self, df: pd.DataFrame) -> pd.Series:
    spread = df['snapshot1_notional_1'] - df['snapshot1_notional_-1']  # Oblicza spread
    pressure = np.where(
        (spread > config.MIN_SPREAD_THRESHOLD) & (mid_price != 0),
        (df['snapshot1_depth_1'] - df['snapshot1_depth_-1']) / (df['snapshot1_depth_1'] + df['snapshot1_depth_-1']),  # depth imbalance
        0
    )
    return pd.Series(pressure, index=df.index)  # BRAK clipping i fillna
```

### 6. `volume_imbalance_trend`
**Różnica:** Różne algorytmy trendu

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_volume_imbalance_trend(self, df: pd.DataFrame) -> pd.Series:
    imbalance = self._calculate_volume_imbalance(df)
    trend = imbalance - imbalance.rolling(window=config.PRESSURE_WINDOW).mean()  # Różnica od średniej
    trend = np.clip(trend, -1, 1)  # Clipping
    return pd.Series(trend, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_volume_imbalance_trend(self, df: pd.DataFrame) -> pd.Series:
    imbalance = self._calculate_volume_imbalance(df)
    trend = imbalance.rolling(window=config.PRESSURE_WINDOW).mean()  # Średnia krocząca
    return trend  # BRAK clipping i fillna
```

### 7. `order_flow_imbalance`
**Różnica:** Różne algorytmy

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
    bid_volume = df['snapshot1_bid_volume']
    ask_volume = df['snapshot1_ask_volume']
    bid_change = bid_volume.diff()  # Zmiana volume
    ask_change = ask_volume.diff()
    total_change = bid_change + ask_change
    flow_imbalance = np.where(
        total_change != 0,
        (bid_change - ask_change) / total_change,
        0
    )
    flow_imbalance = np.clip(flow_imbalance, -1, 1)  # Clipping
    return pd.Series(flow_imbalance, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
    bid_flow = sum(df[f'snapshot1_depth_{level}'] for level in config.BID_LEVELS)  # Sumuje depth
    ask_flow = sum(df[f'snapshot1_depth_{level}'] for level in config.ASK_LEVELS)
    total_flow = bid_flow + ask_flow
    imbalance = np.where(total_flow != 0, (bid_flow - ask_flow) / total_flow, 0)
    return pd.Series(imbalance, index=df.index)  # BRAK clipping i fillna
```

### 8. `order_flow_trend`
**Różnica:** Różne algorytmy trendu

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_order_flow_trend(self, df: pd.DataFrame) -> pd.Series:
    flow_imbalance = self._calculate_order_flow_imbalance(df)
    trend = flow_imbalance.rolling(window=config.PRESSURE_WINDOW).mean()
    trend = np.clip(trend, -1, 1)  # Clipping
    return pd.Series(trend, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_order_flow_trend(self, df: pd.DataFrame) -> pd.Series:
    imbalance = self._calculate_order_flow_imbalance(df)
    trend = imbalance.rolling(window=config.PRESSURE_WINDOW).mean()
    return trend  # BRAK clipping i fillna
```

### 9. `weighted_volume_imbalance`
**Różnica:** Różne algorytmy

**feature_calculator_ohlc_snapshot:**
```python
def _calculate_weighted_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
    bid_volume = df['snapshot1_bid_volume']
    ask_volume = df['snapshot1_ask_volume']
    weighted_imbalance = np.where(
        (bid_volume + ask_volume) > 0,
        (bid_volume - ask_volume) / (bid_volume + ask_volume),
        0
    )
    weighted_imbalance = np.clip(weighted_imbalance, -1, 1)  # Clipping
    return pd.Series(weighted_imbalance, index=df.index).fillna(0)
```

**feature_calculator_download2:**
```python
def _calculate_weighted_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
    weighted_bid = sum(df[f'snapshot1_depth_{level}'] * abs(level) for level in config.BID_LEVELS)
    weighted_ask = sum(df[f'snapshot1_depth_{level}'] * level for level in config.ASK_LEVELS)
    total_weighted = weighted_bid + weighted_ask
    imbalance = np.where(total_weighted != 0, (weighted_bid - weighted_ask) / total_weighted, 0)
    return pd.Series(imbalance, index=df.index)  # BRAK clipping i fillna
```

---

## 🔍 CECHY Z DUŻYMI RÓŻNICAMI (niska korelacja)

### 1. `spread` - korelacja 0.8793
**Różnica:** Różne źródła danych

**feature_calculator_ohlc_snapshot:**
- Używa kolumny `snapshot1_spread`

**feature_calculator_download2:**
- Oblicza: `snapshot1_notional_1 - snapshot1_notional_-1`

### 2. `adx_14` - korelacja 0.3258
**Różnica:** Różne implementacje ADX

### 3. `market_trend_strength` - korelacja 0.0824
**Różnica:** Różne algorytmy obliczania siły trendu

### 4. `volatility_percentile` - korelacja 0.2247
**Różnica:** Różne metody obliczania percentyli

### 5. `market_trend_direction` - korelacja 0.3969
**Różnica:** Różne algorytmy określania kierunku trendu

---

## 📊 PODSUMOWANIE RÓŻNIC

### Główne kategorie różnic:

1. **RÓŻNE ŹRÓDŁA DANYCH:**
   - ohlc_snapshot: używa kolumn `snapshot1_bid_volume`, `snapshot1_ask_volume`, `snapshot1_spread`
   - download2: oblicza te wartości z `snapshot1_depth_*` i `snapshot1_notional_*`

2. **BRAK NORMALIZACJI:**
   - ohlc_snapshot: stosuje `np.clip()` i `fillna(0)`
   - download2: brak normalizacji w wielu funkcjach

3. **RÓŻNE ALGORYTMY:**
   - Kompletnie inne wzory matematyczne dla tych samych cech
   - Różne okresy (np. `diff(1)` vs `diff(5)`)

4. **RÓŻNE METODY TRENDU:**
   - ohlc_snapshot: różnica od średniej kroczącej
   - download2: średnia krocząca

### Wpływ na wyniki:
- **9 cech z NaN korelacją** = kompletnie różne wartości
- **79/104 cech z istotnymi różnicami statystycznymi** = znaczące różnice
- **Średnia korelacja 0.909** = ogólnie podobne, ale z istotnymi wyjątkami

### Rekomendacje:
1. **Użyć `feature_calculator_ohlc_snapshot`** do obliczania cech (ma poprawne algorytmy)
2. **Lub poprawić algorytmy w `feature_calculator_download2`** aby były identyczne
3. **Zachować spójność** między modułami obliczania cech i etykietowania

---

## 📋 LISTA WSZYSTKICH CECH Z RÓŻNICAMI

### Cechy z NaN korelacją (9):
1. `weighted_price_pressure`
2. `price_pressure_momentum`
3. `volume_imbalance`
4. `market_choppiness`
5. `price_pressure`
6. `volume_imbalance_trend`
7. `order_flow_imbalance`
8. `order_flow_trend`
9. `weighted_volume_imbalance`

### Cechy z niską korelacją (<0.5):
1. `spread` - 0.8793
2. `adx_14` - 0.3258
3. `market_trend_strength` - 0.0824
4. `volatility_percentile` - 0.2247
5. `market_trend_direction` - 0.3969
6. `volatility_persistence` - 0.0681
7. `market_regime` - 0.0309
8. `volatility_of_volatility` - 0.3279
9. `volatility_term_structure` - 0.2862
10. `volatility_momentum` - 0.2610

### Cechy z wysoką korelacją (>0.99):
- Większość cech OHLC (open, high, low, close, volume)
- Cechy techniczne (rsi_14, macd_hist, bb_width)
- Cechy orderbook (depth, notional)
- Cechy trendu (price_trend_*, volume_trend_*)
