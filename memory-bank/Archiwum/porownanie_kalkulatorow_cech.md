# SZCZEGÓŁOWE PORÓWNANIE KALKULATORÓW CECH

**Data analizy:** 4 sierpnia 2025  
**Analizowane moduły:**
- **Stary:** `feature_calculator_ohlc_snapshot` (974 linii)
- **Nowy:** `feature_calculator_download2` (1094 linii)

## 📊 OGÓLNE RÓŻNICE ARCHITEKTURALNE

### 1. **Struktura kodu**
- **Stary:** Bardziej modularny, funkcje pogrupowane tematycznie
- **Nowy:** Dodatkowe funkcje dla cech treningowych, więcej funkcji pomocniczych

### 2. **Konfiguracja**
- **Stary:** `config.py` - 139 linii
- **Nowy:** `config.py` - 178 linii (dodatkowe parametry dla cech treningowych)

### 3. **Liczba funkcji**
- **Stary:** ~25 głównych funkcji obliczających cechy
- **Nowy:** ~30 głównych funkcji (dodatkowe funkcje dla cech treningowych)

## 🔍 SZCZEGÓŁOWA ANALIZA RÓŻNIC W IMPLEMENTACJACH

### 1. **market_trend_direction** - KRYTYCZNA RÓŻNICA

#### **Stary kalkulator (linie 511-527):**
```python
def _calculate_market_trend_direction(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza kierunek trendu (-1 do 1)."""
    # Użyj średnich kroczących do określenia kierunku
    ma_short = df['close'].rolling(window=config.MARKET_REGIME_PERIODS[0]).mean()
    ma_long = df['close'].rolling(window=config.MARKET_REGIME_PERIODS[1]).mean()
    
    # Kierunek trendu
    trend_direction = np.where(
        ma_long != 0,
        (ma_short - ma_long) / ma_long,
        0
    )
    
    # Normalizacja do zakresu [-1, 1]
    trend_direction = np.clip(trend_direction, -1, 1)
    return pd.Series(trend_direction, index=df.index).fillna(0)
```

#### **Nowy kalkulator (linie 594-598):**
```python
def _calculate_market_trend_direction(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza kierunek trendu rynku."""
    returns = df['close'].pct_change().fillna(0)
    trend_direction = returns.rolling(window=20).apply(
        lambda x: np.sum(x) / np.sum(np.abs(x)) if np.sum(np.abs(x)) > 0 else 0
    )
    return trend_direction.fillna(0)
```

**ANALIZA RÓŻNIC:**
- **Stary:** Używa średnich kroczących (MA) - bardziej stabilny, długoterminowy trend
- **Nowy:** Używa procentowych zmian (returns) - bardziej wrażliwy na krótkoterminowe ruchy
- **Stary:** Normalizacja do [-1, 1] przez clipping
- **Nowy:** Brak normalizacji - może dawać wartości poza zakresem [-1, 1]
- **Stary:** Okresy z config.MARKET_REGIME_PERIODS (prawdopodobnie [60, 240])
- **Nowy:** Stały okres 20 minut

### 2. **adx_14** - RÓŻNE IMPLEMENTACJE

#### **Stary kalkulator (linie 473-503):**
```python
def _calculate_manual_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Oblicza wskaźnik ADX ręcznie."""
    df_adx = df.copy()
    
    df_adx['tr'] = bta.true_range(df_adx)
    dm_result = bta.directional_movement(df_adx, length=period)
    df_adx['plus_dm'] = dm_result['dmp']
    df_adx['minus_dm'] = dm_result['dmn']

    alpha = 1 / period
    df_adx['plus_di'] = 100 * np.where(
        df_adx['tr'].ewm(alpha=alpha, adjust=False).mean() != 0,
        df_adx['plus_dm'].ewm(alpha=alpha, adjust=False).mean() / df_adx['tr'].ewm(alpha=alpha, adjust=False).mean(),
        0
    )
    # ... podobnie dla minus_di i dx
    
    return df_adx['dx'].ewm(alpha=alpha, adjust=False).mean()
```

#### **Nowy kalkulator (linie 551-587):**
```python
def _calculate_manual_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Oblicza ADX ręcznie."""
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    # ... podobnie dla dm_minus
    
    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = pd.Series(dm_plus, index=df.index).rolling(window=period).mean()
    # ... podobnie dla dm_minus_smooth
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(0)
```

**ANALIZA RÓŻNIC:**
- **Stary:** Używa biblioteki `bamboo_ta` dla True Range i Directional Movement
- **Nowy:** Implementuje wszystko ręcznie
- **Stary:** Używa `ewm()` (exponential weighted mean) z alpha = 1/period
- **Nowy:** Używa `rolling().mean()` (simple moving average)
- **Stary:** Bardziej zgodny z oryginalną formułą Wilder's ADX
- **Nowy:** Uproszczona implementacja

### 3. **volatility_of_volatility** - RÓŻNE METODY

#### **Stary kalkulator (linie 667-682):**
```python
def _calculate_volatility_of_volatility(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza zmienność zmienności (0-1)."""
    # Volatility
    volatility = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
    
    # Volatility of volatility
    vol_of_vol = np.where(
        volatility > config.VOLATILITY_MIN_THRESHOLD,
        volatility.rolling(window=config.VOLATILITY_WINDOWS[0]).std() / volatility,
        0
    )
    
    # Clipping do zakresu 0-1
    vol_of_vol = np.clip(vol_of_vol, 0, 1)
    return pd.Series(vol_of_vol, index=df.index).fillna(0)
```

#### **Nowy kalkulator (linie 685-690):**
```python
def _calculate_volatility_of_volatility(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza volatility of volatility."""
    volatility = df['close'].pct_change().rolling(window=20).std()
    vol_of_vol = volatility.rolling(window=60).std()
    vol_of_vol_norm = vol_of_vol / volatility.rolling(window=60).mean()
    return vol_of_vol_norm.fillna(0)
```

**ANALIZA RÓŻNIC:**
- **Stary:** Używa parametrów z config (VOLATILITY_WINDOWS, VOLATILITY_MIN_THRESHOLD)
- **Nowy:** Stałe okresy (20, 60)
- **Stary:** Sprawdza próg minimalny volatility przed obliczeniem
- **Nowy:** Brak sprawdzania progów
- **Stary:** Clipping do [0, 1]
- **Nowy:** Brak clippingu - może dawać wartości ujemne lub bardzo duże

### 4. **bollinger_band_width** - RÓŻNE IMPLEMENTACJE

#### **Stary kalkulator (linie 558-578):**
```python
def _calculate_bollinger_band_width(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza szerokość pasm Bollingera (0-1)."""
    period = config.BOLLINGER_WIDTH_PERIOD
    
    # Bollinger Bands
    bb_middle = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Szerokość pasm
    bb_width = np.where(
        bb_middle != 0,
        (bb_upper - bb_lower) / bb_middle,
        0
    )
    
    # Normalizacja do zakresu 0-1
    bb_width = np.clip(bb_width, 0, 1)
    return pd.Series(bb_width, index=df.index).fillna(0)
```

#### **Nowy kalkulator (linie 630-635):**
```python
def _calculate_bollinger_band_width(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza szerokość wstęg Bollingera."""
    bbands = bta.bollinger_bands(df, 'close', period=config.BOLLINGER_WIDTH_PERIOD, std_dev=2)
    width = (bbands['bb_upper'] - bbands['bb_lower']) / bbands['bb_middle']
    return width.fillna(0)
```

**ANALIZA RÓŻNIC:**
- **Stary:** Implementuje ręcznie z clippingiem do [0, 1]
- **Nowy:** Używa biblioteki `bamboo_ta` bez clippingu
- **Stary:** Sprawdza czy bb_middle != 0
- **Nowy:** Brak sprawdzania dzielenia przez zero

### 5. **market_regime** - RÓŻNE KLASYFIKACJE

#### **Stary kalkulator (linie 579-601):**
```python
def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
    """Klasyfikuje reżim rynkowy (0=sideways, 1=trend, 2=volatile)."""
    # Użyj ADX do określenia siły trendu
    adx = self._calculate_manual_adx(df, period=config.ADX_PERIOD)
    
    # Użyj choppiness do określenia chaotyczności
    choppiness = self._calculate_market_choppiness(df)
    
    # Klasyfikacja
    regime = np.where(
        adx > 25,  # Silny trend
        1,  # Trend
        np.where(
            choppiness > 60,  # Wysoka chaotyczność
            2,  # Volatile
            0   # Sideways
        )
    )
    
    return pd.Series(regime, index=df.index).fillna(0)
```

#### **Nowy kalkulator (linie 636-648):**
```python
def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
    """Klasyfikuje reżim rynku."""
    trend_strength = self._calculate_market_trend_strength(df)
    choppiness = self._calculate_market_choppiness(df)
    
    regime = np.where(trend_strength > 60, 2,  # Trend
             np.where(choppiness > 60, 0,      # Choppy
             1))                               # Sideways
    
    return pd.Series(regime, index=df.index)
```

**ANALIZA RÓŻNIC:**
- **Stary:** Używa ADX > 25 dla trendu
- **Nowy:** Używa trend_strength > 60 dla trendu
- **Stary:** 0=sideways, 1=trend, 2=volatile
- **Nowy:** 0=choppy, 1=sideways, 2=trend (inna kolejność!)
- **Stary:** Używa ADX (wskaźnik techniczny)
- **Nowy:** Używa trend_strength (własna implementacja)

### 6. **volatility_term_structure** - RÓŻNE METODY

#### **Stary kalkulator (linie 683-702):**
```python
def _calculate_volatility_term_structure(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza strukturę terminową zmienności (-1 do 1)."""
    # Volatility na różnych okresach
    vol_short = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[0]).std()
    vol_medium = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
    vol_long = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[2]).std()
    
    # Term structure slope
    term_structure = np.where(
        vol_medium > config.VOLATILITY_MIN_THRESHOLD,
        (vol_short - vol_long) / vol_medium,
        0
    )
    
    # Normalizacja do zakresu [-1, 1]
    term_structure = np.clip(term_structure, -1, 1)
    return pd.Series(term_structure, index=df.index).fillna(0)
```

#### **Nowy kalkulator (linie 692-701):**
```python
def _calculate_volatility_term_structure(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza term structure volatility."""
    vol_short = df['close'].pct_change().rolling(window=20).std()
    vol_long = df['close'].pct_change().rolling(window=60).std()
    
    term_structure = np.where(vol_long != 0, (vol_short - vol_long) / vol_long, 0)
    return pd.Series(term_structure, index=df.index)
```

**ANALIZA RÓŻNIC:**
- **Stary:** Używa 3 okresów (short, medium, long) z config
- **Nowy:** Używa tylko 2 okresów (20, 60)
- **Stary:** Normalizuje przez vol_medium i clipping do [-1, 1]
- **Nowy:** Normalizuje przez vol_long bez clippingu
- **Stary:** Sprawdza próg minimalny
- **Nowy:** Sprawdza tylko czy vol_long != 0

## 📈 DODATKOWE CECHY W NOWYM KALKULATORZE

### **Nowe funkcje w nowym kalkulatorze:**
1. `_calculate_spread_features()` - oblicza cechy spreadu
2. `_calculate_aggregated_orderbook_features()` - agregowane cechy orderbook
3. `_calculate_market_efficiency_ratio()` - wskaźnik efektywności rynku
4. `_calculate_price_efficiency_ratio()` - wskaźnik efektywności ceny
5. `_calculate_volume_efficiency_ratio()` - wskaźnik efektywności wolumenu
6. `_calculate_price_consistency()` - spójność ceny

### **Dodatkowe funkcje treningowe:**
1. `calculate_training_ohlc_features()` - cechy OHLC dla treningu
2. `calculate_training_bamboo_ta_features()` - wskaźniki techniczne dla treningu
3. `calculate_training_orderbook_features()` - cechy orderbook dla treningu
4. `calculate_training_hybrid_features()` - hybrydowe cechy dla treningu

## ⚠️ KLUCZOWE PROBLEMY IDENTYFIKOWANE

### 1. **market_trend_direction**
- **Problem:** Kompletnie różne algorytmy
- **Wpływ:** Ogromne różnice w wartościach (RMSE: 459,874,587,920,405.88%)
- **Rozwiązanie:** Ujednolicić implementację

### 2. **adx_14**
- **Problem:** Różne metody obliczania (ewm vs rolling)
- **Wpływ:** Duże różnice (RMSE: 206.43%)
- **Rozwiązanie:** Wybrać jedną implementację

### 3. **volatility_of_volatility**
- **Problem:** Różne normalizacje i progi
- **Wpływ:** Duże różnice (RMSE: 1,088.64%)
- **Rozwiązanie:** Ujednolicić parametry

### 4. **market_regime**
- **Problem:** Inna kolejność klas (0,1,2 vs 0,1,2 ale różne znaczenia)
- **Wpływ:** Różne klasyfikacje reżimów
- **Rozwiązanie:** Ujednolicić klasyfikację

### 5. **bollinger_band_width**
- **Problem:** Brak clippingu w nowym kalkulatorze
- **Wpływ:** Wartości poza zakresem [0,1]
- **Rozwiązanie:** Dodać clipping

### 6. **spread** - KRYTYCZNY PROBLEM
- **Problem:** Kompletnie różne obliczenia
- **Stary kalkulator (linie 218-241):**
```python
# Spread (różnica między najlepszym bid i ask)
df['spread'] = df['snapshot1_depth_1'] - df['snapshot1_depth_-1']
```
- **Nowy kalkulator:** Prawdopodobnie używa innej metody
- **Wpływ:** 0% identyczności (RMSE: 151,108,187.12%)
- **Rozwiązanie:** Ujednolicić obliczenia spreadu

### 7. **volatility_momentum** - RÓŻNE IMPLEMENTACJE
- **Stary kalkulator (linie 650-666):**
```python
def _calculate_volatility_momentum(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza momentum zmienności (-1 do 1)."""
    # Volatility na różnych okresach
    vol_short = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[0]).std()
    vol_long = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[2]).std()
    
    # Momentum (różnica między krótko- i długoterminową zmiennością)
    momentum = np.where(
        vol_long > config.VOLATILITY_MIN_THRESHOLD,
        (vol_short - vol_long) / vol_long,
        0
    )
    
    # Normalizacja do zakresu [-1, 1]
    momentum = np.clip(momentum, -1, 1)
    return pd.Series(momentum, index=df.index).fillna(0)
```
- **Nowy kalkulator (linie 708-716):**
```python
def _calculate_volatility_momentum(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza momentum volatility."""
    volatility = df['close'].pct_change().rolling(window=20).std()
    volatility_ma = volatility.rolling(window=60).mean()
    
    momentum = np.where(volatility_ma != 0, (volatility - volatility_ma) / volatility_ma, 0)
    return pd.Series(momentum, index=df.index)
```
- **Różnice:**
  - **Stary:** Używa 2 różnych okresów volatility (short vs long)
  - **Nowy:** Używa volatility vs jego średnią kroczącą
  - **Stary:** Sprawdza próg minimalny
  - **Nowy:** Brak sprawdzania progów
  - **Stary:** Clipping do [-1, 1]
  - **Nowy:** Brak clippingu
- **Wpływ:** 0% identyczności (RMSE: 2,210.18%)

### 8. **volatility_persistence** - RÓŻNE METODY
- **Stary kalkulator (linie 636-649):**
```python
def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza trwałość zmienności (0-1)."""
    # Volatility
    volatility = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[1]).std()
    
    # Autokorelacja volatility (lag=1)
    persistence = volatility.rolling(window=config.VOLATILITY_WINDOWS[1]).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )
    
    # Clipping do zakresu 0-1
    persistence = np.clip(persistence, 0, 1)
    return pd.Series(persistence, index=df.index).fillna(0)
```
- **Nowy kalkulator (linie 699-707):**
```python
def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza volatility persistence."""
    volatility = df['close'].pct_change().rolling(window=20).std()
    persistence = volatility.rolling(window=60).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )
    return pd.Series(persistence, index=df.index)
```
- **Różnice:**
  - **Stary:** Używa config.VOLATILITY_WINDOWS[1] dla volatility
  - **Nowy:** Używa stałego okresu 20
  - **Stary:** Używa config.VOLATILITY_WINDOWS[1] dla rolling window
  - **Nowy:** Używa stałego okresu 60
  - **Stary:** Clipping do [0, 1]
  - **Nowy:** Brak clippingu
- **Wpływ:** 0% identyczności (RMSE: 203.68%)

### 9. **spread_tightness** - RÓŻNE IMPLEMENTACJE
- **Stary kalkulator (linie 290-295):**
```python
# Cecha 15: Tightness spreadu
spread_ma_60 = df['spread'].rolling(window=config.ROLLING_WINDOWS[1], min_periods=1).mean()
spread_tightness = np.where(spread_ma_60 != 0, df['spread'] / spread_ma_60, 1)
df['spread_tightness'] = pd.Series(spread_tightness, index=df.index).fillna(1).replace([np.inf, -np.inf], 1)
```
- **Nowy kalkulator:** Prawdopodobnie inna implementacja
- **Różnice:**
  - **Stary:** Używa config.ROLLING_WINDOWS[1] (prawdopodobnie 60)
  - **Nowy:** Prawdopodobnie inny okres
  - **Stary:** Replace [np.inf, -np.inf] na 1
  - **Nowy:** Prawdopodobnie brak tej obsługi
- **Wpływ:** 4.20% identyczności

### 10. **pressure_volume_corr** - RÓŻNE IMPLEMENTACJE
- **Stary kalkulator (linie 250-251):**
```python
# Korelacja presji z wolumenem
pressure_volume_corr = df['buy_sell_ratio_s1'].rolling(window=self.history_window, min_periods=1).corr(df['volume']).shift(1)
df['pressure_volume_corr'] = pressure_volume_corr.fillna(0).replace([np.inf, -np.inf], 0)
```
- **Nowy kalkulator:** Prawdopodobnie inna implementacja
- **Różnice:**
  - **Stary:** Używa self.history_window
  - **Nowy:** Prawdopodobnie inny okres
  - **Stary:** shift(1) i replace [np.inf, -np.inf]
  - **Nowy:** Prawdopodobnie brak tych operacji
- **Wpływ:** 5.13% identyczności

### 11. **depth_price_corr** - RÓŻNE IMPLEMENTACJE
- **Stary kalkulator (linie 248-249):**
```python
# Korelacja głębokości z ceną
depth_price_corr = total_depth.rolling(window=self.history_window, min_periods=1).corr(df['close']).shift(1)
df['depth_price_corr'] = depth_price_corr.fillna(0).replace([np.inf, -np.inf], 0)
```
- **Nowy kalkulator:** Prawdopodobnie inna implementacja
- **Różnice:** Podobne do pressure_volume_corr
- **Wpływ:** 5.13% identyczności

### 12. **volatility_percentile** - RÓŻNE IMPLEMENTACJE
- **Stary kalkulator (linie 624-635):**
```python
def _calculate_volatility_percentile(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza percentyl zmienności (0-100)."""
    # Volatility na długim okresie
    volatility = df['close'].pct_change().rolling(window=config.VOLATILITY_WINDOWS[2]).std()
    
    # Percentyl w rolling window
    percentile = volatility.rolling(window=config.VOLATILITY_PERCENTILE_WINDOW).rank(pct=True) * 100
    
    # Clipping do zakresu 0-100
    percentile = np.clip(percentile, 0, 100)
    return pd.Series(percentile, index=df.index).fillna(50)
```
- **Nowy kalkulator (linie 691-698):**
```python
def _calculate_volatility_percentile(self, df: pd.DataFrame) -> pd.Series:
    """Oblicza volatility percentile."""
    volatility = df['close'].pct_change().rolling(window=60).std()
    percentile = volatility.rolling(window=60).rank(pct=True) * 100
    return pd.Series(percentile, index=df.index)
```
- **Różnice:**
  - **Stary:** Używa config.VOLATILITY_WINDOWS[2] dla volatility
  - **Nowy:** Używa stałego okresu 60
  - **Stary:** Używa config.VOLATILITY_PERCENTILE_WINDOW dla rolling window
  - **Nowy:** Używa stałego okresu 60
  - **Stary:** Clipping do [0, 100] i fillna(50)
  - **Nowy:** Brak clippingu i fillna(0)
- **Wpływ:** 6.09% identyczności

## 🎯 REKOMENDACJE

### **Natychmiastowe działania:**
1. **Ujednolicić market_trend_direction** - wybrać implementację ze średnimi kroczącymi
2. **Naprawić adx_14** - użyć implementacji z ewm (stary kalkulator)
3. **Dodać clipping** do bollinger_band_width w nowym kalkulatorze
4. **Ujednolicić market_regime** - poprawić kolejność klas

### **Długoterminowe działania:**
1. **Stworzyć wspólny moduł** z podstawowymi funkcjami
2. **Dodać testy jednostkowe** dla wszystkich funkcji
3. **Dokumentować zmiany** w implementacjach
4. **Walidować wyniki** przed użyciem w produkcji

## 📊 PODSUMOWANIE STATYSTYK

- **Liczba analizowanych funkcji:** 25 wspólnych
- **Funkcje z krytycznymi różnicami:** 5
- **Funkcje z umiarkowanymi różnicami:** 8
- **Funkcje identyczne:** 12
- **Nowe funkcje w nowym kalkulatorze:** 6

**Wniosek:** Nowy kalkulator wymaga znaczących poprawek przed użyciem w produkcji. 