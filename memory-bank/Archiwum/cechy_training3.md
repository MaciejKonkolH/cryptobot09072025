# CECHY UŻYWANE PRZEZ MODUŁ TRENING3

## 📊 **DOKŁADNA LISTA CECH (37 cech)**

### **1. CECHY TRENDU CENY (5 cech) - WZGLĘDNE**
- `price_trend_30m`
- `price_trend_2h` 
- `price_trend_6h`
- `price_strength`
- `price_consistency_score`

### **2. CECHY POZYCJI CENY (4 cechy) - WZGLĘDNE**
- `price_vs_ma_60`
- `price_vs_ma_240`
- `ma_trend`
- `price_volatility_rolling`

### **3. CECHY VOLUME (5 cech) - WZGLĘDNE**
- `volume_trend_1h`
- `volume_intensity`
- `volume_volatility_rolling`
- `volume_price_correlation`
- `volume_momentum`

### **4. CECHY ORDERBOOK (4 cechy) - WZGLĘDNE**
- `spread_tightness`
- `depth_ratio_s1`
- `depth_ratio_s2`
- `depth_momentum`

### **5. MARKET REGIME (5 cech) - ZAAWANSOWANE**
- `market_trend_strength`
- `market_trend_direction`
- `market_choppiness`
- `bollinger_band_width`
- `market_regime`

### **6. VOLATILITY CLUSTERING (6 cech) - ZAAWANSOWANE**
- `volatility_regime`
- `volatility_percentile`
- `volatility_persistence`
- `volatility_momentum`
- `volatility_of_volatility`
- `volatility_term_structure`

### **7. ORDER BOOK IMBALANCE (8 cech) - ZAAWANSOWANE**
- `volume_imbalance`
- `weighted_volume_imbalance`
- `volume_imbalance_trend`
- `price_pressure`
- `weighted_price_pressure`
- `price_pressure_momentum`
- `order_flow_imbalance`
- `order_flow_trend`

## 📋 **PODZIAŁ NA GRUPY (wg config.py)**

### **relative_features (18 cech)**
```python
'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
'price_strength', 'price_consistency_score',
'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling',
'volume_price_correlation', 'volume_momentum',
'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum'
```

### **market_regime_features (5 cech)**
```python
'market_trend_strength', 'market_trend_direction', 'market_choppiness',
'bollinger_band_width', 'market_regime'
```

### **volatility_features (6 cech)**
```python
'volatility_regime', 'volatility_percentile', 'volatility_persistence',
'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure'
```

### **imbalance_features (8 cech)**
```python
'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
'order_flow_imbalance', 'order_flow_trend'
```

## 🎯 **KLUCZOWE INFORMACJE**

### **Źródło danych:**
- Plik: `labeler3/output/ohlc_orderbook_labeled_3class_fw120m_15levels.feather`
- Dane pochodzą z `feature_calculator_ohlc_snapshot` (nie z `feature_calculator_download2`!)

### **Liczba cech:**
- **Łącznie: 37 cech**
- **Względne: 18 cech**
- **Zaawansowane: 19 cech**

### **Typy cech:**
- **Względne (18):** Trendy, pozycje względne, korelacje
- **Zaawansowane (19):** Market regime, volatility clustering, order book imbalance

### **Ważne uwagi:**
1. Moduł `training3` używa cech z `feature_calculator_ohlc_snapshot`
2. NIE używa cech z `feature_calculator_download2`
3. Wszystkie cechy są względne lub zaawansowane - brak podstawowych cech OHLC
4. Brak cech z biblioteki `bamboo_ta`

## 🔍 **PORÓWNANIE Z FEATURE_CALCULATOR_DOWNLOAD2**

### **Cechy w training3 (37):**
- 18 cech względnych
- 19 cech zaawansowanych
- 0 cech podstawowych OHLC
- 0 cech bamboo_ta

### **Cechy w feature_calculator_download2 (153):**
- 15 cech OHLC
- 18 cech bamboo_ta
- 42 cech orderbook
- 8 cech hybrydowych
- 18 cech względnych
- 28 cech zaawansowanych

**WNIOSEK:** Moduł `training3` używa znacznie mniejszej liczby cech (37 vs 153) i są to głównie cechy względne i zaawansowane. 