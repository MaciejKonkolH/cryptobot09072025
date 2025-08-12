### 3. ALGORYTM OBLICZANIA FEATURES
```
OBLICZANIE 8 FEATURES TECHNICZNYCH
├── OBLICZ ZMIANY PROCENTOWE (3 features)
│   ├── high_change = (high[t] - close[t-1]) / close[t-1] * 100
│   ├── low_change = (low[t] - close[t-1]) / close[t-1] * 100
│   └── close_change = (close[t] - close[t-1]) / close[t-1] * 100
├── OBLICZ ŚREDNIE KROCZĄCE (na dostępnych danych)
│   ├── MA_1440 (1 dzień = 1440 minut):
│   │   ├── Świeca 1: MA = close[1]
│   │   ├── Świeca 2: MA = (close[1] + close[2]) / 2
│   │   ├── ...
│   │   └── Świeca 1440+: MA = mean(close[t-1439:t+1])
│   └── MA_43200 (30 dni = 43200 minut):
│       ├── Świeca 1: MA = close[1]  
│       ├── Świeca 2: MA = (close[1] + close[2]) / 2
│       ├── ...
│       └── Świeca 43200+: MA = mean(close[t-43199:t+1])
├── OBLICZ STOSUNKI DO MA (2 features)
│   ├── price_to_ma1440 = close[t] / MA_1440[t]
│   └── price_to_ma43200 = close[t] / MA_43200[t]
├── OBLICZ VOLUME FEATURES (3 features)
│   ├── MA_volume_1440 = średnia krocząca volume (okno 1440)
│   ├── MA_volume_43200 = średnia krocząca volume (okno 43200)
│   ├── volume_to_ma1440 = volume[t] / MA_volume_1440[t]
│   ├── volume_to_ma43200 = volume[t] / MA_volume_43200[t]
│   └── volume_change = (volume[t] - volume[t-1]) / volume[t-1] * 100
└── ZWRÓĆ DataFrame z 8 kolumnami features