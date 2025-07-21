#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print('🔍 Ładuję pliki do porównania...')

# Ładuj pliki
validation_file = Path('validation_and_labeling/output/BTCUSDT_TF-1m__FW-120__SL-050__TP-100__single_label.feather')
freqtrade_file = Path('ft_bot_clean/user_data/debug_sequences/freqtrade_features_UNSCALED_BTC_USDT_USDT_20250703_235629.feather')

print(f'📄 Validation file: {validation_file.name}')
print(f'📄 FreqTrade file: {freqtrade_file.name}')

# Załaduj dane
val_df = pd.read_feather(validation_file)
ft_df = pd.read_feather(freqtrade_file)

print(f'📊 Validation: {len(val_df)} wierszy, {len(val_df.columns)} kolumn')
print(f'📊 FreqTrade: {len(ft_df)} wierszy, {len(ft_df.columns)} kolumn')

print('\n📅 Zakresy dat:')
print(f'Validation: {val_df["timestamp"].min()} - {val_df["timestamp"].max()}')
print(f'FreqTrade: {ft_df["timestamp"].min()} - {ft_df["timestamp"].max()}')

print('\n🔧 Kolumny:')
print(f'Validation: {list(val_df.columns)}')
print(f'FreqTrade: {list(ft_df.columns)}')

# Konwertuj timestampy
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
ft_df['timestamp'] = pd.to_datetime(ft_df['timestamp'])

# Znajdź wspólne timestampy
common_timestamps = set(val_df['timestamp']).intersection(set(ft_df['timestamp']))
print(f'\n📊 Wspólne timestampy: {len(common_timestamps)}')

if len(common_timestamps) > 0:
    # Filtruj do wspólnych timestampów
    val_aligned = val_df[val_df['timestamp'].isin(common_timestamps)].copy()
    ft_aligned = ft_df[ft_df['timestamp'].isin(common_timestamps)].copy()
    
    # Sortuj po timestamp
    val_aligned = val_aligned.sort_values('timestamp').reset_index(drop=True)
    ft_aligned = ft_aligned.sort_values('timestamp').reset_index(drop=True)
    
    print(f'✅ Wyrównano do {len(val_aligned)} wierszy')
    
    # Porównaj features
    expected_features = [
        'high_change', 'low_change', 'close_change', 'volume_change',
        'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200'
    ]
    
    print('\n🔍 Porównanie features:')
    for feature in expected_features:
        if feature in val_aligned.columns and feature in ft_aligned.columns:
            val_values = val_aligned[feature].values
            ft_values = ft_aligned[feature].values
            
            identical = np.allclose(val_values, ft_values, rtol=1e-10)
            close = np.allclose(val_values, ft_values, rtol=1e-6)
            max_diff = np.max(np.abs(val_values - ft_values))
            correlation = np.corrcoef(val_values, ft_values)[0, 1]
            
            print(f'  📊 {feature}:')
            print(f'    🎯 Identical: {identical}')
            print(f'    🎯 Close (1e-6): {close}')
            print(f'    📈 Correlation: {correlation:.6f}')
            print(f'    📊 Max diff: {max_diff:.8f}')
        else:
            print(f'  ⚠️ {feature}: NOT FOUND')
    
    # Zapisz pierwsze 10 wierszy do porównania
    print('\n💾 Zapisuję przykładowe dane...')
    sample_data = []
    for i in range(min(10, len(val_aligned))):
        row = {'timestamp': val_aligned.iloc[i]['timestamp']}
        for feature in expected_features:
            if feature in val_aligned.columns and feature in ft_aligned.columns:
                row[f'val_{feature}'] = val_aligned.iloc[i][feature]
                row[f'ft_{feature}'] = ft_aligned.iloc[i][feature]
                row[f'diff_{feature}'] = val_aligned.iloc[i][feature] - ft_aligned.iloc[i][feature]
        sample_data.append(row)
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_comparison.csv', index=False)
    print('✅ Zapisano sample_comparison.csv')
else:
    print('❌ Brak wspólnych timestampów!') 