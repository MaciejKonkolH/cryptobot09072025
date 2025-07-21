#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

def compare_features():
    print("🔍 Feature Comparison - WORKING VERSION")
    print("="*60)
    
    # Ścieżki
    validation_path = Path('..') / 'validation_and_labeling' / 'output'
    freqtrade_path = Path('..') / 'ft_bot_clean' / 'user_data' / 'debug_sequences'
    
    # Znajdź pliki
    val_files = list(validation_path.glob('*BTCUSDT*single_label.feather'))
    ft_files = list(freqtrade_path.glob('freqtrade_features_UNSCALED_*.feather'))
    
    if not val_files or not ft_files:
        print("❌ Nie znaleziono plików!")
        return
    
    val_file = max(val_files, key=lambda x: x.stat().st_mtime)
    ft_file = max(ft_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 Validation: {val_file.name}")
    print(f"📁 FreqTrade: {ft_file.name}")
    
    # Wczytaj dane
    val_df = pd.read_feather(val_file)
    ft_df = pd.read_feather(ft_file)
    
    print(f"\n📊 Rozmiary:")
    print(f"  Validation: {len(val_df):,} wierszy")
    print(f"  FreqTrade: {len(ft_df):,} wierszy")
    
    # Dopasuj timestampy
    val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
    ft_df['timestamp'] = pd.to_datetime(ft_df['timestamp'])
    
    common_timestamps = set(val_df['timestamp']).intersection(set(ft_df['timestamp']))
    print(f"  Wspólne timestampy: {len(common_timestamps):,}")
    
    if len(common_timestamps) == 0:
        print("❌ Brak wspólnych timestampów!")
        return
    
    # Filtruj do wspólnych timestampów
    val_aligned = val_df[val_df['timestamp'].isin(common_timestamps)].sort_values('timestamp')
    ft_aligned = ft_df[ft_df['timestamp'].isin(common_timestamps)].sort_values('timestamp')
    
    # Features do porównania
    features = ['high_change', 'low_change', 'close_change', 'volume_change',
                'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200']
    
    print(f"\n🔍 PORÓWNANIE FEATURES:")
    print("="*60)
    
    identical_features = []
    different_features = []
    
    for feature in features:
        if feature not in val_aligned.columns or feature not in ft_aligned.columns:
            print(f"⚠️  {feature}: BRAK W JEDNYM Z PLIKÓW")
            continue
        
        val_values = val_aligned[feature].values
        ft_values = ft_aligned[feature].values
        
        # Sprawdź różnice
        diff = np.abs(val_values - ft_values)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Sprawdź ile jest identycznych
        identical_count = np.sum(np.isclose(val_values, ft_values, rtol=1e-10, atol=1e-10))
        total_count = len(val_values)
        identical_percent = (identical_count / total_count) * 100
        
        # Przykłady
        val_examples = val_values[:3]
        ft_examples = ft_values[:3]
        
        print(f"\n{feature}:")
        print(f"  Identyczne: {identical_count:,}/{total_count:,} ({identical_percent:.2f}%)")
        print(f"  Max różnica: {max_diff:.6f}")
        print(f"  Średnia różnica: {mean_diff:.6f}")
        print(f"  Przykłady - Val: {val_examples}")
        print(f"  Przykłady - FT:  {ft_examples}")
        
        if identical_percent == 100:
            identical_features.append(feature)
            print(f"  Status: ✅ IDENTYCZNE")
        else:
            different_features.append(feature)
            print(f"  Status: ❌ RÓŻNE")
    
    # Podsumowanie
    print(f"\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    
    print(f"✅ Identyczne features ({len(identical_features)}):")
    for f in identical_features:
        print(f"   - {f}")
    
    print(f"\n❌ Różne features ({len(different_features)}):")
    for f in different_features:
        print(f"   - {f}")
    
    # Diagnoza
    print(f"\n🔍 DIAGNOZA:")
    if len(different_features) == 0:
        print("  🎉 Wszystkie features są identyczne!")
    elif len(different_features) == 4 and all('change' in f for f in different_features):
        print("  🎯 Problem: 4 'change' features są różne")
        print("  💡 Prawdopodobnie FreqTrade nie ma '* 100'")
    else:
        print("  ⚠️  Mieszane wyniki")

if __name__ == "__main__":
    compare_features() 