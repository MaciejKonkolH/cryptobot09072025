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
    val_aligned = val_df[val_df['timestamp'].isin(common_timestamps)].sort_values('timestamp').reset_index(drop=True)
    ft_aligned = ft_df[ft_df['timestamp'].isin(common_timestamps)].sort_values('timestamp').reset_index(drop=True)
    
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
        
        # Znajdź indeksy różniących się wierszy
        different_indices = np.where(~np.isclose(val_values, ft_values, rtol=1e-10, atol=1e-10))[0]
        
        print(f"\n{feature}:")
        print(f"  Identyczne: {identical_count:,}/{total_count:,} ({identical_percent:.2f}%)")
        print(f"  Max różnica: {max_diff:.6f}")
        print(f"  Średnia różnica: {mean_diff:.6f}")
        
        if len(different_indices) > 0:
            print(f"  Różniące się wiersze (pierwsze 5):")
            for i, idx in enumerate(different_indices[:5]):
                timestamp = val_aligned.iloc[idx]['timestamp']
                val_val = val_values[idx]
                ft_val = ft_values[idx]
                diff_val = abs(val_val - ft_val)
                print(f"    [{idx}] {timestamp} | Val: {val_val:.6f} | FT: {ft_val:.6f} | Diff: {diff_val:.6f}")
            
            if len(different_indices) > 5:
                print(f"    ... i {len(different_indices) - 5} więcej")
        else:
            print(f"  Przykłady (pierwsze 3):")
            for i in range(min(3, len(val_values))):
                print(f"    [{i}] Val: {val_values[i]:.6f} | FT: {ft_values[i]:.6f}")
        
        if identical_percent == 100:
            identical_features.append(feature)
            print(f"  Status: ✅ IDENTYCZNE")
        elif identical_percent > 99.9:
            different_features.append(feature)
            print(f"  Status: 🟡 PRAWIE IDENTYCZNE ({identical_percent:.4f}%)")
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
    
    print(f"\n🟡 Różne features ({len(different_features)}):")
    for f in different_features:
        print(f"   - {f}")
    
    # Diagnoza
    print(f"\n🔍 DIAGNOZA:")
    if len(different_features) == 0:
        print("  🎉 Wszystkie features są identyczne!")
    elif all(any(f in feat for feat in different_features) for f in ['high_change', 'low_change', 'close_change', 'volume_change']):
        if all(feat in [f['feature'] if isinstance(f, dict) else f for f in different_features] for feat in ['high_change', 'low_change', 'close_change', 'volume_change']):
            print("  🎯 Problem: 4 'change' features mają minimalne różnice")
            print("  💡 Prawdopodobnie problem z pierwszym timestampem lub edge case")
        else:
            print("  🔧 Częściowy problem z change features")
    else:
        print("  ⚠️  Mieszane wyniki")

if __name__ == "__main__":
    compare_features()
