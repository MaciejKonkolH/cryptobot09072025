#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 FEATURE COMPARISON TOOL V1.0
===============================================
Porównanie 8 cech technicznych między:
- validation_and_labeling output
- FreqTrade features log

Autor: Crypto Trading System
Data: 2025-06-27
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ===== KONFIGURACJA =====
class Config:
    # Ścieżki plików
    VALIDATION_FILE = r"validation_and_labeling/output\BTCUSDT_TF-1m__FW-120__SL-050__TP-100__single_label.feather"
    FREQTRADE_FILE = r"ft_bot_clean/user_data/logs/features\features_BTC_USDT_USDT_20250628_150106.csv"
    
    # 8 cech do porównania
    FEATURES = [
        'high_change',
        'low_change', 
        'close_change',
        'volume_change',
        'price_to_ma1440',
        'price_to_ma43200',
        'volume_to_ma1440',
        'volume_to_ma43200'
    ]
    
    # Progi różnic
    THRESHOLDS = {
        'IDENTICAL': 1e-10,     # Praktycznie identyczne
        'MINOR': 0.01,          # < 1% różnicy
        'MODERATE': 0.05,       # 1-5% różnicy
        'MAJOR': 0.1,           # 5-10% różnicy
        'EXTREME': float('inf') # > 10% różnicy
    }
    
    # Kolumny czasowe
    TIMESTAMP_COLS = {
        'validation': 'timestamp',
        'freqtrade': 'timestamp'
    }

class FeatureComparator:
    """Główna klasa do porównywania cech"""
    
    def __init__(self):
        self.config = Config()
        self.validation_df = None
        self.freqtrade_df = None
        self.synchronized_df = None
        self.comparison_results = {}
        
    def _load_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        """Inteligentnie ładuje DataFrame na podstawie rozszerzenia pliku."""
        try:
            if file_path.endswith('.feather'):
                return pd.read_feather(file_path)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            else:
                print(f"❌ Nieobsługiwany format pliku: {file_path}")
                return None
        except Exception as e:
            print(f"❌ Błąd podczas ładowania pliku {file_path}: {e}")
            return None

    def load_data(self) -> bool:
        """Ładowanie danych z obu plików"""
        print("🔍 FEATURE COMPARISON TOOL V1.0")
        print("=" * 50)
        
        # Sprawdź czy pliki istnieją
        if not os.path.exists(self.config.VALIDATION_FILE):
            print(f"❌ Plik validation nie istnieje: {self.config.VALIDATION_FILE}")
            return False
            
        if not os.path.exists(self.config.FREQTRADE_FILE):
            print(f"❌ Plik FreqTrade nie istnieje: {self.config.FREQTRADE_FILE}")
            return False
        
        try:
            # Ładowanie validation data
            print(f"📁 Ładowanie validation data z: {self.config.VALIDATION_FILE}")
            self.validation_df = self._load_dataframe(self.config.VALIDATION_FILE)
            if self.validation_df is None:
                return False
            print(f"   ✅ Loaded: {len(self.validation_df):,} rows")
            print(f"   📊 Columns: {list(self.validation_df.columns)}")
            
            # Ładowanie FreqTrade data
            print(f"📁 Ładowanie FreqTrade data z: {self.config.FREQTRADE_FILE}")
            self.freqtrade_df = self._load_dataframe(self.config.FREQTRADE_FILE)
            if self.freqtrade_df is None:
                return False
            print(f"   ✅ Loaded: {len(self.freqtrade_df):,} rows")
            print(f"   📊 Columns: {list(self.freqtrade_df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas ładowania danych: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """Przetwarzanie i synchronizacja danych"""
        print("\n🔧 PREPROCESSING DATA...")
        
        try:
            # Konwersja timestampów
            print("📅 Konwersja timestampów...")
            
            # Validation timestamps
            if 'timestamp' in self.validation_df.columns:
                self.validation_df['timestamp'] = pd.to_datetime(self.validation_df['timestamp'])
                # Make timezone-naive if needed
                if self.validation_df['timestamp'].dt.tz is not None:
                    self.validation_df['timestamp'] = self.validation_df['timestamp'].dt.tz_localize(None)
            else:
                print("❌ Brak kolumny timestamp w validation data")
                return False
            
            # FreqTrade timestamps
            if 'timestamp' in self.freqtrade_df.columns:
                self.freqtrade_df['timestamp'] = pd.to_datetime(self.freqtrade_df['timestamp'])
                # Make timezone-naive if needed
                if self.freqtrade_df['timestamp'].dt.tz is not None:
                    self.freqtrade_df['timestamp'] = self.freqtrade_df['timestamp'].dt.tz_localize(None)
            else:
                print("❌ Brak kolumny timestamp w FreqTrade data")
                return False
            
            # Sprawdź zakresy dat
            val_start = self.validation_df['timestamp'].min()
            val_end = self.validation_df['timestamp'].max()
            ft_start = self.freqtrade_df['timestamp'].min()
            ft_end = self.freqtrade_df['timestamp'].max()
            
            print(f"📅 Validation range: {val_start} to {val_end}")
            print(f"📅 FreqTrade range: {ft_start} to {ft_end}")
            
            # Znajdź wspólny okres
            common_start = max(val_start, ft_start)
            common_end = min(val_end, ft_end)
            
            if common_start >= common_end:
                print("❌ Brak wspólnego okresu między plikami!")
                return False
            
            print(f"📅 Wspólny okres: {common_start} to {common_end}")
            
            # Filtruj do wspólnego okresu
            self.validation_df = self.validation_df[
                (self.validation_df['timestamp'] >= common_start) & 
                (self.validation_df['timestamp'] <= common_end)
            ].copy()
            
            self.freqtrade_df = self.freqtrade_df[
                (self.freqtrade_df['timestamp'] >= common_start) & 
                (self.freqtrade_df['timestamp'] <= common_end)
            ].copy()
            
            print(f"📊 Po filtrowaniu:")
            print(f"   Validation: {len(self.validation_df):,} rows")
            print(f"   FreqTrade: {len(self.freqtrade_df):,} rows")
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas preprocessingu: {e}")
            return False
    
    def synchronize_data(self) -> bool:
        """Synchronizacja danych po timestampie"""
        print("\n🔄 SYNCHRONIZING DATA...")
        
        try:
            # Merge po timestamp
            merged_df = pd.merge(
                self.validation_df,
                self.freqtrade_df,
                on='timestamp',
                how='inner',
                suffixes=('_val', '_ft')
            )
            
            if len(merged_df) == 0:
                print("❌ Brak wspólnych timestampów!")
                return False
            
            print(f"✅ Synchronized: {len(merged_df):,} rows")
            
            # Sprawdź dostępność cech
            missing_features = []
            for feature in self.config.FEATURES:
                val_col = f"{feature}_val"
                ft_col = f"{feature}_ft"
                
                if val_col not in merged_df.columns:
                    missing_features.append(f"{feature} (validation)")
                if ft_col not in merged_df.columns:
                    missing_features.append(f"{feature} (freqtrade)")
            
            if missing_features:
                print(f"⚠️ Brakujące cechy: {missing_features}")
                # Kontynuuj z dostępnymi cechami
                available_features = [
                    f for f in self.config.FEATURES 
                    if f"{f}_val" in merged_df.columns and f"{f}_ft" in merged_df.columns
                ]
                print(f"📊 Dostępne cechy: {available_features}")
                self.config.FEATURES = available_features
            
            self.synchronized_df = merged_df
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas synchronizacji: {e}")
            return False
    
    def calculate_differences(self) -> Dict:
        """Obliczanie różnic między cechami"""
        print("\n📊 CALCULATING DIFFERENCES...")
        
        results = {}
        
        for feature in self.config.FEATURES:
            print(f"🔍 Analyzing {feature}...")
            
            val_col = f"{feature}_val"
            ft_col = f"{feature}_ft"
            
            val_data = self.synchronized_df[val_col]
            ft_data = self.synchronized_df[ft_col]
            
            # Oblicz różnice
            abs_diff = np.abs(val_data - ft_data)
            rel_diff = np.abs((val_data - ft_data) / (val_data + 1e-10))  # Avoid division by zero
            
            # Statystyki
            stats = {
                'total_samples': len(val_data),
                'val_mean': val_data.mean(),
                'val_std': val_data.std(),
                'ft_mean': ft_data.mean(),
                'ft_std': ft_data.std(),
                'abs_diff_mean': abs_diff.mean(),
                'abs_diff_std': abs_diff.std(),
                'abs_diff_max': abs_diff.max(),
                'rel_diff_mean': rel_diff.mean(),
                'rel_diff_std': rel_diff.std(),
                'rel_diff_max': rel_diff.max(),
                'correlation': np.corrcoef(val_data, ft_data)[0, 1]
            }
            
            # Kategoryzacja różnic
            categories = {}
            for category, threshold in self.config.THRESHOLDS.items():
                if category == 'EXTREME':
                    mask = rel_diff > self.config.THRESHOLDS['MAJOR']
                else:
                    prev_threshold = 0
                    if category == 'MINOR':
                        prev_threshold = self.config.THRESHOLDS['IDENTICAL']
                    elif category == 'MODERATE':
                        prev_threshold = self.config.THRESHOLDS['MINOR']
                    elif category == 'MAJOR':
                        prev_threshold = self.config.THRESHOLDS['MODERATE']
                    
                    mask = (rel_diff > prev_threshold) & (rel_diff <= threshold)
                
                count = mask.sum()
                percentage = (count / len(rel_diff)) * 100
                categories[category] = {'count': count, 'percentage': percentage}
            
            # Znajdź największe różnice
            top_diffs_idx = abs_diff.nlargest(10).index
            top_differences = []
            for idx in top_diffs_idx:
                top_differences.append({
                    'timestamp': self.synchronized_df.loc[idx, 'timestamp'],
                    'val_value': val_data.loc[idx],
                    'ft_value': ft_data.loc[idx],
                    'abs_diff': abs_diff.loc[idx],
                    'rel_diff': rel_diff.loc[idx]
                })
            
            results[feature] = {
                'stats': stats,
                'categories': categories,
                'top_differences': top_differences
            }
        
        self.comparison_results = results
        return results
    
    def print_summary(self):
        """Wyświetlenie podsumowania wyników"""
        print("\n" + "=" * 60)
        print("📊 FEATURE COMPARISON SUMMARY")
        print("=" * 60)
        
        if not self.comparison_results:
            print("❌ Brak wyników do wyświetlenia")
            return
        
        # Tabela podsumowująca
        print(f"\n{'Feature':<20} {'Correlation':<12} {'Mean Diff':<12} {'Max Diff':<12} {'Major %':<10}")
        print("-" * 66)
        
        for feature, results in self.comparison_results.items():
            stats = results['stats']
            categories = results['categories']
            
            correlation = f"{stats['correlation']:.4f}"
            mean_diff = f"{stats['rel_diff_mean']:.4f}"
            max_diff = f"{stats['rel_diff_max']:.4f}"
            major_pct = f"{categories['MAJOR']['percentage']:.1f}%"
            
            print(f"{feature:<20} {correlation:<12} {mean_diff:<12} {max_diff:<12} {major_pct:<10}")
        
        # Szczegółowe wyniki dla każdej cechy
        for feature, results in self.comparison_results.items():
            print(f"\n🔍 DETAILED ANALYSIS: {feature}")
            print("-" * 40)
            
            stats = results['stats']
            categories = results['categories']
            
            print(f"📊 Basic Statistics:")
            print(f"   Total samples: {stats['total_samples']:,}")
            print(f"   Correlation: {stats['correlation']:.6f}")
            print(f"   Mean abs diff: {stats['abs_diff_mean']:.6f}")
            print(f"   Mean rel diff: {stats['rel_diff_mean']:.4%}")
            print(f"   Max rel diff: {stats['rel_diff_max']:.4%}")
            
            print(f"\n📈 Value Statistics:")
            print(f"   Validation - Mean: {stats['val_mean']:.6f}, Std: {stats['val_std']:.6f}")
            print(f"   FreqTrade  - Mean: {stats['ft_mean']:.6f}, Std: {stats['ft_std']:.6f}")
            
            print(f"\n🎯 Difference Categories:")
            for category, data in categories.items():
                print(f"   {category:<10}: {data['count']:>8,} ({data['percentage']:>5.1f}%)")
            
            print(f"\n🔥 Top 5 Largest Differences:")
            for i, diff in enumerate(results['top_differences'][:5], 1):
                timestamp = diff['timestamp']
                val_val = diff['val_value']
                ft_val = diff['ft_value']
                rel_diff = diff['rel_diff']
                
                print(f"   {i}. {timestamp}: {val_val:.6f} vs {ft_val:.6f} ({rel_diff:.2%} diff)")
    
    def save_detailed_report(self, output_file: str = None):
        """Zapisanie szczegółowego raportu do pliku"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"feature_comparison_report_{timestamp}.txt"
        
        print(f"\n💾 Saving detailed report to: {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("🔍 FEATURE COMPARISON DETAILED REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Validation file: {self.config.VALIDATION_FILE}\n")
                f.write(f"FreqTrade file: {self.config.FREQTRADE_FILE}\n")
                f.write(f"Synchronized samples: {len(self.synchronized_df):,}\n\n")
                
                # Summary table
                f.write("📊 SUMMARY TABLE\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Feature':<20} {'Correlation':<12} {'Mean Diff':<12} {'Max Diff':<12} {'Major %':<10}\n")
                f.write("-" * 66 + "\n")
                
                for feature, results in self.comparison_results.items():
                    stats = results['stats']
                    categories = results['categories']
                    
                    correlation = f"{stats['correlation']:.4f}"
                    mean_diff = f"{stats['rel_diff_mean']:.4f}"
                    max_diff = f"{stats['rel_diff_max']:.4f}"
                    major_pct = f"{categories['MAJOR']['percentage']:.1f}%"
                    
                    f.write(f"{feature:<20} {correlation:<12} {mean_diff:<12} {max_diff:<12} {major_pct:<10}\n")
                
                # Detailed analysis
                for feature, results in self.comparison_results.items():
                    f.write(f"\n🔍 DETAILED ANALYSIS: {feature}\n")
                    f.write("-" * 40 + "\n")
                    
                    stats = results['stats']
                    categories = results['categories']
                    
                    f.write(f"📊 Basic Statistics:\n")
                    f.write(f"   Total samples: {stats['total_samples']:,}\n")
                    f.write(f"   Correlation: {stats['correlation']:.6f}\n")
                    f.write(f"   Mean abs diff: {stats['abs_diff_mean']:.6f}\n")
                    f.write(f"   Mean rel diff: {stats['rel_diff_mean']:.4%}\n")
                    f.write(f"   Max rel diff: {stats['rel_diff_max']:.4%}\n\n")
                    
                    f.write(f"📈 Value Statistics:\n")
                    f.write(f"   Validation - Mean: {stats['val_mean']:.6f}, Std: {stats['val_std']:.6f}\n")
                    f.write(f"   FreqTrade  - Mean: {stats['ft_mean']:.6f}, Std: {stats['ft_std']:.6f}\n\n")
                    
                    f.write(f"🎯 Difference Categories:\n")
                    for category, data in categories.items():
                        f.write(f"   {category:<10}: {data['count']:>8,} ({data['percentage']:>5.1f}%)\n")
                    
                    f.write(f"\n🔥 Top 10 Largest Differences:\n")
                    for i, diff in enumerate(results['top_differences'], 1):
                        timestamp = diff['timestamp']
                        val_val = diff['val_value']
                        ft_val = diff['ft_value']
                        rel_diff = diff['rel_diff']
                        
                        f.write(f"   {i:2}. {timestamp}: {val_val:.6f} vs {ft_val:.6f} ({rel_diff:.2%} diff)\n")
                    
                    f.write("\n")
            
            print(f"✅ Report saved successfully!")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def run_comparison(self):
        """Główna funkcja uruchamiająca porównanie"""
        print("🚀 Starting feature comparison...")
        
        # Ładowanie danych
        if not self.load_data():
            return False
        
        # Preprocessing
        if not self.preprocess_data():
            return False
        
        # Synchronizacja
        if not self.synchronize_data():
            return False
        
        # Obliczanie różnic
        self.calculate_differences()
        
        # Wyświetlenie wyników
        self.print_summary()
        
        # Zapisanie raportu
        self.save_detailed_report()
        
        print("\n✅ Feature comparison completed successfully!")
        return True

def main():
    """Główna funkcja"""
    # Sprawdź czy jesteśmy w odpowiednim katalogu
    if not os.path.exists("validation_and_labeling") or not os.path.exists("ft_bot_clean"):
        print("❌ Uruchom skrypt z głównego katalogu projektu!")
        print("   Oczekiwane katalogi: validation_and_labeling/, ft_bot_clean/")
        return
    
    # Uruchom porównanie
    comparator = FeatureComparator()
    success = comparator.run_comparison()
    
    if success:
        print("\n🎉 Porównanie zakończone pomyślnie!")
    else:
        print("\n❌ Porównanie zakończone błędem!")
        sys.exit(1)

if __name__ == "__main__":
    main()
