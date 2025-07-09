#!/usr/bin/env python3
"""
Detailed Feature Analysis Script
================================
Szczegółowa analiza różnic między features z walidacji i FreqTrade
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

class DetailedFeatureAnalyzer:
    def __init__(self, base_path=None):
        self.base_path = Path(base_path) if base_path else Path('..')
        self.validation_path = self.base_path / 'validation_and_labeling' / 'output'
        self.freqtrade_debug_path = self.base_path / 'ft_bot_clean' / 'user_data' / 'debug_sequences'
        
        self.expected_features = [
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200'
        ]
        
        print(f'🔍 Detailed Feature Analyzer initialized')
        print(f'📁 Validation path: {self.validation_path}')
        print(f'📁 FreqTrade debug path: {self.freqtrade_debug_path}')
    
    def find_latest_files(self):
        print('\n🔍 Searching for files...')
        
        validation_files = list(self.validation_path.glob('*BTCUSDT*single_label.feather'))
        freqtrade_files = list(self.freqtrade_debug_path.glob('freqtrade_features_UNSCALED_*.feather'))
        
        print(f'Found {len(validation_files)} validation files')
        print(f'Found {len(freqtrade_files)} FreqTrade debug files')
        
        if not validation_files or not freqtrade_files:
            return None, None
        
        latest_validation = max(validation_files, key=lambda x: x.stat().st_mtime)
        latest_freqtrade = max(freqtrade_files, key=lambda x: x.stat().st_mtime)
        
        print(f'✅ Selected validation: {latest_validation.name}')
        print(f'✅ Selected FreqTrade: {latest_freqtrade.name}')
        
        return latest_validation, latest_freqtrade
    
    def analyze_feature_differences(self, val_values, ft_values, feature_name):
        """Szczegółowa analiza różnic dla pojedynczego feature"""
        
        # Podstawowe statystyki
        diff = val_values - ft_values
        abs_diff = np.abs(diff)
        
        stats = {
            'feature': feature_name,
            'total_values': len(val_values),
            'validation_mean': np.mean(val_values),
            'validation_std': np.std(val_values),
            'validation_min': np.min(val_values),
            'validation_max': np.max(val_values),
            'freqtrade_mean': np.mean(ft_values),
            'freqtrade_std': np.std(ft_values),
            'freqtrade_min': np.min(ft_values),
            'freqtrade_max': np.max(ft_values),
            'diff_mean': np.mean(diff),
            'diff_std': np.std(diff),
            'diff_min': np.min(diff),
            'diff_max': np.max(diff),
            'abs_diff_mean': np.mean(abs_diff),
            'abs_diff_max': np.max(abs_diff),
            'correlation': np.corrcoef(val_values, ft_values)[0, 1],
            'identical_count': np.sum(np.isclose(val_values, ft_values, rtol=1e-10)),
            'close_count': np.sum(np.isclose(val_values, ft_values, rtol=1e-6)),
            'different_count': np.sum(~np.isclose(val_values, ft_values, rtol=1e-6)),
        }
        
        # Percentyle różnic
        stats['diff_percentiles'] = {
            '1%': np.percentile(abs_diff, 1),
            '5%': np.percentile(abs_diff, 5),
            '10%': np.percentile(abs_diff, 10),
            '25%': np.percentile(abs_diff, 25),
            '50%': np.percentile(abs_diff, 50),
            '75%': np.percentile(abs_diff, 75),
            '90%': np.percentile(abs_diff, 90),
            '95%': np.percentile(abs_diff, 95),
            '99%': np.percentile(abs_diff, 99),
        }
        
        # Sprawdź czy różnice mają stały stosunek (np. zawsze 100x)
        non_zero_val = val_values[val_values != 0]
        non_zero_ft = ft_values[val_values != 0]
        
        if len(non_zero_val) > 0:
            ratios = non_zero_val / non_zero_ft
            ratios = ratios[~np.isinf(ratios)]  # Usuń inf
            
            if len(ratios) > 0:
                stats['ratio_mean'] = np.mean(ratios)
                stats['ratio_std'] = np.std(ratios)
                stats['ratio_min'] = np.min(ratios)
                stats['ratio_max'] = np.max(ratios)
                
                # Sprawdź czy stosunek jest stały (~100)
                ratio_close_to_100 = np.sum(np.isclose(ratios, 100, rtol=0.01))
                stats['ratio_close_to_100_count'] = ratio_close_to_100
                stats['ratio_close_to_100_percent'] = ratio_close_to_100 / len(ratios) * 100
        
        return stats
    
    def print_detailed_analysis(self, stats):
        """Wydrukuj szczegółową analizę"""
        feature = stats['feature']
        
        print(f"\n📊 DETAILED ANALYSIS: {feature}")
        print("=" * 60)
        
        print(f"📈 BASIC STATISTICS:")
        print(f"  Total values: {stats['total_values']:,}")
        print(f"  Validation mean: {stats['validation_mean']:.6f}")
        print(f"  FreqTrade mean: {stats['freqtrade_mean']:.6f}")
        print(f"  Mean difference: {stats['diff_mean']:.6f}")
        print(f"  Correlation: {stats['correlation']:.6f}")
        
        print(f"\n🎯 DIFFERENCE ANALYSIS:")
        print(f"  Identical values: {stats['identical_count']:,} ({stats['identical_count']/stats['total_values']*100:.2f}%)")
        print(f"  Close values (1e-6): {stats['close_count']:,} ({stats['close_count']/stats['total_values']*100:.2f}%)")
        print(f"  Different values: {stats['different_count']:,} ({stats['different_count']/stats['total_values']*100:.2f}%)")
        
        print(f"\n📊 DIFFERENCE PERCENTILES:")
        for pct, val in stats['diff_percentiles'].items():
            print(f"  {pct}: {val:.6f}")
        
        if 'ratio_mean' in stats:
            print(f"\n🔢 RATIO ANALYSIS (Validation/FreqTrade):")
            print(f"  Ratio mean: {stats['ratio_mean']:.2f}")
            print(f"  Ratio std: {stats['ratio_std']:.2f}")
            print(f"  Ratio range: {stats['ratio_min']:.2f} - {stats['ratio_max']:.2f}")
            if 'ratio_close_to_100_count' in stats:
                print(f"  Values close to 100x ratio: {stats['ratio_close_to_100_count']:,} ({stats['ratio_close_to_100_percent']:.1f}%)")
        
        # Diagnoza
        print(f"\n🔍 DIAGNOSIS:")
        if stats['identical_count'] == stats['total_values']:
            print("  ✅ ALL VALUES ARE IDENTICAL")
        elif stats['close_count'] == stats['total_values']:
            print("  ✅ ALL VALUES ARE VERY CLOSE (within 1e-6)")
        elif stats['different_count'] > stats['total_values'] * 0.9:
            print("  ❌ MOST VALUES ARE DIFFERENT")
            if 'ratio_close_to_100_percent' in stats and stats['ratio_close_to_100_percent'] > 90:
                print("  🎯 LIKELY CAUSE: FreqTrade missing '* 100' multiplication")
            elif abs(stats['correlation']) > 0.99:
                print("  🎯 LIKELY CAUSE: Constant scaling factor")
            else:
                print("  🎯 LIKELY CAUSE: Different calculation method")
        else:
            print("  ⚠️ MIXED RESULTS - needs further investigation")
    
    def save_sample_data(self, val_aligned, ft_aligned, n_samples=100):
        """Zapisz przykładowe dane do CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_feature_sample_{timestamp}.csv"
        
        print(f"\n💾 Saving sample data to {filename}")
        
        sample_data = []
        for i in range(min(n_samples, len(val_aligned))):
            row = {
                'timestamp': val_aligned.iloc[i]['timestamp'],
                'row_index': i
            }
            
            for feature in self.expected_features:
                if feature in val_aligned.columns and feature in ft_aligned.columns:
                    val_val = val_aligned.iloc[i][feature]
                    ft_val = ft_aligned.iloc[i][feature]
                    
                    row[f'val_{feature}'] = val_val
                    row[f'ft_{feature}'] = ft_val
                    row[f'diff_{feature}'] = val_val - ft_val
                    row[f'abs_diff_{feature}'] = abs(val_val - ft_val)
                    
                    if ft_val != 0:
                        row[f'ratio_{feature}'] = val_val / ft_val
                    else:
                        row[f'ratio_{feature}'] = np.nan
            
            sample_data.append(row)
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(filename, index=False)
        print(f"✅ Saved {len(sample_df)} rows to {filename}")
    
    def run_detailed_analysis(self):
        print('🚀 Starting detailed feature analysis...')
        
        # Find files
        validation_file, freqtrade_file = self.find_latest_files()
        if not validation_file or not freqtrade_file:
            print('❌ Files not found!')
            return False
        
        # Load files
        print('\n📊 Loading files...')
        val_df = pd.read_feather(validation_file)
        ft_df = pd.read_feather(freqtrade_file)
        
        print(f'  Validation: {len(val_df):,} rows, {len(val_df.columns)} columns')
        print(f'  FreqTrade: {len(ft_df):,} rows, {len(ft_df.columns)} columns')
        
        # Align timestamps
        print('\n🔄 Aligning timestamps...')
        val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
        ft_df['timestamp'] = pd.to_datetime(ft_df['timestamp'])
        
        common_timestamps = set(val_df['timestamp']).intersection(set(ft_df['timestamp']))
        print(f'  Common timestamps: {len(common_timestamps):,}')
        
        if len(common_timestamps) == 0:
            print('❌ No common timestamps!')
            return False
        
        val_aligned = val_df[val_df['timestamp'].isin(common_timestamps)].sort_values('timestamp').reset_index(drop=True)
        ft_aligned = ft_df[ft_df['timestamp'].isin(common_timestamps)].sort_values('timestamp').reset_index(drop=True)
        
        print(f'  Aligned to: {len(val_aligned):,} rows')
        
        # Detailed analysis for each feature
        all_stats = []
        
        for feature in self.expected_features:
            if feature in val_aligned.columns and feature in ft_aligned.columns:
                val_values = val_aligned[feature].values
                ft_values = ft_aligned[feature].values
                
                stats = self.analyze_feature_differences(val_values, ft_values, feature)
                all_stats.append(stats)
                self.print_detailed_analysis(stats)
            else:
                print(f"\n⚠️ {feature}: NOT FOUND in one of the files")
        
        # Save sample data
        self.save_sample_data(val_aligned, ft_aligned)
        
        # Summary
        print(f"\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        identical_features = [s['feature'] for s in all_stats if s['identical_count'] == s['total_values']]
        close_features = [s['feature'] for s in all_stats if s['close_count'] == s['total_values'] and s['identical_count'] < s['total_values']]
        different_features = [s['feature'] for s in all_stats if s['different_count'] > s['total_values'] * 0.1]
        
        print(f"✅ Identical features ({len(identical_features)}): {identical_features}")
        print(f"🟡 Close features ({len(close_features)}): {close_features}")
        print(f"❌ Different features ({len(different_features)}): {different_features}")
        
        # Check for 100x pattern
        ratio_100_features = []
        for s in all_stats:
            if 'ratio_close_to_100_percent' in s and s['ratio_close_to_100_percent'] > 80:
                ratio_100_features.append(s['feature'])
        
        if ratio_100_features:
            print(f"\n🎯 FEATURES WITH ~100x RATIO: {ratio_100_features}")
            print("   This suggests FreqTrade is missing '* 100' multiplication!")
        
        return True

# Run the analysis
if __name__ == "__main__":
    analyzer = DetailedFeatureAnalyzer()
    analyzer.run_detailed_analysis() 