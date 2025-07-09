"""
🔍 PREDICTION COMPARISON TOOL V2
Narzędzie do porównywania predykcji ML między backtestingiem FreqTrade a walidacją podczas treningu

USAGE:
    python compare_predictions.py --backtesting path/to/ml_predictions_backtesting.csv --validation path/to/ml_predictions_validation.csv
    
    # Z porównaniem cech:
    python compare_predictions.py --backtesting path/to/ml_predictions_backtesting.csv --validation path/to/ml_predictions_validation.csv --backtesting-features path/to/features_backtesting.csv --validation-features path/to/features_validation.csv

FEATURES:
- Synchronizacja czasowa plików CSV
- Porównanie predykcji minuta po minucie  
- Analiza rozbieżności w prawdopodobieństwach
- Identyfikacja zmian sygnałów (SHORT/HOLD/LONG)
- Porównanie 8 cech wejściowych modelu ML
- Analiza korelacji między różnicami w cechach a predykcjach
- Szczegółowe raporty diagnostyczne
- Eksport krytycznych różnic
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore')


class PredictionComparator:
    """
    🔍 PREDICTION COMPARATOR
    Klasa do porównywania predykcji ML między różnymi źródłami
    """
    
    def __init__(self, backtesting_csv: str, validation_csv: str, 
                 backtesting_features_csv: str = None, validation_features_csv: str = None, 
                 verbose: bool = True):
        """
        Initialize Prediction Comparator
        
        Args:
            backtesting_csv: Ścieżka do pliku CSV z backtestingu FreqTrade
            validation_csv: Ścieżka do pliku CSV z walidacji ML
            backtesting_features_csv: Ścieżka do pliku CSV z cechami z backtestingu
            validation_features_csv: Ścieżka do pliku CSV z cechami z walidacji
            verbose: Czy wyświetlać szczegółowe komunikaty
        """
        self.backtesting_file = backtesting_csv
        self.validation_file = validation_csv
        self.backtesting_features_file = backtesting_features_csv
        self.validation_features_file = validation_features_csv
        self.verbose = verbose
        
        # DataFrames
        self.backtesting_df = None
        self.validation_df = None
        self.backtesting_features_df = None
        self.validation_features_df = None
        self.synchronized_df = None
        self.differences_df = None
        self.feature_differences_df = None
        
        # Statistics
        self.stats = {}
        self.feature_stats = {}
        
        # Feature columns
        self.feature_columns = [
            'high_change', 'low_change', 'close_change', 'volume_change',
            'price_to_ma1440', 'price_to_ma43200', 'volume_to_ma1440', 'volume_to_ma43200'
        ]
        
        if self.verbose:
            print("🔍 PREDICTION COMPARATOR V2 - INITIALIZED")
            print("=" * 60)
    
    def load_data(self) -> bool:
        """
        📂 LOAD DATA FROM CSV FILES
        Załaduj dane z obu plików CSV
        
        Returns:
            bool: True jeśli oba pliki zostały załadowane pomyślnie
        """
        if self.verbose:
            print("📂 Loading CSV files...")
        
        try:
            # Load backtesting file
            if not os.path.exists(self.backtesting_file):
                print(f"❌ Backtesting file not found: {self.backtesting_file}")
                return False
            
            self.backtesting_df = pd.read_csv(self.backtesting_file)
            if self.verbose:
                print(f"   ✅ Backtesting: {len(self.backtesting_df):,} rows loaded")
                print(f"      File: {os.path.basename(self.backtesting_file)}")
            
            # Load validation file
            if not os.path.exists(self.validation_file):
                print(f"❌ Validation file not found: {self.validation_file}")
                return False
            
            self.validation_df = pd.read_csv(self.validation_file)
            if self.verbose:
                print(f"   ✅ Validation: {len(self.validation_df):,} rows loaded")
                print(f"      File: {os.path.basename(self.validation_file)}")
            
            # Load feature files if provided
            if self.backtesting_features_file and self.validation_features_file:
                if self.verbose:
                    print("\n📊 Loading feature files...")
                
                # Load backtesting features
                if os.path.exists(self.backtesting_features_file):
                    self.backtesting_features_df = pd.read_csv(self.backtesting_features_file)
                    if self.verbose:
                        print(f"   ✅ Backtesting features: {len(self.backtesting_features_df):,} rows loaded")
                        print(f"      File: {os.path.basename(self.backtesting_features_file)}")
                else:
                    print(f"⚠️ Backtesting features file not found: {self.backtesting_features_file}")
                
                # Load validation features
                if os.path.exists(self.validation_features_file):
                    self.validation_features_df = pd.read_csv(self.validation_features_file)
                    if self.verbose:
                        print(f"   ✅ Validation features: {len(self.validation_features_df):,} rows loaded")
                        print(f"      File: {os.path.basename(self.validation_features_file)}")
                else:
                    print(f"⚠️ Validation features file not found: {self.validation_features_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """
        🔍 VALIDATE CSV STRUCTURE
        Sprawdź czy struktury plików są kompatybilne
        
        Returns:
            bool: True jeśli struktury są kompatybilne
        """
        if self.verbose:
            print("\n🔍 Validating CSV structure...")
        
        if self.backtesting_df is None or self.validation_df is None:
            print("❌ Data not loaded")
            return False
        
        # Expected columns
        expected_columns = [
            'pair', 'chunk_id', 'pred_idx', 'short_prob', 'hold_prob', 'long_prob',
            'best_class', 'confidence', 'final_signal', 'threshold_short', 'threshold_long', 'timestamp'
        ]
        
        # Check backtesting columns
        missing_bt = [col for col in expected_columns if col not in self.backtesting_df.columns]
        if missing_bt:
            print(f"❌ Missing columns in backtesting file: {missing_bt}")
            return False
        
        # Check validation columns
        missing_val = [col for col in expected_columns if col not in self.validation_df.columns]
        if missing_val:
            print(f"❌ Missing columns in validation file: {missing_val}")
            return False
        
        if self.verbose:
            print("   ✅ Both files have required columns")
            print(f"   📊 Backtesting columns: {list(self.backtesting_df.columns)}")
            print(f"   📊 Validation columns: {list(self.validation_df.columns)}")
        
        return True
    
    def synchronize_data(self) -> bool:
        """
        📅 SYNCHRONIZE DATA BY TIMESTAMP
        Zsynchronizuj dane po timestampach
        
        Returns:
            bool: True jeśli synchronizacja się powiodła
        """
        if self.verbose:
            print("\n📅 Synchronizing data by timestamp...")
        
        try:
            # Convert timestamps to datetime
            self.backtesting_df['timestamp_dt'] = pd.to_datetime(self.backtesting_df['timestamp'])
            self.validation_df['timestamp_dt'] = pd.to_datetime(self.validation_df['timestamp'])

            # 🎯 FIX: STANDARDIZE TIMEZONES TO UTC
            # Ensure both dataframes are timezone-aware (UTC) to prevent merge errors
            self.backtesting_df['timestamp_dt'] = self.backtesting_df['timestamp_dt'].dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            self.validation_df['timestamp_dt'] = self.validation_df['timestamp_dt'].dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            
            # Get time ranges
            bt_start = self.backtesting_df['timestamp_dt'].min()
            bt_end = self.backtesting_df['timestamp_dt'].max()
            val_start = self.validation_df['timestamp_dt'].min()
            val_end = self.validation_df['timestamp_dt'].max()
            
            if self.verbose:
                print(f"   📅 Backtesting range: {bt_start} to {bt_end}")
                print(f"   📅 Validation range: {val_start} to {val_end}")
            
            # Find common time range
            common_start = max(bt_start, val_start)
            common_end = min(bt_end, val_end)
            
            if common_start >= common_end:
                print("❌ No overlapping time range found")
                return False
            
            if self.verbose:
                print(f"   📅 Common range: {common_start} to {common_end}")
            
            # Filter both datasets to common range
            bt_filtered = self.backtesting_df[
                (self.backtesting_df['timestamp_dt'] >= common_start) &
                (self.backtesting_df['timestamp_dt'] <= common_end)
            ].copy()
            
            val_filtered = self.validation_df[
                (self.validation_df['timestamp_dt'] >= common_start) &
                (self.validation_df['timestamp_dt'] <= common_end)
            ].copy()
            
            # Merge on timestamp
            self.synchronized_df = pd.merge(
                bt_filtered,
                val_filtered,
                on='timestamp_dt',
                suffixes=('_bt', '_val'),
                how='inner'
            )
            
            if len(self.synchronized_df) == 0:
                print("❌ No matching timestamps found")
                return False
            
            if self.verbose:
                print(f"   ✅ Synchronized {len(self.synchronized_df):,} rows")
                print(f"   📊 Backtesting rows in range: {len(bt_filtered):,}")
                print(f"   📊 Validation rows in range: {len(val_filtered):,}")
                print(f"   📊 Matched rows: {len(self.synchronized_df):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error synchronizing data: {e}")
            return False
    
    def calculate_differences(self) -> bool:
        """
        🧮 CALCULATE DIFFERENCES
        Oblicz różnice między predykcjami
        
        Returns:
            bool: True jeśli obliczenia się powiodły
        """
        if self.verbose:
            print("\n🧮 Calculating prediction differences...")
        
        if self.synchronized_df is None:
            print("❌ Data not synchronized")
            return False
        
        try:
            df = self.synchronized_df.copy()
            
            # Calculate probability differences
            df['short_prob_diff'] = abs(df['short_prob_bt'] - df['short_prob_val'])
            df['hold_prob_diff'] = abs(df['hold_prob_bt'] - df['hold_prob_val'])
            df['long_prob_diff'] = abs(df['long_prob_bt'] - df['long_prob_val'])
            df['confidence_diff'] = abs(df['confidence_bt'] - df['confidence_val'])
            
            # Calculate maximum difference across all probabilities
            df['max_prob_diff'] = df[['short_prob_diff', 'hold_prob_diff', 'long_prob_diff']].max(axis=1)
            
            # Check for signal changes
            df['best_class_changed'] = df['best_class_bt'] != df['best_class_val']
            df['final_signal_changed'] = df['final_signal_bt'] != df['final_signal_val']
            
            # Categorize differences
            df['difference_category'] = 'UNKNOWN'
            df.loc[df['max_prob_diff'] < 0.001, 'difference_category'] = 'IDENTICAL'
            df.loc[(df['max_prob_diff'] >= 0.001) & (df['max_prob_diff'] < 0.01), 'difference_category'] = 'MINOR'
            df.loc[(df['max_prob_diff'] >= 0.01) & (df['max_prob_diff'] < 0.05), 'difference_category'] = 'MODERATE'
            df.loc[df['max_prob_diff'] >= 0.05, 'difference_category'] = 'MAJOR'
            df.loc[df['final_signal_changed'], 'difference_category'] = 'SIGNAL_CHANGE'
            
            # Store results
            self.differences_df = df
            
            # Calculate statistics
            total_rows = len(df)
            self.stats = {
                'total_comparisons': total_rows,
                'identical': len(df[df['difference_category'] == 'IDENTICAL']),
                'minor_diff': len(df[df['difference_category'] == 'MINOR']),
                'moderate_diff': len(df[df['difference_category'] == 'MODERATE']),
                'major_diff': len(df[df['difference_category'] == 'MAJOR']),
                'signal_changes': len(df[df['difference_category'] == 'SIGNAL_CHANGE']),
                'avg_short_diff': df['short_prob_diff'].mean(),
                'avg_hold_diff': df['hold_prob_diff'].mean(),
                'avg_long_diff': df['long_prob_diff'].mean(),
                'avg_confidence_diff': df['confidence_diff'].mean(),
                'max_difference': df['max_prob_diff'].max()
            }
            
            if self.verbose:
                print(f"   ✅ Differences calculated for {total_rows:,} rows")
                print(f"   📊 Categories:")
                print(f"      IDENTICAL: {self.stats['identical']:,} ({self.stats['identical']/total_rows*100:.1f}%)")
                print(f"      MINOR (<1%): {self.stats['minor_diff']:,} ({self.stats['minor_diff']/total_rows*100:.1f}%)")
                print(f"      MODERATE (1-5%): {self.stats['moderate_diff']:,} ({self.stats['moderate_diff']/total_rows*100:.1f}%)")
                print(f"      MAJOR (>5%): {self.stats['major_diff']:,} ({self.stats['major_diff']/total_rows*100:.1f}%)")
                print(f"      SIGNAL_CHANGE: {self.stats['signal_changes']:,} ({self.stats['signal_changes']/total_rows*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error calculating differences: {e}")
            return False
    
    def compare_features(self) -> bool:
        """
        🔍 COMPARE FEATURES
        Porównaj wartości cech między backtestingiem a walidacją
        
        Returns:
            bool: True jeśli porównanie się powiodło
        """
        if self.backtesting_features_df is None or self.validation_features_df is None:
            if self.verbose:
                print("⚠️ Feature files not loaded - skipping feature comparison")
            return True
        
        if self.verbose:
            print("\n🔍 Comparing features...")
        
        try:
            # Convert timestamps to datetime
            bt_features = self.backtesting_features_df.copy()
            val_features = self.validation_features_df.copy()
            
            # Convert timestamps and normalize timezone
            bt_features['timestamp_dt'] = pd.to_datetime(bt_features['timestamp'])
            val_features['timestamp_dt'] = pd.to_datetime(val_features['timestamp'])
            
            # Remove timezone from backtesting if present to match validation
            if bt_features['timestamp_dt'].dt.tz is not None:
                bt_features['timestamp_dt'] = bt_features['timestamp_dt'].dt.tz_convert(None)
            
            # Ensure validation timestamps don't have timezone
            if val_features['timestamp_dt'].dt.tz is not None:
                val_features['timestamp_dt'] = val_features['timestamp_dt'].dt.tz_convert(None)
            
            # Merge features on timestamp
            features_merged = pd.merge(
                bt_features,
                val_features,
                on='timestamp_dt',
                suffixes=('_bt', '_val'),
                how='inner'
            )
            
            if len(features_merged) == 0:
                print("❌ No matching timestamps in feature files")
                return False
            
            # Calculate feature differences
            feature_diffs = {}
            for feature in self.feature_columns:
                bt_col = f"{feature}_bt"
                val_col = f"{feature}_val"
                
                if bt_col in features_merged.columns and val_col in features_merged.columns:
                    diff_col = f"{feature}_diff"
                    features_merged[diff_col] = abs(features_merged[bt_col] - features_merged[val_col])
                    
                    feature_diffs[feature] = {
                        'mean_diff': features_merged[diff_col].mean(),
                        'max_diff': features_merged[diff_col].max(),
                        'std_diff': features_merged[diff_col].std(),
                        'correlation': features_merged[bt_col].corr(features_merged[val_col])
                    }
            
            self.feature_differences_df = features_merged
            self.feature_stats = feature_diffs
            
            if self.verbose:
                print(f"   ✅ Features compared for {len(features_merged):,} timestamps")
                print("   📊 Feature differences summary:")
                for feature, stats in feature_diffs.items():
                    print(f"      {feature}: mean={stats['mean_diff']:.6f}, max={stats['max_diff']:.6f}, corr={stats['correlation']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error comparing features: {e}")
            return False
    
    def find_critical_differences(self, threshold: float = 0.05) -> pd.DataFrame:
        """
        🚨 FIND CRITICAL DIFFERENCES
        Znajdź krytyczne różnice przekraczające próg
        
        Args:
            threshold: Próg różnicy prawdopodobieństwa
            
        Returns:
            DataFrame z krytycznymi różnicami
        """
        if self.differences_df is None:
            print("❌ Differences not calculated")
            return pd.DataFrame()
        
        # Find critical differences
        critical = self.differences_df[
            (self.differences_df['max_prob_diff'] >= threshold) |
            (self.differences_df['final_signal_changed'])
        ].copy()
        
        # Sort by maximum difference
        critical = critical.sort_values('max_prob_diff', ascending=False)
        
        return critical
    
    def generate_report(self, output_file: str = None) -> str:
        """
        📊 GENERATE COMPARISON REPORT
        Wygeneruj szczegółowy raport porównawczy
        
        Args:
            output_file: Ścieżka do pliku wyjściowego (opcjonalne)
            
        Returns:
            str: Treść raportu
        """
        if self.stats is None or len(self.stats) == 0:
            return "❌ No statistics available - run analysis first"
        
        report_lines = []
        report_lines.append("🔍 RAPORT PORÓWNANIA PREDYKCJI ML")
        report_lines.append("=" * 60)
        report_lines.append(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Backtesting: {os.path.basename(self.backtesting_file)}")
        report_lines.append(f"Walidacja: {os.path.basename(self.validation_file)}")
        report_lines.append("")
        
        # Summary statistics
        total = self.stats['total_comparisons']
        report_lines.append("📊 PODSUMOWANIE PORÓWNANIA:")
        report_lines.append(f"   Łączna liczba porównań: {total:,}")
        report_lines.append(f"   Identyczne predykcje: {self.stats['identical']:,} ({self.stats['identical']/total*100:.1f}%)")
        report_lines.append(f"   Małe różnice (<1%): {self.stats['minor_diff']:,} ({self.stats['minor_diff']/total*100:.1f}%)")
        report_lines.append(f"   Umiarkowane różnice (1-5%): {self.stats['moderate_diff']:,} ({self.stats['moderate_diff']/total*100:.1f}%)")
        report_lines.append(f"   Duże różnice (>5%): {self.stats['major_diff']:,} ({self.stats['major_diff']/total*100:.1f}%)")
        report_lines.append(f"   Zmiany sygnału: {self.stats['signal_changes']:,} ({self.stats['signal_changes']/total*100:.1f}%)")
        report_lines.append("")
        
        # Average differences
        report_lines.append("📈 ŚREDNIE RÓŻNICE:")
        report_lines.append(f"   SHORT prawdopodobieństwo: {self.stats['avg_short_diff']:.6f}")
        report_lines.append(f"   HOLD prawdopodobieństwo: {self.stats['avg_hold_diff']:.6f}")
        report_lines.append(f"   LONG prawdopodobieństwo: {self.stats['avg_long_diff']:.6f}")
        report_lines.append(f"   Pewność predykcji: {self.stats['avg_confidence_diff']:.6f}")
        report_lines.append(f"   Maksymalna różnica: {self.stats['max_difference']:.6f}")
        report_lines.append("")
        
        # Feature comparison results
        if self.feature_stats:
            report_lines.append("🔍 PORÓWNANIE CECH WEJŚCIOWYCH:")
            for feature, stats in self.feature_stats.items():
                report_lines.append(f"   {feature}:")
                report_lines.append(f"      Średnia różnica: {stats['mean_diff']:.8f}")
                report_lines.append(f"      Maksymalna różnica: {stats['max_diff']:.8f}")
                report_lines.append(f"      Korelacja: {stats['correlation']:.4f}")
            report_lines.append("")
        
        # Critical differences
        critical = self.find_critical_differences(0.05)
        if len(critical) > 0:
            report_lines.append("🚨 KRYTYCZNE RÓŻNICE (top 10):")
            for i, (_, row) in enumerate(critical.head(10).iterrows()):
                timestamp = row['timestamp_dt'].strftime('%Y-%m-%d %H:%M:%S')
                bt_signal = ['SHORT', 'HOLD', 'LONG'][int(row['final_signal_bt'])]
                val_signal = ['SHORT', 'HOLD', 'LONG'][int(row['final_signal_val'])]
                max_diff = row['max_prob_diff']
                
                report_lines.append(f"   {i+1}. {timestamp}: BT={bt_signal}({row['confidence_bt']:.3f}), VAL={val_signal}({row['confidence_val']:.3f}), diff={max_diff:.3f}")
        else:
            report_lines.append("✅ Brak krytycznych różnic")
        
        report_lines.append("")
        report_lines.append("💡 REKOMENDACJE:")
        
        if self.stats['signal_changes'] > total * 0.1:
            report_lines.append("   ⚠️ Wysoki odsetek zmian sygnałów - sprawdź spójność modeli")
        
        if self.stats['major_diff'] > total * 0.05:
            report_lines.append("   ⚠️ Wiele dużych różnic - sprawdź preprocessing danych")
        
        if self.stats['identical'] < total * 0.2:
            report_lines.append("   ⚠️ Mało identycznych predykcji - sprawdź wersje modelu")
        
        if self.stats['identical'] > total * 0.8:
            report_lines.append("   ✅ Wysoka zgodność predykcji")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                # Create directory only if path contains directory
                output_dir = os.path.dirname(output_file)
                if output_dir:  # Only create directory if it's not empty
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                if self.verbose:
                    print(f"📄 Raport zapisany: {output_file}")
            except Exception as e:
                print(f"❌ Błąd zapisu raportu: {e}")
        
        return report_text
    
    def export_discrepancies(self, output_file: str, threshold: float = 0.01):
        """
        📤 EXPORT DISCREPANCIES TO CSV
        Wyeksportuj rozbieżności do pliku CSV
        
        Args:
            output_file: Ścieżka do pliku wyjściowego
            threshold: Minimalny próg różnicy do eksportu
        """
        if self.differences_df is None:
            print("❌ Differences not calculated")
            return
        
        try:
            # Filter significant differences
            significant = self.differences_df[
                (self.differences_df['max_prob_diff'] >= threshold) |
                (self.differences_df['final_signal_changed'])
            ].copy()
            
            # Select relevant columns
            export_columns = [
                'timestamp_dt', 'difference_category', 'max_prob_diff',
                'short_prob_bt', 'short_prob_val', 'short_prob_diff',
                'hold_prob_bt', 'hold_prob_val', 'hold_prob_diff',
                'long_prob_bt', 'long_prob_val', 'long_prob_diff',
                'confidence_bt', 'confidence_val', 'confidence_diff',
                'best_class_bt', 'best_class_val', 'best_class_changed',
                'final_signal_bt', 'final_signal_val', 'final_signal_changed'
            ]
            
            export_df = significant[export_columns].copy()
            export_df = export_df.sort_values('max_prob_diff', ascending=False)
            
            # Save to CSV
            output_dir = os.path.dirname(output_file)
            if output_dir:  # Only create directory if it's not empty
                os.makedirs(output_dir, exist_ok=True)
            export_df.to_csv(output_file, index=False)
            
            if self.verbose:
                print(f"📤 Wyeksportowano {len(export_df):,} rozbieżności do: {output_file}")
                
        except Exception as e:
            print(f"❌ Błąd eksportu: {e}")
    
    def run_full_analysis(self, report_file: str = None, discrepancies_file: str = None) -> bool:
        """
        🎯 RUN FULL ANALYSIS
        Uruchom pełną analizę porównawczą
        
        Args:
            report_file: Ścieżka do pliku raportu (opcjonalne)
            discrepancies_file: Ścieżka do pliku rozbieżności (opcjonalne)
            
        Returns:
            bool: True jeśli analiza się powiodła
        """
        try:
            # Step 1: Load data
            if not self.load_data():
                return False
            
            # Step 2: Validate structure
            if not self.validate_structure():
                return False
            
            # Step 3: Synchronize data
            if not self.synchronize_data():
                return False
            
            # Step 4: Calculate differences
            if not self.calculate_differences():
                return False
            
            # Step 5: Compare features (if available)
            if not self.compare_features():
                return False
            
            # Step 6: Generate report
            if report_file:
                self.generate_report(report_file)
            else:
                report = self.generate_report()
                print("\n" + report)
            
            # Step 6: Export discrepancies
            if discrepancies_file:
                self.export_discrepancies(discrepancies_file)
            
            if self.verbose:
                print("\n✅ ANALIZA ZAKOŃCZONA POMYŚLNIE!")
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas analizy: {e}")
            return False


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Compare ML predictions between backtesting and validation')
    parser.add_argument('--backtesting', '-b', required=True, help='Path to backtesting CSV file')
    parser.add_argument('--validation', '-v', required=True, help='Path to validation CSV file')
    parser.add_argument('--backtesting-features', '-bf', help='Path to backtesting features CSV file')
    parser.add_argument('--validation-features', '-vf', help='Path to validation features CSV file')
    parser.add_argument('--report', '-r', help='Output path for comparison report')
    parser.add_argument('--discrepancies', '-d', help='Output path for discrepancies CSV')
    parser.add_argument('--threshold', '-t', type=float, default=0.01, help='Threshold for discrepancy export (default: 0.01)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = PredictionComparator(
        backtesting_csv=args.backtesting,
        validation_csv=args.validation,
        backtesting_features_csv=args.backtesting_features,
        validation_features_csv=args.validation_features,
        verbose=not args.quiet
    )
    
    # Set default output files if not specified
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = args.report or f"prediction_comparison_report_{timestamp}.txt"
    discrepancies_file = args.discrepancies or f"prediction_discrepancies_{timestamp}.csv"
    
    # Run analysis
    success = comparator.run_full_analysis(report_file, discrepancies_file)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 