"""
🚀 QUICK COMPARISON RUNNER
Prosty skrypt do uruchamiania porównania predykcji z domyślnymi ścieżkami

USAGE:
    python run_comparison.py
    
Automatycznie znajdzie najnowsze pliki CSV i uruchomi porównanie.
"""

import os
import glob
from datetime import datetime
from compare_predictions import PredictionComparator


def find_latest_backtesting_csv():
    """Znajdź najnowszy plik CSV z backtestingu FreqTrade"""
    pattern = "../ft_bot_clean/user_data/logs/ml_predictions_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ Nie znaleziono plików backtestingu w: ../ft_bot_clean/user_data/logs/")
        return None
    
    # Sortuj po dacie modyfikacji
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    
    print(f"📁 Znaleziono backtesting CSV: {os.path.basename(latest)}")
    return latest


def find_latest_validation_csv():
    """Znajdź najnowszy plik CSV z walidacji ML"""
    # Sprawdź różne możliwe lokalizacje
    patterns = [
        "../Kaggle/models/ml_predictions_validation_*.csv",
        "../Kaggle/output/ml_predictions_validation_*.csv",
        "../ft_bot_clean/user_data/strategies/inputs/*/ml_predictions_validation_*.csv"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        print("❌ Nie znaleziono plików walidacji")
        print("   Sprawdzone lokalizacje:")
        for pattern in patterns:
            print(f"   - {pattern}")
        return None
    
    # Sortuj po dacie modyfikacji
    all_files.sort(key=os.path.getmtime, reverse=True)
    latest = all_files[0]
    
    print(f"📁 Znaleziono validation CSV: {os.path.basename(latest)}")
    return latest


def find_latest_backtesting_features_csv():
    """Znajdź najnowszy plik CSV z cechami z backtestingu FreqTrade"""
    pattern = "../ft_bot_clean/user_data/logs/features/features_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("⚠️ Nie znaleziono plików z cechami backtestingu w: ../ft_bot_clean/user_data/logs/features/")
        return None
    
    # Sortuj po dacie modyfikacji
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    
    print(f"📁 Znaleziono backtesting features CSV: {os.path.basename(latest)}")
    return latest


def find_latest_validation_features_csv():
    """Znajdź najnowszy plik CSV z cechami z walidacji ML"""
    # Sprawdź różne możliwe lokalizacje
    patterns = [
        "../validation_and_labeling/output/*single_label.csv",
        "../validation_and_labeling/output/*features*.csv",
        "../Kaggle/input/*features*.csv"
    ]
    
    all_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        print("⚠️ Nie znaleziono plików z cechami walidacji")
        print("   Sprawdzone lokalizacje:")
        for pattern in patterns:
            print(f"   - {pattern}")
        return None
    
    # Sortuj po dacie modyfikacji
    all_files.sort(key=os.path.getmtime, reverse=True)
    latest = all_files[0]
    
    print(f"📁 Znaleziono validation features CSV: {os.path.basename(latest)}")
    return latest


def main():
    """Main function"""
    print("🚀 QUICK PREDICTION COMPARISON WITH FEATURES")
    print("=" * 55)
    
    # Znajdź najnowsze pliki predykcji
    backtesting_file = find_latest_backtesting_csv()
    validation_file = find_latest_validation_csv()
    
    if not backtesting_file or not validation_file:
        print("\n❌ Nie można znaleźć wymaganych plików CSV z predykcjami")
        print("\nManualne uruchomienie:")
        print("python compare_predictions.py --backtesting path/to/backtesting.csv --validation path/to/validation.csv")
        return 1
    
    # Znajdź pliki z cechami (opcjonalne)
    print("\n📊 Szukanie plików z cechami...")
    backtesting_features_file = find_latest_backtesting_features_csv()
    validation_features_file = find_latest_validation_features_csv()
    
    if backtesting_features_file and validation_features_file:
        print("✅ Znaleziono pliki z cechami - będą porównane")
    else:
        print("⚠️ Nie znaleziono wszystkich plików z cechami - porównanie tylko predykcji")
    
    # Utwórz nazwy plików wyjściowych
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comparison_report_{timestamp}.txt"
    discrepancies_file = f"discrepancies_{timestamp}.csv"
    
    print(f"\n📊 Uruchamianie porównania...")
    print(f"   Backtesting: {os.path.basename(backtesting_file)}")
    print(f"   Validation: {os.path.basename(validation_file)}")
    if backtesting_features_file:
        print(f"   Backtesting features: {os.path.basename(backtesting_features_file)}")
    if validation_features_file:
        print(f"   Validation features: {os.path.basename(validation_features_file)}")
    print(f"   Raport: {report_file}")
    print(f"   Rozbieżności: {discrepancies_file}")
    
    # Uruchom porównanie
    comparator = PredictionComparator(
        backtesting_csv=backtesting_file,
        validation_csv=validation_file,
        backtesting_features_csv=backtesting_features_file,
        validation_features_csv=validation_features_file,
        verbose=True
    )
    
    success = comparator.run_full_analysis(report_file, discrepancies_file)
    
    if success:
        print(f"\n✅ PORÓWNANIE ZAKOŃCZONE!")
        print(f"📄 Raport: {report_file}")
        print(f"📊 Rozbieżności: {discrepancies_file}")
        
        # Pokaż szybkie statystyki
        if comparator.stats:
            total = comparator.stats['total_comparisons']
            identical = comparator.stats['identical']
            signal_changes = comparator.stats['signal_changes']
            
            print(f"\n📈 SZYBKIE STATYSTYKI PREDYKCJI:")
            print(f"   Porównanych predykcji: {total:,}")
            print(f"   Identycznych: {identical:,} ({identical/total*100:.1f}%)")
            print(f"   Zmian sygnału: {signal_changes:,} ({signal_changes/total*100:.1f}%)")
            
            if signal_changes > total * 0.1:
                print(f"   ⚠️ UWAGA: Wysoki odsetek zmian sygnałów!")
            elif identical > total * 0.8:
                print(f"   ✅ Wysoka zgodność predykcji")
        
        # Pokaż statystyki cech jeśli dostępne
        if comparator.feature_stats:
            print(f"\n🔍 SZYBKIE STATYSTYKI CECH:")
            worst_features = []
            for feature, stats in comparator.feature_stats.items():
                corr = stats['correlation']
                mean_diff = stats['mean_diff']
                print(f"   {feature}: korelacja={corr:.3f}, śr.różnica={mean_diff:.6f}")
                
                if corr < 0.95 or mean_diff > 0.01:
                    worst_features.append((feature, corr, mean_diff))
            
            if worst_features:
                print(f"\n⚠️ PROBLEMATYCZNE CECHY:")
                for feature, corr, diff in worst_features:
                    print(f"   - {feature}: niska korelacja ({corr:.3f}) lub duże różnice ({diff:.6f})")
            else:
                print(f"\n✅ Wszystkie cechy mają wysoką zgodność")
    else:
        print(f"\n❌ PORÓWNANIE NIEUDANE!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 