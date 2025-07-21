#!/usr/bin/env python3
"""
🔍 TEST DIAGNOSTIC SYSTEM
Skrypt testowy do porównania plików audit generowanych przez oba moduły
"""

import os
import sys
import glob
from datetime import datetime

# Import diagnostic utils
from diagnostic_utils import compare_audit_files, compare_scaled_features

def find_latest_audit_files():
    """
    Znajduje najnowsze pliki audit z obu modułów
    """
    raporty_dir = "raporty"
    
    # Znajdź pliki audit
    trainer_files = glob.glob(os.path.join(raporty_dir, "model_scaler_audit_trainer_*.json"))
    freqtrade_files = glob.glob(os.path.join(raporty_dir, "model_scaler_audit_freqtrade_*.json"))
    
    # Znajdź pliki scaled features
    trainer_features = glob.glob(os.path.join(raporty_dir, "scaled_features_sample_trainer.json"))
    freqtrade_features = glob.glob(os.path.join(raporty_dir, "scaled_features_sample_freqtrade.json"))
    
    # Posortuj po dacie (najnowsze pierwsze)
    trainer_files.sort(key=os.path.getmtime, reverse=True)
    freqtrade_files.sort(key=os.path.getmtime, reverse=True)
    
    return {
        'trainer_audit': trainer_files[0] if trainer_files else None,
        'freqtrade_audit': freqtrade_files[0] if freqtrade_files else None,
        'trainer_features': trainer_features[0] if trainer_features else None,
        'freqtrade_features': freqtrade_features[0] if freqtrade_features else None
    }

def main():
    """
    Główna funkcja testowa
    """
    print("🔍 DIAGNOSTIC SYSTEM TEST")
    print("=" * 50)
    
    # Sprawdź czy katalog raporty istnieje
    if not os.path.exists("raporty"):
        print("❌ Katalog 'raporty' nie istnieje!")
        print("   Uruchom najpierw trening i backtesting aby wygenerować pliki audit.")
        return
    
    # Znajdź najnowsze pliki
    files = find_latest_audit_files()
    
    print("\n📁 Znalezione pliki:")
    for key, path in files.items():
        if path:
            print(f"   ✅ {key}: {os.path.basename(path)}")
        else:
            print(f"   ❌ {key}: BRAK")
    
    # Porównaj pliki audit
    if files['trainer_audit'] and files['freqtrade_audit']:
        print("\n🔍 Porównywanie plików audit...")
        comparison_report = compare_audit_files(
            files['trainer_audit'],
            files['freqtrade_audit'],
            "raporty"
        )
        
        if comparison_report:
            print(f"   ✅ Raport porównawczy zapisany: {os.path.basename(comparison_report)}")
        else:
            print("   ❌ Nie udało się utworzyć raportu porównawczego")
    else:
        print("\n⚠️ Brak plików audit do porównania")
        print("   Uruchom trening i backtesting aby wygenerować pliki audit.")
    
    # Porównaj scaled features
    if files['trainer_features'] and files['freqtrade_features']:
        print("\n🔍 Porównywanie scaled features...")
        features_comparison = compare_scaled_features(
            files['trainer_features'],
            files['freqtrade_features'],
            "raporty"
        )
        
        if features_comparison:
            print(f"   ✅ Raport porównawczy features zapisany: {os.path.basename(features_comparison)}")
        else:
            print("   ❌ Nie udało się utworzyć raportu porównawczego features")
    else:
        print("\n⚠️ Brak plików scaled features do porównania")
        print("   Uruchom trening i backtesting aby wygenerować pliki features.")
    
    print("\n✅ Test diagnostic system zakończony!")
    print("\nInstrukcje:")
    print("1. Uruchom trening: cd Kaggle && python trainer.py")
    print("2. Uruchom backtesting: cd ft_bot_clean && python -m freqtrade backtesting ...")
    print("3. Uruchom ponownie ten skrypt: python test_diagnostic.py")

if __name__ == "__main__":
    main() 