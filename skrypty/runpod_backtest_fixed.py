#!/usr/bin/env python3
"""
RunPod Backtest Runner - FIXED VERSION
Zawiera poprawioną komendę backtestingu z flagą --datadir
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Uruchamia backtest z poprawioną komendą zawierającą --datadir"""
    
    print("🎯 RUNPOD BACKTEST - FIXED VERSION")
    print("🔧 Dodano flagę --datadir dla poprawnego wskazania ścieżki danych")
    print("=" * 70)
    
    # Komenda backtestingu z flagą --datadir
    cmd = [
        'freqtrade', 'backtesting',
        '--config', 'user_data/config.json',
        '--strategy', 'Enhanced_ML_MA43200_Buffer_Strategy',
        '--timerange', '20240101-20240102',
        '--dry-run-wallet', '1000',
        '--datadir', 'user_data/strategies/inputs'  # <-- TO JEST KLUCZOWE!
    ]
    
    print("📋 Komenda backtestingu:")
    print(" ".join(cmd))
    print("=" * 70)
    
    # Sprawdź czy plik z danymi istnieje
    data_file = Path("user_data/strategies/inputs/BTC_USDT_USDT/raw_validated.feather")
    if data_file.exists():
        print(f"✅ Plik z danymi znaleziony: {data_file}")
        print(f"📊 Rozmiar pliku: {data_file.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ BŁĄD: Plik z danymi nie istnieje: {data_file}")
        return False
    
    print("\n🚀 Uruchamianie backtestingu...")
    print("-" * 50)
    
    # Uruchom backtest
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Backtest completed successfully!")
        else:
            print(f"\n❌ Backtest failed with code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Błąd podczas uruchamiania: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 