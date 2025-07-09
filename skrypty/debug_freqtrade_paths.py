#!/usr/bin/env python3
"""
Debug Freqtrade Paths - pokazuje dokładnie gdzie Freqtrade szuka plików
"""

import os
from pathlib import Path

def pair_to_filename(pair: str) -> str:
    """Kopiuję funkcję z Freqtrade misc.py"""
    for ch in ["/", " ", ".", "@", "$", "+", ":"]:
        pair = pair.replace(ch, "_")
    return pair

def show_freqtrade_expected_paths():
    """Pokazuje dokładnie gdzie Freqtrade oczekuje plików"""
    
    print("🔍 FREQTRADE PATH ANALYSIS")
    print("=" * 60)
    
    # Parametry
    datadir = Path("user_data/strategies/inputs")
    pair = "BTC/USDT:USDT"
    timeframe = "1m"
    candle_type = "futures"  # CandleType.FUTURES
    file_extension = "feather"
    
    # Krok 1: Konwersja pary na nazwę pliku
    pair_filename = pair_to_filename(pair)
    print(f"1. Pair conversion: '{pair}' → '{pair_filename}'")
    
    # Krok 2: Dodanie katalogu futures
    futures_datadir = datadir / "futures"
    print(f"2. Futures datadir: {futures_datadir}")
    
    # Krok 3: Konstruowanie pełnej nazwy pliku
    expected_filename = f"{pair_filename}-{timeframe}-{candle_type}.{file_extension}"
    expected_full_path = futures_datadir / expected_filename
    print(f"3. Expected filename: {expected_filename}")
    print(f"4. Expected full path: {expected_full_path}")
    
    print("\n" + "=" * 60)
    print("📁 CURRENT FILE STRUCTURE:")
    
    # Sprawdź co faktycznie istnieje
    current_file = Path("user_data/strategies/inputs/BTC_USDT_USDT/raw_validated.feather")
    print(f"Current file: {current_file}")
    print(f"Current file exists: {current_file.exists()}")
    
    print(f"Expected file: {expected_full_path}")
    print(f"Expected file exists: {expected_full_path.exists()}")
    
    print("\n" + "=" * 60)
    print("🔧 SOLUTION OPTIONS:")
    
    print("\nOption 1: Create symbolic link")
    print(f"mkdir -p {futures_datadir}")
    print(f"ln -s ../../BTC_USDT_USDT/raw_validated.feather {expected_full_path}")
    
    print("\nOption 2: Copy file to expected location")
    print(f"mkdir -p {futures_datadir}")
    print(f"cp {current_file} {expected_full_path}")
    
    print("\nOption 3: Move file to expected location")
    print(f"mkdir -p {futures_datadir}")
    print(f"mv {current_file} {expected_full_path}")
    
    return expected_full_path, current_file

def create_proper_structure():
    """Tworzy właściwą strukturę katalogów i link"""
    expected_path, current_file = show_freqtrade_expected_paths()
    
    print("\n" + "=" * 60)
    print("🚀 CREATING PROPER STRUCTURE...")
    
    if not current_file.exists():
        print(f"❌ Source file doesn't exist: {current_file}")
        return False
    
    # Utwórz katalog futures
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {expected_path.parent}")
    
    # Utwórz link symboliczny (Windows wymaga uprawnień administratora)
    try:
        if expected_path.exists():
            expected_path.unlink()
        
        # Na Windows użyjemy kopii zamiast linku
        if os.name == 'nt':
            import shutil
            shutil.copy2(current_file, expected_path)
            print(f"✅ Copied file to: {expected_path}")
        else:
            expected_path.symlink_to(current_file.resolve())
            print(f"✅ Created symlink: {expected_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating link/copy: {e}")
        return False

if __name__ == "__main__":
    success = create_proper_structure()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS! Freqtrade should now find the data file.")
        print("Run your backtest again.")
    else:
        print("❌ FAILED! Check the errors above.") 