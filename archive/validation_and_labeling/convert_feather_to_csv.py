#!/usr/bin/env python3
"""
🔧 Konwerter plików .feather do CSV

Konwertuje pliki wyjściowe z modułu etykietowania (format .feather) 
do formatu CSV z formatowaniem zgodnym z backtestingiem (12 miejsc po przecinku).

Użycie:
    python convert_feather_to_csv.py
    python convert_feather_to_csv.py path/to/file.feather
    python convert_feather_to_csv.py --all  # konwertuje wszystkie pliki .feather w output/
"""

import pandas as pd
import os
import sys
import glob
import time
from pathlib import Path
from tqdm import tqdm

def format_file_size(size_bytes):
    """Formatuje rozmiar pliku do czytelnej formy"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def convert_feather_to_csv(feather_path, output_path=None, show_progress=True):
    """
    Konwertuje plik .feather do CSV z formatowaniem
    
    Args:
        feather_path (str): Ścieżka do pliku .feather
        output_path (str): Ścieżka wyjściowa CSV (opcjonalna)
        show_progress (bool): Czy pokazywać progress bar
    """
    try:
        # Informacje o pliku wejściowym
        file_size = os.path.getsize(feather_path)
        print(f"📖 Reading: {feather_path}")
        print(f"   📏 File size: {format_file_size(file_size)}")
        
        start_time = time.time()
        
        # Wczytaj plik feather z progress bar
        if show_progress:
            print("   🔄 Loading feather file...")
        
        df = pd.read_feather(feather_path)
        load_time = time.time() - start_time
        
        print(f"   ✅ Loaded in {load_time:.2f}s")
        print(f"   📊 Shape: {df.shape} ({df.shape[0]:,} rows × {df.shape[1]} columns)")
        print(f"   💾 Memory usage: {format_file_size(df.memory_usage(deep=True).sum())}")
        
        # Przygotuj ścieżkę wyjściową
        if output_path is None:
            output_path = feather_path.replace('.feather', '.csv')
        
        # Sprawdź kolumny numeryczne
        numeric_columns = df.select_dtypes(include=['float64', 'float32']).columns
        print(f"   🔢 Numeric columns to format: {len(numeric_columns)}")
        
        # Formatuj kolumny numeryczne z progress bar
        if show_progress and len(numeric_columns) > 0:
            print("   🎯 Formatting numeric columns...")
            for col in tqdm(numeric_columns, desc="   Formatting", unit="col", leave=False):
                df[col] = df[col].round(12)
        else:
            for col in numeric_columns:
                df[col] = df[col].round(12)
        
        # Zapisz jako CSV z progress bar
        print("   💾 Saving to CSV...")
        save_start = time.time()
        
        # Progress bar dla zapisu (symulowany przez chunki)
        if show_progress and len(df) > 10000:
            # Dla dużych plików - zapisuj w chunkach z progress bar
            chunk_size = 10000
            total_chunks = (len(df) + chunk_size - 1) // chunk_size
            
            # Zapisz header
            df.iloc[:0].to_csv(output_path, index=False, float_format='%.12f')
            
            # Zapisuj chunki
            with tqdm(total=total_chunks, desc="   Writing", unit="chunk", leave=False) as pbar:
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk.to_csv(output_path, mode='a', header=False, index=False, float_format='%.12f')
                    pbar.update(1)
        else:
            # Dla małych plików - zapisuj normalnie
            df.to_csv(output_path, index=False, float_format='%.12f')
        
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        
        # Informacje o pliku wyjściowym
        output_size = os.path.getsize(output_path)
        
        print(f"✅ Converted successfully!")
        print(f"   📁 Output: {output_path}")
        print(f"   📏 Output size: {format_file_size(output_size)}")
        print(f"   ⏱️  Save time: {save_time:.2f}s")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📋 Columns: {df.columns.tolist()}")
        
        # Pokaż pierwsze kilka wierszy
        print(f"   👀 Preview (first 3 rows):")
        print(df.head(3).to_string(index=False))
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting {feather_path}: {e}")
        return False

def convert_all_feather_files(directory):
    """
    Konwertuje wszystkie pliki .feather w podanym katalogu
    
    Args:
        directory (str): Katalog do przeszukania
    """
    print(f"🔍 Scanning directory: {directory}")
    feather_files = glob.glob(os.path.join(directory, "*.feather"))
    
    if not feather_files:
        print(f"⚠️ No .feather files found in {directory}")
        return
    
    # Sortuj pliki według rozmiaru (największe pierwsze)
    feather_files_with_size = [(f, os.path.getsize(f)) for f in feather_files]
    feather_files_with_size.sort(key=lambda x: x[1], reverse=True)
    
    total_size = sum(size for _, size in feather_files_with_size)
    
    print(f"📁 Found {len(feather_files)} .feather files")
    print(f"📏 Total size: {format_file_size(total_size)}")
    print()
    
    # Progress bar dla wszystkich plików
    success_count = 0
    with tqdm(total=len(feather_files), desc="Converting files", unit="file") as pbar:
        for i, (feather_file, file_size) in enumerate(feather_files_with_size, 1):
            print(f"🔄 File {i}/{len(feather_files)}: {os.path.basename(feather_file)} ({format_file_size(file_size)})")
            
            if convert_feather_to_csv(feather_file, show_progress=True):
                success_count += 1
            
            pbar.update(1)
            print("-" * 80)
    
    print(f"🎉 Conversion complete!")
    print(f"   ✅ Success: {success_count}/{len(feather_files)} files")
    if success_count < len(feather_files):
        print(f"   ❌ Failed: {len(feather_files) - success_count} files")

def main():
    """Główna funkcja skryptu"""
    
    print("🔧 Feather to CSV Converter with Progress Tracking")
    print("=" * 60)
    
    # Sprawdź czy tqdm jest dostępne
    try:
        import tqdm
    except ImportError:
        print("⚠️ Warning: tqdm not installed. Install with: pip install tqdm")
        print("   Progress bars will be disabled.")
        print()
    
    # Sprawdź argumenty
    if len(sys.argv) == 1:
        # Brak argumentów - konwertuj domyślny plik
        default_file = "output/BTCUSDT_TF-1m__FW-120__SL-050__TP-100__single_label.feather"
        
        if os.path.exists(default_file):
            print(f"🎯 Converting default file: {default_file}")
            print()
            convert_feather_to_csv(default_file)
        else:
            print(f"❌ Default file not found: {default_file}")
            print()
            print("💡 Usage:")
            print("   python convert_feather_to_csv.py                    # convert default file")
            print("   python convert_feather_to_csv.py path/to/file.feather  # convert specific file")
            print("   python convert_feather_to_csv.py --all              # convert all .feather files in output/")
            
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        
        if arg == "--all":
            # Konwertuj wszystkie pliki w output/
            output_dir = "output"
            if os.path.exists(output_dir):
                convert_all_feather_files(output_dir)
            else:
                print(f"❌ Output directory not found: {output_dir}")
                
        elif arg.endswith('.feather'):
            # Konwertuj konkretny plik
            if os.path.exists(arg):
                print(f"🎯 Converting file: {arg}")
                print()
                convert_feather_to_csv(arg)
            else:
                print(f"❌ File not found: {arg}")
                
        else:
            print(f"❌ Invalid argument: {arg}")
            print("💡 Expected .feather file path or --all")
            
    else:
        print("❌ Too many arguments")
        print("💡 Usage: python convert_feather_to_csv.py [file.feather|--all]")

if __name__ == "__main__":
    main() 