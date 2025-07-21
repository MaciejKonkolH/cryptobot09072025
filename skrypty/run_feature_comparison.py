#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 FEATURE COMPARISON RUNNER V1.0
=====================================
Automatyczne wykrywanie i uruchamianie porównania cech

Autor: Crypto Trading System
Data: 2025-06-27
"""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path

# --- DYNAMIC PATH RESOLUTION ---
# Ustawia bieżący katalog roboczy na główny katalog projektu,
# co pozwala na uruchomienie skryptu z dowolnego miejsca.
try:
    # Znajdź ścieżkę do bieżącego skryptu
    script_path = Path(__file__).resolve()
    # Przejdź do katalogu nadrzędnego (główny katalog projektu)
    project_root = script_path.parent.parent
    # Zmień bieżący katalog roboczy
    os.chdir(project_root)
    print(f"✅ Ustawiono katalog roboczy na: {project_root}")
except NameError:
    # Fallback dla interaktywnych środowisk, gdzie __file__ nie jest zdefiniowane
    print("⚠️ Nie można automatycznie ustawić ścieżki (prawdopodobnie tryb interaktywny).")
    pass
# -----------------------------

def find_validation_files():
    """
    Znajdź pliki validation_and_labeling, priorytetyzując .feather.
    """
    # Priorytet: szukaj plików .feather
    pattern_feather = "validation_and_labeling/output/*single_label.feather"
    files = glob.glob(pattern_feather)
    
    # Fallback: jeśli nie ma .feather, szukaj .csv
    if not files:
        pattern_csv = "validation_and_labeling/output/*single_label.csv"
        files = glob.glob(pattern_csv)
    
    # Sortuj po dacie modyfikacji
    return sorted(files, key=os.path.getmtime, reverse=True)

def find_freqtrade_files():
    """Znajdź pliki FreqTrade features"""
    pattern = "ft_bot_clean/user_data/logs/features/features_*.csv"
    files = glob.glob(pattern)
    return sorted(files, key=os.path.getmtime, reverse=True)

def select_files():
    """Interaktywny wybór plików"""
    print("🔍 FEATURE COMPARISON RUNNER V1.0")
    print("=" * 50)
    
    # Znajdź pliki validation
    val_files = find_validation_files()
    if not val_files:
        print("❌ Nie znaleziono plików validation_and_labeling!")
        return None, None
    
    print(f"\n📁 Znalezione pliki validation ({len(val_files)}):")
    for i, file in enumerate(val_files, 1):
        size = os.path.getsize(file) / (1024*1024)  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"   {i}. {file} ({size:.1f} MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    # Znajdź pliki FreqTrade
    ft_files = find_freqtrade_files()
    if not ft_files:
        print("❌ Nie znaleziono plików FreqTrade features!")
        return None, None
    
    print(f"\n📁 Znalezione pliki FreqTrade ({len(ft_files)}):")
    for i, file in enumerate(ft_files, 1):
        size = os.path.getsize(file) / (1024*1024)  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"   {i}. {file} ({size:.1f} MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    # Wybór plików
    print(f"\n🎯 WYBÓR PLIKÓW:")
    
    # Validation file
    if len(val_files) == 1:
        val_choice = 1
        print(f"📊 Auto-wybór validation: {val_files[0]}")
    else:
        while True:
            try:
                val_choice = int(input(f"Wybierz plik validation (1-{len(val_files)}): "))
                if 1 <= val_choice <= len(val_files):
                    break
                print("❌ Nieprawidłowy numer!")
            except ValueError:
                print("❌ Wprowadź liczbę!")
    
    # FreqTrade file
    if len(ft_files) == 1:
        ft_choice = 1
        print(f"📊 Auto-wybór FreqTrade: {ft_files[0]}")
    else:
        while True:
            try:
                ft_choice = int(input(f"Wybierz plik FreqTrade (1-{len(ft_files)}): "))
                if 1 <= ft_choice <= len(ft_files):
                    break
                print("❌ Nieprawidłowy numer!")
            except ValueError:
                print("❌ Wprowadź liczbę!")
    
    selected_val = val_files[val_choice - 1]
    selected_ft = ft_files[ft_choice - 1]
    
    print(f"\n✅ WYBRANE PLIKI:")
    print(f"   Validation: {selected_val}")
    print(f"   FreqTrade:  {selected_ft}")
    
    return selected_val, selected_ft

def update_config_file(val_file, ft_file):
    """Aktualizuj plik konfiguracyjny compare_features.py"""
    config_file = "skrypty/compare_features.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Nie znaleziono pliku: {config_file}")
        return False
    
    try:
        # Wczytaj plik
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Znajdź i zamień ścieżki
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            if line.strip().startswith('VALIDATION_FILE = '):
                updated_lines.append(f'    VALIDATION_FILE = r\"{val_file}\"')
            elif line.strip().startswith('FREQTRADE_FILE = '):
                updated_lines.append(f'    FREQTRADE_FILE = r\"{ft_file}\"')
            else:
                updated_lines.append(line)
        
        # Zapisz zaktualizowany plik
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"✅ Zaktualizowano konfigurację w {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas aktualizacji konfiguracji: {e}")
        return False

def run_comparison():
    """Uruchom porównanie cech"""
    print(f"\n🚀 URUCHAMIANIE PORÓWNANIA...")
    
    try:
        # Import i uruchomienie
        sys.path.append('skrypty')
        from compare_features import FeatureComparator
        
        comparator = FeatureComparator()
        success = comparator.run_comparison()
        
        return success
        
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        return False
    except Exception as e:
        print(f"❌ Błąd podczas porównania: {e}")
        return False

def main():
    """Główna funkcja"""
    # Wybierz pliki
    val_file, ft_file = select_files()
    if not val_file or not ft_file:
        print("❌ Nie udało się wybrać plików!")
        return
    
    # Aktualizuj konfigurację
    if not update_config_file(val_file, ft_file):
        print("❌ Nie udało się zaktualizować konfiguracji!")
        return
    
    # Uruchom porównanie
    success = run_comparison()
    
    if success:
        print("\n🎉 Porównanie zakończone pomyślnie!")
        print("📄 Sprawdź wygenerowany raport w katalogu skrypty/")
    else:
        print("\n❌ Porównanie zakończone błędem!")
        sys.exit(1)

if __name__ == "__main__":
    main() 