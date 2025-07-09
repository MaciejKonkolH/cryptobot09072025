#!/usr/bin/env python3
"""
Porównanie struktur plików feather
"""
import pandas as pd
import numpy as np
import os

def analyze_feather_file(filepath, name):
    """Analizuje strukturę pliku feather"""
    print(f"\n📁 {name}")
    print("-" * 60)
    
    if not os.path.exists(filepath):
        print(f"❌ Plik nie istnieje: {filepath}")
        return None
    
    try:
        df = pd.read_feather(filepath)
        print(f"✅ Załadowano pomyślnie")
        print(f"📊 Shape: {df.shape}")
        print(f"📏 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Sprawdź kolumny z datami
        date_col = None
        if 'date' in df.columns:
            date_col = 'date'
        elif 'datetime' in df.columns:
            date_col = 'datetime'
        
        if date_col:
            print(f"📅 Date range: {df[date_col].iloc[0]} to {df[date_col].iloc[-1]}")
        
        print(f"🗂️ Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"🔢 Data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        print(f"📋 First 3 rows:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def compare_structures(df1, df2, name1, name2):
    """Porównuje struktury dwóch DataFrame"""
    print(f"\n🔍 PORÓWNANIE: {name1} vs {name2}")
    print("=" * 80)
    
    if df1 is None or df2 is None:
        print("❌ Nie można porównać - jeden z plików się nie załadował")
        return
    
    # Porównaj kolumny
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"🔗 Wspólne kolumny ({len(common_cols)}): {sorted(common_cols)}")
    if only_in_1:
        print(f"📁 Tylko w {name1} ({len(only_in_1)}): {sorted(only_in_1)}")
    if only_in_2:
        print(f"📁 Tylko w {name2} ({len(only_in_2)}): {sorted(only_in_2)}")
    
    # Porównaj typy danych dla wspólnych kolumn
    print(f"\n🔢 Porównanie typów danych:")
    for col in sorted(common_cols):
        type1 = df1[col].dtype
        type2 = df2[col].dtype
        match = "✅" if type1 == type2 else "❌"
        print(f"  {col}: {type1} vs {type2} {match}")
    
    # Porównaj zakresy dat
    print(f"\n📅 Porównanie zakresów dat:")
    date_cols = ['date', 'datetime']
    for date_col in date_cols:
        if date_col in df1.columns and date_col in df2.columns:
            print(f"  {date_col}:")
            print(f"    {name1}: {df1[date_col].iloc[0]} to {df1[date_col].iloc[-1]}")
            print(f"    {name2}: {df2[date_col].iloc[0]} to {df2[date_col].iloc[-1]}")
    
    # Sprawdź kompatybilność
    print(f"\n✨ OCENA KOMPATYBILNOŚCI:")
    if len(common_cols) >= 5:  # OHLCV minimum
        print("✅ Podstawowe kolumny OHLCV są obecne")
    else:
        print("❌ Brakuje podstawowych kolumn OHLCV")
    
    if len(only_in_1) == 0 and len(only_in_2) == 0:
        print("✅ Identyczne kolumny - pełna kompatybilność")
    elif len(only_in_2) == 0:
        print("⚠️ raw_validated ma dodatkowe kolumny - można skopiować")
    else:
        print("❌ FreqTrade ma dodatkowe kolumny - potrzebna konwersja")

def check_freqtrade_compatibility(df):
    """Sprawdza czy plik jest kompatybilny z FreqTrade"""
    print(f"\n🔍 SPRAWDZENIE KOMPATYBILNOŚCI Z FREQTRADE:")
    print("-" * 50)
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if not missing_cols:
        print("✅ Wszystkie wymagane kolumny FreqTrade są obecne")
        return True
    else:
        print(f"❌ Brakuje kolumn: {missing_cols}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("PORÓWNANIE STRUKTUR PLIKÓW FEATHER")
    print("=" * 80)
    
    # Ścieżki do plików
    file1 = "user_data/strategies/inputs/BTC_USDT_USDT/raw_validated.feather"
    file2 = "user_data/data/binanceusdm/futures/BTC_USDT_USDT-1m-futures.feather"
    
    # Sprawdź alternatywne lokalizacje jeśli standardowa nie istnieje
    if not os.path.exists(file2):
        alt_file2 = "user_data/data/binanceusdm/BTC_USDT-1m-futures2/BTC_USDT-1m-futures.feather"
        if os.path.exists(alt_file2):
            file2 = alt_file2
            print(f"📍 Używam alternatywnej lokalizacji FreqTrade: {file2}")
    
    # Analizuj pliki
    df1 = analyze_feather_file(file1, "raw_validated.feather")
    df2 = analyze_feather_file(file2, "freqtrade.feather")
    
    # Sprawdź kompatybilność z FreqTrade
    if df1 is not None:
        check_freqtrade_compatibility(df1)
    if df2 is not None:
        check_freqtrade_compatibility(df2)
    
    # Porównaj struktury
    compare_structures(df1, df2, "raw_validated", "freqtrade")
    
    print("\n" + "=" * 80)
    print("REKOMENDACJE:")
    print("=" * 80)
    
    if df1 is not None and df2 is not None:
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        if cols1 == cols2:
            print("✅ Pliki mają identyczne struktury - można bezpiecznie skopiować")
        elif cols2.issubset(cols1):
            print("⚠️ raw_validated ma więcej kolumn - można skopiować, ale FreqTrade użyje tylko podstawowe")
        else:
            print("❌ Struktury są niezgodne - potrzebna konwersja")
    
    print("\n" + "=" * 80)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 80) 