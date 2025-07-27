"""
Skrypt do sprawdzania jakości danych po etykietowaniu w labeler3.
Sprawdza czy plik wyjściowy nie ma pustych wartości.
"""
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def check_labeled_data_quality(input_file="output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather"):
    """
    Sprawdza jakość danych po etykietowaniu.
    
    Args:
        input_file: Ścieżka do pliku z danymi po etykietowaniu
    """
    print("🔍 SPRAWDZANIE JAKOŚCI DANYCH PO ETYKIETOWANIU")
    print("=" * 60)
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(input_file):
        print(f"❌ Plik nie istnieje: {input_file}")
        return False
    
    try:
        # Wczytaj dane
        print(f"📂 Wczytywanie danych z: {input_file}")
        df = pd.read_feather(input_file)
        
        print(f"📊 Dane wczytane: {len(df):,} wierszy, {len(df.columns)} kolumn")
        print(f"⏰ Zakres czasowy: {df.index.min()} do {df.index.max()}")
        
        # Kategorie kolumn
        ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
        label_cols = [col for col in df.columns if col.startswith('label_')]
        feature_cols = [col for col in df.columns if col not in ohlc_cols + label_cols and not col.startswith('snapshot')]
        snapshot_cols = [col for col in df.columns if col.startswith('snapshot')]
        other_cols = [col for col in df.columns if col not in ohlc_cols + label_cols + feature_cols + snapshot_cols]
        
        print(f"\n📋 KATEGORIE KOLUMN:")
        print(f"  📈 OHLC: {len(ohlc_cols)} kolumn")
        print(f"  🏷️  Etykiety: {len(label_cols)} kolumn")
        print(f"  🔧 Cechy: {len(feature_cols)} kolumn")
        print(f"  📊 Snapshot: {len(snapshot_cols)} kolumn")
        print(f"  🔍 Inne: {len(other_cols)} kolumn")
        
        # Sprawdź brakujące wartości
        print(f"\n🔍 SPRAWDZANIE BRAKUJĄCYCH WARTOŚCI:")
        missing_data = df.isnull().sum()
        columns_with_missing = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(columns_with_missing) == 0:
            print("  ✅ Brak brakujących wartości!")
            return True
        else:
            print(f"  ⚠️  Znaleziono {len(columns_with_missing)} kolumn z brakującymi wartościami:")
            for col, count in columns_with_missing.items():
                percentage = (count / len(df)) * 100
                print(f"    {col}: {count:,} ({percentage:.2f}%)")
        
        # Sprawdź wartości infinity
        print(f"\n🔍 SPRAWDZANIE WARTOŚCI INFINITY:")
        infinity_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    infinity_cols.append(col)
                    percentage = (inf_count / len(df)) * 100
                    print(f"    ⚠️  {col}: {inf_count:,} infinity ({percentage:.2f}%)")
        
        if not infinity_cols:
            print("  ✅ Brak wartości infinity")
        
        # Sprawdź wartości bardzo duże/małe
        print(f"\n🔍 SPRAWDZANIE WARTOŚCI EKSTREMALNYCH:")
        extreme_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                large_count = (df[col] > 1e10).sum()
                small_count = (df[col] < -1e10).sum()
                if large_count > 0 or small_count > 0:
                    extreme_cols.append(col)
                    print(f"    ⚠️  {col}: {large_count:,} >1e10, {small_count:,} <-1e10")
        
        if not extreme_cols:
            print("  ✅ Brak wartości ekstremalnych")
        
        # Sprawdź etykiety
        print(f"\n🏷️  SPRAWDZANIE ETYKIET:")
        for label_col in label_cols:
            unique_values = df[label_col].value_counts().sort_index()
            print(f"  {label_col}: {unique_values.to_dict()}")
            
            # Sprawdź czy etykiety są w zakresie [0, 1, 2]
            if not set(unique_values.index).issubset({0, 1, 2}):
                print(f"    ⚠️  Nieoczekiwane wartości etykiet: {set(unique_values.index)}")
        
        # Sprawdź cechy
        print(f"\n🔧 SPRAWDZANIE CECH:")
        print(f"  Liczba cech: {len(feature_cols)}")
        if feature_cols:
            print(f"  Przykładowe cechy: {feature_cols[:5]}")
            if len(feature_cols) > 5:
                print(f"  ... i {len(feature_cols) - 5} więcej")
        
        # Sprawdź snapshot
        print(f"\n📊 SPRAWDZANIE SNAPSHOT:")
        print(f"  Liczba kolumn snapshot: {len(snapshot_cols)}")
        if snapshot_cols:
            snapshot1_cols = [col for col in snapshot_cols if col.startswith('snapshot1_')]
            snapshot2_cols = [col for col in snapshot_cols if col.startswith('snapshot2_')]
            print(f"  Snapshot1: {len(snapshot1_cols)} kolumn")
            print(f"  Snapshot2: {len(snapshot2_cols)} kolumn")
        
        # Podsumowanie
        print(f"\n📋 PODSUMOWANIE:")
        total_issues = len(columns_with_missing) + len(infinity_cols) + len(extreme_cols)
        if total_issues == 0:
            print("  ✅ DANE SĄ CZYSTE - gotowe do treningu!")
            return True
        else:
            print(f"  ⚠️  Znaleziono {total_issues} problemów z jakością danych")
            return False
            
    except Exception as e:
        print(f"❌ Błąd podczas sprawdzania danych: {e}")
        return False

def main():
    """Główna funkcja"""
    parser = argparse.ArgumentParser(description='Sprawdza jakość danych po etykietowaniu')
    parser.add_argument('--input', default='output/ohlc_orderbook_labeled_3class_fw60m_5levels.feather', 
                       help='Ścieżka do pliku z danymi po etykietowaniu')
    parser.add_argument('--report', default='data_quality_report.txt',
                       help='Nazwa pliku raportu')
    
    args = parser.parse_args()
    
    # Sprawdź jakość danych
    is_clean = check_labeled_data_quality(args.input)
    
    # Zapisz raport
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("RAPORT JAKOŚCI DANYCH PO ETYKIETOWANIU\n")
            f.write("=" * 50 + "\n")
            f.write(f"Plik: {args.input}\n")
            f.write(f"Status: {'CZYSTE' if is_clean else 'PROBLEMY'}\n")
        print(f"\n📄 Raport zapisany: {args.report}")
    
    print(f"\n{'🎉' if is_clean else '⚠️'} SPRAWDZANIE ZAKOŃCZONE")
    
    return 0 if is_clean else 1

if __name__ == "__main__":
    exit(main()) 