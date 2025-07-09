"""
🔬 NANO-TEST: AUDYT CZYSTOŚCI DANYCH
========================================
Test, który weryfikuje hipotezę o istnieniu wartości NaN w finalnym
zbiorze cech generowanym przez `feature_calculator.py`.

Cel: Sprawdzić, czy dane opuszczające moduł walidacji są w 100% czyste.
"""

import os
import sys
import pandas as pd
import numpy as np

def run_test():
    """Główna funkcja testująca"""
    print("🔬 NANO-TEST: AUDYT CZYSTOŚCI DANYCH")
    print("=" * 40)

    try:
        # --- KROK 0: Konfiguracja ścieżek ---
        # Ustawia bieżący katalog roboczy na główny katalog projektu
        try:
            from pathlib import Path
            script_path = Path(__file__).resolve()
            project_root = script_path.parent.parent
            os.chdir(project_root)
            print(f"✅ Ustawiono katalog roboczy na: {project_root}")
        except NameError:
            print("⚠️ Nie można automatycznie ustawić ścieżki (tryb interaktywny).")
            # Zakładamy, że skrypt jest uruchamiany z głównego katalogu
            if not os.path.exists("validation_and_labeling"):
                print("❌ Błąd: Uruchom skrypt z głównego katalogu projektu.")
                return

        # Dodaj `validation_and_labeling` do ścieżki, aby umożliwić import
        sys.path.insert(0, os.path.abspath("validation_and_labeling"))
        from feature_calculator import FeatureCalculator
        print("✅ Pomyślnie zaimportowano FeatureCalculator.")

        # --- KROK 1: Załaduj surowe, zwalidowane dane ---
        input_file = "validation_and_labeling/raw_validated/BTCUSDT_raw_validated.feather"
        print(f"\n1. Ładowanie danych wejściowych z:\n   {input_file}")
        if not os.path.exists(input_file):
            print(f"❌ Plik wejściowy nie istnieje! Uruchom najpierw pełny pipeline walidacji.")
            return

        df_raw_validated = pd.read_feather(input_file)
        # Ustaw 'timestamp' jako indeks, tak jak robi to pipeline
        if 'timestamp' in df_raw_validated.columns:
            df_raw_validated.set_index('timestamp', inplace=True)
        print(f"   ✅ Załadowano {len(df_raw_validated):,} wierszy.")

        # --- KROK 2: Uruchom FeatureCalculator ---
        print("\n2. Uruchamianie `feature_calculator.calculate_features()`...")
        calculator = FeatureCalculator()
        df_features, report = calculator.calculate_features(df_raw_validated, "BTCUSDT_Test")
        print("   ✅ Obliczanie cech zakończone.")

        # --- KROK 3: Sprawdź wartości NaN ---
        print("\n3. Sprawdzanie obecności wartości NaN w wynikowym zbiorze cech...")
        nan_sum = df_features.isnull().sum().sum()

        # --- KROK 4: Raport końcowy ---
        print("\n" + "="*20 + " WYNIK TESTU " + "="*20)
        if nan_sum == 0:
            print("✅ HIPOTEZA OBALONA!")
            print("   Moduł `feature_calculator.py` produkuje dane w 100% czyste.")
            print("   Nie znaleziono żadnych wartości NaN.")
            print("\n   Wniosek: Problem musi leżeć gdzieś indziej, a nie w niespójnym")
            print("   traktowaniu NaN, ponieważ nie ma czego traktować.")
        else:
            print("❌ HIPOTEZA POTWIERDZONA!")
            print(f"   Znaleziono łącznie {nan_sum} wartości NaN w finalnym zbiorze cech.")
            print("\n   Wniosek: Niespójne traktowanie tych wartości (`fillna(0)` vs `bfill/ffill`)")
            print("   jest zidentyfikowanym źródłem rozbieżności w predykcjach.")
            
            print("\n   Szczegóły (NaN na kolumnę):")
            print(df_features.isnull().sum()[df_features.isnull().sum() > 0])

        print("=" * 53)

    except Exception as e:
        print(f"\n❌ WYSTĄPIŁ KRYTYCZNY BŁĄD PODCZAS TESTU:")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test() 