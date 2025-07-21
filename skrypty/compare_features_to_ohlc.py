import sys
import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURACJA ŚCIEŻEK ---
# Upewnij się, że ścieżki są poprawne dla Twojego projektu

# Ścieżka do pliku z surowymi danymi OHLCV używanymi w backtestingu
RAW_OHLC_FILE = Path('ft_bot_clean/user_data/strategies/inputs/BTC_USDT_USDT/BTCUSDT-1m-futures.feather')

# Ścieżka do pliku z gotowymi cechami i etykietami, który był użyty do treningu
# Nazwa pliku może się różnić w zależności od daty/godziny generowania
# Użyjemy `rglob` do znalezienia odpowiedniego pliku.
FEATURES_DIR = Path('validation_and_labeling/output/')
# Wzorzec do znalezienia pliku, np. '*_single_label.feather'
FEATURES_FILE_PATTERN = '*_single_label.feather'

# --- POCZĄTEK SKRYPTU ---

def find_latest_features_file(directory: Path, pattern: str) -> Path:
    """Znajduje najnowszy plik z cechami pasujący do wzorca."""
    print(f"🔍 Przeszukuję katalog: {directory} dla wzorca: {pattern}")
    files = list(directory.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"Nie znaleziono żadnych plików z cechami w '{directory}' pasujących do '{pattern}'")
    
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"✅ Znaleziono najnowszy plik z cechami: {latest_file.name}")
    return latest_file

def main():
    """Główna funkcja porównująca cechy."""
    
    # --- KROK 0: Dodaj moduł `validation_and_labeling` do ścieżki, aby móc go importować ---
    try:
        # __file__ odnosi się do bieżącego pliku, więc przechodzimy o jeden katalog w górę
        project_root = Path(__file__).parent.parent
        validation_module_path = project_root / 'validation_and_labeling'
        if str(validation_module_path) not in sys.path:
            sys.path.insert(0, str(validation_module_path))
        
        from feature_calculator import FeatureCalculator
        print("✅ Pomyślnie zaimportowano FeatureCalculator.")
    except ImportError as e:
        print(f"❌ BŁĄD: Nie można zaimportować FeatureCalculator. Upewnij się, że skrypt jest w katalogu 'skrypty'.")
        print(f"Błąd importu: {e}")
        return

    # --- KROK 1: Wczytaj oba pliki ---
    print("\n--- KROK 1: Wczytywanie plików ---")
    try:
        # Znajdź najnowszy plik z cechami
        features_file_path = find_latest_features_file(FEATURES_DIR, FEATURES_FILE_PATTERN)
        
        df_original_features = pd.read_feather(features_file_path)
        print(f"📄 Wczytano oryginalne cechy: {features_file_path.name} ({len(df_original_features):,} wierszy)")
        
        df_raw_ohlc = pd.read_feather(RAW_OHLC_FILE)
        print(f"📄 Wczytano surowe dane OHLC: {RAW_OHLC_FILE.name} ({len(df_raw_ohlc):,} wierszy)")
        
    except FileNotFoundError as e:
        print(f"❌ BŁĄD: Plik nie został znaleziony: {e}")
        return

    # --- KROK 2: Przygotowanie i wyrównanie danych ---
    print("\n--- KROK 2: Przygotowanie i wyrównanie danych ---")
    
    # Standaryzacja kolumn `date` i `timestamp`
    if 'date' in df_raw_ohlc.columns:
        df_raw_ohlc.rename(columns={'date': 'timestamp'}, inplace=True)
    
    df_original_features['timestamp'] = pd.to_datetime(df_original_features['timestamp'])
    df_raw_ohlc['timestamp'] = pd.to_datetime(df_raw_ohlc['timestamp'])
    
    # Ustawienie indeksu na timestamp
    df_original_features.set_index('timestamp', inplace=True)
    df_raw_ohlc.set_index('timestamp', inplace=True)

    # Wyrównanie stref czasowych (jeśli jedna jest świadoma, a druga nie)
    if df_original_features.index.tz is not None and df_raw_ohlc.index.tz is None:
        df_raw_ohlc.index = df_raw_ohlc.index.tz_localize(df_original_features.index.tz)
        print("   - Wyrównano strefy czasowe: dodano TZ do danych OHLC.")
    elif df_raw_ohlc.index.tz is not None and df_original_features.index.tz is None:
        df_original_features.index = df_original_features.index.tz_localize(df_raw_ohlc.index.tz)
        print("   - Wyrównano strefy czasowe: dodano TZ do danych z cechami.")

    # Znalezienie wspólnego zakresu dat
    common_index = df_original_features.index.intersection(df_raw_ohlc.index)
    if common_index.empty:
        print("❌ BŁĄD: Brak wspólnego zakresu dat między plikami. Nie można porównać.")
        return
        
    df_original_features = df_original_features.loc[common_index]
    df_raw_ohlc = df_raw_ohlc.loc[common_index]
    
    print(f"   - Znaleziono wspólny zakres: {len(common_index):,} wierszy.")
    print(f"   - Od: {common_index.min()}")
    print(f"   - Do: {common_index.max()}")
    
    # --- KROK 3: Obliczenie cech na nowo z surowych danych ---
    print("\n--- KROK 3: Obliczanie cech na nowo z surowych danych OHLCV ---")
    feature_calculator = FeatureCalculator()
    # Przekazujemy tylko dane OHLCV, ponieważ FeatureCalculator sam sobie je zwaliduje
    # Ważne: `calculate_features` odrzuca okres "rozgrzewkowy", więc wynikowy DataFrame będzie krótszy
    df_recalculated, report = feature_calculator.calculate_features(df_raw_ohlc, "recalculation")
    
    print(f"   - Obliczono na nowo cechy. Wynik: {len(df_recalculated):,} wierszy (po odrzuceniu okresu rozgrzewkowego).")

    # --- KROK 4: Porównanie cech ---
    print("\n--- KROK 4: Porównanie cech (oryginalne vs. nowo obliczone) ---")
    
    # Ponownie wyrównaj indeksy, ponieważ recalculate mogło skrócić dane
    final_common_index = df_original_features.index.intersection(df_recalculated.index)
    df_original = df_original_features.loc[final_common_index]
    df_recalculated = df_recalculated.loc[final_common_index]
    
    print(f"   - Ostateczny wspólny zakres do porównania: {len(final_common_index):,} wierszy.\n")
    
    feature_columns = [
        'high_change', 'low_change', 'close_change', 'volume_change',
        'price_to_ma1440', 'price_to_ma43200',
        'volume_to_ma1440', 'volume_to_ma43200'
    ]
    
    total_mismatches = 0
    
    for feature in feature_columns:
        if feature not in df_original.columns or feature not in df_recalculated.columns:
            print(f"⚠️ OSTRZEŻENIE: Kolumna '{feature}' nie istnieje w obu plikach.")
            continue
            
        # Użyj `np.isclose` do bezpiecznego porównywania liczb zmiennoprzecinkowych
        are_close = np.isclose(df_original[feature], df_recalculated[feature])
        mismatches = ~are_close
        mismatch_count = mismatches.sum()
        total_mismatches += mismatch_count
        
        if mismatch_count == 0:
            print(f"✅ ZGODNOŚĆ: Cechy '{feature}' są identyczne.")
        else:
            print(f"❌ NIEZGODNOŚĆ: Znaleziono {mismatch_count} różnic w cesze '{feature}'.")
            
            # Pokaż kilka przykładów
            df_diff = pd.DataFrame({
                'original': df_original.loc[mismatches, feature],
                'recalculated': df_recalculated.loc[mismatches, feature],
            })
            df_diff['difference'] = df_diff['recalculated'] - df_diff['original']
            
            print("   Przykłady różnic (pierwsze 5):")
            print(df_diff.head().to_string())
            print("-" * 50)
            
    # --- KROK 5: Ostateczny werdykt ---
    print("\n--- KROK 5: Ostateczny werdykt ---")
    if total_mismatches == 0:
        print("\n🎉 SUKCES! Plik z cechami jest w 100% zgodny z surowymi danymi historycznymi.")
        print("Oznacza to, że oba moduły (trening i backtesting) operują na tych samych danych bazowych.")
    else:
        print(f"\n🚨 PORAŻKA! Wykryto łącznie {total_mismatches} niezgodności.")
        print("Oznacza to, że plik z cechami został wygenerowany na podstawie INNYCH danych niż te,")
        print("których używasz do backtestingu. To jest fundamentalna przyczyna obserwowanych anomalii.")

if __name__ == '__main__':
    main() 