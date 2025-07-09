import os
import pandas as pd
import sys

# --- Konfiguracja ---
# Upewnij się, że nazwy plików odpowiadają tym z Twojego ostatniego przebiegu
ROW_INDEX_TO_TEST = 1000
FT_BOT_PATH = '../ft_bot_clean/'

# Plik z cechami z Freqtrade, z którego pobierzemy timestamp
FEATURES_FILE_FT = os.path.join(FT_BOT_PATH, 'user_data/logs/features/features_BTC_USDT_USDT_20250628_231551.csv')

# Plik z predykcjami z walidacji (RunPod)
PREDICTIONS_FILE_VALIDATION = os.path.join(FT_BOT_PATH, 'user_data/strategies/inputs/BTC_USDT_USDT/ml_predictions_validation_BTCUSDT_20250628_193853.csv')

# Wartości predykcji z lokalnego skryptu debug_single_prediction.py (wklejone dla porównania)
LOCAL_PREDICTION = {
    'short_prob': 0.282514959574,
    'hold_prob': 0.268984496593,
    'long_prob': 0.448500484228,
    'predicted_class': 2
}

def print_header(title):
    print("\n" + "="*80)
    print(f"--- {title} ---")
    print("="*80)

def main():
    print_header("PORÓWNANIE ŚRODOWISK PREDYKCJI (LOKALNE vs RUNPOD)")

    # --- 1. Sprawdzenie plików ---
    print("1. Sprawdzanie dostępności plików...")
    for f_path, name in [(FEATURES_FILE_FT, "Cechy Freqtrade"), (PREDICTIONS_FILE_VALIDATION, "Predykcje RunPod")]:
        if not os.path.exists(f_path):
            print(f"❌ KRYTYCZNY BŁĄD: Plik '{name}' nie znaleziony w: {f_path}")
            sys.exit(1)
        print(f"   ✅ Plik '{name}' znaleziony.")

    # --- 2. Pobranie docelowego timestampu ---
    print_header("2. POBIERANIE DOCELOWEGO TIMESTAMP'U")
    try:
        df_features = pd.read_csv(FEATURES_FILE_FT)
        target_timestamp_str = df_features.loc[ROW_INDEX_TO_TEST, 'timestamp']
        # Konwersja do obiektu datetime z UTC dla pewnego porównania
        target_timestamp = pd.to_datetime(target_timestamp_str, utc=True)
        print(f"🎯 Docelowy timestamp dla indeksu {ROW_INDEX_TO_TEST} to: {target_timestamp}")
    except Exception as e:
        print(f"❌ Błąd podczas odczytu pliku z cechami Freqtrade: {e}")
        sys.exit(1)

    # --- 3. Wyszukanie predykcji w pliku z RunPod ---
    print_header("3. WYSZUKIWANIE PREDYKCJI W PLIKU Z RUNPOD")
    try:
        df_validation = pd.read_csv(PREDICTIONS_FILE_VALIDATION)
        # Upewnij się, że kolumna timestamp jest w formacie datetime UTC
        df_validation['timestamp'] = pd.to_datetime(df_validation['timestamp'], utc=True)

        # Znajdź wiersz z pasującym timestampem
        validation_row = df_validation[df_validation['timestamp'] == target_timestamp]

        if validation_row.empty:
            print(f"❌ BŁĄD: Nie znaleziono predykcji dla timestampu {target_timestamp} w pliku z walidacji.")
            print("   Możliwe przyczyny:")
            print("   - Pliki pochodzą z różnych okresów danych.")
            print("   - Występują minimalne różnice w zaokrągleniach sekund.")
            sys.exit(1)
        
        runpod_prediction = validation_row.iloc[0]
        print(f"✅ Znaleziono pasującą predykcję w pliku z walidacji.")

    except Exception as e:
        print(f"❌ Błąd podczas odczytu pliku z predykcjami z walidacji: {e}")
        sys.exit(1)

    # --- 4. OSTATECZNE PORÓWNANIE ---
    print_header("4. OSTATECZNE PORÓWNANIE")

    print(f"{'':<20} | {'LOKALNIE (debug_script)':<30} | {'RUNPOD (validation_file)':<30}")
    print("-" * 85)
    print(f"{'Prawd. SHORT':<20} | {LOCAL_PREDICTION['short_prob']:<30.12f} | {runpod_prediction['short_prob']:<30.12f}")
    print(f"{'Prawd. HOLD':<20} | {LOCAL_PREDICTION['hold_prob']:<30.12f} | {runpod_prediction['hold_prob']:<30.12f}")
    print(f"{'Prawd. LONG':<20} | {LOCAL_PREDICTION['long_prob']:<30.12f} | {runpod_prediction['long_prob']:<30.12f}")
    print("-" * 85)
    
    local_class = ["SHORT", "HOLD", "LONG"][LOCAL_PREDICTION['predicted_class']]
    runpod_class = ["SHORT", "HOLD", "LONG"][runpod_prediction['best_class']]
    
    print(f"{'Przewidziana klasa':<20} | {local_class:<30} | {runpod_class:<30}")
    
    print("\n" + "="*80)
    if local_class == runpod_class and abs(LOCAL_PREDICTION['long_prob'] - runpod_prediction['long_prob']) < 0.0001:
         print("✅ WNIOSEK: Wyniki są ZGODNE. Rozbieżność leży w innym miejscu.")
    else:
         print("❌ WNIOSEK: Wyniki są ROZBIEŻNE. To potwierdza, że środowisko wykonawcze (RunPod vs Lokalnie) jest przyczyną problemu.")
    print("="*80)


if __name__ == "__main__":
    main() 