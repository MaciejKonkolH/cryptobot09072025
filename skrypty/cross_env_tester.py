import os
import sys
import numpy as np
import tensorflow as tf
import joblib

# --- NOWA KONCEPCJA: CZYSTY SKRYPT TESTOWY ---
# Ten skrypt będzie identyczny dla obu środowisk.
# Jedyną rzeczą do przeniesienia będzie plik z danymi 'test_sequence.npy'.

def get_paths():
    """Automatycznie wyszukuje komponenty (model, skaler, dane) w predefiniowanych lokalizacjach."""

    # --- Wyszukiwanie artefaktów modelu ---
    # Lista potencjalnych ścieżek do folderu z modelem, od najbardziej prawdopodobnej
    potential_model_paths = [
        '/workspace/kaggle/working/models/BTCUSDT/',  # Priorytet 1: Podana przez użytkownika ścieżka RunPod/Kaggle
        '/kaggle/working/models/BTCUSDT/',  # Priorytet 2: Poprzednia próba
        '/workspace/crypto/ft_bot_clean/user_data/strategies/inputs/BTC_USDT_USDT/', # Priorytet 3: Inna możliwa ścieżka RunPod
        os.path.join(os.getcwd(), 'ft_bot_clean/user_data/strategies/inputs/BTC_USDT_USDT/') # Priorytet 4: Lokalna ścieżka (relatywna)
    ]
    
    model_artifacts_path = ""
    for path in potential_model_paths:
        # Sprawdzamy istnienie pliku modelu, bo sam folder może istnieć, ale być pusty
        if os.path.exists(os.path.join(path, 'best_model.h5')):
            model_artifacts_path = path
            print(f"✅ Znaleziono artefakty modelu w: {path}")
            break

    if not model_artifacts_path:
        # Ostatnia deska ratunku - ścieżka lokalna względem pliku skryptu
        try:
            # Działa przy `python skrypty/f.py`, szuka ../ft_bot_clean...
            local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ft_bot_clean', 'user_data', 'strategies', 'inputs', 'BTC_USDT_USDT'))
            if os.path.exists(os.path.join(local_path, 'best_model.h5')):
                 model_artifacts_path = local_path
                 print(f"✅ Znaleziono artefakty modelu w (fallback): {local_path}")
        except NameError:
             pass # Ignorujemy błąd __file__ w trybie exec

    # --- Wyszukiwanie pliku z danymi ---
    sequence_data_path = ''
    data_search_paths = [
        os.getcwd(),                                # Bieżący katalog roboczy
        os.path.join(os.getcwd(), 'skrypty'),       # Podfolder 'skrypty' w bieżącym katalogu
        '/kaggle/working/',                         # Główny folder Kaggle
        '/kaggle/working/skrypty/',                 # Folder skryptów w Kaggle
        '/workspace/crypto/skrypty/',               # Poprzednia domyślna ścieżka RunPod
    ]
    try:
        data_search_paths.append(os.path.dirname(__file__)) # Folder samego skryptu
    except NameError:
        pass # Ignorujemy błąd __file__ w trybie exec

    for path in data_search_paths:
        candidate_path = os.path.join(path, 'test_sequence.npy')
        if os.path.exists(candidate_path):
            sequence_data_path = candidate_path
            print(f"✅ Znaleziono dane testowe w: {sequence_data_path}")
            break

    # --- Komunikaty o błędach ---
    if not model_artifacts_path:
        print(f"❌ KRYTYCZNY BŁĄD: Nie udało się zlokalizować folderu z plikami 'best_model.h5' i 'scaler.pkl'.")
        print("   Sprawdzone ścieżki:")
        for p in potential_model_paths: print(f"   - {p}")
        
    if not sequence_data_path:
        print(f"❌ KRYTYCZNY BŁĄD: Nie można zlokalizować pliku 'test_sequence.npy'.")
        print("   Przeszukano m.in.: bieżący folder, folder 'skrypty'.")

    return {
        "model": os.path.join(model_artifacts_path, 'best_model.h5') if model_artifacts_path else '',
        "scaler": os.path.join(model_artifacts_path, 'scaler.pkl') if model_artifacts_path else '',
        "sequence_data": sequence_data_path
    }

def print_header(title):
    print("\n" + "="*80)
    print(f"--- {title} ---")
    print("="*80)

def main():
    paths = get_paths()
    
    print_header("1. WERSJE BIBLIOTEK I ŚCIEŻKI")
    import sklearn
    print(f"🐍 Python: {sys.version.split()[0]}, TensorFlow: {tf.__version__}, scikit-learn: {sklearn.__version__}, NumPy: {np.__version__}")
    print(f"   - Ścieżka modelu: {paths['model']}")
    print(f"   - Ścieżka skalera: {paths['scaler']}")
    print(f"   - Ścieżka danych: {paths['sequence_data']}")

    # --- ŁADOWANIE ---
    print_header("2. ŁADOWANIE KOMPONENTÓW")
    try:
        model = tf.keras.models.load_model(paths['model'], compile=False)
        scaler_dict = joblib.load(paths['scaler'])
        scaler = scaler_dict['scaler']
        raw_sequence_data = np.load(paths['sequence_data'])
        print("✅ Model, skaler i dane testowe załadowane pomyślnie.")
        print(f"   - Kształt wczytanych danych: {raw_sequence_data.shape}")
    except Exception as e:
        print(f"❌ KRYTYCZNY BŁĄD podczas ładowania: {e}")
        sys.exit(1)

    if raw_sequence_data.shape != (120, 8):
        print(f"❌ BŁĄD KSZTAŁTU DANYCH! Oczekiwano (120, 8), otrzymano {raw_sequence_data.shape}")
        sys.exit(1)

    # --- PRZETWARZANIE I PREDYKCJA ---
    print_header("3. PRZETWARZANIE I PREDYKCJA")
    scaled_data = scaler.transform(raw_sequence_data)
    final_input = scaled_data.reshape(1, 120, 8)
    prediction_vector = model.predict(final_input)[0]
    
    # --- WYNIKI ---
    print_header("4. OSTATECZNY WYNIK PREDYKCJI")
    class_map = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
    predicted_class_idx = np.argmax(prediction_vector)
    
    print(f"🎯 Surowy wektor predykcji (prawdopodobieństwa):")
    print(f"   - SHORT (0): {prediction_vector[0]:.12f}")
    print(f"   - HOLD  (1): {prediction_vector[1]:.12f}")
    print(f"   - LONG  (2): {prediction_vector[2]:.12f}")
    print("\n" + "-"*40)
    print(f"🏆 PRZEWIDZIANA KLASA: {class_map.get(predicted_class_idx, 'UNKNOWN')}")
    print("-" * 40)

if __name__ == "__main__":
    main() 