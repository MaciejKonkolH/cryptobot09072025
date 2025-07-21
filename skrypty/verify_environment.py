#!/usr/bin/env python3
"""
Skrypt do weryfikacji zgodności środowiska WSL z RunPod
"""

import subprocess
import sys
from pathlib import Path

def get_installed_packages():
    """Pobiera listę zainstalowanych pakietów"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True, check=True)
        return {line.split('==')[0]: line.split('==')[1] for line in result.stdout.strip().split('\n') if '==' in line}
    except Exception as e:
        print(f"❌ Błąd podczas pobierania pakietów: {e}")
        return {}

def load_runpod_requirements():
    """Ładuje requirements z pliku RunPod"""
    runpod_file = Path(__file__).parent / "runpod_dep.txt"
    try:
        with open(runpod_file, 'r') as f:
            lines = f.read().strip().split('\n')
        return {line.split('==')[0]: line.split('==')[1] for line in lines if '==' in line}
    except Exception as e:
        print(f"❌ Błąd podczas czytania {runpod_file}: {e}")
        return {}

def compare_environments():
    """Porównuje środowiska"""
    print("🔍 Porównywanie środowisk WSL vs RunPod...")
    
    local_packages = get_installed_packages()
    runpod_packages = load_runpod_requirements()
    
    print(f"📦 Lokalnie zainstalowane: {len(local_packages)} pakietów")
    print(f"📦 RunPod requirements: {len(runpod_packages)} pakietów")
    
    # Kluczowe pakiety ML do sprawdzenia
    critical_packages = [
        'tensorflow', 'keras', 'numpy', 'pandas', 'scipy', 
        'scikit-learn', 'matplotlib', 'jupyter'
    ]
    
    print("\n🎯 Sprawdzanie kluczowych pakietów ML:")
    all_match = True
    
    for package in critical_packages:
        if package in runpod_packages:
            runpod_version = runpod_packages[package]
            local_version = local_packages.get(package, "BRAK")
            
            if local_version == runpod_version:
                print(f"✅ {package}: {local_version}")
            else:
                print(f"❌ {package}: RunPod={runpod_version}, Lokalnie={local_version}")
                all_match = False
        else:
            print(f"⚠️  {package}: Nie znaleziono w RunPod requirements")
    
    # Sprawdź GPU
    print("\n🚀 Sprawdzanie dostępności GPU:")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow wykrył GPU: {len(gpus)} urządzeń")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  TensorFlow nie wykrył GPU")
    except ImportError:
        print("❌ TensorFlow nie zainstalowany")
    
    return all_match

def test_ml_consistency():
    """Test podstawowej zgodności ML"""
    print("\n🧪 Test zgodności obliczeń ML...")
    try:
        import tensorflow as tf
        import numpy as np
        
        # Test deterministyczny
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Prosty test sieci
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        X = np.random.random((100, 5)).astype(np.float32)
        y = model.predict(X, verbose=0)
        
        print(f"✅ Test ML zakończony. Checksum wyników: {np.sum(y):.6f}")
        print("   (Ta wartość powinna być identyczna na RunPod)")
        
    except Exception as e:
        print(f"❌ Błąd podczas testu ML: {e}")

if __name__ == "__main__":
    print("🔬 Weryfikacja środowiska WSL vs RunPod")
    print("=" * 50)
    
    matches = compare_environments()
    test_ml_consistency()
    
    print("\n" + "=" * 50)
    if matches:
        print("🎉 Środowiska są zgodne!")
    else:
        print("⚠️  Wykryto różnice - mogą wystąpić rozbieżności w predykcjach")
        print("💡 Uruchom ponownie instalację lub zaktualizuj pakiety") 