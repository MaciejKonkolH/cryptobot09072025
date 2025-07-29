"""
Skrypt do eksportu modelu XGBoost z training3 do struktury FreqTrade
"""

import os
import sys
import json
import pickle
import shutil
from pathlib import Path
import pandas as pd

# Dodaj ścieżkę do modułu training3
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def export_xgboost_model():
    """Eksportuje model XGBoost z training3 do struktury FreqTrade"""
    
    # Ścieżki
    training3_output = project_root / "training3" / "output" / "models"
    freqtrade_inputs = project_root / "ft_bot_clean" / "user_data" / "strategies" / "inputs"
    
    # Znajdź model XGBoost w training3
    model_file = training3_output / "model_multioutput.pkl"
    if not model_file.exists():
        print("❌ Nie znaleziono model_multioutput.pkl w training3/output/models/")
        return
    
    print(f"📦 Znaleziono model: {model_file.name}")
    
    # Znajdź odpowiadający scaler
    scaler_file = training3_output / "scaler.pkl"
    if not scaler_file.exists():
        print("❌ Nie znaleziono scaler.pkl")
        return
    
    # Utwórz katalog dla BTCUSDT jeśli nie istnieje
    btcusdt_dir = freqtrade_inputs / "BTCUSDT"
    btcusdt_dir.mkdir(parents=True, exist_ok=True)
    
    # Skopiuj model i scaler
    model_dest = btcusdt_dir / "xgboost_model.pkl"
    scaler_dest = btcusdt_dir / "scaler.pkl"
    
    shutil.copy2(model_file, model_dest)
    shutil.copy2(scaler_file, scaler_dest)
    
    print(f"✅ Skopiowano model do: {model_dest}")
    print(f"✅ Skopiowano scaler do: {scaler_dest}")
    
    # Sprawdź czy metadata.json już istnieje
    metadata_file = btcusdt_dir / "metadata.json"
    if not metadata_file.exists():
        print("❌ Brak metadata.json - utwórz go ręcznie")
    else:
        print(f"✅ Metadata już istnieje: {metadata_file}")
    
    print("\n🎯 Eksport zakończony!")
    print("📁 Struktura w ft_bot_clean/user_data/strategies/inputs/BTCUSDT/:")
    print("   ├── xgboost_model.pkl")
    print("   ├── scaler.pkl")
    print("   └── metadata.json")

if __name__ == "__main__":
    export_xgboost_model() 