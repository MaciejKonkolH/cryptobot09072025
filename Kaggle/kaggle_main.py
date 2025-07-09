"""
🎯 KAGGLE MAIN - Training Module Entry Point
Uruchamianie: exec(open('/kaggle/input/trening/kaggle_main.py').read())

DATASET SEPARATION:
- CODE: /kaggle/input/trening/ (kaggle_main.py, config.py, trainer.py, etc.)
- DATA: /kaggle/input/crypto-historical-data/ (.feather files)

The script automatically finds data files and imports code modules from correct paths.

UPDATED: Adapted for crypto-historical-data dataset
- Primary path: /kaggle/input/crypto-historical-data/
- Legacy fallback: /kaggle/input/trening/
"""

import os
import sys
import traceback
from pathlib import Path

def detect_kaggle_environment():
    """Detect and setup Kaggle environment"""
    print("🔍 Detecting environment...")
    
    # Check if we're in Kaggle
    if '/kaggle' in os.getcwd():
        print("✅ Kaggle environment detected")
        return True
    else:
        print("ℹ️ Local environment detected")
        return False

def find_dataset_files():
    """Find available .feather files in Kaggle dataset"""
    print("📂 Looking for training data...")
    
    # Check multiple possible paths - updated for crypto-historical-data dataset
    possible_paths = [
        "/kaggle/input/crypto-historical-data/",
        "/kaggle/input/trening/",  # Legacy fallback
        "/kaggle/input/",
        "/kaggle/input/crypto-historical-data/input/",
        "./input/",
        "./"
    ]
    
    found_files = []
    
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            print(f"   Checking: {path}")
            feather_files = list(path.glob("*.feather"))
            if feather_files:
                print(f"   ✅ Found {len(feather_files)} .feather files in {path}")
                found_files.extend([(str(f), f.name) for f in feather_files])
        else:
            print(f"   ❌ Path not found: {path}")
    
    return found_files

def setup_kaggle_paths(files_found):
    """Setup paths based on found files"""
    if not files_found:
        raise FileNotFoundError("No .feather files found in any expected locations")
    
    # Use the first file's directory as base path
    first_file_path = files_found[0][0]
    base_path = str(Path(first_file_path).parent) + "/"
    
    print(f"📍 Using dataset path: {base_path}")
    return base_path

def patch_config(dataset_path):
    """Dynamically patch config with correct paths"""
    print("⚙️ Updating configuration for Kaggle...")
    
    # FIXED: Add CODE path to sys.path for imports (where config.py is located)
    code_path = "Kaggle"  # Use relative path since __file__ is not defined in exec
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
        print(f"   Added CODE path to Python path: {code_path}")
    
    import config
    
    # Update DATA path in config (where .feather files are located)
    config.TRAINING_DATA_PATH = dataset_path  # Where the data files are located
    config.OUTPUT_BASE_PATH = "Kaggle/output/"  # Use a relative path
    
    print(f"   Code path: {code_path}")
    print(f"   Data path: {config.TRAINING_DATA_PATH}")
    print(f"   Output path: {config.OUTPUT_BASE_PATH}")
    
    return config

def import_training_modules():
    """Import all required training modules"""
    print("📦 Importing training modules...")
    
    try:
        from trainer import StandaloneTrainer
        print("   ✅ trainer module imported")
        
        from data_loader import TrainingDataLoader  
        print("   ✅ data_loader module imported")
        
        from model_builder import DualWindowLSTMBuilder
        print("   ✅ model_builder module imported")
        
        from sequence_generator import MemoryEfficientDataLoader
        print("   ✅ sequence_generator module imported")
    
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def check_tensorflow():
    """Check TensorFlow availability and setup"""
    print("🧠 Checking TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__} available")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   🚀 GPU available: {len(gpus)} device(s)")
            # Try to enable memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("   ✅ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"   ⚠️ GPU setup warning: {e}")
        else:
            print("   💻 Using CPU (no GPU detected)")
            
        return True
        
    except ImportError:
        print("   ❌ TensorFlow not available")
        return False

def main():
    """Main function for Kaggle execution"""
    print("🚀 KAGGLE TRAINING MODULE V3")
    print("=" * 60)
    
    try:
        # Step 1: Environment detection
        is_kaggle = detect_kaggle_environment()
        
        # Step 2: Find dataset files
        files_found = find_dataset_files()
        if files_found:
            print(f"📋 Available files:")
            for file_path, file_name in files_found:
                file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
                print(f"   • {file_name} ({file_size:.1f} MB)")
        
        # Step 3: Setup paths
        dataset_path = setup_kaggle_paths(files_found)
        
        # Step 4: Patch configuration (handles Python path setup)
        config = patch_config(dataset_path)
        
        # Step 5: Check TensorFlow
        tf_available = check_tensorflow()
        if not tf_available:
            raise RuntimeError("TensorFlow not available")
        
        # Step 6: Import modules
        modules_ok = import_training_modules()
        if not modules_ok:
            raise RuntimeError("Failed to import training modules")
        
        # Step 7: Start training
        print("=" * 60)
        print("🎯 STARTING TRAINING...")
        print("=" * 60)
        
        from trainer import StandaloneTrainer
        
        # Initialize trainer
        trainer = StandaloneTrainer()
        
        # Run complete training pipeline
        success = trainer.run_training()
        
        if success:
            print("=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
        else:
            print("=" * 60)
            print("❌ TRAINING FAILED!")
            print("=" * 60)
            
        return success
        
    except Exception as e:
        print("=" * 60)
        print(f"💥 ERROR: {str(e)}")
        print("=" * 60)
        print("🔍 Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 