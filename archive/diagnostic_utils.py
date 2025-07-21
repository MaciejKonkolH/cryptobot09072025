"""
🔍 DIAGNOSTIC UTILS - Wspólny moduł diagnostyczny
Używany przez oba moduły (trainer.py i signal_generator.py) do zapewnienia
identycznej metodologii zapisywania danych diagnostycznych.
"""

import json
import hashlib
import numpy as np
import os
from datetime import datetime
from typing import Any, Optional, Dict, List

def save_model_scaler_audit(model: Any, scaler: Any, module_name: str, output_dir: str = "./raporty/") -> str:
    """
    🔍 ZAPISUJE KOMPLETNY FINGERPRINT MODELU I SCALERA
    
    Args:
        model: Model TensorFlow/Keras
        scaler: Scaler (sklearn lub podobny)
        module_name: Nazwa modułu ("trainer" lub "freqtrade")
        output_dir: Katalog wyjściowy
        
    Returns:
        str: Ścieżka do zapisanego pliku
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # MODEL FINGERPRINT
        model_weights = model.get_weights()
        model_weights_str = str([w.tolist() for w in model_weights])
        model_weights_hash = hashlib.md5(model_weights_str.encode()).hexdigest()
        
        # Collect model layers info
        layers_info = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': type(layer).__name__,
                'trainable': layer.trainable,
                'input_shape': str(layer.input_shape) if hasattr(layer, 'input_shape') else None,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else None
            }
            
            # Add layer-specific info
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units
            if hasattr(layer, 'activation'):
                layer_info['activation'] = str(layer.activation)
            if hasattr(layer, 'dropout'):
                layer_info['dropout'] = layer.dropout
                
            layers_info.append(layer_info)
        
        # SCALER FINGERPRINT
        scaler_info = {
            'type': type(scaler).__name__,
            'n_features_in': getattr(scaler, 'n_features_in_', None),
            'feature_names_in': getattr(scaler, 'feature_names_in_', None),
        }
        
        # Add scaler-specific attributes
        if hasattr(scaler, 'mean_'):
            scaler_info['mean'] = scaler.mean_.tolist()
        if hasattr(scaler, 'scale_'):
            scaler_info['scale'] = scaler.scale_.tolist()
        if hasattr(scaler, 'center_'):
            scaler_info['center'] = scaler.center_.tolist()
        if hasattr(scaler, 'quantile_range'):
            scaler_info['quantile_range'] = scaler.quantile_range
            
        # Create audit data
        audit_data = {
            'metadata': {
                'module': module_name,
                'timestamp': timestamp,
                'tensorflow_version': None,
                'numpy_version': np.__version__
            },
            'model': {
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'total_params': model.count_params(),
                'layers_count': len(model.layers),
                'layers_info': layers_info,
                'weights_hash': model_weights_hash,
                'optimizer': str(model.optimizer) if hasattr(model, 'optimizer') else None,
                'loss': str(model.loss) if hasattr(model, 'loss') else None
            },
            'scaler': scaler_info
        }
        
        # Try to get TensorFlow version
        try:
            import tensorflow as tf
            audit_data['metadata']['tensorflow_version'] = tf.__version__
        except ImportError:
            audit_data['metadata']['tensorflow_version'] = "Not available"
        
        # Save to file
        filename = f"model_scaler_audit_{module_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)
        
        print(f"🔍 Model & Scaler audit saved: {filename}")
        print(f"   📊 Model hash: {model_weights_hash[:16]}...")
        print(f"   📊 Scaler type: {scaler_info['type']}")
        print(f"   📊 Features: {scaler_info.get('n_features_in', 'Unknown')}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving model/scaler audit: {e}")
        return ""

def save_scaled_features_sample(scaled_features: np.ndarray, timestamps: Optional[List], 
                               module_name: str, output_dir: str = "./raporty/", 
                               sample_size: int = 100) -> str:
    """
    🔍 ZAPISUJE PRÓBKĘ PRZESKALOWANYCH FEATURES
    
    Args:
        scaled_features: Przeskalowane features (numpy array)
        timestamps: Lista timestampów (opcjonalna)
        module_name: Nazwa modułu ("trainer" lub "freqtrade")
        output_dir: Katalog wyjściowy
        sample_size: Liczba próbek do zapisania
        
    Returns:
        str: Ścieżka do zapisanego pliku
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit sample size to available data
        actual_sample_size = min(sample_size, len(scaled_features))
        
        # Prepare sample data
        sample_data = {
            'metadata': {
                'module': module_name,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'original_shape': list(scaled_features.shape),
                'sample_size': actual_sample_size,
                'numpy_version': np.__version__
            },
            'scaled_features': scaled_features[:actual_sample_size].tolist(),
            'timestamps': timestamps[:actual_sample_size].tolist() if timestamps is not None else None,
            'statistics': {
                'mean': np.mean(scaled_features[:actual_sample_size], axis=0).tolist(),
                'std': np.std(scaled_features[:actual_sample_size], axis=0).tolist(),
                'min': np.min(scaled_features[:actual_sample_size], axis=0).tolist(),
                'max': np.max(scaled_features[:actual_sample_size], axis=0).tolist()
            }
        }
        
        # Save to file
        filename = f"scaled_features_sample_{module_name}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"🔍 Scaled features sample saved: {filename}")
        print(f"   📊 Sample size: {actual_sample_size}")
        print(f"   📊 Features shape: {scaled_features.shape}")
        print(f"   📊 Mean range: [{np.min(sample_data['statistics']['mean']):.6f}, {np.max(sample_data['statistics']['mean']):.6f}]")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving scaled features sample: {e}")
        return ""

def compare_audit_files(file1: str, file2: str, output_dir: str = "./raporty/") -> str:
    """
    🔍 PORÓWNUJE DWA PLIKI AUDIT I POKAZUJE RÓŻNICE
    
    Args:
        file1: Ścieżka do pierwszego pliku audit
        file2: Ścieżka do drugiego pliku audit
        output_dir: Katalog wyjściowy dla raportu
        
    Returns:
        str: Ścieżka do raportu porównawczego
    """
    try:
        # Load both files
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        # Compare data
        comparison_result = {
            'metadata': {
                'file1': file1,
                'file2': file2,
                'comparison_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            'model_comparison': {},
            'scaler_comparison': {},
            'summary': {
                'models_identical': True,
                'scalers_identical': True,
                'differences_found': []
            }
        }
        
        # Compare model weights hash
        hash1 = data1.get('model', {}).get('weights_hash')
        hash2 = data2.get('model', {}).get('weights_hash')
        
        if hash1 != hash2:
            comparison_result['summary']['models_identical'] = False
            comparison_result['summary']['differences_found'].append('Model weights hash mismatch')
            comparison_result['model_comparison']['weights_hash'] = {
                'file1': hash1,
                'file2': hash2,
                'match': False
            }
        else:
            comparison_result['model_comparison']['weights_hash'] = {
                'value': hash1,
                'match': True
            }
        
        # Compare model architecture
        arch_fields = ['input_shape', 'output_shape', 'total_params', 'layers_count']
        for field in arch_fields:
            val1 = data1.get('model', {}).get(field)
            val2 = data2.get('model', {}).get(field)
            
            if val1 != val2:
                comparison_result['summary']['models_identical'] = False
                comparison_result['summary']['differences_found'].append(f'Model {field} mismatch')
                comparison_result['model_comparison'][field] = {
                    'file1': val1,
                    'file2': val2,
                    'match': False
                }
            else:
                comparison_result['model_comparison'][field] = {
                    'value': val1,
                    'match': True
                }
        
        # Compare scaler
        scaler_fields = ['type', 'n_features_in', 'mean', 'scale', 'center']
        for field in scaler_fields:
            val1 = data1.get('scaler', {}).get(field)
            val2 = data2.get('scaler', {}).get(field)
            
            if val1 != val2:
                comparison_result['summary']['scalers_identical'] = False
                comparison_result['summary']['differences_found'].append(f'Scaler {field} mismatch')
                comparison_result['scaler_comparison'][field] = {
                    'file1': val1,
                    'file2': val2,
                    'match': False
                }
            else:
                comparison_result['scaler_comparison'][field] = {
                    'value': val1,
                    'match': True
                }
        
        # Save comparison report
        os.makedirs(output_dir, exist_ok=True)
        filename = f"audit_comparison_report_{comparison_result['metadata']['comparison_timestamp']}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"🔍 Audit comparison report saved: {filename}")
        print(f"   📊 Models identical: {comparison_result['summary']['models_identical']}")
        print(f"   📊 Scalers identical: {comparison_result['summary']['scalers_identical']}")
        
        if comparison_result['summary']['differences_found']:
            print(f"   ⚠️ Differences found:")
            for diff in comparison_result['summary']['differences_found']:
                print(f"      - {diff}")
        else:
            print(f"   ✅ No differences found!")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error comparing audit files: {e}")
        return ""

def compare_scaled_features(file1: str, file2: str, output_dir: str = "./raporty/") -> str:
    """
    🔍 PORÓWNUJE PRZESKALOWANE FEATURES Z DWÓCH PLIKÓW
    
    Args:
        file1: Ścieżka do pierwszego pliku z features
        file2: Ścieżka do drugiego pliku z features
        output_dir: Katalog wyjściowy dla raportu
        
    Returns:
        str: Ścieżka do raportu porównawczego
    """
    try:
        # Load both files
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        # Convert to numpy arrays
        features1 = np.array(data1['scaled_features'])
        features2 = np.array(data2['scaled_features'])

        if features1.shape != features2.shape:
            report = {
                "metadata": {
                    "file1": file1,
                    "file2": file2,
                    "comparison_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                },
                "error": "Shapes of scaled_features are different",
                "shape1": list(features1.shape),
                "shape2": list(features2.shape)
            }
        else:
            diff = np.abs(features1 - features2)
            total_elements = features1.size
            differing_elements = np.sum(diff > 1e-9) # Use a small tolerance for "identical"

            # Tolerance analysis
            tolerances = {
                "strict (1e-6)": 1e-6,
                "medium (1e-4)": 1e-4,
                "loose (1%)": 0.01
            }
            tolerance_results = {}
            for name, tol in tolerances.items():
                # Using relative tolerance for percentage check
                rtol = tol if name == "loose (1%)" else 0
                atol = tol if name != "loose (1%)" else 0
                
                close_elements = np.isclose(features1, features2, rtol=rtol, atol=atol)
                num_within_tolerance = np.sum(close_elements)
                
                tolerance_results[name] = {
                    "elements_within_tolerance": int(num_within_tolerance),
                    "percentage_within_tolerance": (num_within_tolerance / total_elements) * 100 if total_elements > 0 else 100,
                    "elements_outside_tolerance": int(total_elements - num_within_tolerance),
                    "percentage_outside_tolerance": ((total_elements - num_within_tolerance) / total_elements) * 100 if total_elements > 0 else 0
                }

            report = {
                "metadata": {
                    "file1": os.path.basename(file1),
                    "file2": os.path.basename(file2),
                    "comparison_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "compared_samples": features1.shape[0],
                    "compared_features_per_sample": features1.shape[1] if features1.ndim > 1 else 1,
                },
                "summary": {
                    "total_values": int(total_elements),
                    "identical_values": int(total_elements - differing_elements),
                    "differing_values": int(differing_elements),
                    "percentage_identical": ((total_elements - differing_elements) / total_elements) * 100 if total_elements > 0 else 100,
                    "percentage_differing": (differing_elements / total_elements) * 100 if total_elements > 0 else 0,
                },
                "statistics": {
                     "max_absolute_difference": float(np.max(diff)) if differing_elements > 0 else 0.0,
                     "mean_absolute_difference": float(np.mean(diff)) if differing_elements > 0 else 0.0,
                },
                "tolerance_analysis": tolerance_results
            }

        # Save comparison report
        os.makedirs(output_dir, exist_ok=True)
        filename = f"scaled_features_comparison_{report['metadata']['comparison_timestamp']}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"🔍 Scaled features comparison saved: {filename}")
        if "error" in report:
            print(f"   ❌ Error: {report['error']}")
        else:
            print(f"   📊 Differing values: {report['summary']['differing_values']}/{report['summary']['total_values']} ({report['summary']['percentage_differing']:.2f}%)")
            print(f"   📊 Max difference: {report['statistics']['max_absolute_difference']:.10f}")
            print(f"   📊 Within 1% tolerance: {report['tolerance_analysis']['loose (1%)']['percentage_within_tolerance']:.2f}%")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error comparing scaled features: {e}")
        return ""

# Convenience function for complete diagnostic workflow
def run_complete_diagnostic(model, scaler, scaled_features, timestamps, module_name, output_dir="./raporty/"):
    """
    🔍 URUCHAMIA KOMPLETNĄ DIAGNOSTYKĘ
    Zapisuje audit modelu/scalera oraz próbkę przeskalowanych features
    """
    print(f"\n🔍 Running complete diagnostic for {module_name}...")
    
    audit_file = save_model_scaler_audit(model, scaler, module_name, output_dir)
    features_file = save_scaled_features_sample(scaled_features, timestamps, module_name, output_dir)
    
    print(f"✅ Complete diagnostic finished for {module_name}")
    return audit_file, features_file 