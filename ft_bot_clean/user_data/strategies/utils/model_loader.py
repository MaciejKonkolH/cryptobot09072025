"""
Model Loader - Ładowanie modeli ML dla Enhanced ML Strategy

Odpowiedzialny za:
- Ładowanie modeli .h5 format z nowej struktury inputs/ (zmienione z .keras na .h5)
- Ładowanie scalerów i metadanych
- Dynamiczny window_size z metadata.json
- Error handling per para
- Cache modeli w pamięci

NOWA STRUKTURA v2.1 (ZAKTUALIZOWANA DLA H5):
user_data/strategies/inputs/BTCUSDT/
├── best_model.h5      # Zmienione z .keras na .h5 (fix dla TensorFlow 2.15.0 LSTM bug)
├── scaler.pkl  
└── metadata.json
"""

import os
import json
import logging
import joblib
import tensorflow as tf
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader:
    """Ładuje i zarządza modelami ML per para z nowej struktury inputs/"""
    
    def __init__(self, base_artifacts_path: str = "user_data/strategies/inputs"):
        """
        Inicjalizacja Model Loader
        
        Args:
            base_artifacts_path: Bazowa ścieżka do modeli (domyślnie: user_data/strategies/inputs)
        """
        self.base_artifacts_path = base_artifacts_path
        self.models_cache = {}
        self.scalers_cache = {}
        self.metadata_cache = {}
        
    def _convert_pair_to_directory_name(self, pair: str) -> str:
        """
        Konwertuje nazwę pary na nazwę katalogu
        
        Args:
            pair: Nazwa pary (np. "BTC/USDT")
            
        Returns:
            str: Nazwa katalogu (np. "BTCUSDT")
        """
        # Usuń "/" i zamień na format BTCUSDT
        return pair.replace('/', '').replace(':', '')
        
    def load_model_for_pair(self, pair: str, model_dir: str = None) -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
        """
        Ładuje model, scaler i metadata dla konkretnej pary
        
        Args:
            pair: Nazwa pary (np. "BTC/USDT")
            model_dir: Nazwa katalogu z modelami (opcjonalne - auto-generowana z pair)
            
        Returns:
            Tuple[model, scaler, metadata]: Załadowane komponenty lub None jeśli błąd
        """
        # Auto-generuj model_dir jeśli nie podano
        if model_dir is None:
            model_dir = self._convert_pair_to_directory_name(pair)
            
        cache_key = f"{pair}_{model_dir}"
        
        try:
            # Sprawdź cache
            if cache_key in self.models_cache:
                logger.debug(f"📋 Using cached model for {pair}")
                return (
                    self.models_cache[cache_key],
                    self.scalers_cache[cache_key], 
                    self.metadata_cache[cache_key]
                )
            
            # Waliduj że katalog istnieje
            artifacts_dir = os.path.join(self.base_artifacts_path, model_dir)
            if not os.path.exists(artifacts_dir):
                error_msg = f"Model directory not found: {artifacts_dir}"
                logger.error(f"❌ {pair}: {error_msg}")
                return None, None, None
            
            # Załaduj metadata
            metadata = self._load_metadata(artifacts_dir, pair)
            if metadata is None:
                return None, None, None
            
            # Załaduj model .h5
            model = self._load_keras_model(artifacts_dir, pair)
            if model is None:
                return None, None, None
            
            # Załaduj scaler
            scaler = self._load_scaler(artifacts_dir, pair)
            if scaler is None:
                return None, None, None
            
            # Waliduj kompatybilność modelu
            if not self._validate_model_compatibility(metadata, pair):
                return None, None, None
            
            # Cache załadowane komponenty
            self.models_cache[cache_key] = model
            self.scalers_cache[cache_key] = scaler
            self.metadata_cache[cache_key] = metadata
            
            window_size = self._extract_window_size(metadata)
            logger.info(f"✅ {pair}: Model loaded successfully from {artifacts_dir} (window_size: {window_size})")
            
            return model, scaler, metadata
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error loading model: {e}")
            return None, None, None
    
    def _load_metadata(self, artifacts_dir: str, pair: str) -> Optional[Dict]:
        """Ładuje metadata.json (NOWA NAZWA PLIKU)"""
        try:
            # 🔥 ZMIANA: metadata.json zamiast model_metadata.json
            metadata_path = os.path.join(artifacts_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.error(f"❌ {pair}: metadata.json not found in {artifacts_dir}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.debug(f"✅ {pair}: Metadata loaded from metadata.json")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error loading metadata: {e}")
            return None
    
    def _load_keras_model(self, artifacts_dir: str, pair: str) -> Optional[Any]:
        """Ładuje best_model.h5 (ZMIENIONY FORMAT Z KERAS NA H5)"""
        try:
            # 🔥 ZMIANA: best_model.h5 zamiast best_model.keras (fix dla TensorFlow 2.15.0 LSTM bug)
            model_filename = "best_model.h5"
            model_path = os.path.join(artifacts_dir, model_filename)
            
            if not os.path.exists(model_path):
                logger.error(f"❌ {pair}: Model file not found: {model_path}")
                return None
            
            # Załaduj model .h5 z opcją compile=False, aby uniknąć błędów kompatybilności
            model = tf.keras.models.load_model(model_path, compile=False)
            
            logger.debug(f"✅ {pair}: H5 model loaded from {model_filename}")
            return model
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error loading H5 model: {e}")
            return None
    
    def _load_scaler(self, artifacts_dir: str, pair: str) -> Optional[Any]:
        """Ładuje scaler.pkl - obsługuje nowy format z metadata"""
        try:
            scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
            
            if not os.path.exists(scaler_path):
                logger.error(f"❌ {pair}: scaler.pkl not found in {artifacts_dir}")
                return None

            scaler_data = joblib.load(scaler_path)
            
            # Obsługa nowego formatu z metadata
            if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
                # Nowy format: {'scaler': actual_scaler, 'scaler_type': ..., ...}
                scaler = scaler_data['scaler']
                logger.debug(f"✅ {pair}: Scaler loaded from metadata format (type: {scaler_data.get('scaler_type', 'unknown')})")
            else:
                # Stary format: bezpośrednio scaler object
                scaler = scaler_data
                logger.debug(f"✅ {pair}: Scaler loaded from legacy format")
            
            return scaler
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error loading scaler: {e}")
            return None
    
    def _validate_model_compatibility(self, metadata: Dict, pair: str) -> bool:
        """Waliduje kompatybilność modelu z strategią"""
        try:
            # 🔥 OBSŁUGA NOWEJ STRUKTURY METADATA.JSON
            
            # Sprawdź czy ma training_config (nowa struktura)
            if 'training_config' in metadata:
                # Nowa struktura z training_config.sequence_length
                training_config = metadata['training_config']
                
                if 'sequence_length' not in training_config:
                    logger.error(f"❌ {pair}: Missing sequence_length in training_config")
                    return False
                
                window_size = training_config['sequence_length']
                num_features = 8  # Default dla strategii (z features_shape można sprawdzić)
                
                # Sprawdź data_info jeśli dostępne
                if 'data_info' in metadata and 'features_shape' in metadata['data_info']:
                    features_shape = metadata['data_info']['features_shape']
                    if len(features_shape) >= 2:
                        num_features = features_shape[1]
                
            else:
                # Stara struktura z input_shape (backward compatibility)
                required_fields = ['input_shape']
                for field in required_fields:
                    if field not in metadata:
                        logger.error(f"❌ {pair}: Missing required metadata field: {field}")
                        return False
                
                input_shape = metadata['input_shape']
                if not isinstance(input_shape, list) or len(input_shape) != 2:
                    logger.error(f"❌ {pair}: Invalid input_shape format: {input_shape}")
                    return False
                
                window_size, num_features = input_shape
            
            # Waliduj window_size
            if window_size < 60 or window_size > 240:
                logger.warning(f"⚠️ {pair}: Unusual window_size: {window_size} (expected 60-240)")
            
            # Waliduj num_features (strategia oczekuje 8 cech)
            if num_features != 8:
                logger.error(f"❌ {pair}: Invalid num_features: {num_features} (expected 8)")
                return False
            
            logger.debug(f"✅ {pair}: Model compatibility validated (window: {window_size}, features: {num_features})")
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error validating model compatibility: {e}")
            return False
    
    def _extract_window_size(self, metadata: Dict) -> int:
        """
        Wyciąga window_size z metadanych (obsługuje obie struktury)
        
        Args:
            metadata: Metadane modelu
            
        Returns:
            int: Window size lub default 60
        """
        try:
            # 🔥 OBSŁUGA NOWEJ STRUKTURY METADATA.JSON
            
            # Sprawdź nową strukturę z training_config.sequence_length
            if 'training_config' in metadata and 'sequence_length' in metadata['training_config']:
                window_size = int(metadata['training_config']['sequence_length'])
                logger.debug(f"✅ Window size extracted from training_config.sequence_length: {window_size}")
                return window_size
            
            # Sprawdź starą strukturę z input_shape (backward compatibility)
            elif 'input_shape' in metadata:
                input_shape = metadata.get('input_shape', [60, 8])
                window_size = int(input_shape[0])
                logger.debug(f"✅ Window size extracted from input_shape: {window_size}")
                return window_size
            
            else:
                logger.warning("⚠️ Could not extract window_size from metadata, using default: 60")
                return 60
                
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f"⚠️ Error extracting window_size from metadata: {e}, using default: 60")
            return 60
    
    def get_window_size_for_pair(self, pair: str, model_dir: str = None) -> int:
        """
        Pobiera window_size dla pary bez ładowania całego modelu
        
        Args:
            pair: Nazwa pary
            model_dir: Katalog modelu (opcjonalne - auto-generowany)
            
        Returns:
            int: Window size lub default 60
        """
        try:
            if model_dir is None:
                model_dir = self._convert_pair_to_directory_name(pair)
                
            artifacts_dir = os.path.join(self.base_artifacts_path, model_dir)
            metadata = self._load_metadata(artifacts_dir, pair)
            
            if metadata:
                return self._extract_window_size(metadata)
            else:
                logger.warning(f"⚠️ {pair}: Could not load metadata, using default window_size: 60")
                return 60
                
        except Exception as e:
            logger.warning(f"⚠️ {pair}: Error getting window_size: {e}, using default: 60")
            return 60
    
    def clear_cache_for_pair(self, pair: str, model_dir: str = None) -> None:
        """Usuwa z cache model dla konkretnej pary"""
        if model_dir is None:
            model_dir = self._convert_pair_to_directory_name(pair)
            
        cache_key = f"{pair}_{model_dir}"
        
        if cache_key in self.models_cache:
            del self.models_cache[cache_key]
        if cache_key in self.scalers_cache:
            del self.scalers_cache[cache_key]
        if cache_key in self.metadata_cache:
            del self.metadata_cache[cache_key]
            
        logger.info(f"🗑️ {pair}: Cache cleared")
    
    def clear_all_cache(self) -> None:
        """Usuwa cały cache modeli"""
        self.models_cache.clear()
        self.scalers_cache.clear()
        self.metadata_cache.clear()
        logger.info("🗑️ All model cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Zwraca statystyki cache"""
        return {
            'models_cached': len(self.models_cache),
            'scalers_cached': len(self.scalers_cache),
            'metadata_cached': len(self.metadata_cache),
            'cached_pairs': list(self.models_cache.keys())
        }
    
    def validate_artifacts_directory(self, model_dir: str, pair: str) -> Tuple[bool, List[str]]:
        """
        Waliduje czy katalog artifacts ma wszystkie wymagane pliki
        
        Args:
            model_dir: Nazwa katalogu modelu (lub None dla auto-generacji)
            pair: Nazwa pary
            
        Returns:
            Tuple[bool, List[str]]: (success, missing_files)
        """
        if model_dir is None:
            model_dir = self._convert_pair_to_directory_name(pair)
            
        artifacts_dir = os.path.join(self.base_artifacts_path, model_dir)
        
        if not os.path.exists(artifacts_dir):
            return False, [f"Directory not found: {artifacts_dir}"]
        
        # 🔥 NOWE NAZWY PLIKÓW (ZAKTUALIZOWANE DLA H5 FORMAT)
        required_files = [
            "best_model.h5",       # Zmienione z .keras na .h5 (fix dla TensorFlow 2.15.0 LSTM bug)
            "scaler.pkl",          # Bez zmian
            "metadata.json"        # Zamiast model_metadata.json
        ]
        
        missing_files = []
        for filename in required_files:
            file_path = os.path.join(artifacts_dir, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)
        
        return len(missing_files) == 0, missing_files 