"""
Model Loader - Ładowanie modeli XGBoost dla Enhanced ML Strategy v7.0

Aktualizacja pod training5 pipeline:
- Modele zapisywane jako model_{index+1}.json w katalogu: crypto/training5/output/models/{SYMBOL}
- Skaler: scaler.pkl
- Metadane: metadata.json (zawiera feature_names, tp_sl_levels, label_columns)
"""

import os
import json
import logging
import joblib
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader:
    """Ładuje i zarządza modelami ML per para z training5 pipeline"""
    
    def __init__(self, base_artifacts_path: str = "C:/Users/macie/OneDrive/Python/Binance/crypto/training5/output/models"):
        """
        Inicjalizacja Model Loader
        
        Args:
            base_artifacts_path: Bazowa ścieżka do modeli training4
        """
        self.base_artifacts_path = base_artifacts_path
        self.models_cache = {}
        self.scalers_cache = {}
        self.metadata_cache = {}
        
        # Informacyjnie: training5 również ma 15 poziomów TP/SL (kolejność w training5.config.TP_SL_LEVELS)
        self.model_levels = [
            {"index": i, "desc": f"model_{i+1}.json"} for i in range(15)
        ]
        
    def _convert_pair_to_directory_name(self, pair: str) -> str:
        """
        Konwertuje nazwę pary na nazwę katalogu
        
        Args:
            pair: Nazwa pary (np. "ETH/USDT")
            
        Returns:
            str: Nazwa katalogu (np. "ETHUSDT")
        """
        # Usuń wszystkie znaki specjalne i zostaw tylko litery/cyfry
        # Dla BTC/USDT:USDT -> BTCUSDT
        result = pair.replace('/', '').replace(':', '').replace('-', '')
        
        # Usuń duplikaty (BTCUSDTUSDT -> BTCUSDT)
        if result.endswith('USDTUSDT'):
            result = result[:-4]  # Usuń ostatnie 4 znaki (USDT)
        
        return result
        
    def load_model_for_pair(self, pair: str, model_index: int = 3, use_basic_features: bool = False) -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
        """
        Ładuje model XGBoost, scaler i metadata dla konkretnej pary i modelu
        
        Args:
            pair: Nazwa pary (np. "ETH/USDT")
            model_index: Indeks modelu (0-14)
            use_basic_features: True = 37 cech, False = 71 cech
            
        Returns:
            Tuple[model, scaler, metadata]: Załadowane komponenty lub None jeśli błąd
        """
        model_dir = self._convert_pair_to_directory_name(pair)
        cache_key = f"{pair}_{model_dir}_{model_index}_{use_basic_features}"
        
        try:
            # Bez cache (diagnostyka)
            
            # Waliduj model_index
            if model_index < 0 or model_index >= len(self.model_levels):
                logger.error(f"❌ {pair}: Invalid model_index: {model_index}. Must be 0-{len(self.model_levels)-1}")
                return None, None, None
            
            # Informacja o modelu
            model_info = self.model_levels[model_index]
            desc = model_info['desc']
            
            # Waliduj że katalog istnieje
            artifacts_dir = os.path.join(self.base_artifacts_path, model_dir)
            if not os.path.exists(artifacts_dir):
                error_msg = f"Model directory not found: {artifacts_dir}"
                logger.error(f"❌ {pair}: {error_msg}")
                return None, None, None
            
            # Załaduj metadata (training5: metadata.json)
            metadata = self._load_metadata(artifacts_dir)
            if metadata is None:
                logger.warning(f"⚠️ Metadata not found for {pair} in {artifacts_dir}. Proceeding without feature_names.")
                metadata = {}
            
            # Załaduj model XGBoost (training5: model_{index+1}.json)
            model = self._load_xgboost_model(artifacts_dir, model_index, metadata)
            if model is None:
                return None, None, None
            
            # Załaduj scaler
            scaler = self._load_scaler(artifacts_dir, pair)
            if scaler is None:
                return None, None, None
            
            # Waliduj kompatybilność modelu
            if not self._validate_model_compatibility(metadata, pair, use_basic_features):
                return None, None, None
            
            # Brak cache (diagnostyka)
            
            logger.info(f"✅ {pair}: XGBoost model {model_index} loaded successfully ({desc})")
            logger.info(f"   - Features: {metadata.get('n_features', 'N/A')}")
            
            return model, scaler, metadata
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error loading model {model_index}: {e}")
            import traceback
            logger.error(f"❌ {pair}: Full traceback: {traceback.format_exc()}")
            return None, None, None
    
    def _load_metadata(self, artifacts_dir: str) -> Optional[Dict]:
        """Ładuje metadata z training5 (metadata.json)"""
        try:
            metadata_path = os.path.join(artifacts_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.error(f"❌ Metadata not found in {artifacts_dir}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.debug(f"✅ Metadata loaded from metadata.json")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Error loading metadata: {e}")
            return None
    
    def _load_xgboost_model(self, artifacts_dir: str, model_index: int, metadata: Optional[Dict]) -> Optional[Any]:
        """Ładuje konkretny model XGBoost training5 (model_{index+1}.json)"""
        try:
            import xgboost as xgb
            import json
            
            # training5 nazewnictwo modeli
            model_filename = f"model_{model_index+1}.json"
            model_path = os.path.join(artifacts_dir, model_filename)
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return None
            
            logger.info(f"🔄 Loading XGBoost model: {model_filename}")
            
            # Załaduj model
            model = xgb.Booster()
            model.load_model(model_path)
            
            # Stwórz wrapper kompatybilny z sklearn
            class IndividualXGBoostWrapper:
                def __init__(self, model, feature_names, model_desc):
                    self.model = model
                    self.feature_names = feature_names
                    self.model_desc = model_desc
                    self.classes_ = np.array([0, 1, 2])  # LONG, SHORT, NEUTRAL
                
                def predict_proba(self, X):
                    import xgboost as xgb
                    dtest = xgb.DMatrix(X, feature_names=self.feature_names)
                    probs = self.model.predict(dtest)
                    return probs.reshape(-1, 3)
                
                def predict(self, X):
                    probs = self.predict_proba(X)
                    return np.argmax(probs, axis=1)
            
            # feature_names z metadata
            feature_names = metadata.get('feature_names') if metadata else None
            
            wrapper = IndividualXGBoostWrapper(model, feature_names, f"model_{model_index+1}.json")
            logger.info(f"✅ XGBoost model loaded successfully")
            logger.info(f"  - Model: {wrapper.model_desc}")
            logger.info(f"  - Features: {len(feature_names)}")
            return wrapper
            
        except Exception as e:
            logger.error(f"❌ Error loading XGBoost model: {e}")
            return None
    
    def _load_scaler(self, artifacts_dir: str, pair: str) -> Optional[Any]:
        """Ładuje scaler.pkl"""
        try:
            scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
            
            if not os.path.exists(scaler_path):
                logger.error(f"❌ {pair}: scaler.pkl not found in {artifacts_dir}")
                return None

            scaler = joblib.load(scaler_path)
            logger.debug(f"✅ {pair}: Scaler loaded successfully")
            return scaler
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error loading scaler: {e}")
            return None
    
    def _validate_model_compatibility(self, metadata: Dict, pair: str, use_basic_features: bool) -> bool:
        """Waliduje kompatybilność modelu z konfiguracją cech"""
        try:
            # Sprawdź czy ma wymagane pola
            required_fields = ['n_features', 'feature_names', 'model_type']
            for field in required_fields:
                if field not in metadata:
                    logger.error(f"❌ {pair}: Missing required field in metadata: {field}")
                    return False
            
            # Sprawdź typ modelu
            model_type = metadata.get('model_type')
            if model_type != 'xgboost_individual':
                logger.error(f"❌ {pair}: Unsupported model type: {model_type}")
                return False
            
            # Training5: akceptuj dowolną liczbę cech > 0 (lista cech przekazana w metadata)
            n_features = metadata.get('n_features', 0)
            if not isinstance(n_features, int) or n_features <= 0:
                logger.error(f"❌ {pair}: Invalid n_features in metadata: {n_features}")
                return False
            
            logger.debug(f"✅ {pair}: Model compatibility validated ({n_features} features)")
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error validating model compatibility: {e}")
            return False
    
    def get_model_info(self, model_index: int) -> Optional[Dict]:
        """Pobiera informacje o modelu na podstawie indeksu"""
        if 0 <= model_index < len(self.model_levels):
            return self.model_levels[model_index]
        return None
    
    def get_available_models(self) -> List[Dict]:
        """Zwraca listę wszystkich dostępnych modeli"""
        return self.model_levels.copy()
    
    def clear_cache_for_pair(self, pair: str, model_index: int = None, use_basic_features: bool = None) -> None:
        """Usuwa z cache model dla konkretnej pary"""
        model_dir = self._convert_pair_to_directory_name(pair)
        
        if model_index is not None and use_basic_features is not None:
            # Usuń konkretny model
            cache_key = f"{pair}_{model_dir}_{model_index}_{use_basic_features}"
            if cache_key in self.models_cache:
                del self.models_cache[cache_key]
            if cache_key in self.scalers_cache:
                del self.scalers_cache[cache_key]
            if cache_key in self.metadata_cache:
                del self.metadata_cache[cache_key]
        else:
            # Usuń wszystkie modele dla tej pary
            keys_to_remove = [k for k in self.models_cache.keys() if k.startswith(f"{pair}_{model_dir}_")]
            for key in keys_to_remove:
                del self.models_cache[key]
                del self.scalers_cache[key]
                del self.metadata_cache[key]
            
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
            'cached_pairs': list(set([k.split('_')[0] for k in self.models_cache.keys()]))
        } 