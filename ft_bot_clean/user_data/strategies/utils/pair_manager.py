"""
Pair Manager - Zarządzanie parami walutowymi dla Enhanced ML Strategy v7.0

Odpowiedzialny za:
- Zarządzanie aktywnymi parami walutowymi
- Konfiguracja per para (model_index, use_basic_features)
- Walidacja konfiguracji
- Dynamiczne włączanie/wyłączanie par
- Priority management

NOWA STRUKTURA v7.0:
- Obsługa 15 modeli (różne poziomy TP/SL)
- Obsługa 37 lub 71 cech (konfigurowalne)
- Konfiguracja per para w pair_config.json
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class PairManager:
    """Zarządza parami walutowymi i ich konfiguracją"""
    
    def __init__(self, config_path: str = "user_data/strategies/config/pair_config.json"):
        """
        Inicjalizacja Pair Manager
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego par
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.active_pairs = []
        self.pair_settings = {}
        self.global_settings = {}
        
        # Inicjalizuj ustawienia
        self._initialize_settings()
        
    def _load_config(self) -> dict:
        """Ładuje konfigurację z pliku JSON"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Pair config loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading pair config: {e}")
            return {}
    
    def _initialize_settings(self):
        """Inicjalizuje ustawienia z konfiguracji"""
        try:
            # Pobierz globalne ustawienia
            self.global_settings = self.config.get('global_settings', {})
            
            # Pobierz ustawienia par
            pair_settings = self.config.get('pair_settings', {})
            
            # Inicjalizuj aktywne pary
            active_pairs = self.config.get('active_pairs', [])
            
            for pair in active_pairs:
                if pair in pair_settings:
                    settings = pair_settings[pair]
                    if settings.get('enabled', False):
                        # Dodaj domyślne wartości jeśli brakuje
                        if 'selected_model_index' not in settings:
                            settings['selected_model_index'] = self.global_settings.get('default_model_index', 3)
                        if 'use_basic_features_only' not in settings:
                            settings['use_basic_features_only'] = self.global_settings.get('default_use_basic_features', False)
                        
                        self.pair_settings[pair] = settings
                        self.active_pairs.append(pair)
                        
                        logger.info(f"✅ {pair}: Activated with model_index={settings['selected_model_index']}, features={'basic' if settings['use_basic_features_only'] else 'extended'}")
            
            logger.info(f"✅ Pair Manager initialized: {len(self.active_pairs)} active pairs")
            
        except Exception as e:
            logger.error(f"❌ Error initializing pair settings: {e}")
    
    def get_active_pairs(self) -> List[str]:
        """Zwraca listę aktywnych par"""
        return self.active_pairs.copy()
    
    def get_pair_settings(self, pair: str) -> Optional[Dict]:
        """Pobiera ustawienia dla konkretnej pary"""
        return self.pair_settings.get(pair)
    
    def get_model_index_for_pair(self, pair: str) -> int:
        """Pobiera indeks modelu dla pary"""
        settings = self.get_pair_settings(pair)
        if settings:
            return settings.get('selected_model_index', self.global_settings.get('default_model_index', 3))
        return self.global_settings.get('default_model_index', 3)
    
    def get_feature_mode_for_pair(self, pair: str) -> bool:
        """Pobiera tryb cech dla pary (True = basic, False = extended)"""
        settings = self.get_pair_settings(pair)
        if settings:
            return settings.get('use_basic_features_only', self.global_settings.get('default_use_basic_features', False))
        return self.global_settings.get('default_use_basic_features', False)
    
    def get_risk_multiplier(self, pair: str) -> float:
        """Pobiera mnożnik ryzyka dla pary"""
        settings = self.get_pair_settings(pair)
        if settings:
            return settings.get('risk_multiplier', 1.0)
        return 1.0
    
    def get_priority(self, pair: str) -> int:
        """Pobiera priorytet pary"""
        settings = self.get_pair_settings(pair)
        if settings:
            return settings.get('priority', 999)
        return 999
    
    def is_pair_enabled(self, pair: str) -> bool:
        """Sprawdza czy para jest aktywna"""
        return pair in self.active_pairs
    
    def get_max_active_pairs(self) -> int:
        """Pobiera maksymalną liczbę aktywnych par"""
        return self.global_settings.get('max_active_pairs', 3)
    
    def get_model_dir(self, pair: str) -> str:
        """Pobiera nazwę katalogu modelu dla pary"""
        settings = self.get_pair_settings(pair)
        if settings:
            return settings.get('model_dir', pair.replace('/', ''))
        return pair.replace('/', '')
    
    def update_pair_settings(self, pair: str, settings: Dict) -> bool:
        """Aktualizuje ustawienia dla pary"""
        try:
            if pair not in self.pair_settings:
                logger.warning(f"⚠️ {pair}: Pair not found in settings")
                return False
            
            # Aktualizuj ustawienia
            self.pair_settings[pair].update(settings)
            
            # Zapisz do pliku konfiguracyjnego
            self._save_config()
            
            logger.info(f"✅ {pair}: Settings updated")
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error updating settings: {e}")
            return False
    
    def enable_pair(self, pair: str) -> bool:
        """Włącza parę"""
        try:
            if pair not in self.pair_settings:
                logger.error(f"❌ {pair}: Pair not found in configuration")
                return False
            
            if pair not in self.active_pairs:
                self.active_pairs.append(pair)
                self.pair_settings[pair]['enabled'] = True
                self._save_config()
                logger.info(f"✅ {pair}: Enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error enabling pair: {e}")
            return False
    
    def disable_pair(self, pair: str) -> bool:
        """Wyłącza parę"""
        try:
            if pair in self.active_pairs:
                self.active_pairs.remove(pair)
                self.pair_settings[pair]['enabled'] = False
                self._save_config()
                logger.info(f"✅ {pair}: Disabled")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error disabling pair: {e}")
            return False
    
    def set_model_index(self, pair: str, model_index: int) -> bool:
        """Ustawia indeks modelu dla pary"""
        try:
            if pair not in self.pair_settings:
                logger.error(f"❌ {pair}: Pair not found in configuration")
                return False
            
            # Waliduj model_index
            if model_index < 0 or model_index > 14:
                logger.error(f"❌ {pair}: Invalid model_index: {model_index}. Must be 0-14")
                return False
            
            self.pair_settings[pair]['selected_model_index'] = model_index
            self._save_config()
            
            logger.info(f"✅ {pair}: Model index set to {model_index}")
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error setting model index: {e}")
            return False
    
    def set_feature_mode(self, pair: str, use_basic_features: bool) -> bool:
        """Ustawia tryb cech dla pary"""
        try:
            if pair not in self.pair_settings:
                logger.error(f"❌ {pair}: Pair not found in configuration")
                return False
            
            self.pair_settings[pair]['use_basic_features_only'] = use_basic_features
            self._save_config()
            
            mode = "basic" if use_basic_features else "extended"
            logger.info(f"✅ {pair}: Feature mode set to {mode}")
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error setting feature mode: {e}")
            return False
    
    def get_pairs_by_priority(self) -> List[str]:
        """Zwraca pary posortowane według priorytetu"""
        return sorted(self.active_pairs, key=lambda p: self.get_priority(p))
    
    def get_pairs_with_model_info(self) -> List[Dict]:
        """Zwraca listę par z informacjami o modelach"""
        pairs_info = []
        for pair in self.active_pairs:
            settings = self.get_pair_settings(pair)
            if settings:
                pairs_info.append({
                    'pair': pair,
                    'model_index': settings.get('selected_model_index', 3),
                    'feature_mode': 'basic' if settings.get('use_basic_features_only', False) else 'extended',
                    'risk_multiplier': settings.get('risk_multiplier', 1.0),
                    'priority': settings.get('priority', 999),
                    'model_dir': settings.get('model_dir', pair.replace('/', ''))
                })
        return pairs_info
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Waliduje konfigurację par"""
        errors = []
        
        try:
            # Sprawdź czy są aktywne pary
            if not self.active_pairs:
                errors.append("No active pairs found")
            
            # Sprawdź maksymalną liczbę par
            max_pairs = self.get_max_active_pairs()
            if len(self.active_pairs) > max_pairs:
                errors.append(f"Too many active pairs: {len(self.active_pairs)} > {max_pairs}")
            
            # Sprawdź ustawienia każdej pary
            for pair in self.active_pairs:
                settings = self.get_pair_settings(pair)
                if not settings:
                    errors.append(f"{pair}: Missing settings")
                    continue
                
                # Sprawdź model_index
                model_index = settings.get('selected_model_index', 3)
                if model_index < 0 or model_index > 14:
                    errors.append(f"{pair}: Invalid model_index: {model_index}")
                
                # Sprawdź model_dir
                model_dir = settings.get('model_dir')
                if not model_dir:
                    errors.append(f"{pair}: Missing model_dir")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
            return False, errors
    
    def _save_config(self):
        """Zapisuje konfigurację do pliku"""
        try:
            # Aktualizuj konfigurację
            self.config['active_pairs'] = self.active_pairs
            self.config['pair_settings'] = self.pair_settings
            self.config['global_settings'] = self.global_settings
            
            # Zapisz do pliku
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✅ Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
    
    def get_config_summary(self) -> Dict:
        """Zwraca podsumowanie konfiguracji"""
        return {
            'version': self.config.get('version', 'unknown'),
            'active_pairs_count': len(self.active_pairs),
            'max_active_pairs': self.get_max_active_pairs(),
            'active_pairs': self.active_pairs,
            'pairs_info': self.get_pairs_with_model_info(),
            'global_settings': self.global_settings
        } 