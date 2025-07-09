"""
Pair Manager - Zarządzanie parami walutowymi dla Enhanced ML Strategy

Odpowiedzialny za:
- Ładowanie konfiguracji par z pair_config.json
- Walidację ustawień par
- Status tracking aktywnych par
- Error handling per para
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class PairManager:
    """Zarządza konfiguracją i statusem par walutowych"""
    
    def __init__(self, config_path: str = "user_data/strategies/config/pair_config.json"):
        """
        Inicjalizacja Pair Manager
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego par
        """
        self.config_path = config_path
        self.config = {}
        self.active_pairs = {}
        self.failed_pairs = {}
        self.last_config_reload = None
        
        # Load initial configuration
        self.reload_config()
    
    def reload_config(self) -> bool:
        """
        Przeładowuje konfigurację par z pliku
        
        Returns:
            bool: True jeśli sukces, False jeśli błąd
        """
        try:
            if not os.path.exists(self.config_path):
                logger.error(f"❌ Pair config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Waliduj konfigurację
            if not self._validate_config():
                logger.error("❌ Invalid pair configuration")
                return False
            
            logger.info(f"✅ Pair configuration loaded: {len(self.get_enabled_pairs())} enabled pairs")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading pair config: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Waliduje strukturę konfiguracji"""
        required_keys = ['active_pairs', 'pair_settings', 'global_settings']
        
        for key in required_keys:
            if key not in self.config:
                logger.error(f"❌ Missing required config key: {key}")
                return False
        
        # Sprawdź czy wszystkie active_pairs mają ustawienia
        for pair in self.config['active_pairs']:
            if pair not in self.config['pair_settings']:
                logger.error(f"❌ Missing settings for pair: {pair}")
                return False
        
        return True
    
    def get_enabled_pairs(self) -> List[str]:
        """
        Zwraca listę włączonych par
        
        Returns:
            List[str]: Lista par które są enabled=true
        """
        enabled_pairs = []
        
        for pair in self.config.get('active_pairs', []):
            pair_settings = self.config['pair_settings'].get(pair, {})
            if pair_settings.get('enabled', False):
                enabled_pairs.append(pair)
        
        return enabled_pairs
    
    def get_pair_settings(self, pair: str) -> Optional[Dict]:
        """
        Pobiera ustawienia dla konkretnej pary
        
        Args:
            pair: Nazwa pary (np. "BTC/USDT")
            
        Returns:
            Dict: Ustawienia pary lub None jeśli nie znaleziono
        """
        return self.config['pair_settings'].get(pair)
    
    def get_model_dir(self, pair: str) -> Optional[str]:
        """
        Pobiera katalog modelu dla pary
        
        Args:
            pair: Nazwa pary
            
        Returns:
            str: Nazwa katalogu modelu lub None
        """
        pair_settings = self.get_pair_settings(pair)
        if pair_settings:
            return pair_settings.get('model_dir')
        return None
    
    def mark_pair_as_failed(self, pair: str, error: str) -> None:
        """
        Oznacza parę jako failed z powodu błędu
        
        Args:
            pair: Nazwa pary
            error: Opis błędu
        """
        self.failed_pairs[pair] = {
            'error': error,
            'timestamp': self._get_current_timestamp(),
            'retry_count': self.failed_pairs.get(pair, {}).get('retry_count', 0) + 1
        }
        
        # Usuń z aktywnych jeśli była tam
        if pair in self.active_pairs:
            del self.active_pairs[pair]
        
        logger.warning(f"⚠️ Pair {pair} marked as failed: {error}")
    
    def mark_pair_as_active(self, pair: str, model_info: Dict) -> None:
        """
        Oznacza parę jako aktywną z działającym modelem
        
        Args:
            pair: Nazwa pary
            model_info: Informacje o modelu (window_size, etc.)
        """
        self.active_pairs[pair] = {
            'model_info': model_info,
            'timestamp': self._get_current_timestamp(),
            'status': 'active'
        }
        
        # Usuń z failed jeśli była tam
        if pair in self.failed_pairs:
            del self.failed_pairs[pair]
        
        logger.info(f"✅ Pair {pair} marked as active with model: {model_info}")
    
    def is_pair_enabled(self, pair: str) -> bool:
        """Sprawdza czy para jest włączona w konfiguracji"""
        pair_settings = self.get_pair_settings(pair)
        return pair_settings.get('enabled', False) if pair_settings else False
    
    def is_pair_active(self, pair: str) -> bool:
        """Sprawdza czy para ma załadowany i działający model"""
        return pair in self.active_pairs
    
    def is_pair_failed(self, pair: str) -> bool:
        """Sprawdza czy para jest oznaczona jako failed"""
        return pair in self.failed_pairs
    
    def get_active_pairs_count(self) -> int:
        """Zwraca liczbę aktywnych par"""
        return len(self.active_pairs)
    
    def get_failed_pairs_count(self) -> int:
        """Zwraca liczbę failed par"""
        return len(self.failed_pairs)
    
    def get_status_summary(self) -> Dict:
        """
        Zwraca podsumowanie statusu wszystkich par
        
        Returns:
            Dict: Podsumowanie z aktywne/failed/disabled pary
        """
        enabled_pairs = self.get_enabled_pairs()
        
        return {
            'total_configured': len(self.config.get('active_pairs', [])),
            'enabled': len(enabled_pairs),
            'active': len(self.active_pairs),
            'failed': len(self.failed_pairs),
            'disabled': len([p for p in self.config.get('active_pairs', []) if not self.is_pair_enabled(p)]),
            'active_pairs': list(self.active_pairs.keys()),
            'failed_pairs': list(self.failed_pairs.keys()),
            'enabled_pairs': enabled_pairs
        }
    
    def should_retry_failed_pair(self, pair: str) -> bool:
        """
        Sprawdza czy powinna być próba retry dla failed pary
        
        Args:
            pair: Nazwa pary
            
        Returns:
            bool: True jeśli należy spróbować retry
        """
        if not self.config.get('global_settings', {}).get('retry_failed_models', False):
            return False
        
        if pair not in self.failed_pairs:
            return False
        
        # Implementacja prostej logiki retry (można rozszerzyć)
        failed_info = self.failed_pairs[pair]
        retry_count = failed_info.get('retry_count', 0)
        
        # Maksymalnie 3 próby retry
        return retry_count < 3
    
    def _get_current_timestamp(self) -> str:
        """Zwraca aktualny timestamp jako string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def log_status_summary(self) -> None:
        """Loguje podsumowanie statusu par"""
        summary = self.get_status_summary()
        
        logger.info("📊 PAIR MANAGER STATUS:")
        logger.info(f"   ✅ Active pairs: {summary['active']} ({summary['active_pairs']})")
        logger.info(f"   ❌ Failed pairs: {summary['failed']} ({summary['failed_pairs']})")
        logger.info(f"   📋 Enabled pairs: {summary['enabled']} ({summary['enabled_pairs']})")
        logger.info(f"   🚫 Disabled pairs: {summary['disabled']}") 