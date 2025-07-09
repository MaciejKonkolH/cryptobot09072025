"""
🚀 Enhanced ML MA43200 Buffer Strategy v2.0 - PRODUCTION READY

🚀 NOWA FUNKCJONALNOŚĆ: Multi-pair support z modeli .h5 (zmienione z .keras na .h5)

Strategia łączy:
1. Precyzyjną MA43200 z buffer systemem (najlepszy long-term trend indicator)
2. Modele .h5 format (nowe, fix dla TensorFlow 2.15.0 LSTM bug)
3. Multi-pair support z konfiguracją per para
4. Inteligentny risk management
5. Zaawansowany backtesting support

🎯 KLUCZOWE FUNKCJE:
- MA43200 Buffer System (3-dniowy buffer dla stabilności)
- ML sygnały z modeli .h5 (precision/recall based)
- Multi-pair support (każda para ma swój model)
- Smart position sizing
- Zaawansowane risk management

STRUKTURA MODELI v2.1:
user_data/strategies/inputs/BTCUSDT/
├── best_model.h5      # Model ML (zmienione z .keras na .h5)
├── scaler.pkl         # Feature scaler
└── metadata.json      # Model metadata

KONFIGURACJA:
user_data/strategies/config/
└── pairs_config.json  # Multi-pair configuration

FEATURES UŻYWANE PRZEZ MODEL (8 features):
1. returns_1m, returns_5m, returns_15m, returns_1h (4 returns)
2. rsi_14 (RSI indicator)  
3. bb_position (Bollinger Bands position)
4. volume_sma_ratio (Volume vs SMA ratio)
5. price_sma_ratio (Price vs SMA ratio)

SYGNAŁY:
- LONG: MA43200 trend UP + ML confidence > threshold
- EXIT: MA43200 trend DOWN lub ML exit signal

RISK MANAGEMENT:
- Stop loss: 0.5% (tight risk control)
- Take profit: 1.0% (2:1 risk/reward)
- Max open trades: 2
- Position sizing: fixed 100 USDT

BACKTESTING SUPPORT:
- Automatyczne rozpoznawanie trybu backtest
- Optymalizacja ładowania danych dla backtest
- Pełna kompatybilność z Freqtrade backtesting

🎯 PRODUCTION FEATURES:
- Zero dependencies na validation module
- Stabilny MA43200 calculation
- Multi-pair model support
- Error handling per para
- Intelligent data loading
- Memory efficient processing

WERSJE:
v1.0 - Podstawowy MA43200 + ML
v2.0 - Multi-pair support + .h5 format + production ready
"""

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys
import logging
import json
import talib.abstract as ta
from datetime import datetime
from pathlib import Path
from typing import Optional

from freqtrade.persistence import Trade
from freqtrade.exceptions import DependencyException

# 🚀 SETUP PROJECT PATH 🚀
# To zapewnia, że importy działają poprawnie, niezależnie od sposobu uruchomienia.
# ft_bot_clean/user_data/strategies/ -> ft_bot_clean/user_data/ -> ft_bot_clean/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Teraz wszystkie importy mogą być absolutne z roota projektu
from user_data.buffer.dataframe_extender import get_dataframe_extender
from user_data.strategies.utils.pair_manager import PairManager
from user_data.strategies.utils.model_loader import ModelLoader
from user_data.strategies.components.signal_generator import SignalGenerator


logger = logging.getLogger(__name__)

class Enhanced_ML_MA43200_Buffer_Strategy(IStrategy):
    strategy_name = "Enhanced_ML_MA43200_Buffer_Strategy_v4"
    timeframe = '1m'
    startup_candle_count: int = 0

    minimal_roi = {"0": 0.01}
    stoploss = -0.005
    can_short = True
    position_adjustment_enable = False
    use_exit_signal = False
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self._load_backtest_config()
        
        inputs_path = Path(self.config['user_data_dir'], "strategies", "inputs")
        
        self.pair_manager = PairManager()
        self.model_loader = ModelLoader(base_artifacts_path=str(inputs_path))
        self.signal_generator = SignalGenerator()
        self.buffer_service = get_dataframe_extender(self.config)

        self.signal_generator.set_thresholds(
            short_threshold=self.ml_confidence_short,
            long_threshold=self.ml_confidence_long,
            hold_threshold=self.ml_confidence_hold
        )
        
        self.models_cache = {}
        self.scalers_cache = {}
        self.pair_window_sizes = {}

        logger.info("🚀 Enhanced ML Strategy v4.0 (Feature Service) initialized!")

    def _load_backtest_config(self) -> None:
        ml_config = self.config.get('ml_config', {})
        self.ml_confidence_short = ml_config.get('confidence_threshold_short', 0.42)
        self.ml_confidence_long = ml_config.get('confidence_threshold_long', 0.42)
        self.ml_confidence_hold = ml_config.get('confidence_threshold_hold', 0.30)

    def bot_start(self, **kwargs) -> None:
        self._initialize_multi_pair_system()

    def _initialize_multi_pair_system(self) -> None:
        if not self.pair_manager.reload_config():
            logger.error("❌ Failed to load pair configuration")
            return
        enabled_pairs = self.pair_manager.get_enabled_pairs()
        self._initialize_models_for_pairs(enabled_pairs)
        self.pair_manager.log_status_summary()

    def _initialize_models_for_pairs(self, pairs: list) -> None:
        for pair in pairs:
            model_dir = self.pair_manager.get_model_dir(pair)
            if not model_dir: continue
            
            model, scaler, metadata = self.model_loader.load_model_for_pair(pair, model_dir=model_dir)
            if model and scaler and metadata:
                normalized_pair = self._normalize_pair_name(pair)
                self.models_cache[normalized_pair] = model
                self.scalers_cache[normalized_pair] = scaler
                window_size = self.model_loader._extract_window_size(metadata)
                self.pair_window_sizes[normalized_pair] = window_size
                model_info = {'window_size': window_size, 'model_dir': model_dir}
                self.pair_manager.mark_pair_as_active(normalized_pair, model_info)
            else:
                self.pair_manager.mark_pair_as_failed(self._normalize_pair_name(pair), "Model loading failed")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        logger.info(f"📊 Processing {pair} - DataFrame size: {len(dataframe)}")

        # Krok 1: Cała logika przygotowania danych (w tym buforów i cech)
        # jest teraz w serwisie bufora, zgodnie z nowym, precyzyjnym planem.
        dataframe = self.buffer_service.extend_and_calculate_features(dataframe, pair)

        # Krok 2: Dodanie sygnałów ML na podstawie świeżo obliczonych cech.
        # SignalGenerator jest już przygotowany do obsługi ramki z buforem
        # i poprawnie zignoruje wiersze, dla których nie da się zrobić predykcji.
        dataframe = self.add_ml_signals(dataframe, pair)
        
        return dataframe

    def add_ml_signals(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        normalized_pair = self._normalize_pair_name(pair)
        if not self.pair_manager.is_pair_active(normalized_pair):
            return self._add_default_ml_signals(dataframe)
            
        model = self.models_cache.get(normalized_pair)
        scaler = self.scalers_cache.get(normalized_pair)
        window_size = self.pair_window_sizes.get(normalized_pair)

        if not all([model, scaler, window_size]):
            return self._add_default_ml_signals(dataframe)
        
        return self.signal_generator.generate_ml_signals(
            dataframe=dataframe, pair=pair, model=model, scaler=scaler, window_size=window_size
        )
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        dataframe['enter_long'], dataframe['enter_short'], dataframe['enter_tag'] = 0, 0, ''
        
        long_condition = (dataframe['ml_signal'] == 2)
        short_condition = (dataframe['ml_signal'] == 0)
        
        dataframe.loc[long_condition, ['enter_long', 'enter_tag']] = (1, f'{pair}_long')
        dataframe.loc[short_condition, ['enter_short', 'enter_tag']] = (1, f'{pair}_short')
        
        return dataframe
     
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'], dataframe['exit_short'] = 0, 0
        return dataframe

    def _normalize_pair_name(self, pair: str) -> str:
        return pair.split(':')[0]



    def bot_loop_start(self, current_time, **kwargs) -> None:
        """Optimized progress counter - minimal overhead"""
        if not hasattr(self, '_progress_counter'):
            self._progress_counter = 0
            self._last_log_counter = 0
        
        self._progress_counter += 1
        
        # Log every 1000 candles but with minimal overhead
        if self._progress_counter - self._last_log_counter >= 1000:
            logger.info(f"🚀 PROGRESS: {self._progress_counter} candles - {current_time}")
            self._last_log_counter = self._progress_counter

    def _add_default_ml_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for prob_col in ['ml_short_prob', 'ml_hold_prob', 'ml_long_prob', 'ml_buy_prob', 'ml_sell_prob']:
            dataframe[prob_col] = 0.0
        dataframe['ml_signal'] = 1
        dataframe['ml_confidence'] = 0.0
        return dataframe