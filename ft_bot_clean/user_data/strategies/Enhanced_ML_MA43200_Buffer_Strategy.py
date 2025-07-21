"""
🚀 ARCHITEKTURA V5 - Enhanced ML Strategy 🚀

Zgodnie z ostatecznym, precyzyjnym planem, ta strategia implementuje
dwumodalną logikę przetwarzania danych.

✅ TRYB BACKTEST:
- Przetwarzanie "świeca po świecy".
- Dla każdego punktu w czasie, Bufor buduje od zera pełny kontekst historyczny
  (43,200+ świec), oblicza wskaźniki i 8 kluczowych cech.
- Zwracana jest gotowa do predykcji tablica (120, 8), co gwarantuje
  maksymalną spójność i eliminuje błędy przesunięcia czasowego.

📝 TRYB LIVE (w przygotowaniu):
- Przetwarzanie na paczkach 60 świec.
- Jednorazowa synchronizacja na starcie w celu wypełnienia "luki" w danych.
- Inteligentne łączenie danych i okresowy zapis na dysk w celu zapewnienia
  ciągłości i wydajności.
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
    strategy_name = "Enhanced_ML_MA43200_Buffer_Strategy_v5"
    timeframe = '1m'
    startup_candle_count: int = 0

    # ROI table:
    minimal_roi = {
        "0": 0.01
    }

    # Stoploss:
    stoploss = -0.005

    # Trailing Stoploss
    trailing_stop = False
    
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
        
        # Inicjalizacja nowych, przebudowanych serwisów
        self.pair_manager = PairManager()
        self.model_loader = ModelLoader(base_artifacts_path=str(inputs_path))
        self.signal_generator = SignalGenerator()
        
        # Krok 1: Pobierz instancję singletona
        self.buffer_service = get_dataframe_extender(self.config)
        
        # Krok 2: Wstrzyknij poprawną, rozwiązaną przez Freqtrade ścieżkę do danych
        # To jest gwarancja, że będziemy szukać plików tam, gdzie trzeba.
        correct_datadir = Path(self.config['datadir'])
        self.buffer_service.set_resolved_datadir(correct_datadir)

        self.signal_generator.set_thresholds(
            short_threshold=self.ml_confidence_short,
            long_threshold=self.ml_confidence_long,
            hold_threshold=self.ml_confidence_hold
        )
        
        self.models_cache = {}
        self.scalers_cache = {}
        self.pair_window_sizes = {}
        self.predictions_log = [] # Lista do logowania predykcji

        logger.info("🚀 Enhanced ML Strategy v5.0 (Precise Architecture) initialized!")

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
        """
        Główna metoda strategii, zaimplementowana zgodnie z nową logiką.
        Wykrywa tryb (backtest/live) i deleguje zadania do odpowiednich serwisów.
        """
        pair = metadata['pair']

        # Prosta heurystyka do rozróżnienia trybów
        # W przyszłości można to oprzeć o stan bota z Freqtrade
        is_backtest = len(dataframe) > 500

        if is_backtest:
            logger.info(f"🚀 {pair}: Wykryto tryb BACKTEST. Uruchamianie logiki 'świeca po świecy'...")
            dataframe = self._populate_for_backtest(dataframe, pair)
            
            # Po zakończeniu backtestu dla pary, zapisz logi predykcji do pliku
            if self.predictions_log:
                log_df = pd.DataFrame(self.predictions_log)
                filename = Path(self.config['user_data_dir']) / 'backtest_results' / f"predictions_{metadata['pair'].replace('/', '_').replace(':', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                try:
                    log_df.to_csv(filename, index=False)
                    logger.info(f"✅ Zapisano log predykcji do pliku: {filename}")
                except Exception as e:
                    logger.error(f"❌ Nie udało się zapisać logu predykcji: {e}")
                
                # Wyczyść log dla następnej pary (jeśli będzie)
                self.predictions_log = []
                
        else:
            logger.info(f"🚀 {pair}: Wykryto tryb LIVE. Uruchamianie logiki na paczkach danych...")
            dataframe = self._populate_for_live(dataframe, pair)
            
        return dataframe

    def _populate_for_backtest(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        🚀 OPTYMALIZOWANA implementacja logiki "świeca po świecy" dla trybu backtest.
        Używa batch processing - oblicza MA43200 raz na starcie, potem używa pre-obliczonych wartości.
        Zachowuje dokładnie tę samą logikę, ale dramatycznie poprawia wydajność.
        """
        normalized_pair = self._normalize_pair_name(pair)
        
        # Inicjalizacja kolumn wyjściowych
        self._initialize_signal_columns(dataframe)

        # Sprawdzenie, czy para jest aktywna i ma załadowany model/scaler
        if not self.pair_manager.is_pair_active(normalized_pair):
            return dataframe
            
        model = self.models_cache.get(normalized_pair)
        scaler = self.scalers_cache.get(normalized_pair)
        
        if not all([model, scaler]):
            logger.warning(f"⚠️ {pair}: Brak modelu lub scalera, pomijanie generowania sygnałów.")
            return dataframe

        logger.info(f"🚀 {pair}: Rozpoczynanie OPTYMALIZOWANEGO batch processing dla {len(dataframe)} świec...")
        
        # Krok 1: Przygotowanie listy timestampów
        timestamps = dataframe['date'].tolist()
        
        # Krok 2: Batch processing - oblicz MA43200 raz, potem wszystkie cechy naraz
        features_batch = self.buffer_service.get_features_for_backtest_batch(pair, timestamps)
        
        if not features_batch:
            logger.warning(f"⚠️ {pair}: Batch processing nie zwrócił żadnych cech.")
            return dataframe
        
        # Krok 3: Przygotowanie danych do batch prediction
        logger.info(f"🤖 {pair}: Przygotowywanie batch prediction dla {len(features_batch)} próbek...")
        
        valid_features = []
        valid_indices = []
        timestamp_to_index = {row.date: idx for idx, row in enumerate(dataframe.itertuples())}
        
        for timestamp, features in features_batch.items():
            if features is not None:
                idx = timestamp_to_index.get(timestamp)
                if idx is not None:
                    valid_features.append(features)
                    valid_indices.append(idx)
        
        if not valid_features:
            logger.warning(f"⚠️ {pair}: Brak prawidłowych cech do predykcji.")
            return dataframe
        
        # Krok 4: Batch prediction
        logger.info(f"🧠 {pair}: Wykonywanie batch prediction dla {len(valid_features)} próbek...")
        all_signal_dicts = self.signal_generator.generate_signals_for_batch(model, scaler, valid_features)
        
        # Krok 5: Przypisanie sygnałów z powrotem do głównej ramki danych
        logger.info(f"✍️ {pair}: Przypisywanie {len(all_signal_dicts)} sygnałów do ramki danych...")
        
        for i, signal_data in enumerate(all_signal_dicts):
            # Pobierz oryginalny indeks z `dataframe` i timestamp
            original_idx = valid_indices[i]
            timestamp = dataframe.at[original_idx, 'date']
            
            # Przypisz sygnały
            signal = signal_data.get('signal')
            if signal == 'long':
                dataframe.at[original_idx, 'enter_long'] = 1
            elif signal == 'short':
                dataframe.at[original_idx, 'enter_short'] = 1
            
            # Zapisz dane do logu, jeśli istnieją
            if signal_data and 'probabilities' in signal_data:
                self.predictions_log.append({
                    'timestamp': timestamp,
                    'pair': pair,
                    'signal': signal,
                    'confidence': signal_data.get('confidence'),
                    'prob_short': signal_data['probabilities'][0],
                    'prob_hold': signal_data['probabilities'][1],
                    'prob_long': signal_data['probabilities'][2]
                })

        logger.info(f"✅ {pair}: Zakończono OPTYMALIZOWANY backtesting.")
        return dataframe

    def _populate_for_live(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Logika dla trybu LIVE - na razie uproszczona.
        W przyszłości będzie używać `process_live_data` z bufora.
        """
        self._initialize_signal_columns(dataframe)
        
        # Pobierz ostatnią świecę
        last_candle_timestamp = dataframe.iloc[-1]['date']
        
        # Użyj trybu "świeca po świecy" z bufora
        features = self.buffer_service.get_features_for_backtest(pair, last_candle_timestamp)
        
        if features is not None and features.any():
            model = self.models_cache.get(self._normalize_pair_name(pair))
            scaler = self.scalers_cache.get(self._normalize_pair_name(pair))
            
            if model and scaler:
                signal_data = self.signal_generator.generate_signal(model, scaler, features)
                self._assign_signals_to_row(
                    dataframe,
                    dataframe.index[-1],
                    signal_data
                )
        return dataframe
     
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Sygnały 'enter_long' i 'enter_short' są już obliczone w `populate_indicators`.
        # Ta metoda służy teraz tylko do ewentualnego dodania tagów lub innej logiki
        # bazującej na istniejących sygnałach.
        pair = metadata['pair']
        
        long_condition = dataframe['enter_long'] == 1
        short_condition = dataframe['enter_short'] == 1
        
        dataframe.loc[long_condition, 'enter_tag'] = f'{pair}_long'
        dataframe.loc[short_condition, 'enter_tag'] = f'{pair}_short'
        
        return dataframe
     
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Wyłącza niestandardowe sygnały wyjścia. Strategia opiera się wyłącznie na
        wbudowanych mechanizmach ROI i stop-loss dla maksymalnej wydajności.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def _normalize_pair_name(self, pair: str) -> str:
        return pair.split(':')[0]

    def _initialize_signal_columns(self, dataframe: pd.DataFrame):
        """Inicjalizuje kolumny sygnałowe, jeśli nie istnieją."""
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        if 'exit_long' not in dataframe.columns:
            dataframe['exit_long'] = 0
        if 'exit_short' not in dataframe.columns:
            dataframe['exit_short'] = 0
        if 'enter_tag' not in dataframe.columns:
            dataframe['enter_tag'] = ''
            
    def _assign_signals_to_row(self, df: pd.DataFrame, index, signal_data: dict):
        """Helper do przypisywania sygnałów do wiersza ramki danych."""
        signal = signal_data.get('signal')

        if signal == 'long':
            df.loc[index, 'enter_long'] = 1
        elif signal == 'short':
            df.loc[index, 'enter_short'] = 1
        
        # Logika wyjścia jest wyłączona (use_exit_signal=False), więc nie ma potrzeby jej implementować
        # df.loc[index, 'exit_long'] = ...
        # df.loc[index, 'exit_short'] = ...

    def bot_loop_start(self, current_time, **kwargs) -> None:
        """Wywoływane na początku każdej iteracji pętli bota."""
        pass