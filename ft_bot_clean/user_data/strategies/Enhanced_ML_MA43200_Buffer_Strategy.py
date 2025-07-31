"""
🚀 ARCHITEKTURA V7 - Enhanced XGBoost Strategy 🚀

Zgodnie z ostatecznym, precyzyjnym planem, ta strategia implementuje
XGBoost z 37 cechami i pojedynczym modelem.

✅ TRYB BACKTEST:
- Przetwarzanie "świeca po świecy".
- Dla każdego punktu w czasie, Bufor buduje od zera pełny kontekst historyczny
  (43,200+ świec), oblicza wskaźniki i 37 kluczowych cech.
- Zwracana jest gotowa do predykcji dataframe z 37 cechami.

📝 TRYB LIVE:
- Przetwarzanie na paczkach 60 świec.
- Jednorazowa synchronizacja na starcie w celu wypełnienia "luki" w danych.
- Inteligentne łączenie danych i okresowy zapis na dysk w celu zapewnienia
  ciągłości i wydajności.
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.enums import RunMode
from freqtrade.exceptions import DependencyException

# 🚀 SETUP PROJECT PATH 🚀
# To zapewnia, że importy działają poprawnie, niezależnie od sposobu uruchomienia.
# ft_bot_clean/user_data/strategies/ -> ft_bot_clean/user_data/ -> ft_bot_clean/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importy komponentów
from user_data.strategies.components.signal_generator import SignalGenerator
from user_data.strategies.utils.model_loader import ModelLoader
from user_data.strategies.utils.pair_manager import PairManager

logger = logging.getLogger(__name__)

class Enhanced_ML_MA43200_Buffer_Strategy(IStrategy):
    strategy_name = "Enhanced_ML_MA43200_Buffer_Strategy_v7"
    timeframe = '1m'
    startup_candle_count: int = 0

    # ROI table:
    minimal_roi = {
        "0": 0.008
    }

    # Stoploss:
    stoploss = -0.003

    # Trailing stop:
    trailing_stop = False

    # Wymuś opłaty na poziomie 0
    def custom_fee(self, pair: str, side: str, amount: float, price: float, 
                   taker_or_maker: str) -> float:
        """Wymusza opłaty na poziomie 0 dla backtestingu"""
        return 0.0
    
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
        'stoploss_on_exchange_interval': 60,
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC',
    }

    # Parametry strategii
    stake_currency = 'USDT'
    stake_amount = 120
    unfilledtimeout = {
        'entry': 10,
        'unit': 'minutes',
        'exit_timeout_count': 0,
    }
    max_open_trades = 100

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        
        # Konfiguracja ML
        self.ml_config = config.get('ml_config', {}) if config else {}
        self.enabled = self.ml_config.get('enabled', True)
        
        # Inicjalizacja komponentów
        self.signal_generator = SignalGenerator(self.ml_config)
        self.model_loader = ModelLoader()
        self.pair_manager = PairManager()
        
        # Cache dla modeli i scalerów
        self.models_cache = {}
        self.scalers_cache = {}
        
        # Dane z cechami (załadowane raz na początku)
        self.features_data = None
        self.features_loaded = False
        
        # Log predykcji
        self.predictions_log = []
        
        # Konfiguracja backtestingu
        self._load_backtest_config()

    def _load_backtest_config(self) -> None:
        """Ładuje konfigurację dla backtestingu."""
        self.ml_confidence_short = self.ml_config.get('confidence_threshold_short', 0.40)
        self.ml_confidence_long = self.ml_config.get('confidence_threshold_long', 0.40)
        self.ml_confidence_neutral = self.ml_config.get('confidence_threshold_neutral', 0.30)


    def bot_start(self, **kwargs) -> None:
        """Inicjalizacja na starcie bota."""
        if not self.enabled:
            logger.warning("❌ ML Strategy jest wyłączona w konfiguracji.")
            return
            
        logger.info("🚀 Enhanced ML Strategy v7.0 (Single Model Architecture) initialized!")
        
        # Logowanie konfiguracji ML
        logger.info(f"🔧 Konfiguracja ML:")
        logger.info(f"   - Progi pewności: SHORT={self.ml_confidence_short}, LONG={self.ml_confidence_long}, NEUTRAL={self.ml_confidence_neutral}")
        logger.info(f"   - Model: Pojedynczy model ładowany z metadata.json")
        
        # Przekaż progi pewności do SignalGenerator
        self.signal_generator.set_thresholds(
            short_threshold=self.ml_confidence_short,
            long_threshold=self.ml_confidence_long,
            neutral_threshold=self.ml_confidence_neutral
        )
        
        # Inicjalizacja systemu wieloparowego
        self._initialize_multi_pair_system()
        
        # Ładowanie danych z cechami
        self._load_features_data()

    def _initialize_multi_pair_system(self) -> None:
        """Inicjalizuje system wieloparowy."""
        logger.info("🔄 Inicjalizacja systemu wieloparowego...")
        
        # Pobierz listę par z konfiguracji
        pairs = self.config.get('pair_whitelist', [])
        if not pairs:
            # Fallback - użyj par z pairlist
            pairs = ["BTC/USDT:USDT"]
            logger.info(f"🔄 Używam domyślnej pary: {pairs}")
            
        # Inicjalizuj pary
        self._initialize_models_for_pairs(pairs)

    def _initialize_models_for_pairs(self, pairs: list) -> None:
        """Inicjalizuje modele dla wszystkich par."""
        logger.info(f"🔄 Inicjalizacja modeli dla {len(pairs)} par...")
        
        for pair in pairs:
            try:
                # Załaduj model, scaler i metadata
                model, scaler, metadata = self.model_loader.load_model_for_pair(pair)
                
                if model and scaler:
                    normalized_pair = self._normalize_pair_name(pair)
                    self.models_cache[normalized_pair] = model
                    self.scalers_cache[normalized_pair] = scaler
                    
                    logger.info(f"✅ {pair}: Model i scaler załadowane pomyślnie.")
                else:
                    logger.error(f"❌ {pair}: Nie udało się załadować modelu lub scalera.")
                    
            except Exception as e:
                logger.error(f"❌ {pair}: Błąd podczas ładowania modelu: {e}")

    def _load_features_data(self) -> None:
        """Ładuje dane z cechami z pliku labeler3 (ten sam plik co używany podczas treningu)."""
        try:
            # Ścieżka do pliku z cechami z labeler3 (bezwzględna)
            features_path = Path("C:/Users/macie/OneDrive/Python/Binance/crypto/labeler3/output/ohlc_orderbook_labeled_3class_fw120m_5levels.feather")
            
            if not features_path.exists():
                logger.error(f"❌ Plik z cechami nie istnieje: {features_path}")
                return
                
            # Wczytaj dane
            self.features_data = pd.read_feather(features_path)
            
            # Konwertuj timestamp na datetime z UTC
            if 'timestamp' in self.features_data.columns:
                self.features_data['date'] = pd.to_datetime(self.features_data['timestamp'], utc=True)
                self.features_data.set_index('date', inplace=True)
            elif 'date' in self.features_data.columns:
                self.features_data['date'] = pd.to_datetime(self.features_data['date'], utc=True)
                self.features_data.set_index('date', inplace=True)
            else:
                logger.error("❌ Brak kolumny timestamp lub date w pliku z cechami")
                return
            
            # Sprawdź czy wszystkie wymagane cechy są dostępne
            required_features = self.signal_generator.feature_columns
            missing_features = [f for f in required_features if f not in self.features_data.columns]
            
            if missing_features:
                logger.error(f"❌ Brakujące cechy: {missing_features}")
                return
                
            self.features_loaded = True
            logger.info(f"✅ Dane z cechami załadowane z pliku labeler3: {len(self.features_data)} wierszy, {len(required_features)} cech")
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas ładowania danych z cechami: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Główna metoda FreqTrade - wywoływana dla każdej świecy.
        """
        pair = metadata['pair']
        
        if not self.enabled:
            logger.warning(f"❌ {pair}: ML Strategy jest wyłączona.")
            return dataframe
            
        if not self.features_loaded:
            logger.error(f"❌ {pair}: Dane z cechami nie zostały załadowane.")
            return dataframe
        
        # Wybierz tryb działania na podstawie runmode
        if self.dp.runmode == RunMode.BACKTEST:
            return self._populate_for_backtest(dataframe, pair)
        else:
            return self._populate_for_live(dataframe, pair)

    def _populate_for_backtest(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Logika dla trybu BACKTEST - przetwarzanie wszystkich świec na raz.
        """
        logger.info(f"🚀 {pair}: Wykryto tryb BACKTEST. Uruchamianie logiki 'świeca po świecy'...")
        
        # Inicjalizuj kolumny sygnałowe
        self._initialize_signal_columns(dataframe)
        
        # Pobierz wszystkie timestampy z dataframe
        timestamps = dataframe['date'].tolist()
        logger.info(f"🚀 {pair}: Rozpoczynanie przetwarzania dla {len(timestamps)} świec...")
        
        # SPRAWDZENIE DANYCH OHLC - porównanie z danymi z pliku z cechami
        logger.info(f"🔍 {pair}: Sprawdzanie zgodności danych OHLC...")
        mismatches = 0
        total_checked = 0
        
        for i, timestamp in enumerate(timestamps[:100]):  # Sprawdź pierwsze 100 świec
            if timestamp in self.features_data.index:
                # Pobierz dane OHLC z FreqTrade
                ft_open = dataframe.iloc[i]['open']
                ft_high = dataframe.iloc[i]['high']
                ft_low = dataframe.iloc[i]['low']
                ft_close = dataframe.iloc[i]['close']
                ft_volume = dataframe.iloc[i]['volume']
                
                # Pobierz dane OHLC z pliku z cechami
                feat_open = self.features_data.loc[timestamp, 'open']
                feat_high = self.features_data.loc[timestamp, 'high']
                feat_low = self.features_data.loc[timestamp, 'low']
                feat_close = self.features_data.loc[timestamp, 'close']
                feat_volume = self.features_data.loc[timestamp, 'volume']
                
                # Sprawdź czy są identyczne
                if (abs(ft_open - feat_open) > 0.01 or 
                    abs(ft_high - feat_high) > 0.01 or 
                    abs(ft_low - feat_low) > 0.01 or 
                    abs(ft_close - feat_close) > 0.01 or 
                    abs(ft_volume - feat_volume) > 0.01):
                    mismatches += 1
                    if mismatches <= 5:  # Pokaż tylko pierwsze 5 różnic
                        logger.warning(f"⚠️ {pair}: Różnica OHLC dla {timestamp}:")
                        logger.warning(f"   FreqTrade: O={ft_open:.2f}, H={ft_high:.2f}, L={ft_low:.2f}, C={ft_close:.2f}, V={ft_volume:.2f}")
                        logger.warning(f"   Features:  O={feat_open:.2f}, H={feat_high:.2f}, L={feat_low:.2f}, C={feat_close:.2f}, V={feat_volume:.2f}")
                
                total_checked += 1
        
        logger.info(f"🔍 {pair}: Sprawdzono {total_checked} świec, znaleziono {mismatches} różnic w danych OHLC")
        
        # SPRAWDZENIE TIMESTAMPÓW - ile z FreqTrade nie ma w danych z cechami
        logger.info(f"🔍 {pair}: Sprawdzanie dostępności timestampów...")
        missing_timestamps = 0
        available_timestamps = 0
        
        for timestamp in timestamps:
            if timestamp in self.features_data.index:
                available_timestamps += 1
            else:
                missing_timestamps += 1
                if missing_timestamps <= 5:  # Pokaż tylko pierwsze 5 brakujących
                    logger.warning(f"⚠️ {pair}: Timestamp {timestamp} nie znaleziony w danych z cechami")
        
        logger.info(f"🔍 {pair}: Dostępne timestampy: {available_timestamps}, brakujące: {missing_timestamps}")
        logger.info(f"🔍 {pair}: Zakres timestampów FreqTrade: {timestamps[0]} - {timestamps[-1]}")
        logger.info(f"🔍 {pair}: Zakres timestampów Features: {self.features_data.index.min()} - {self.features_data.index.max()}")
        
        # Pobierz model i scaler
        model = self.models_cache.get(self._normalize_pair_name(pair))
        scaler = self.scalers_cache.get(self._normalize_pair_name(pair))
        
        if not model or not scaler:
            logger.error(f"❌ {pair}: Brak modelu lub scalera.")
            return dataframe
        
        # Przygotuj cechy dla wszystkich timestampów
        features_list = []
        valid_indices = []
        
        for i, timestamp in enumerate(timestamps):
            try:
                # Znajdź cechy dla tego timestampu
                if timestamp in self.features_data.index:
                    features = self.features_data.loc[timestamp, self.signal_generator.feature_columns].values
                    features = features.astype(np.float64)
                    
                    if np.isfinite(features).all():
                        features_list.append(features)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"⚠️ {pair}: Nieprawidłowe cechy dla {timestamp}")
                else:
                    logger.warning(f"⚠️ {pair}: Timestamp {timestamp} nie znaleziony w danych z cechami")
                    
            except Exception as e:
                logger.warning(f"⚠️ {pair}: Błąd dla timestamp {timestamp}: {e}")
        
        if not features_list:
            logger.error(f"❌ {pair}: Brak prawidłowych cech do predykcji.")
            return dataframe
        
        # Utwórz DataFrame z cechami
        features_df = pd.DataFrame(features_list, columns=self.signal_generator.feature_columns)
        logger.info(f"🤖 {pair}: Przygotowywanie batch prediction dla {len(features_list)} próbek...")
        
        # Generuj sygnały dla wszystkich świec w batch
        all_signal_dicts = self.signal_generator.generate_signals_for_batch(model, scaler, features_df)
        
        # Przypisz sygnały do odpowiednich wierszy
        logger.info(f"✍️ {pair}: Przypisywanie {len(all_signal_dicts)} sygnałów do ramki danych...")
        
        for i, signal_data in enumerate(all_signal_dicts):
            if i < len(valid_indices):
                # Pobierz oryginalny indeks z `dataframe` i timestamp
                original_idx = valid_indices[i]
                timestamp = dataframe.at[original_idx, 'date']
                
                # Przypisz sygnały
                signal = signal_data.get('signal')
                if signal == 'long':
                    dataframe.at[original_idx, 'enter_long'] = 1
                elif signal == 'short':
                    dataframe.at[original_idx, 'enter_short'] = 1
                # neutral nie generuje sygnałów wejścia
                
                # Zapisz dane do logu, jeśli istnieją
                if signal_data and 'probabilities' in signal_data:
                    self.predictions_log.append({
                        'timestamp': timestamp,
                        'pair': pair,
                        'signal': signal,
                        'confidence': signal_data.get('confidence'),
                        'prob_SHORT': signal_data['probabilities'][1],
                        'prob_LONG': signal_data['probabilities'][0],
                        'prob_NEUTRAL': signal_data['probabilities'][2]
                    })

        logger.info(f"✅ {pair}: Zakończono backtesting.")
        
        # Zapisz log predykcji
        if self.predictions_log:
            self._save_predictions_log(pair)
        
        return dataframe

    def _populate_for_live(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Logika dla trybu LIVE - przetwarzanie tylko ostatniej świecy.
        """
        self._initialize_signal_columns(dataframe)
        
        # Pobierz ostatnią świecę
        last_candle_timestamp = dataframe.iloc[-1]['date']
        
        # Znajdź cechy dla ostatniego timestampu
        if last_candle_timestamp in self.features_data.index:
            features = self.features_data.loc[last_candle_timestamp, self.signal_generator.feature_columns].values
            features = features.astype(np.float64)
            
            if np.isfinite(features).all():
                model = self.models_cache.get(self._normalize_pair_name(pair))
                scaler = self.scalers_cache.get(self._normalize_pair_name(pair))
                
                if model and scaler:
                    # Utwórz dataframe z cechami dla ostatniego wiersza
                    features_df = pd.DataFrame([features], columns=self.signal_generator.feature_columns)
                    
                    signal_data = self.signal_generator.generate_signal(model, scaler, features_df)
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
        
        # Inicjalizuj kolumny jeśli nie istnieją
        self._initialize_signal_columns(dataframe)
        
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

        if signal == 'LONG':
            df.loc[index, 'enter_long'] = 1
        elif signal == 'SHORT':
            df.loc[index, 'enter_short'] = 1
        
    def _save_predictions_log(self, pair: str):
        """Zapisuje log predykcji do pliku CSV."""
        if not self.predictions_log:
            return
            
        try:
            # Utwórz katalog jeśli nie istnieje
            log_dir = Path("user_data/backtest_results")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Nazwa pliku z timestampem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{pair.replace('/', '').replace(':', '')}_{timestamp}.csv"
            filepath = log_dir / filename
            
            # Zapisz do CSV
            df_log = pd.DataFrame(self.predictions_log)
            df_log.to_csv(filepath, index=False)
            
            logger.info(f"✅ Zapisano log predykcji do pliku: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas zapisywania logu predykcji: {e}")

    def bot_loop_start(self, current_time, **kwargs) -> None:
        """Wywoływane na początku każdej pętli bota."""
        pass