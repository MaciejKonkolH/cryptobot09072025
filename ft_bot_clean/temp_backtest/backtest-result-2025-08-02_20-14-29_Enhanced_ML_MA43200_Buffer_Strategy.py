"""
Enhanced ML Strategy v7.0 - FreqTrade Strategy with XGBoost ML Models

OPTYMALIZOWANA WERSJA V7.0:
- XGBoost Single Model predictions z training4 pipeline
- 37 lub 71 cech (konfigurowalne per para)
- 15 modeli (różne poziomy TP/SL)
- Wybór modelu per para
- Konfiguracja w pair_config.json

Odpowiedzialny za:
- Ładowanie modeli XGBoost z training4 pipeline
- Generowanie sygnałów ML per para
- Zarządzanie parami walutowymi
- Error handling i fallback
- Backtest i live trading

NOWA STRUKTURA v7.0:
- ModelLoader: Ładowanie modeli z training4
- SignalGenerator: Generowanie sygnałów (37/71 cech)
- PairManager: Zarządzanie parami i konfiguracją
- Konfiguracja per para w pair_config.json
"""

import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.parameters import CategoricalParameter
from freqtrade.persistence import Trade

# Import custom components
from utils.model_loader import ModelLoader
from utils.pair_manager import PairManager
from components.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

class Enhanced_ML_MA43200_Buffer_Strategy(IStrategy):
    """
    Enhanced ML Strategy v7.0 - FreqTrade Strategy with XGBoost ML Models
    
    NOWA STRUKTURA v7.0:
    - Obsługa 15 modeli (różne poziomy TP/SL)
    - Obsługa 37 lub 71 cech (konfigurowalne)
    - Konfiguracja per para w pair_config.json
    - Wybór modelu per para
    """
    
    # === STRATEGY METADATA ===
    INTERFACE_VERSION = 3
    minimal_roi = {
        "0": 0.01
    }
    
    stoploss = -0.02
    
    # === TIMEFRAME & CANDLE SETTINGS ===
    timeframe = '5m'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # === POSITION SIZING ===
    position_adjustment_enable = False
    use_custom_stoploss = False
    
    # === TRADING RULES ===
    startup_candle_count = 240  # 20 godzin danych historycznych
    can_short = True  # Włącz obsługę pozycji SHORT
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
    
    # === ML CONFIGURATION ===
    # Parametry ML będą pobierane z konfiguracji par
    ml_enabled = True
    ml_fallback_enabled = True
    

    
    # === ML THRESHOLDS ===
    # Progi pewności dla sygnałów ML
    ml_long_threshold = DecimalParameter(0.3, 0.7, default=0.4, space="buy", decimals=2)
    ml_short_threshold = DecimalParameter(0.3, 0.7, default=0.4, space="buy", decimals=2)
    ml_neutral_threshold = DecimalParameter(0.3, 0.7, default=0.4, space="buy", decimals=2)
    
    # === STRATEGY COMPONENTS ===
    model_loader = None
    pair_manager = None
    signal_generator = None
    
    def __init__(self, config: dict) -> None:
        """
        Inicjalizacja strategii v7.0
        """
        super().__init__(config)
        
        # DEBUG: Sprawdzamy can_short
        logger.info(f"🔍 DEBUG: can_short = {self.can_short}")
        
        # Status tracking
        self.ml_models_loaded = {}
        self.ml_errors = {}
        self.last_model_reload = None
        
        # Dane z cechami
        self.features_data = None
        self.features_loaded = False
        
        # Cache dla modeli i scalerów
        self.models_cache = {}
        self.scalers_cache = {}
        
        # Log predykcji
        self.predictions_log = []
        
        # Inicjalizuj komponenty ML
        self._initialize_ml_components()
        
        logger.info("🚀 Enhanced ML Strategy v7.0 initialized")
    
    def _initialize_ml_components(self):
        """Inicjalizuje komponenty ML"""
        try:
            # Inicjalizuj Pair Manager
            self.pair_manager = PairManager()
            
            # Inicjalizuj Model Loader
            self.model_loader = ModelLoader()
            
            # Inicjalizuj Signal Generator
            self.signal_generator = SignalGenerator()
            
            # Ustaw progi ML
            self.signal_generator.set_thresholds(
                self.ml_long_threshold.value,
                self.ml_short_threshold.value,
                self.ml_neutral_threshold.value
            )
            
            logger.info("✅ ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ML components: {e}")
            self.ml_enabled = False
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Główna metoda FreqTrade - wywoływana dla każdej świecy.
        """
        pair = metadata['pair']
        logger.info(f"🔍 {pair}: populate_indicators - ml_enabled={self.ml_enabled}, features_loaded={self.features_loaded}")
        
        if not self.ml_enabled:
            logger.warning(f"❌ {pair}: ML Strategy jest wyłączona.")
            return dataframe
            
        if not self.features_loaded:
            logger.error(f"❌ {pair}: Dane z cechami nie zostały załadowane.")
            return dataframe
        
        # Wybierz tryb działania na podstawie runmode
        if hasattr(self, 'dp') and self.dp and hasattr(self.dp, 'runmode'):
            if self.dp.runmode.value == 'backtest':
                return self._populate_for_backtest(dataframe, pair)
            else:
                return self._populate_for_live(dataframe, pair)
        else:
            # Domyślnie backtest
            return self._populate_for_backtest(dataframe, pair)

    def _load_features_data(self) -> None:
        """Ładuje dane z cechami z pliku labeler4."""
        try:
            # Ścieżka do pliku z cechami z labeler4
            features_path = Path("C:/Users/macie/OneDrive/Python/Binance/crypto/labeler4/output/labeled_BTCUSDT.feather")
            
            if not features_path.exists():
                logger.error(f"❌ Plik z cechami nie istnieje: {features_path}")
                return
                
            # Wczytaj dane
            self.features_data = pd.read_feather(features_path)
            
            # Konwertuj timestamp na datetime z UTC
            if 'timestamp' in self.features_data.columns:
                self.features_data['date'] = pd.to_datetime(self.features_data['timestamp'], utc=True)
                self.features_data.set_index('date', inplace=True)
            else:
                logger.error("❌ Brak kolumny timestamp w pliku z cechami")
                return
            
            # Sprawdź czy wszystkie wymagane cechy są dostępne
            required_features = self.signal_generator.feature_columns
            missing_features = [f for f in required_features if f not in self.features_data.columns]
            
            logger.info(f"🔍 DEBUGGING: Wymagane cechy: {len(required_features)}, Dostępne cechy: {len(self.features_data.columns)}")
            
            if missing_features:
                logger.error(f"❌ Brakujące cechy ({len(missing_features)}): {missing_features[:10]}...")
                return
                
            self.features_loaded = True
            logger.info(f"✅ Dane z cechami załadowane z pliku labeler4: {len(self.features_data)} wierszy, {len(required_features)} cech")
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas ładowania danych z cechami: {e}")

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
        

        
                # Pobierz model i scaler
        model = self.models_cache.get(self._normalize_pair_name(pair))
        scaler = self.scalers_cache.get(self._normalize_pair_name(pair))
        
        logger.info(f"🔍 {pair}: DEBUGGING - model={model is not None}, scaler={scaler is not None}")
        
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
        
        logger.info(f"✅ {pair}: Znaleziono {len(features_list)} prawidłowych próbek z {len(timestamps)} świec")
    
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
                
                # Zapisz WSZYSTKIE predykcje do logu (LONG, SHORT, NEUTRAL)
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

    def _normalize_pair_name(self, pair: str) -> str:
        """Konwertuje nazwę pary na format używany w cache."""
        return pair.replace('/', '').replace(':', '')

    def _initialize_signal_columns(self, dataframe: pd.DataFrame):
        """Inicjalizuje kolumny sygnałowe."""
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        if 'enter_tag' not in dataframe.columns:
            dataframe['enter_tag'] = ''

    def _assign_signals_to_row(self, df: pd.DataFrame, index, signal_data: dict):
        """Przypisuje sygnały do konkretnego wiersza."""
        signal = signal_data.get('signal')
        if signal == 'long':
            df.at[index, 'enter_long'] = 1
            df.at[index, 'enter_tag'] = 'long'
        elif signal == 'short':
            df.at[index, 'enter_short'] = 1
            df.at[index, 'enter_tag'] = 'short'

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
        Generuje sygnały wyjścia - wyłącza niestandardowe sygnały wyjścia.
        Strategia opiera się wyłącznie na wbudowanych mechanizmach ROI i stop-loss.
        """
        # Wyłącz niestandardowe sygnały wyjścia
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str], 
                           side: str, **kwargs) -> bool:
        """
        Walidacja przed wejściem w pozycję - TYLKO ML
        """
        try:
            # Sprawdź czy ML jest dostępne dla tej pary
            if not self.ml_enabled:
                logger.warning(f"⚠️ {pair}: ML disabled, rejecting trade")
                return False
                
            if not self.pair_manager or not self.pair_manager.is_pair_enabled(pair):
                logger.warning(f"⚠️ {pair}: Pair not enabled, rejecting trade")
                return False
                
            if pair not in self.ml_models_loaded:
                logger.warning(f"⚠️ {pair}: ML model not loaded, rejecting trade")
                return False
            
            logger.info(f"✅ {pair}: ML trade confirmed ({side})")
            return True
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error confirming trade: {e}")
            return False
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, 
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic
        """
        # Domyślny stoploss z konfiguracji
        return self.stoploss
    
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float, 
                          entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Custom entry price logic
        """
        return proposed_rate
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        Custom exit logic
        """
        return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: Optional[float],
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Custom stake amount based on risk_multiplier from pair configuration
        """
        try:
            if self.pair_manager and self.pair_manager.is_pair_enabled(pair):
                risk_multiplier = self.pair_manager.get_risk_multiplier(pair)
                adjusted_stake = proposed_stake * risk_multiplier
                
                # Apply min/max constraints
                if min_stake is not None:
                    adjusted_stake = max(adjusted_stake, min_stake)
                if max_stake is not None:
                    adjusted_stake = min(adjusted_stake, max_stake)
                
                logger.debug(f"💰 {pair}: Stake adjusted by risk_multiplier {risk_multiplier}: {proposed_stake} -> {adjusted_stake}")
                return adjusted_stake
            
            return proposed_stake
            
        except Exception as e:
            logger.error(f"❌ {pair}: Error calculating custom stake: {e}")
            return proposed_stake

    def bot_start(self, **kwargs) -> None:
        """
        Wywoływane na starcie bota
        """
        try:
            logger.info("🚀 Bot start - inicjalizacja strategii...")
            
            # Załaduj dane z cechami
            self._load_features_data()
            
            # Załaduj modele dla aktywnych par
            self._load_models_for_active_pairs()
            
            logger.info("✅ Bot start - inicjalizacja zakończona")
            
        except Exception as e:
            logger.error(f"❌ Error in bot_start: {e}")

    def _load_models_for_active_pairs(self) -> None:
        """Ładuje modele dla aktywnych par"""
        try:
            active_pairs = self.pair_manager.get_active_pairs()
            
            for pair in active_pairs:
                try:
                    # Pobierz ustawienia pary
                    model_index = self.pair_manager.get_model_index_for_pair(pair)
                    use_basic_features = self.pair_manager.get_feature_mode_for_pair(pair)
                    
                    # Załaduj model
                    model, scaler, metadata = self.model_loader.load_model_for_pair(
                        pair, model_index, use_basic_features
                    )
                    
                    if model and scaler and metadata:
                        # Zapisz w cache
                        normalized_pair = self._normalize_pair_name(pair)
                        self.models_cache[normalized_pair] = model
                        self.scalers_cache[normalized_pair] = scaler
                        
                        # Zapisz w ml_models_loaded dla kompatybilności
                        self.ml_models_loaded[pair] = {
                            'model': model,
                            'scaler': scaler,
                            'metadata': metadata,
                            'model_index': model_index,
                            'use_basic_features': use_basic_features
                        }
                        
                        logger.info(f"✅ {pair}: Model {model_index} loaded successfully")
                    else:
                        logger.error(f"❌ {pair}: Failed to load model")
                
                except Exception as e:
                    logger.error(f"❌ {pair}: Error loading model: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models for active pairs: {e}")

    def bot_loop_start(self, **kwargs) -> None:
        """
        Wywoływane na początku każdej pętli bota
        """
        pass
    

    
    def get_strategy_stats(self) -> Dict:
        """
        Zwraca statystyki strategii
        """
        stats = {
            'strategy_version': '7.0',
            'ml_enabled': self.ml_enabled,
            'ml_models_loaded': len(self.ml_models_loaded),
            'ml_errors': len(self.ml_errors),
            'active_pairs': len(self.pair_manager.get_active_pairs()) if self.pair_manager else 0
        }
        
        if self.pair_manager:
            stats['pair_config'] = self.pair_manager.get_config_summary()
        
        return stats