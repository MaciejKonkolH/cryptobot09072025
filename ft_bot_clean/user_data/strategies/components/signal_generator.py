"""
Signal Generator - Generowanie sygnałów XGBoost per para

OPTIMIZED VERSION V7.0:
- XGBoost Single Model predictions z training4 pipeline
- 37 lub 71 cech (konfigurowalne)
- Pojedynczy model ładowany z training4
- Batch predictions dla backtest
- Single predictions dla live

Odpowiedzialny za:
- Generowanie sygnałów XGBoost per para
- Obsługa 37 lub 71 cech (konfigurowalne)
- Pojedynczy model z training4
- Normalizacja danych przez RobustScaler
- Error handling w przypadku problemów z predykcją
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Klasa odpowiedzialna za generowanie sygnałów transakcyjnych na podstawie
    przewidywań pojedynczego modelu XGBoost z training4 pipeline.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.short_threshold = 0.4
        self.long_threshold = 0.4
        self.neutral_threshold = 0.4
        self.use_basic_features = False
        
        # Lista 37 podstawowych cech (z training3)
        self.basic_feature_columns = [
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
            'market_trend_strength', 'market_trend_direction', 'market_choppiness', 'bollinger_band_width', 'market_regime',
            'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend', 'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend'
        ]
        
        # Lista 71 rozszerzonych cech (z training4)
        self.extended_feature_columns = [
            # Cechy trendu ceny (5 cech) - PODSTAWOWE
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
            
            # Cechy pozycji ceny (4 cechy) - PODSTAWOWE
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            
            # Cechy wolumenu (5 cech) - PODSTAWOWE
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
            
            # Cechy orderbook (4 cechy) - PODSTAWOWE
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
            
            # Market regime (5 cech) - PODSTAWOWE
            'market_trend_strength', 'market_trend_direction', 'market_choppiness',
            'bollinger_band_width', 'market_regime',
            
            # Volatility clustering (6 cech) - PODSTAWOWE
            'volatility_regime', 'volatility_percentile', 'volatility_persistence',
            'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            
            # Order book imbalance (8 cech) - PODSTAWOWE
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
            'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
            'order_flow_imbalance', 'order_flow_trend',
            
            # Dodatkowe cechy OHLC (12) - DODATKOWE
            'bb_width', 'bb_position', 'rsi_14', 'macd_hist', 'adx_14',
            'price_to_ma_60', 'price_to_ma_240', 'ma_60_to_ma_240', 'price_to_ma_1440',
            'volume_change_norm', 'upper_wick_ratio_5m', 'lower_wick_ratio_5m',
            
            # Dodatkowe cechy bamboo_ta (6) - DODATKOWE
            'stoch_k', 'stoch_d', 'cci', 'williams_r', 'mfi', 'trange',
            
            # Dodatkowe cechy orderbook (6) - DODATKOWE
            'buy_sell_ratio_s1', 'buy_sell_ratio_s2', 'imbalance_s1', 'imbalance_s2',
            'spread_pct', 'price_imbalance',
            
            # Cechy hybrydowe (10) - DODATKOWE
            'market_microstructure_score', 'liquidity_score', 'depth_price_corr',
            'pressure_volume_corr', 'hour_of_day', 'day_of_week', 'price_momentum',
            'market_efficiency_ratio', 'price_efficiency_ratio', 'volume_efficiency_ratio'
        ]
        
        # Domyślnie używaj rozszerzonych cech
        self.feature_columns = self.extended_feature_columns

    def set_thresholds(self, short_threshold: float, long_threshold: float, neutral_threshold: float):
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        self.neutral_threshold = neutral_threshold
        logger.info(f"✅ Updated ML thresholds: SHORT={self.short_threshold}, LONG={self.long_threshold}, NEUTRAL={self.neutral_threshold}")

    def set_feature_mode(self, use_basic_features: bool):
        """Ustawia tryb cech (37 lub 71)"""
        self.use_basic_features = use_basic_features
        if use_basic_features:
            self.feature_columns = self.basic_feature_columns
            logger.info(f"✅ Using basic features mode: {len(self.feature_columns)} features")
        else:
            self.feature_columns = self.extended_feature_columns
            logger.info(f"✅ Using extended features mode: {len(self.feature_columns)} features")

    def set_feature_names(self, feature_names: List[str]):
        """Ustaw listę cech dynamicznie (np. z metadata training5)."""
        if feature_names and isinstance(feature_names, list):
            self.feature_columns = feature_names
            self.use_basic_features = False
            logger.info(f"✅ Using dynamic feature list from metadata: {len(self.feature_columns)} features")

    def generate_signal(self, model, scaler, dataframe: pd.DataFrame) -> Dict:
        """
        Generuje pojedynczy sygnał dla ostatniego wiersza dataframe.
        """
        # Sprawdź czy wszystkie wymagane cechy są dostępne
        if not all(col in dataframe.columns for col in self.feature_columns):
            missing_features = [col for col in self.feature_columns if col not in dataframe.columns]
            logger.error(f"❌ Brak wymaganych cech w dataframe: {missing_features}")
            return {'signal': 'neutral', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
        
        # Pobierz ostatni wiersz z cechami
        features = dataframe[self.feature_columns].iloc[-1].values
        
        # Sprawdź czy cechy są prawidłowe
        if not np.isfinite(features).all():
            logger.error("❌ Nieprawidłowe wartości w cechach (inf/nan)")
            return {'signal': 'neutral', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
        
        # Skaluj cechy z zachowaniem nazw kolumn
        features_df = pd.DataFrame(features.reshape(1, -1), columns=self.feature_columns)
        scaled_arr = scaler.transform(features_df)
        scaled_features = pd.DataFrame(scaled_arr, columns=self.feature_columns)
        
        # Predykcja XGBoost
        try:
            probabilities = model.predict_proba(scaled_features)
            probs = probabilities[0]
        except Exception as e:
            logger.error(f"❌ Błąd podczas predykcji: {e}")
            return {'signal': 'neutral', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
        
        signal, confidence = self._get_signal_from_probabilities(probs)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probabilities': probs
        }

    def generate_signals_for_batch(self, model, scaler, dataframe: pd.DataFrame) -> list:
        """
        Generuje sygnały dla całego dataframe.
        Zoptymalizowane pod kątem wydajności w backtestingu.
        """
        if dataframe.empty:
            logger.error("❌ Pusty dataframe")
            return []

        # Sprawdź czy wszystkie wymagane cechy są dostępne
        if not all(col in dataframe.columns for col in self.feature_columns):
            missing_features = [col for col in self.feature_columns if col not in dataframe.columns]
            logger.error(f"❌ Brak wymaganych cech w dataframe: {missing_features}")
            return []

        # Pobierz wszystkie cechy jako DataFrame, by zachować kolejność i nazwy
        X_df = dataframe[self.feature_columns].copy()
        
        # Sprawdź czy cechy są prawidłowe
        if not np.isfinite(X_df.values).all():
            logger.error("❌ Nieprawidłowe wartości w cechach (inf/nan)")
            return []
        
        # Skaluj cechy z zachowaniem nazw kolumn
        scaled_arr = scaler.transform(X_df)
        scaled_features = pd.DataFrame(scaled_arr, index=X_df.index, columns=self.feature_columns)
        
        # Predykcja XGBoost
        try:
            probabilities = model.predict_proba(scaled_features)
        except Exception as e:
            logger.error(f"❌ Błąd podczas batch predykcji: {e}")
            return []
        
        # Generuj sygnały dla każdej próbki
        signals = []
        for i, probs in enumerate(probabilities):
            signal, confidence = self._get_signal_from_probabilities(probs)
            signals.append({
                'signal': signal,
                'confidence': confidence,
                'probabilities': probs
            })
        
        return signals

    def _get_signal_from_probabilities(self, probabilities) -> Tuple[str, float]:
        """
        Konwertuje prawdopodobieństwa XGBoost na sygnał i pewność.
        
        XGBoost output mapping:
        - Klasa 0: LONG
        - Klasa 1: SHORT  
        - Klasa 2: NEUTRAL
        """
        long_prob, short_prob, neutral_prob = probabilities
        
        # Znajdź klasę z najwyższym prawdopodobieństwem
        best_class = np.argmax(probabilities)
        confidence = probabilities[best_class]
        
        # Mapowanie klas na sygnały z progami pewności
        if best_class == 0 and confidence >= self.long_threshold:
            return 'long', confidence
        elif best_class == 1 and confidence >= self.short_threshold:
            return 'short', confidence
        else:
            return 'neutral', confidence

    def get_feature_count(self) -> int:
        """Zwraca liczbę aktualnie używanych cech"""
        return len(self.feature_columns)

    def get_feature_mode(self) -> str:
        """Zwraca aktualny tryb cech"""
        return "basic" if self.use_basic_features else "extended" 