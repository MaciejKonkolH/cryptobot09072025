"""
Signal Generator - Generowanie sygnałów XGBoost per para

OPTIMIZED VERSION V3.0:
- XGBoost Multi-Output predictions
- 37 cech z dataframe
- 5 poziomów TP/SL → 1 wybrany poziom
- Batch predictions dla backtest
- Single predictions dla live

Odpowiedzialny za:
- Generowanie sygnałów XGBoost per para
- Obsługa 37 cech z dataframe
- Wybór poziomu TP/SL z konfiguracji
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
    przewidywań modelu XGBoost.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.short_threshold = 0.4
        self.long_threshold = 0.4
        self.hold_threshold = 0.4
        

        
        # Lista 37 cech
        self.feature_columns = [
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h', 'price_strength', 'price_consistency_score',
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
            'market_trend_strength', 'market_trend_direction', 'market_choppiness', 'bollinger_band_width', 'market_regime',
            'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend', 'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend'
        ]

    def set_thresholds(self, short_threshold: float, long_threshold: float, hold_threshold: float):
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        self.hold_threshold = hold_threshold
        logger.info(f"✅ Updated ML thresholds: SHORT={self.short_threshold}, LONG={self.long_threshold}, HOLD={self.hold_threshold}")

    def generate_signal(self, model, scaler, dataframe: pd.DataFrame, selected_model_index: int = 2) -> Dict:
        """
        Generuje pojedynczy sygnał dla ostatniego wiersza dataframe z 37 cechami.
        """
        # Pobierz ostatni wiersz z cechami
        if not all(col in dataframe.columns for col in self.feature_columns):
            logger.error("❌ Brak wymaganych cech w dataframe")
            return {'signal': 'hold', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.34]}
        
        features = dataframe[self.feature_columns].iloc[-1].values
        
        # Skaluj cechy - przekaż nazwy cech żeby uniknąć ostrzeżenia
        features_df = pd.DataFrame(features.reshape(1, -1), columns=self.feature_columns)
        scaled_features = scaler.transform(features_df)
        
        # Predykcja XGBoost (wybrany model z MultiOutputClassifier)
        if hasattr(model, 'estimators_') and len(model.estimators_) > selected_model_index:
            # MultiOutputClassifier - użyj wybranego modelu
            selected_model = model.estimators_[selected_model_index]
            probabilities = selected_model.predict_proba(scaled_features)
        else:
            # Pojedynczy model
            probabilities = model.predict_proba(scaled_features)
        
        # Pobierz prawdopodobieństwa dla pierwszej (i jedynej) próbki
        probs = probabilities[0]
        
        signal, confidence = self._get_signal_from_probabilities(probs)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probabilities': probs
        }

    def generate_signals_for_batch(self, model, scaler, dataframe: pd.DataFrame, selected_model_index: int = 2) -> list:
        """
        Generuje sygnały dla całego dataframe.
        Zoptymalizowane pod kątem wydajności w backtestingu.
        """
        if dataframe.empty or not all(col in dataframe.columns for col in self.feature_columns):
            logger.error("❌ Brak wymaganych cech w dataframe")
            return []

        # Pobierz wszystkie cechy
        features = dataframe[self.feature_columns].values
        
        # Skaluj cechy - przekaż nazwy cech żeby uniknąć ostrzeżenia
        features_df = pd.DataFrame(features, columns=self.feature_columns)
        scaled_features = scaler.transform(features_df)
        
        # Predykcja XGBoost (wybrany model z MultiOutputClassifier)
        if hasattr(model, 'estimators_') and len(model.estimators_) > selected_model_index:
            # MultiOutputClassifier - użyj wybranego modelu
            selected_model = model.estimators_[selected_model_index]
            probabilities = selected_model.predict_proba(scaled_features)
        else:
            # Pojedynczy model
            probabilities = model.predict_proba(scaled_features)
        
        results = []
        for prob in probabilities:
            signal, confidence = self._get_signal_from_probabilities(prob)
            results.append({
                'signal': signal,
                'confidence': confidence,
                'probabilities': prob
            })
            
        return results

    def _get_signal_from_probabilities(self, probabilities: np.ndarray) -> Tuple[str, float]:
        """Logika konwersji prawdopodobieństw na sygnał."""
        long_prob, short_prob, neutral_prob = probabilities  # XGBoost: 0=LONG, 1=SHORT, 2=NEUTRAL
        
        best_class = np.argmax(probabilities)
        confidence = probabilities[best_class]

        if best_class == 0 and confidence >= self.long_threshold:
            return "LONG", confidence
        elif best_class == 1 and confidence >= self.short_threshold:
            return "SHORT", confidence
        else:
            return "NEUTRAL", neutral_prob

    def _setup_gpu(self):
        """Konfiguruje pamięć GPU, aby zapobiec błędom OOM."""
        # XGBoost nie wymaga GPU setup
        logger.info("✅ XGBoost model - no GPU setup required") 