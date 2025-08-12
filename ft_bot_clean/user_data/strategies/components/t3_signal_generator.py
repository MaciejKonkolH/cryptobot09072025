from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
import xgboost as xgb


class Training3SignalGenerator:
    def __init__(self, long_threshold: float = 0.5, short_threshold: float = 0.5) -> None:
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)

    def _predict_proba(self, model: xgb.Booster, scaler, features_df: pd.DataFrame) -> np.ndarray:
        # Scale preserving columns
        Xs = scaler.transform(features_df)
        dtest = xgb.DMatrix(Xs, feature_names=list(features_df.columns))
        proba = model.predict(dtest)
        # Ensure shape (n,3)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 3)
        return proba

    def _decide_signal(self, p_long: float, p_short: float, p_neutral: float) -> str:
        maxp = max(p_long, p_short, p_neutral)
        if maxp < min(self.long_threshold, self.short_threshold):
            return "neutral"
        if p_long == maxp and p_long >= self.long_threshold:
            return "long"
        if p_short == maxp and p_short >= self.short_threshold:
            return "short"
        return "neutral"

    def generate_signals_for_batch(self, model: xgb.Booster, scaler, features_df: pd.DataFrame) -> List[str]:
        if features_df is None or features_df.empty:
            return []
        proba = self._predict_proba(model, scaler, features_df)
        signals: List[str] = []
        for i in range(proba.shape[0]):
            pL, pS, pN = float(proba[i, 0]), float(proba[i, 1]), float(proba[i, 2])
            signals.append(self._decide_signal(pL, pS, pN))
        return signals

    def generate_signals_with_proba_for_batch(self, model: xgb.Booster, scaler, features_df: pd.DataFrame) -> List[dict]:
        if features_df is None or features_df.empty:
            return []
        proba = self._predict_proba(model, scaler, features_df)
        out: List[dict] = []
        for i in range(proba.shape[0]):
            pL, pS, pN = float(proba[i, 0]), float(proba[i, 1]), float(proba[i, 2])
            maxp = max(pL, pS, pN)
            signal = self._decide_signal(pL, pS, pN)
            out.append({
                'signal': signal,
                'confidence': maxp,
                'prob_LONG': pL,
                'prob_SHORT': pS,
                'prob_NEUTRAL': pN,
            })
        return out

