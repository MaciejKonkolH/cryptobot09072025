from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(numer: pd.Series | float, denom: pd.Series | float, eps: float = 1e-9) -> pd.Series:
    return numer / (denom + eps)


def softmax_shares(values: pd.DataFrame | pd.Series) -> pd.Series:
    v = values.astype(float)
    v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exp_v = np.exp(v - v.max())
    s = exp_v.sum()
    if s == 0:
        return v * 0.0
    return exp_v / s


def rolling_stats(series: pd.Series, window: int) -> dict[str, pd.Series]:
    roll = series.rolling(window=window, min_periods=window)
    return {
        "mean": roll.mean(),
        "std": roll.std(ddof=0),
        "min": roll.min(),
        "max": roll.max(),
    }


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def make_lag(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    return df[col].shift(k)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    # Parkinson estimator (annualization not applied; relative measure)
    hl = (high / (low + 1e-9)).apply(np.log)
    return (hl ** 2).rolling(window=window, min_periods=window).mean()


def bollinger_width(series: pd.Series, window: int, nstd: float = 2.0) -> pd.Series:
    roll = series.rolling(window=window, min_periods=window)
    m = roll.mean()
    s = roll.std(ddof=0)
    upper = m + nstd * s
    lower = m - nstd * s
    return (upper - lower) / (m.abs() + 1e-9)


def bollinger_pos(series: pd.Series, window: int, nstd: float = 2.0) -> pd.Series:
    roll = series.rolling(window=window, min_periods=window)
    m = roll.mean()
    s = roll.std(ddof=0)
    upper = m + nstd * s
    lower = m - nstd * s
    return safe_div(series - lower, (upper - lower).abs() + 1e-9)


def donchian_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def donchian_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()

