from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(numer: pd.Series | float, denom: pd.Series | float, eps: float = 1e-9) -> pd.Series:
    return numer / (denom + eps)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).rolling(window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = safe_div(up, down)
    return 100.0 - (100.0 / (1.0 + rs))


def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_window, min_periods=k_window).min()
    highest = high.rolling(k_window, min_periods=k_window).max()
    k = safe_div(close - lowest, highest - lowest) * 100.0
    d = k.rolling(d_window, min_periods=d_window).mean()
    return k, d


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def bollinger_pos_width(series: pd.Series, window: int = 20, nstd: float = 2.0) -> tuple[pd.Series, pd.Series]:
    roll = series.rolling(window=window, min_periods=window)
    m = roll.mean()
    s = roll.std(ddof=0)
    upper = m + nstd * s
    lower = m - nstd * s
    pos = safe_div(series - lower, (upper - lower).abs())
    width = (upper - lower)
    return pos, width


def donchian_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def donchian_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def linear_regression_slope(series: pd.Series) -> pd.Series:
    # Rolling slope via simple OLS on index 0..n-1 per window is heavy; we use endpoints delta as proxy
    return series - series.shift(series.index.to_series().diff().dt.total_seconds().fillna(60) / 60.0)


def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute +DI, -DI and ADX using simple rolling sums (approximation of Wilder's smoothing)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0).fillna(0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0).fillna(0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    sum_tr = tr.rolling(window, min_periods=window).sum()
    sum_plus_dm = plus_dm.rolling(window, min_periods=window).sum()
    sum_minus_dm = minus_dm.rolling(window, min_periods=window).sum()

    plus_di = 100.0 * safe_div(sum_plus_dm, sum_tr)
    minus_di = 100.0 * safe_div(sum_minus_dm, sum_tr)
    dx = 100.0 * (plus_di.subtract(minus_di).abs()) / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(window, min_periods=window).mean()
    return plus_di, minus_di, adx

