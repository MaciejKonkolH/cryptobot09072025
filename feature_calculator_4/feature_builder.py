from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import (
    safe_div,
    ema,
    atr as fn_atr,
    rsi,
    stoch_kd,
    macd_hist,
    bollinger_pos_width,
    donchian_high,
    donchian_low,
    adx_di,
)


def _fit_parallel_channel(close_window: np.ndarray) -> Tuple[float, float, float, float]:
    n = close_window.shape[0]
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, close_window, 1)
    center = slope * x + intercept
    resid = close_window - center
    offset_low = float(resid.min())
    offset_high = float(resid.max())
    return float(slope), float(intercept), offset_low, offset_high


def _channel_metrics(df: pd.DataFrame, window: int, progress: bool = False) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Rolling ATR for normalization (window-specific)
    atr_w = fn_atr(high, low, close, window)

    # Prepare containers
    pos_vals: List[float] = []
    width_over_atr_vals: List[float] = []
    slope_over_atr_window_vals: List[float] = []
    fit_score_vals: List[float] = []

    closes = close.values
    iterator = range(len(df))
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(iterator, desc=f"kanał {window}m", unit="bar")
        except Exception:
            pass
    for end_idx in iterator:
        start_idx = end_idx - window + 1
        if start_idx < 0:
            pos_vals.append(np.nan)
            width_over_atr_vals.append(np.nan)
            slope_over_atr_window_vals.append(np.nan)
            fit_score_vals.append(np.nan)
            continue
        segment = closes[start_idx : end_idx + 1]
        slope, intercept, off_low, off_high = _fit_parallel_channel(segment)
        n = len(segment)
        x = np.arange(n, dtype=float)
        center = slope * x + intercept
        support = center + off_low
        resist = center + off_high
        last_close = float(segment[-1])
        sup_last = float(support[-1])
        res_last = float(resist[-1])
        width_now = float(res_last - sup_last)
        pos = (last_close - sup_last) / width_now if width_now > 0 else np.nan
        pos = float(np.clip(pos, 0.0, 1.0)) if np.isfinite(pos) else np.nan

        # width over ATR(window) using ATR at end_idx
        atr_val = float(atr_w.iloc[end_idx]) if not np.isnan(atr_w.iloc[end_idx]) else np.nan
        width_over_atr = (width_now / atr_val) if (atr_val and atr_val > 0) else np.nan

        # slope over ATR(window) across whole window (delta center vs first point)
        delta_center = float(center[-1] - center[0])
        slope_over_atr_window = (delta_center / atr_val) if (atr_val and atr_val > 0) else np.nan

        # fit score: 1 - IQR(residuals) / channel_width
        resid = segment - center
        q75, q25 = np.percentile(resid, [75, 25])
        iqr = float(q75 - q25)
        channel_width = float(resid.max() - resid.min())
        if channel_width <= 0:
            fit_score = np.nan
        else:
            fit_score = float(np.clip(1.0 - (iqr / channel_width), 0.0, 1.0))

        pos_vals.append(pos)
        width_over_atr_vals.append(width_over_atr)
        slope_over_atr_window_vals.append(slope_over_atr_window)
        fit_score_vals.append(fit_score)

    out = pd.DataFrame(index=df.index)
    out[f"pos_in_channel_{window}"] = pos_vals
    out[f"width_over_ATR_{window}"] = width_over_atr_vals
    out[f"slope_over_ATR_window_{window}"] = slope_over_atr_window_vals
    out[f"channel_fit_score_{window}"] = fit_score_vals
    return out


def _ohlc_ta(df: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df.get("volume", pd.Series(index=df.index, dtype=float)).astype(float)

    # ATR for normalization
    atr_120 = fn_atr(high, low, close, 120)

    # Optional step-progress bar
    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=15, desc="ohlc/ta", unit="step")
        except Exception:
            pbar = None

    # Candle shape
    range_ = (high - low).abs()
    out["body_ratio"] = safe_div((close - open_).abs(), range_)
    out["wick_up_ratio"] = safe_div((high - np.maximum(open_, close)), range_)
    out["wick_down_ratio"] = safe_div((np.minimum(open_, close) - low), range_)
    if pbar: pbar.update(1)

    # Returns and trend
    out["r_1"] = safe_div(close, close.shift(1)) - 1.0
    out["r_5"] = safe_div(close, close.shift(5)) - 1.0
    out["r_15"] = safe_div(close, close.shift(15)) - 1.0
    out["slope_return_120"] = safe_div(close - close.shift(120), close.shift(120))
    if pbar: pbar.update(1)

    # Volatility regime
    ret1 = np.log(safe_div(close, close.shift(1)).replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    out["vol_regime_120"] = ret1.rolling(120, min_periods=120).std(ddof=0)
    roll_std = ret1.rolling(30, min_periods=30).std(ddof=0)
    out["vol_of_vol_120"] = roll_std.rolling(120, min_periods=120).std(ddof=0)
    if pbar: pbar.update(1)

    # Rolling R^2 approximation via correlation^2 with time index
    x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    corr = close.rolling(120, min_periods=120).corr(x)
    out["r2_trend_120"] = (corr ** 2).clip(lower=0.0, upper=1.0)
    if pbar: pbar.update(1)

    # RSI
    out["RSI_14"] = rsi(close, 14)
    out["RSI_30"] = rsi(close, 30)
    if pbar: pbar.update(1)

    # Stochastic
    k14, d14 = stoch_kd(high, low, close, 14, 3)
    out["StochK_14_3"] = k14
    out["StochD_14_3"] = d14
    if pbar: pbar.update(1)

    # MACD histogram normalized by ATR
    macd_h = macd_hist(close, 12, 26, 9)
    out["MACD_hist_over_ATR"] = safe_div(macd_h, atr_120)
    if pbar: pbar.update(1)

    # ADX and DI
    plus_di, minus_di, adx = adx_di(high, low, close, 14)
    out["ADX_14"] = adx
    out["di_diff_14"] = (plus_di - minus_di)
    if pbar: pbar.update(1)

    # CCI over ATR
    tp = (high + low + close) / 3.0
    sma_tp_20 = tp.rolling(20, min_periods=20).mean()
    mad_tp_20 = (tp - sma_tp_20).abs().rolling(20, min_periods=20).mean()
    cci_20 = safe_div(tp - sma_tp_20, 0.015 * mad_tp_20)
    out["CCI_20_over_ATR"] = safe_div(cci_20, atr_120)
    if pbar: pbar.update(1)

    # Bollinger 20
    bb_pos_20, bb_width_20 = bollinger_pos_width(close, 20, 2.0)
    out["bb_pos_20"] = bb_pos_20
    out["bb_width_over_ATR_20"] = safe_div(bb_width_20, atr_120)
    if pbar: pbar.update(1)

    # Donchian 60
    dh60 = donchian_high(high, 60)
    dl60 = donchian_low(low, 60)
    out["donch_pos_60"] = safe_div(close - dl60, (dh60 - dl60))
    out["donch_width_over_ATR_60"] = safe_div((dh60 - dl60).abs(), atr_120)
    if pbar: pbar.update(1)

    # Price vs EMA and slope
    ema60 = ema(close, 60)
    ema120 = ema(close, 120)
    out["close_vs_ema_60"] = safe_div(close - ema60, ema60)
    out["close_vs_ema_120"] = safe_div(close - ema120, ema120)
    out["slope_ema_60_over_ATR"] = safe_div(ema60 - ema60.shift(cfg.EMA_SLOPE_K), atr_120)
    if pbar: pbar.update(1)

    # MFI 14
    tp_price = (high + low + close) / 3.0
    raw_mf = tp_price * volume
    pos_flow = raw_mf.where(tp_price >= tp_price.shift(1), 0.0)
    neg_flow = raw_mf.where(tp_price < tp_price.shift(1), 0.0)
    sum_pos = pos_flow.rolling(14, min_periods=14).sum()
    sum_neg = neg_flow.rolling(14, min_periods=14).sum()
    mfr = safe_div(sum_pos, sum_neg)
    out["MFI_14"] = 100.0 - (100.0 / (1.0 + mfr))
    if pbar: pbar.update(1)

    # OBV slope over ATR
    obv = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
    out["OBV_slope_over_ATR"] = safe_div(obv - obv.shift(cfg.EMA_SLOPE_K), atr_120)
    if pbar:
        pbar.update(1)
        pbar.close()

    return out


def _orderbook_features(df: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Expect columns aggregated by percentage buckets if available; fall back to depth-based approximations if not present
    # Preferred columns
    A1 = df.get("notional_pct_p1")
    A2 = df.get("notional_pct_p2")
    B1 = df.get("notional_pct_m1")
    B2 = df.get("notional_pct_m2")

    # If percentage-aggregated columns are missing, attempt to approximate using level sums per side if present
    if A1 is None and all(col in df.columns for col in [f"notional_1_p{i}" for i in range(1, 6)]):
        A1 = df[[f"notional_1_p{i}" for i in range(1, 6)]].sum(axis=1)
    if A2 is None and all(col in df.columns for col in [f"notional_2_p{i}" for i in range(1, 6)]):
        A2 = df[[f"notional_2_p{i}" for i in range(1, 6)]].sum(axis=1)
    if B1 is None and all(col in df.columns for col in [f"notional_1_m{i}" for i in range(1, 6)]):
        B1 = df[[f"notional_1_m{i}" for i in range(1, 6)]].sum(axis=1)
    if B2 is None and all(col in df.columns for col in [f"notional_2_m{i}" for i in range(1, 6)]):
        B2 = df[[f"notional_2_m{i}" for i in range(1, 6)]].sum(axis=1)

    # Only proceed if we have base series
    has_core = A1 is not None and A2 is not None and B1 is not None and B2 is not None
    if not has_core:
        return out  # empty; OB features unavailable with current inputs

    eps = 1e-9

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
            # Base steps: imbalance, log ratios, concentration, COM, pressure, side_skew, persistence (L), deltas, ema deltas (L), optional TP/SL
            total_steps = 7 + len(cfg.IMB_EMA_LENS) + 1 + len(cfg.IMB_EMA_LENS) + 1
            pbar = tqdm(total=total_steps, desc="orderbook", unit="step")
        except Exception:
            pbar = None

    # Imbalance 1% notional
    imbalance_1pct = safe_div(B1 - A1, B1 + A1)
    out["imbalance_1pct_notional"] = imbalance_1pct
    if pbar: pbar.update(1)

    # Log ratios (liquidity slope proxies)
    out["log_ratio_ask_2_over_1"] = np.log(safe_div(A2 + eps, A1 + eps))
    out["log_ratio_bid_2_over_1"] = np.log(safe_div(B2 + eps, B1 + eps))
    if pbar: pbar.update(1)

    # Concentration near market
    out["ask_near_ratio"] = safe_div(A1, A1 + A2)
    out["bid_near_ratio"] = safe_div(B1, B1 + B2)
    out["concentration_near_mkt"] = safe_div(A1 + B1, A1 + A2 + B1 + B2)
    if pbar: pbar.update(1)

    # Center of mass
    out["ask_com"] = safe_div(1.0 * A1 + 2.0 * A2, A1 + A2)
    out["bid_com"] = safe_div(1.0 * B1 + 2.0 * B2, B1 + B2)
    out["com_diff"] = out["bid_com"] - out["ask_com"]
    if pbar: pbar.update(1)

    # Pressure
    pressure_12 = (B1 + 0.5 * B2) - (A1 + 0.5 * A2)
    out["pressure_12"] = pressure_12
    out["pressure_12_norm"] = safe_div(pressure_12, A1 + A2 + B1 + B2)
    if pbar: pbar.update(1)

    # Side skew
    out["side_skew"] = safe_div((B1 + B2) - (A1 + A2), A1 + A2 + B1 + B2)
    if pbar: pbar.update(1)

    # Persistence of imbalance
    for L in cfg.IMB_EMA_LENS:
        out[f"persistence_imbalance_1pct_ema{L}"] = out["imbalance_1pct_notional"].ewm(span=L, adjust=False).mean()
        if pbar: pbar.update(1)

    # Deltas and EMA of deltas (impulses)
    out["dA1"] = A1.diff()
    out["dB1"] = B1.diff()
    out["dImb1"] = imbalance_1pct.diff()
    if pbar: pbar.update(1)
    for L in cfg.IMB_EMA_LENS:
        out[f"ema_dImb1_{L}"] = out["dImb1"].ewm(span=L, adjust=False).mean()
        if pbar: pbar.update(1)

    # Optional per-model features if TP/SL provided in config
    tp = getattr(cfg, "TP_PARAM", None)
    sl = getattr(cfg, "SL_PARAM", None)
    if tp is not None and sl is not None:
        tp = float(tp)
        sl = float(sl)
        w1 = min(tp, 1.0)
        w2 = max(tp - 1.0, 0.0)
        s1 = min(sl, 1.0)
        s2 = max(sl - 1.0, 0.0)
        reach_TP = w1 * A1 + w2 * A2
        reach_SL = s1 * B1 + s2 * B2
        out["reach_TP_notional"] = reach_TP
        out["reach_SL_notional"] = reach_SL
        out["resistance_vs_support"] = safe_div(reach_TP, reach_SL)
        if pbar: pbar.update(1)

    if pbar: pbar.close()

    return out


def compute_features(df: pd.DataFrame, progress: bool = False) -> pd.DataFrame:
    """Compute ONLY the agreed features from cechy_do_treningu.md."""
    frames: List[pd.DataFrame] = []

    # Channel features
    channel_windows = list(cfg.CHANNEL_WINDOWS)
    if getattr(cfg, "ENABLE_EXTENDED_CHANNELS", False):
        channel_windows = sorted(set(channel_windows + list(getattr(cfg, "EXTENDED_CHANNEL_WINDOWS", []))), reverse=False)
    for w in channel_windows:
        frames.append(_channel_metrics(df, w, progress=progress))

    # OHLC-based technical features
    frames.append(_ohlc_ta(df, progress=progress))

    # Orderbook features (if inputs available)
    frames.append(_orderbook_features(df, progress=progress))

    # Optional: add training3 37-core features for compatibility
    if getattr(cfg, "ENABLE_T3_FEATURES", False):
        frames.append(_t3_core_features(df))

    out = pd.concat(frames, axis=1)

    # Interactions: channel × OB imbalance
    if "imbalance_1pct_notional" in out.columns:
        for w in channel_windows:
            pos_col = f"pos_in_channel_{w}"
            slope_col = f"slope_over_ATR_window_{w}"
            width_col = f"width_over_ATR_{w}"
            if pos_col in out.columns:
                out[f"{pos_col}_x_imbalance_1pct"] = out[pos_col] * out["imbalance_1pct_notional"]
            if slope_col in out.columns:
                out[f"{slope_col}_x_imbalance_1pct"] = out[slope_col] * out["imbalance_1pct_notional"]
            if width_col in out.columns:
                out[f"{width_col}_x_imbalance_1pct"] = out[width_col] * out["imbalance_1pct_notional"]
    return out


def _t3_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of 37 legacy training3 features computed from merged inputs.
    Names align with training3/config.py FEATURES.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    volume = df.get("volume", pd.Series(index=df.index, dtype=float)).astype(float)

    # Moving averages required
    ma60 = close.rolling(cfg.ROLLING_WINDOWS[1], min_periods=1).mean()
    ma240 = close.rolling(240, min_periods=1).mean()

    # 1) Price trend features
    out["price_trend_30m"] = close.pct_change(cfg.PRICE_TREND_PERIODS[0]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["price_trend_2h"] = close.pct_change(cfg.PRICE_TREND_PERIODS[1]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["price_trend_6h"] = close.pct_change(cfg.PRICE_TREND_PERIODS[2]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    eps = 1e-3
    out["price_strength"] = (out["price_trend_2h"].abs() / (out["price_trend_30m"].abs() + eps)).where(out["price_trend_30m"].abs() > 0.001, 0.0)
    def _sign(s: pd.Series) -> pd.Series:
        return (s > 0).astype(int) - (s < 0).astype(int)
    out["price_consistency_score"] = (
        _sign(out["price_trend_30m"]) + _sign(out["price_trend_2h"]) + _sign(out["price_trend_6h"]) 
    ) / 3.0

    # 2) Price position features
    out["price_vs_ma_60"] = safe_div(close - ma60, ma60).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["price_vs_ma_240"] = safe_div(close - ma240, ma240).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["ma_trend"] = safe_div(ma60 - ma240, ma240).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["price_volatility_rolling"] = close.pct_change().rolling(cfg.ROLLING_WINDOWS[0], min_periods=1).std().fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 3) Volume features
    out["volume_trend_1h"] = volume.pct_change(cfg.VOLUME_TREND_PERIODS[0]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    vol_ma60 = volume.rolling(cfg.ROLLING_WINDOWS[1], min_periods=1).mean()
    out["volume_intensity"] = safe_div(volume, vol_ma60).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    out["volume_volatility_rolling"] = volume.pct_change().rolling(cfg.ROLLING_WINDOWS[0], min_periods=1).std().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["volume_price_correlation"] = volume.rolling(cfg.ROLLING_WINDOWS[1], min_periods=1).corr(close).shift(1).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out["volume_momentum"] = (out["volume_trend_1h"] - out["volume_trend_1h"].shift(cfg.MOMENTUM_PERIODS[0]).fillna(0.0)).fillna(0.0)

    # 4) Orderbook relative features
    # Ensure 'spread' like in training3 sources if missing: snapshot1_depth_1 - snapshot1_depth_-1
    if "spread" not in df.columns and {"snapshot1_depth_1", "snapshot1_depth_-1"}.issubset(df.columns):
        df = df.copy()
        df["spread"] = (df["snapshot1_depth_1"].astype(float) - df["snapshot1_depth_-1"].astype(float))
    if "spread" in df.columns:
        spread_ma60 = df["spread"].rolling(cfg.ROLLING_WINDOWS[1], min_periods=1).mean()
        out["spread_tightness"] = safe_div(df["spread"], spread_ma60).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    else:
        # Synthetic spread proxy from near-market notional (A1/B1): tighter book ⇒ larger A1+B1 ⇒ smaller synthetic spread
        A1_proxy = None
        B1_proxy = None
        if all(col in df.columns for col in [f"notional_1_p{i}" for i in range(1, 6)]):
            A1_proxy = df[[f"notional_1_p{i}" for i in range(1, 6)]].sum(axis=1)
        if all(col in df.columns for col in [f"notional_1_m{i}" for i in range(1, 6)]):
            B1_proxy = df[[f"notional_1_m{i}" for i in range(1, 6)]].sum(axis=1)
        if A1_proxy is not None and B1_proxy is not None:
            synt_spread = safe_div(1.0, (A1_proxy + B1_proxy))
            spread_ma60 = synt_spread.rolling(cfg.ROLLING_WINDOWS[1], min_periods=1).mean()
            out["spread_tightness"] = safe_div(synt_spread, spread_ma60).fillna(1.0).replace([np.inf, -np.inf], 1.0)

    # depth ratios require snapshot depth columns
    have_s1 = all((f"snapshot1_depth_{lvl}" in df.columns) for lvl in cfg.BID_LEVELS + cfg.ASK_LEVELS)
    have_s2 = all((f"snapshot2_depth_{lvl}" in df.columns) for lvl in cfg.BID_LEVELS + cfg.ASK_LEVELS)
    if have_s1:
        bid_depth_s1 = sum(df[f"snapshot1_depth_{lvl}"] for lvl in cfg.BID_LEVELS)
        ask_depth_s1 = sum(df[f"snapshot1_depth_{lvl}"] for lvl in cfg.ASK_LEVELS)
        out["depth_ratio_s1"] = safe_div(bid_depth_s1, ask_depth_s1).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    elif all(col in df.columns for col in [f"notional_1_m{i}" for i in range(1,6)]) and all(col in df.columns for col in [f"notional_1_p{i}" for i in range(1,6)]):
        bid_depth_s1 = df[[f"notional_1_m{i}" for i in range(1,6)]].sum(axis=1)
        ask_depth_s1 = df[[f"notional_1_p{i}" for i in range(1,6)]].sum(axis=1)
        out["depth_ratio_s1"] = safe_div(bid_depth_s1, ask_depth_s1).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    if have_s2:
        bid_depth_s2 = sum(df[f"snapshot2_depth_{lvl}"] for lvl in cfg.BID_LEVELS)
        ask_depth_s2 = sum(df[f"snapshot2_depth_{lvl}"] for lvl in cfg.ASK_LEVELS)
        out["depth_ratio_s2"] = safe_div(bid_depth_s2, ask_depth_s2).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    elif all(col in df.columns for col in [f"notional_2_m{i}" for i in range(1,6)]) and all(col in df.columns for col in [f"notional_2_p{i}" for i in range(1,6)]):
        bid_depth_s2 = df[[f"notional_2_m{i}" for i in range(1,6)]].sum(axis=1)
        ask_depth_s2 = df[[f"notional_2_p{i}" for i in range(1,6)]].sum(axis=1)
        out["depth_ratio_s2"] = safe_div(bid_depth_s2, ask_depth_s2).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    if "depth_ratio_s1" in out.columns:
        lag = cfg.MOMENTUM_PERIODS[0]
        prev = out["depth_ratio_s1"].shift(lag).fillna(1.0)
        out["depth_momentum"] = ((out["depth_ratio_s1"] - prev) / prev).where(prev != 0, 0.0).fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 5) Market regime features
    # ADX based strength
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    _, _, adx = adx_di(high, low, close, cfg.ADX_PERIOD)
    out["market_trend_strength"] = adx.clip(lower=0, upper=100).fillna(0.0)

    # Trend direction via MA slope
    ma_short = close.rolling(cfg.MARKET_REGIME_PERIODS[0]).mean()
    ma_long = close.rolling(cfg.MARKET_REGIME_PERIODS[1]).mean()
    out["market_trend_direction"] = safe_div(ma_short - ma_long, ma_long).clip(lower=-1, upper=1).fillna(0.0)

    # Choppiness index — EXACT as in feature_calculator_download2
    period = cfg.CHOPPINESS_PERIOD
    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs())
    )
    tr_sum = tr.rolling(window=period).sum()
    path_length = (high - low).rolling(window=period).sum()
    choppiness = np.where(
        tr_sum > 0,
        100 * np.log10(path_length / tr_sum) / np.log10(period),
        0
    )
    ch_series = pd.Series(choppiness, index=df.index)
    out["market_choppiness"] = (
        ch_series.replace([np.inf, -np.inf], 0.0)
        .clip(lower=0, upper=100)
        .fillna(0.0)
    )

    # Bollinger band width (relative)
    bb_mid = close.rolling(cfg.BOLLINGER_WIDTH_PERIOD).mean()
    bb_std = close.rolling(cfg.BOLLINGER_WIDTH_PERIOD).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bollinger_band_width"] = safe_div(bb_upper - bb_lower, bb_mid).clip(lower=0, upper=1).fillna(0.0)

    # Market regime classification via ADX/Choppiness
    regime = np.where(adx > 25, 1, np.where(out["market_choppiness"] > 60, 2, 0))
    out["market_regime"] = pd.Series(regime, index=df.index).astype(float)

    # 6) Volatility clustering features
    vol_short = close.pct_change().rolling(cfg.VOLATILITY_WINDOWS[0]).std()
    vol_med = close.pct_change().rolling(cfg.VOLATILITY_WINDOWS[1]).std()
    vol_long = close.pct_change().rolling(cfg.VOLATILITY_WINDOWS[2]).std()
    out["volatility_regime"] = (vol_med.rolling(cfg.VOLATILITY_PERCENTILE_WINDOW).rank(pct=True) * 100).fillna(1.0)
    out["volatility_percentile"] = (vol_long.rolling(cfg.VOLATILITY_PERCENTILE_WINDOW).rank(pct=True) * 100).clip(lower=0, upper=100).fillna(50.0)
    # Persistence EXACT as in training3: rolling autocorr(lag=1)
    def _rolling_autocorr_lag1(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan, raw=False)
    out["volatility_persistence"] = (
        _rolling_autocorr_lag1(vol_med, cfg.VOLATILITY_WINDOWS[1])
        .clip(lower=0, upper=1)
        .fillna(0.0)
    )
    out["volatility_momentum"] = safe_div(vol_short - vol_long, vol_long.where(vol_long > cfg.VOLATILITY_MIN_THRESHOLD, np.nan)).clip(lower=-1, upper=1).fillna(0.0)
    out["volatility_of_volatility"] = safe_div(vol_med.rolling(cfg.VOLATILITY_WINDOWS[0]).std(), vol_med).clip(lower=0, upper=1).fillna(0.0)
    out["volatility_term_structure"] = safe_div(vol_short - vol_long, vol_med.where(vol_med > cfg.VOLATILITY_MIN_THRESHOLD, np.nan)).clip(lower=-1, upper=1).fillna(0.0)

    # 7) Imbalance/flow features (prefer bid/ask volume; fallback to near-market notional A1/B1)
    have_vol_cols = {"snapshot1_bid_volume", "snapshot1_ask_volume"}.issubset(df.columns)
    if have_vol_cols or True:
        if have_vol_cols:
            bid_base = df["snapshot1_bid_volume"].astype(float)
            ask_base = df["snapshot1_ask_volume"].astype(float)
        else:
            # Fallback to near-market notional (A1/B1) derived like in _orderbook_features
            A1 = df.get("notional_pct_p1")
            B1 = df.get("notional_pct_m1")
            if A1 is None and all(col in df.columns for col in [f"notional_1_p{i}" for i in range(1, 6)]):
                A1 = df[[f"notional_1_p{i}" for i in range(1, 6)]].sum(axis=1)
            if B1 is None and all(col in df.columns for col in [f"notional_1_m{i}" for i in range(1, 6)]):
                B1 = df[[f"notional_1_m{i}" for i in range(1, 6)]].sum(axis=1)
            bid_base = (B1.astype(float) if B1 is not None else pd.Series(0.0, index=df.index))
            ask_base = (A1.astype(float) if A1 is not None else pd.Series(0.0, index=df.index))

        total_base = bid_base + ask_base
        vol_imb = ((bid_base - ask_base) / total_base.where(total_base != 0, np.nan)).clip(-1, 1).fillna(0.0)
        out["volume_imbalance"] = vol_imb
        out["weighted_volume_imbalance"] = vol_imb
        out["volume_imbalance_trend"] = (vol_imb - vol_imb.rolling(cfg.PRESSURE_WINDOW).mean()).clip(-1, 1).fillna(0.0)

        # Price pressure using spread (or synthetic) if present
        spr_col = "snapshot1_spread" if "snapshot1_spread" in df.columns else ("spread" if "spread" in df.columns else None)
        if spr_col is not None:
            spread_vals = df[spr_col].astype(float)
            price_pressure = (vol_imb / spread_vals.where(spread_vals > cfg.MIN_SPREAD_THRESHOLD, np.nan))
        else:
            # Synthetic spread proxy from near-market notional
            near_total = (bid_base + ask_base)
            synt_spread = safe_div(1.0, near_total)
            price_pressure = (vol_imb / synt_spread.where(synt_spread > 0, np.nan))
        out["price_pressure"] = price_pressure.clip(-1, 1).fillna(0.0)
        out["weighted_price_pressure"] = out["price_pressure"]
        out["price_pressure_momentum"] = out["price_pressure"].diff().clip(-1, 1).fillna(0.0)

        # Order flow imbalance/trend via first differences of bases
        bid_change = bid_base.diff()
        ask_change = ask_base.diff()
        tot_change = bid_change + ask_change
        flow_imb = ((bid_change - ask_change) / tot_change.where(tot_change != 0, np.nan)).clip(-1, 1).fillna(0.0)
        out["order_flow_imbalance"] = flow_imb
        out["order_flow_trend"] = flow_imb.rolling(cfg.PRESSURE_WINDOW).mean().clip(-1, 1).fillna(0.0)

    return out

