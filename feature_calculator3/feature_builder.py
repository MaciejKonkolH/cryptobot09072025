from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from . import config as cfg
from .utils import (
    safe_div,
    ema,
    make_lag,
    true_range,
    atr as fn_atr,
    parkinson_vol,
    bollinger_width,
    bollinger_pos,
    donchian_high,
    donchian_low,
)


ORDERBOOK_COLS = {
    "depth": {
        1: {"m": [f"depth_1_m{k}" for k in range(1, 6)], "p": [f"depth_1_p{k}" for k in range(1, 6)]},
        2: {"m": [f"depth_2_m{k}" for k in range(1, 6)], "p": [f"depth_2_p{k}" for k in range(1, 6)]},
    },
    "notional": {
        1: {"m": [f"notional_1_m{k}" for k in range(1, 6)], "p": [f"notional_1_p{k}" for k in range(1, 6)]},
        2: {"m": [f"notional_2_m{k}" for k in range(1, 6)], "p": [f"notional_2_p{k}" for k in range(1, 6)]},
    },
}


def compute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Sums per side per snapshot
    sum_bid_1 = df[ORDERBOOK_COLS["depth"][1]["m"]].sum(axis=1)
    sum_ask_1 = df[ORDERBOOK_COLS["depth"][1]["p"]].sum(axis=1)
    sum_bid_2 = df[ORDERBOOK_COLS["depth"][2]["m"]].sum(axis=1)
    sum_ask_2 = df[ORDERBOOK_COLS["depth"][2]["p"]].sum(axis=1)

    # Imbalance
    out["imb_s1"] = safe_div(sum_bid_1 - sum_ask_1, sum_bid_1 + sum_ask_1)
    out["imb_s2"] = safe_div(sum_bid_2 - sum_ask_2, sum_bid_2 + sum_ask_2)
    out["imb_delta"] = out["imb_s2"] - out["imb_s1"]

    # Near-touch shares
    out["near_bid_share_s1"] = safe_div(df["depth_1_m1"], sum_bid_1)
    out["near_ask_share_s1"] = safe_div(df["depth_1_p1"], sum_ask_1)
    out["near_pressure_ratio_s1"] = safe_div(out["near_ask_share_s1"], out["near_bid_share_s1"])

    # WADL shares (expected level index)
    idx = np.arange(1, 6)
    bid_share_1 = df[ORDERBOOK_COLS["depth"][1]["m"]].div(sum_bid_1.replace(0, np.nan), axis=0).fillna(0.0)
    ask_share_1 = df[ORDERBOOK_COLS["depth"][1]["p"]].div(sum_ask_1.replace(0, np.nan), axis=0).fillna(0.0)
    out["wadl_bid_share_s1"] = (bid_share_1 * idx).sum(axis=1)
    out["wadl_ask_share_s1"] = (ask_share_1 * idx).sum(axis=1)
    out["wadl_diff_share_s1"] = out["wadl_ask_share_s1"] - out["wadl_bid_share_s1"]

    # Intraminute deltas (relative)
    out["delta_depth_bid_rel"] = safe_div(sum_bid_2 - sum_bid_1, sum_bid_1)
    out["delta_depth_ask_rel"] = safe_div(sum_ask_2 - sum_ask_1, sum_ask_1)
    out["delta_depth_ratio"] = safe_div(
        out["delta_depth_ask_rel"] - out["delta_depth_bid_rel"],
        (out["delta_depth_ask_rel"].abs() + out["delta_depth_bid_rel"].abs()),
    )

    # Asymmetries
    out["ask_over_bid_s1"] = safe_div(sum_ask_1, sum_bid_1)

    # Microprice proxy (notional at ±1)
    not_bid1_1 = df["notional_1_m1"]
    not_ask1_1 = df["notional_1_p1"]
    out["microprice_proxy_s1"] = safe_div(not_ask1_1, not_ask1_1 + not_bid1_1)

    # Tightness / steepness (shares)
    out["tightness_rel"] = out["near_ask_share_s1"] + out["near_bid_share_s1"]

    sum_bid_1_ex1 = sum_bid_1 - df["depth_1_m1"]
    sum_ask_1_ex1 = sum_ask_1 - df["depth_1_p1"]
    out["steep_bid_share_s1"] = safe_div(df["depth_1_m1"], sum_bid_1_ex1)
    out["steep_ask_share_s1"] = safe_div(df["depth_1_p1"], sum_ask_1_ex1)

    # OHLC-derived
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    out["ret_1m"] = np.log(close / close.shift(1))
    ret1 = out["ret_1m"].fillna(0.0)
    # Realized volatility short and extended
    out["rv_5m"] = ret1.rolling(5, min_periods=5).std(ddof=0)
    out["rv_30"] = ret1.rolling(30, min_periods=30).std(ddof=0)
    out["rv_60"] = ret1.rolling(60, min_periods=60).std(ddof=0)
    out["rv_120"] = ret1.rolling(120, min_periods=120).std(ddof=0)

    # ATR and derivatives
    out["atr_14"] = fn_atr(high, low, close, 14)
    out["atr_30"] = fn_atr(high, low, close, 30)
    out["atr_60"] = fn_atr(high, low, close, 60)
    out["atr_pct_14"] = safe_div(out["atr_14"], close)

    # Parkinson volatility
    out["parkinson_60"] = parkinson_vol(high, low, 60)
    out["parkinson_120"] = parkinson_vol(high, low, 120)

    # Bollinger bands (on close)
    out["bb_width_60"] = bollinger_width(close, 60, nstd=2.0)
    out["bb_pos_60"] = bollinger_pos(close, 60, nstd=2.0)

    # Donchian channels and distances
    dh60 = donchian_high(high, 60)
    dl60 = donchian_low(low, 60)
    dh120 = donchian_high(high, 120)
    dl120 = donchian_low(low, 120)
    out["donchian_high_60"] = dh60
    out["donchian_low_60"] = dl60
    out["donchian_high_120"] = dh120
    out["donchian_low_120"] = dl120
    out["dist_to_high_60"] = safe_div(dh60 - close, close)
    out["dist_to_low_60"] = safe_div(close - dl60, close)
    out["dist_to_high_60_atr"] = safe_div(dh60 - close, out["atr_14"] + 1e-9)
    out["range_60_atr"] = safe_div((dh60 - dl60).abs(), out["atr_14"] + 1e-9)
    out["range_120_atr"] = safe_div((dh120 - dl120).abs(), out["atr_14"] + 1e-9)
    out["breakout_score_60"] = safe_div(close - dh60, out["atr_14"] + 1e-9)
    out["breakdown_score_60"] = safe_div(dl60 - close, out["atr_14"] + 1e-9)
    out["since_high_break_60"] = (close > dh60.shift(1)).astype(int)
    out["since_low_break_60"] = (close < dl60.shift(1)).astype(int)
    # transform since_* into time since last event
    for col in ["since_high_break_60", "since_low_break_60"]:
        # cumulative count since last 1
        mask = out[col] == 1
        grp = (~mask).cumsum()
        out[col] = mask.groupby(grp).cumsum()

    # True range sums
    tr = true_range(high, low, close)
    out["tr_sum_30"] = tr.rolling(30, min_periods=30).sum()
    out["tr_sum_60"] = tr.rolling(60, min_periods=60).sum()
    out["tr_sum_120"] = tr.rolling(120, min_periods=120).sum()

    # Price vs MA (SMA windows) – extended per config
    for w, name in cfg.MA_WINDOWS.items():
        ma = close.rolling(w, min_periods=w).mean()
        out[name] = safe_div(close, ma)

    # OB slow aggregations (rolling means/stds/trends)
    out["imb_mean_15"] = out["imb_s1"].rolling(15, min_periods=15).mean()
    out["imb_mean_30"] = out["imb_s1"].rolling(30, min_periods=30).mean()
    out["imb_mean_60"] = out["imb_s1"].rolling(60, min_periods=60).mean()
    out["imb_persistence_30"] = (out["imb_s1"].abs() > 0.2).rolling(30, min_periods=30).mean()
    out["imb_sign_consistency_30"] = (np.sign(out["imb_s1"]) == np.sign(out["imb_s1"].shift(1))).rolling(30, min_periods=30).mean()
    # microprice trend and std
    if "microprice_proxy_s1" in out.columns:
        mp = out["microprice_proxy_s1"]
        # slope via simple diff over window endpoints
        out["microprice_trend_30"] = safe_div(mp - mp.shift(30), 30)
        out["microprice_std_30"] = mp.rolling(30, min_periods=30).std(ddof=0)

    # Reachability TP/SL based on sigma and ATR
    sigma_1m_240 = ret1.rolling(240, min_periods=240).std(ddof=0)
    expected_sigma_120 = np.sqrt(120.0) * sigma_1m_240
    out["sigma_1m_240"] = sigma_1m_240
    out["expected_sigma_120"] = expected_sigma_120
    for tp in [0.6, 0.8, 1.0, 1.2, 1.4]:
        out[f"tp_sigma_ratio_{str(tp).replace('.', 'p')}"] = safe_div(tp / 100.0, expected_sigma_120 + 1e-12)
    for sl in [0.3, 0.4, 0.5, 0.6, 0.7]:
        out[f"sl_sigma_ratio_{str(sl).replace('.', 'p')}"] = safe_div(sl / 100.0, expected_sigma_120 + 1e-12)
    for tp in [0.6, 0.8, 1.0]:
        out[f"tp_atr_ratio_{str(tp).replace('.', 'p')}"] = safe_div(tp / 100.0, out["atr_14"] / (close + 1e-9) + 1e-12)
    for sl in [0.3]:
        out[f"sl_atr_ratio_{str(sl).replace('.', 'p')}"] = safe_div(sl / 100.0, out["atr_14"] / (close + 1e-9) + 1e-12)

    return out


def add_short_lags(df_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_feat.copy()
    for col in cfg.KEY_LAG_FEATURES:
        if col not in out.columns:
            continue
        for k in cfg.SHORT_LAGS:
            out[f"lag{k}_{col}"] = make_lag(out, col, k)
    return out


def add_time_binning(df_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_feat.copy()
    for col in cfg.BIN_FEATURES:
        if col not in out.columns:
            continue
        for bin_name, (a, b) in cfg.BIN_BUCKETS.items():
            window_vals = [out[col].shift(k) for k in range(a, b + 1)]
            stack = pd.concat(window_vals, axis=1)
            out[f"{bin_name}_mean_{col}"] = stack.mean(axis=1)
            out[f"{bin_name}_std_{col}"] = stack.std(axis=1, ddof=0)
    return out


def finalize(df_feat: pd.DataFrame) -> pd.DataFrame:
    # Remove warm-up rows: ensure full windows for longest MAs and 120m blocks
    longest_ma = max(cfg.MA_WINDOWS.keys()) if len(cfg.MA_WINDOWS) else 30
    max_warmup = max(60, 120, longest_ma, max(cfg.SHORT_LAGS))
    df_feat = df_feat.iloc[int(max_warmup):].copy()

    # Replace inf/nan with neutral
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.fillna(0.0)
    return df_feat

