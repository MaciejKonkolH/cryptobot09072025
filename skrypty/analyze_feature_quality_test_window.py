from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training5 import config as cfg
from training5.data_loader import load_labeled, split_scale


KEY_FEATURES = [
    # outliers / TA
    "OBV_slope_over_ATR",
    "bb_pos_20",
    "close_vs_ema_120",
    "slope_ema_60_over_ATR",
    # channels
    "pos_in_channel_240",
    "pos_in_channel_180",
    "pos_in_channel_120",
    "width_over_ATR_240",
    "width_over_ATR_180",
    "width_over_ATR_120",
    "slope_over_ATR_window_240",
    "slope_over_ATR_window_180",
    "slope_over_ATR_window_120",
    # interactions
    "pos_in_channel_240_x_imbalance_1pct",
    "pos_in_channel_180_x_imbalance_1pct",
    "width_over_ATR_240_x_imbalance_1pct",
    "width_over_ATR_180_x_imbalance_1pct",
    "slope_over_ATR_window_240_x_imbalance_1pct",
    "slope_over_ATR_window_180_x_imbalance_1pct",
]


def summarize_series(s: pd.Series) -> dict:
    s_raw = s.copy()
    nan_count = int(s_raw.isna().sum())
    inf_count = int(np.isinf(s_raw.values).sum())
    non_na = s_raw.dropna()
    zero_share = float((non_na == 0).mean()) if len(non_na) else 0.0
    q = non_na.quantile([0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995]) if len(non_na) else pd.Series(dtype=float)
    out = {
        "count": int(non_na.size),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_share": round(zero_share, 6),
        "mean": float(non_na.mean()) if len(non_na) else np.nan,
        "std": float(non_na.std()) if len(non_na) else np.nan,
        "min": float(non_na.min()) if len(non_na) else np.nan,
        "p0_5": float(q.get(0.005, np.nan)),
        "p1": float(q.get(0.01, np.nan)),
        "p5": float(q.get(0.05, np.nan)),
        "p50": float(q.get(0.5, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "p99_5": float(q.get(0.995, np.nan)),
        "max": float(non_na.max()) if len(non_na) else np.nan,
    }
    return out


def analyze_symbol(symbol: str, out_dir: Path) -> Path:
    df_all = load_labeled(symbol)
    # Determine test window via split_scale
    X_train, X_val, X_test, *_ = split_scale(df_all)
    if len(X_test) == 0:
        raise RuntimeError(f"Empty test split for {symbol}")
    test_start = X_test.index.min()
    test_end = X_test.index.max()
    df_test = df_all.loc[(df_all.index >= test_start) & (df_all.index <= test_end)].copy()

    rows = []
    for feat in KEY_FEATURES:
        if feat in df_test.columns:
            stats = summarize_series(df_test[feat])
        else:
            stats = {"count": 0, "nan_count": np.nan, "inf_count": np.nan, "zero_share": np.nan,
                     "mean": np.nan, "std": np.nan, "min": np.nan, "p0_5": np.nan, "p1": np.nan,
                     "p5": np.nan, "p50": np.nan, "p95": np.nan, "p99": np.nan, "p99_5": np.nan, "max": np.nan}
        stats["feature"] = feat
        rows.append(stats)

    out_df = pd.DataFrame(rows)[[
        "feature", "count", "nan_count", "inf_count", "zero_share", "mean", "std", "min",
        "p0_5", "p1", "p5", "p50", "p95", "p99", "p99_5", "max"
    ]]

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fp = out_dir / f"feature_quality_{symbol}_{ts}.csv"
    out_df.to_csv(out_fp, index=False)
    return out_fp


def main():
    parser = argparse.ArgumentParser(description="Analyze feature quality on test window for given symbols")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,XRPUSDT", help="Comma-separated symbols")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "training5" / "output" / "reports" / "diagnostics"
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    paths = []
    for sym in syms:
        try:
            fp = analyze_symbol(sym, out_dir)
            print(f"Saved: {fp}")
            paths.append(fp)
        except Exception as e:
            print(f"Error analyzing {sym}: {e}")

    if paths:
        print("\nGenerated files:")
        for p in paths:
            print(str(p))


if __name__ == "__main__":
    main()

