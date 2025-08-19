import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TP_SL_RE = pd.Series.str.extract


def parse_tp_sl_from_filename(path: Path) -> Tuple[float, float]:
    name = path.name
    try:
        # name contains ...label_tp1p4_sl0p6_...
        tp_part = name.split("label_tp")[1].split("_sl")[0]
        sl_part = name.split("_sl")[1].split("_")[0]
        tp = float(tp_part.replace("p", "."))
        sl = float(sl_part.replace("p", "."))
        return tp, sl
    except Exception as exc:
        raise ValueError(f"Cannot parse TP/SL from filename: {path}") from exc


def compute_net_per_trade(df: pd.DataFrame, tp_pct: float, sl_pct: float, fee_bps: float) -> np.ndarray:
    fee_pct = fee_bps / 100.0
    is_win = df["result"].astype(str).str.upper().eq("WIN")
    gross = np.where(is_win, tp_pct, -sl_pct)
    net = gross - fee_pct
    return net


def load_predictions(paths: List[Path], fee_bps: float) -> Dict[str, pd.DataFrame]:
    """Load predictions_trades files keyed by label string (e.g., 'tp1p4_sl0p6').

    Each DataFrame contains added columns: 'tp_pct','sl_pct','net','margin'.
    margin = prob(predicted class) - prob_NEUTRAL.
    """
    label_to_df: Dict[str, pd.DataFrame] = {}
    for p in sorted(paths):
        df = pd.read_csv(p)
        tp_pct, sl_pct = parse_tp_sl_from_filename(p)
        # net per trade
        net = compute_net_per_trade(df, tp_pct=tp_pct, sl_pct=sl_pct, fee_bps=fee_bps)
        df["net"] = net
        # margin vs NEUTRAL
        prob_short = df.get("prob_SHORT")
        prob_long = df.get("prob_LONG")
        prob_neutral = df.get("prob_NEUTRAL")
        is_short = df["signal"].astype(str).str.lower().eq("short")
        pred_prob = np.where(is_short, prob_short, prob_long)
        df["margin_vs_neutral"] = pred_prob - prob_neutral
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        label_key = f"tp{str(tp_pct).replace('.', 'p')}_sl{str(sl_pct).replace('.', 'p')}"
        label_to_df[label_key] = df
    return label_to_df


def summarize_by_month(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["month"] = g["timestamp"].dt.to_period("M").dt.to_timestamp()
    agg = (
        g.groupby("month")
        .agg(trades=("net", "size"), precision=("correct", "mean"), mean_net=("net", "mean"))
        .reset_index()
        .sort_values("month")
    )
    return agg


def summarize_by_week(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    # ISO week start Monday
    g["week"] = g["timestamp"].dt.to_period("W-MON").dt.start_time
    agg = (
        g.groupby("week")
        .agg(trades=("net", "size"), precision=("correct", "mean"), mean_net=("net", "mean"))
        .reset_index()
        .sort_values("week")
    )
    return agg


def hourly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["hour"] = g["timestamp"].dt.hour
    agg = g.groupby("hour").agg(trades=("net", "size"), precision=("correct", "mean"), mean_net=("net", "mean")).reset_index()
    return agg.sort_values("hour")


def long_short_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    agg = g.groupby("signal").agg(trades=("net", "size"), precision=("correct", "mean"), mean_net=("net", "mean")).reset_index()
    return agg.sort_values("signal")


def threshold_curve_confidence(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        sel = df[df["confidence"] >= t]
        if len(sel) == 0:
            rows.append({"threshold": t, "trades": 0, "precision": np.nan, "mean_net": np.nan})
        else:
            rows.append({
                "threshold": t,
                "trades": int(len(sel)),
                "precision": float(sel["correct"].mean()),
                "mean_net": float(sel["net"].mean()),
            })
    return pd.DataFrame(rows)


def threshold_curve_margin(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        sel = df[df["margin_vs_neutral"] >= t]
        if len(sel) == 0:
            rows.append({"threshold": t, "trades": 0, "precision": np.nan, "mean_net": np.nan})
        else:
            rows.append({
                "threshold": t,
                "trades": int(len(sel)),
                "precision": float(sel["correct"].mean()),
                "mean_net": float(sel["net"].mean()),
            })
    return pd.DataFrame(rows)


def fee_sensitivity(df: pd.DataFrame, tp_pct: float, sl_pct: float, fee_bps_list: List[float]) -> pd.DataFrame:
    rows = []
    for fee in fee_bps_list:
        net = compute_net_per_trade(df, tp_pct=tp_pct, sl_pct=sl_pct, fee_bps=fee)
        rows.append({"fee_bps": fee, "trades": int(len(df)), "precision": float(df["correct"].mean()), "mean_net": float(np.mean(net))})
    return pd.DataFrame(rows)


def rolling_window_time(df: pd.DataFrame, window_days: int = 14, step_days: int = 7) -> pd.DataFrame:
    g = df.sort_values("timestamp").reset_index(drop=True)
    start = g["timestamp"].min().normalize()
    end = g["timestamp"].max().normalize()
    rows = []
    current = start
    while current <= end:
        win_end = current + pd.Timedelta(days=window_days)
        sel = g[(g["timestamp"] >= current) & (g["timestamp"] < win_end)]
        rows.append({
            "window_start": current,
            "window_end": win_end,
            "trades": int(len(sel)),
            "precision": float(sel["correct"].mean()) if len(sel) else np.nan,
            "mean_net": float(sel["net"].mean()) if len(sel) else np.nan,
        })
        current = current + pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)


def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("timestamp").reset_index(drop=True)
    g["cum_net"] = g["net"].cumsum()
    return g[["timestamp", "net", "cum_net"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality/stability reports from predictions_trades files (no retrain)")
    parser.add_argument("--symbol", required=True, help="Symbol like ETHUSDT")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Round-trip fee in bps (default 0)")
    parser.add_argument("--out-subdir", type=str, default="analysis/quality", help="Subfolder under run dir for outputs")
    parser.add_argument("--conf-thr", type=str, default="0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70", help="Comma-separated confidence thresholds")
    parser.add_argument("--margin-thr", type=str, default="0.00,0.02,0.04,0.06,0.08,0.10,0.12", help="Comma-separated margin thresholds (p_top1 - p_NEUTRAL)")
    parser.add_argument("--fee-grid", type=str, default="0,5,10,15,20", help="Comma-separated fee bps for sensitivity")
    parser.add_argument("--win-days", type=int, default=14, help="Rolling window size in days")
    parser.add_argument("--step-days", type=int, default=7, help="Rolling step in days")
    args = parser.parse_args()

    reports_root = Path("training5") / "output" / "reports" / args.symbol
    run_dirs = [p for p in reports_root.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories under {reports_root}")
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    in_files = sorted(latest_run.glob(f"predictions_trades_{args.symbol}_label_*.csv"))
    # filter out any analyzer outputs if pattern overlaps
    in_files = [p for p in in_files if "__" not in p.name]
    if not in_files:
        raise FileNotFoundError(f"No predictions_trades files in {latest_run}")

    out_dir = latest_run / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_to_df = load_predictions(in_files, fee_bps=args.fee_bps)
    conf_thr = [float(x) for x in args.conf_thr.split(",") if x.strip()]
    margin_thr = [float(x) for x in args.margin_thr.split(",") if x.strip()]
    fee_grid = [float(x) for x in args.fee_grid.split(",") if x.strip()]

    index_rows = []
    for label, df in label_to_df.items():
        # Create per-level subdirectory to keep files organized
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        tp_pct, sl_pct = parse_tp_sl_from_filename(Path(f"predictions_trades_{args.symbol}_label_{label}_X.csv"))
        # Monthly/weekly stability
        monthly = summarize_by_month(df)
        weekly = summarize_by_week(df)
        monthly.to_csv(label_dir / f"stability_monthly_{args.symbol}_{label}.csv", index=False)
        weekly.to_csv(label_dir / f"stability_weekly_{args.symbol}_{label}.csv", index=False)

        # Threshold curves
        conf_curve = threshold_curve_confidence(df, thresholds=conf_thr)
        conf_curve.to_csv(label_dir / f"curve_confidence_{args.symbol}_{label}.csv", index=False)

        margin_curve = threshold_curve_margin(df, thresholds=margin_thr)
        margin_curve.to_csv(label_dir / f"curve_margin_{args.symbol}_{label}.csv", index=False)

        # Fee sensitivity
        fee_sens = fee_sensitivity(df, tp_pct=tp_pct, sl_pct=sl_pct, fee_bps_list=fee_grid)
        fee_sens.to_csv(label_dir / f"fee_sensitivity_{args.symbol}_{label}.csv", index=False)

        # Direction/hour breakdown
        ls = long_short_breakdown(df)
        ls.to_csv(label_dir / f"breakdown_long_short_{args.symbol}_{label}.csv", index=False)

        hod = hourly_breakdown(df)
        hod.to_csv(label_dir / f"breakdown_hour_of_day_{args.symbol}_{label}.csv", index=False)

        # Rolling window and equity curve
        rolling = rolling_window_time(df, window_days=args.win_days, step_days=args.step_days)
        rolling.to_csv(label_dir / f"rolling_window_{args.symbol}_{label}.csv", index=False)

        eq = equity_curve(df)
        eq.to_csv(label_dir / f"equity_curve_{args.symbol}_{label}.csv", index=False)

        # Index row
        index_rows.append({
            "label": label,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "trades": int(len(df)),
            "precision": float(df["correct"].mean()),
            "mean_net": float(df["net"].mean()),
        })

    pd.DataFrame(index_rows).sort_values("label").to_csv(out_dir / f"index_{args.symbol}.csv", index=False)
    print(f"Saved quality reports to: {out_dir}")


if __name__ == "__main__":
    main()

