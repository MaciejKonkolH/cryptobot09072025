import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


TP_SL_RE = re.compile(r"label_tp(?P<tp>\d+p\d+)_sl(?P<sl>\d+p\d+)")


def parse_tp_sl_from_filename(path: Path) -> Tuple[float, float]:
    """Extract TP/SL percentages from filename like ...label_tp1p4_sl0p6_....csv -> (1.4, 0.6)."""
    m = TP_SL_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse tp/sl from filename: {path}")

    def to_float(text: str) -> float:
        # '1p4' -> '1.4'
        return float(text.replace('p', '.'))

    tp = to_float(m.group('tp'))
    sl = to_float(m.group('sl'))
    return tp, sl


def analyze_file(path: Path, fee_bps: float, out_dir: Optional[Path] = None) -> dict:
    df = pd.read_csv(path)
    # Basic sanity
    required_cols = {"timestamp", "pair", "signal", "confidence", "prob_SHORT", "prob_LONG", "prob_NEUTRAL", "true_label", "correct", "result"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")

    # Parse TP/SL from filename (percentage points)
    tp_pct, sl_pct = parse_tp_sl_from_filename(path)
    fee_pct = fee_bps / 100.0  # bps -> percent

    # Compute gross/net returns per trade (in percentage points)
    # WIN => +TP, LOSS => -SL; subtract round-trip fee from both
    is_win = df["result"].astype(str).str.upper().eq("WIN")
    gross = np.where(is_win, tp_pct, -sl_pct)
    net = gross - fee_pct

    # Prepare outputs
    total = len(df)
    prec_overall = df["correct"].mean() if total > 0 else np.nan
    prec_by_signal = df.groupby("signal")["correct"].mean().to_dict()
    count_by_signal = df["signal"].value_counts().to_dict()

    mean_gross = float(np.mean(gross)) if total else np.nan
    mean_net = float(np.mean(net)) if total else np.nan

    # Confidence deciles: precision and mean net by decile
    decile_stats = []
    if total:
        try:
            quantiles = np.quantile(df["confidence"], q=np.linspace(0, 1, 11))
        except Exception:
            quantiles = None
        if quantiles is not None:
            bins = quantiles
            # ensure strictly increasing
            bins = np.unique(bins)
            # if duplicates collapse bins, fall back to 5 bins
            if len(bins) < 3:
                bins = np.quantile(df["confidence"], q=np.linspace(0, 1, 6))
                bins = np.unique(bins)
            df_dec = df.copy()
            df_dec["_net"] = net
            df_dec["conf_bin"] = pd.cut(df_dec["confidence"], bins=bins, include_lowest=True, duplicates="drop")
            grouped = df_dec.groupby("conf_bin")
            for bin_key, g in grouped:
                if len(g) == 0:
                    continue
                decile_stats.append({
                    "conf_bin": str(bin_key),
                    "trades": int(len(g)),
                    "precision": float(g["correct"].mean()),
                    "mean_net": float(g["_net"].mean()),
                })

    # Hour-of-day distribution (UTC)
    hod_counts = {}
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        hod_counts = ts.dt.hour.value_counts().sort_index().to_dict()
    except Exception:
        hod_counts = {}

    # Cumulative net return series by time (sorted)
    cum_path = None
    try:
        df_ts = df.copy()
        df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], utc=True)
        df_ts = df_ts.sort_values("timestamp").reset_index(drop=True)
        df_ts["net"] = net
        df_ts["cum_net"] = df_ts["net"].cumsum()
        target_dir = out_dir if out_dir is not None else path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        cum_path = target_dir / (path.stem + "__cum_series.csv")
        df_ts[["timestamp", "net", "cum_net"]].to_csv(cum_path, index=False)
    except Exception:
        cum_path = None

    return {
        "file": str(path),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "fee_bps": fee_bps,
        "total_trades": total,
        "precision_overall": float(prec_overall) if total else np.nan,
        "precision_by_signal": {k: float(v) for k, v in prec_by_signal.items()},
        "count_by_signal": {k: int(v) for k, v in count_by_signal.items()},
        "mean_gross_pct": mean_gross,
        "mean_net_pct": mean_net,
        "decile_stats": decile_stats,
        "hour_of_day_counts": {int(k): int(v) for k, v in hod_counts.items()},
        "cum_series_path": str(cum_path) if cum_path else None,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze trades-only prediction CSVs")
    parser.add_argument("files", nargs="*", help="Paths to predictions_trades_*.csv (optional if --symbol provided)")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol like ETHUSDT; analyzes all predictions_trades_* in the latest run folder")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Round-trip fee in bps (default 10 = 0.10%)")
    parser.add_argument("--save-summary", action="store_true", help="Save per-file summary CSVs next to inputs or in out-subdir if provided")
    parser.add_argument("--out-subdir", type=str, default="analysis", help="Subfolder under run dir to save outputs (default: analysis)")
    args = parser.parse_args(argv)

    # Resolve input files
    input_files: List[Path] = []
    if args.files:
        input_files = [Path(f) for f in args.files]
    elif args.symbol:
        # Find latest run directory for the symbol
        reports_root = Path("training5") / "output" / "reports" / args.symbol
        if not reports_root.exists():
            raise FileNotFoundError(f"Reports directory not found for symbol {args.symbol}: {reports_root}")
        run_dirs = [p for p in reports_root.glob("run_*") if p.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"No run_* directories under {reports_root}")
        latest_run: Optional[Path] = max(run_dirs, key=lambda p: p.stat().st_mtime)
        # Pick all predictions_trades files for this symbol in the latest run
        input_files = sorted(latest_run.glob(f"predictions_trades_{args.symbol}_label_*.csv"))
        # Exclude derived outputs created by this analyzer
        input_files = [p for p in input_files if "__cum_series" not in p.name and "__summary" not in p.name and "__deciles" not in p.name]
        if not input_files:
            raise FileNotFoundError(f"No predictions_trades files found in {latest_run}")
        print(f"Analyzing latest run for {args.symbol}: {latest_run}")
    else:
        parser.error("Provide either explicit files or --symbol to auto-select latest run files.")

    # Determine output directory within the run folder
    out_dir: Optional[Path] = None
    if input_files:
        run_dir = input_files[0].parent
        out_dir = run_dir / args.out_subdir if args.out_subdir else run_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in input_files:
        res = analyze_file(path, fee_bps=args.fee_bps, out_dir=out_dir)
        rows.append(res)

    # Print concise text summary
    for r in rows:
        print("\n== ", r["file"], "==")
        print(f"TP={r['tp_pct']}%, SL={r['sl_pct']}%, fee={r['fee_bps']} bps")
        print(f"trades={r['total_trades']}, precision={r['precision_overall']:.3f}, mean_gross%={r['mean_gross_pct']:.3f}, mean_net%={r['mean_net_pct']:.3f}")
        print("by_signal precision:", r["precision_by_signal"])
        print("by_signal counts:", r["count_by_signal"])
        if r["decile_stats"]:
            # show top and bottom confidence bins
            first = r["decile_stats"][0]
            last = r["decile_stats"][-1]
            print("low-conf bin:", {k: first[k] for k in ("conf_bin", "trades", "precision", "mean_net")})
            print("high-conf bin:", {k: last[k] for k in ("conf_bin", "trades", "precision", "mean_net")})
        if r["cum_series_path"]:
            print("cum_series:", r["cum_series_path"]) 

    # Optionally save per-file summaries
    if args.save_summary:
        for r in rows:
            src = Path(r["file"]) 
            target_dir = out_dir if out_dir is not None else src.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            out = target_dir / (src.stem + "__summary.csv")
            # flatten basics + deciles to CSV
            base = {
                "file": r["file"],
                "tp_pct": r["tp_pct"],
                "sl_pct": r["sl_pct"],
                "fee_bps": r["fee_bps"],
                "total_trades": r["total_trades"],
                "precision_overall": r["precision_overall"],
                "mean_gross_pct": r["mean_gross_pct"],
                "mean_net_pct": r["mean_net_pct"],
            }
            dec = pd.DataFrame(r["decile_stats"]) if r["decile_stats"] else pd.DataFrame()
            if not dec.empty:
                # write base row and deciles below
                base_df = pd.DataFrame([base])
                dec_cols = ["conf_bin", "trades", "precision", "mean_net"]
                dec = dec[dec_cols] if all(c in dec.columns for c in dec_cols) else dec
                tmp = target_dir / (out.stem + "__deciles.csv")
                base_df.to_csv(out, index=False)
                dec.to_csv(tmp, index=False)
                print("saved summary:", out)
                print("saved deciles:", tmp)
            else:
                pd.DataFrame([base]).to_csv(out, index=False)
                print("saved summary:", out)


if __name__ == "__main__":
    main()

