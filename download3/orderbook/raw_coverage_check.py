from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS


TARGET_LEVELS: List[int] = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def setup_logging() -> logging.Logger:
    PATHS["ob_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_logs"] / "raw_coverage_check.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def list_raw_days(symbol: str) -> List[Path]:
    return sorted(PATHS["ob_raw_csv"].glob(f"{symbol}-bookDepth-*.csv"))


def analyze_day(fp: Path) -> Dict[str, Any]:
    # Fast CSV read: only required columns; try pyarrow engine if available
    try:
        df = pd.read_csv(
            fp,
            usecols=["timestamp", "percentage", "depth", "notional"],
            engine="pyarrow",
        )
    except Exception:
        df = pd.read_csv(fp, usecols=["timestamp", "percentage", "depth", "notional"]) 
    day_str = fp.stem.split("-bookDepth-")[-1]
    if df.empty:
        return {
            "day": day_str,
            "snapshots_total": 0,
            "snapshots_with_all_levels": 0,
            "missing_snapshots": 0,
            "missing_examples": [],
            "nan_in_levels_snapshots": 0,
            "minutes_total": 0,
            "minutes_both_complete": 0,
        }

    # Normalize dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    # Round to nearest integer levels (handles -1.0 etc.)
    df["percentage"] = df["percentage"].round().astype("Int64")
    # Drop rows with invalid percentage
    df = df[df["percentage"].isin(TARGET_LEVELS)].copy()
    # Deduplicate potential duplicates per (timestamp, percentage)
    df = df.sort_values(["timestamp", "percentage"]).drop_duplicates(subset=["timestamp", "percentage"], keep="last")

    snapshots_total = int(df["timestamp"].nunique())
    missing_snapshots = 0
    nan_in_levels_snapshots = 0
    missing_examples: List[Dict[str, Any]] = []

    # Group by exact snapshot
    for ts, g in df.groupby("timestamp"):
        present_levels = set(g["percentage"].astype(int).tolist())
        missing_levels = [lvl for lvl in TARGET_LEVELS if lvl not in present_levels]
        if missing_levels:
            missing_snapshots += 1
            if len(missing_examples) < 5:
                missing_examples.append({
                    "timestamp": ts.isoformat(),
                    "missing_levels": missing_levels,
                    "present_levels": sorted(present_levels),
                })
            continue
        # All levels present; check NaN inside level rows
        if g[["depth", "notional"]].isnull().any(axis=None):
            nan_in_levels_snapshots += 1

    snapshots_with_all_levels = snapshots_total - missing_snapshots

    # Minute-level check: are both first and last snapshot complete?
    snaps_idx = df["timestamp"].sort_values()
    minutes = snaps_idx.dt.floor("min")
    first_ts = snaps_idx.groupby(minutes).min()
    last_ts = snaps_idx.groupby(minutes).max()

    # Build a set of complete snapshot timestamps for quick lookup
    complete_ts = set(
        t for t, g in df.groupby("timestamp")
        if set(g["percentage"].astype(int)) == set(TARGET_LEVELS) and not g[["depth", "notional"]].isnull().any(axis=None)
    )
    both_complete = sum((t1 in complete_ts) and (t2 in complete_ts) for t1, t2 in zip(first_ts.values, last_ts.values))

    return {
        "day": day_str,
        "snapshots_total": snapshots_total,
        "snapshots_with_all_levels": snapshots_with_all_levels,
        "missing_snapshots": missing_snapshots,
        "missing_examples": missing_examples,
        "nan_in_levels_snapshots": nan_in_levels_snapshots,
        "minutes_total": int(minutes.nunique()),
        "minutes_both_complete": int(both_complete),
    }


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Check raw orderbook coverage of percentage levels per snapshot and per minute")
    parser.add_argument("--symbol", default=PAIRS[0], help="Symbol to analyze")
    args = parser.parse_args()

    files = list_raw_days(args.symbol)
    if not files:
        logger.error("No raw CSV files found")
        return

    summary: Dict[str, Any] = {
        "symbol": args.symbol,
        "target_levels": TARGET_LEVELS,
        "days": 0,
        "totals": {
            "snapshots_total": 0,
            "snapshots_with_all_levels": 0,
            "missing_snapshots": 0,
            "nan_in_levels_snapshots": 0,
            "minutes_total": 0,
            "minutes_both_complete": 0,
        },
        "per_day": {},
    }

    total = len(files)
    # Parallel processing per day
    workers = min(8, max(1, (os.cpu_count() or 2)))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_path = {ex.submit(analyze_day, fp): fp for fp in files}
        done = 0
        for fut in as_completed(future_to_path):
            done += 1
            fp = future_to_path[fut]
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"[{done}/{total}] Error {fp.name}: {e}")
                continue
            summary["per_day"][res["day"]] = res
            summary["days"] += 1
            summary["totals"]["snapshots_total"] += res["snapshots_total"]
            summary["totals"]["snapshots_with_all_levels"] += res["snapshots_with_all_levels"]
            summary["totals"]["missing_snapshots"] += res["missing_snapshots"]
            summary["totals"]["nan_in_levels_snapshots"] += res["nan_in_levels_snapshots"]
            summary["totals"]["minutes_total"] += res["minutes_total"]
            summary["totals"]["minutes_both_complete"] += res["minutes_both_complete"]
            logger.info(
                f"[{done}/{total}] Done {fp.name} | snaps={res['snapshots_total']:,}, missing_snaps={res['missing_snapshots']:,}, "
                f"both_complete={res['minutes_both_complete']:,}/{res['minutes_total']:,}"
            )

    out = PATHS["ob_metadata"] / f"raw_coverage_{args.symbol}.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"Raw coverage written: {out} | snapshots_total={summary['totals']['snapshots_total']:,}, "
        f"snapshots_with_all_levels={summary['totals']['snapshots_with_all_levels']:,}, missing_snapshots={summary['totals']['missing_snapshots']:,}, "
        f"minutes_both_complete={summary['totals']['minutes_both_complete']:,}"
    )


if __name__ == "__main__":
    main()

