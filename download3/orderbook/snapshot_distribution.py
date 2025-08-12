from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS


def setup_logging() -> logging.Logger:
    PATHS["ob_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_logs"] / "snapshot_distribution.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def list_raw_days(symbol: str) -> List[str]:
    days = []
    for p in sorted(PATHS["ob_raw_csv"].glob(f"{symbol}-bookDepth-*.csv")):
        days.append(p.stem.split("-bookDepth-")[-1])
    return days


def load_day(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    fp = PATHS["ob_raw_csv"] / f"{symbol}-bookDepth-{date_str}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None


def day_distribution(df: pd.DataFrame, day_date: datetime) -> Dict[str, int]:
    # Count unique timestamps per minute
    if df is None or df.empty:
        return {"0": 1440, "1": 0, "2": 0, "3": 0, "4+": 0}
    df = df.copy()
    df["minute"] = df["timestamp"].dt.floor("min")
    counts = df.groupby("minute")["timestamp"].nunique()
    # Build full-day index
    start = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
    full_index = pd.date_range(start=start, periods=1440, freq="min", tz="UTC")
    counts = counts.reindex(full_index, fill_value=0)

    bins = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
    for v in counts.values:
        if v == 0:
            bins["0"] += 1
        elif v == 1:
            bins["1"] += 1
        elif v == 2:
            bins["2"] += 1
        elif v == 3:
            bins["3"] += 1
        else:
            bins["4+"] += 1
    return bins


def add_bins(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    for k in a.keys():
        a[k] += b.get(k, 0)
    return a


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Compute snapshot-per-minute distribution from raw CSVs")
    parser.add_argument("--symbol", help="Symbol to process (default: all in config)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else PAIRS
    PATHS["ob_metadata"].mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        days = list_raw_days(symbol)
        if args.start or args.end:
            s = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
            e = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
            def in_range(d: str) -> bool:
                dd = datetime.strptime(d, "%Y-%m-%d")
                if s and dd < s:
                    return False
                if e and dd > e:
                    return False
                return True
            days = [d for d in days if in_range(d)]

        overall = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
        per_day: Dict[str, Dict[str, int]] = {}

        for ds in days:
            day_dt = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            df = load_day(symbol, ds)
            dist = day_distribution(df, day_dt)
            per_day[ds] = dist
            overall = add_bins(overall, dist)

        report = {
            "symbol": symbol,
            "days": len(days),
            "overall": overall,
            "per_day": per_day,
        }
        out = PATHS["ob_metadata"] / f"snapshot_distribution_{symbol}.json"
        out.write_text(json.dumps(report, indent=2))
        logger.info(f"Report written: {out}")


if __name__ == "__main__":
    main()