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
    log_file = PATHS["ob_logs"] / "validate_normalized.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def list_normalized_days(symbol: str) -> List[str]:
    days = []
    for p in sorted(PATHS["ob_normalized_csv"].glob(f"{symbol}-bookDepth-*.csv")):
        days.append(p.stem.split("-bookDepth-")[-1])
    return days


def load_normalized_day(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    fp = PATHS["ob_normalized_csv"] / f"{symbol}-bookDepth-{date_str}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None


def validate_day(df: pd.DataFrame, day_date: datetime) -> Dict:
    result = {
        "minutes_total": 1440,
        "minutes_with_0": 0,
        "minutes_with_1": 0,
        "minutes_with_2": 0,
        "minutes_with_3": 0,
        "minutes_with_4_plus": 0,
        "nan_rows": 0,
        "negative_depth_rows": 0,
        "negative_notional_rows": 0,
        "example_minutes_not_2": [],
    }
    if df is None or df.empty:
        result["minutes_with_0"] = 1440
        return result

    # Basic sanity
    nan_rows = int(df[["timestamp", "percentage", "depth", "notional"]].isnull().any(axis=1).sum())
    result["nan_rows"] = nan_rows
    result["negative_depth_rows"] = int((df["depth"] < 0).sum())
    result["negative_notional_rows"] = int((df["notional"] < 0).sum())

    df = df.copy()
    df["minute"] = df["timestamp"].dt.floor("min")
    counts = df.groupby("minute")["timestamp"].nunique()

    start = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
    full_index = pd.date_range(start=start, periods=1440, freq="min", tz="UTC")
    counts = counts.reindex(full_index, fill_value=0)

    # Distribution
    result["minutes_with_0"] = int((counts == 0).sum())
    result["minutes_with_1"] = int((counts == 1).sum())
    result["minutes_with_2"] = int((counts == 2).sum())
    result["minutes_with_3"] = int((counts == 3).sum())
    result["minutes_with_4_plus"] = int((counts >= 4).sum())

    # Examples of minutes not equal to 2
    not2 = counts[counts != 2].index[:10]
    result["example_minutes_not_2"] = [ts.isoformat() for ts in not2]

    return result


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Validate normalized per-day orderbook CSVs (2 snapshots per minute)")
    parser.add_argument("--symbol", help="Symbol to process (default: all in config)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else PAIRS
    PATHS["ob_metadata"].mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        days = list_normalized_days(symbol)
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

        summary = {
            "symbol": symbol,
            "days": len(days),
            "totals": {
                "minutes_total": 0,
                "minutes_with_0": 0,
                "minutes_with_1": 0,
                "minutes_with_2": 0,
                "minutes_with_3": 0,
                "minutes_with_4_plus": 0,
                "nan_rows": 0,
                "negative_depth_rows": 0,
                "negative_notional_rows": 0,
            },
            "per_day": {},
        }

        for ds in days:
            df = load_normalized_day(symbol, ds)
            day_dt = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            res = validate_day(df, day_dt)
            summary["per_day"][ds] = res
            for k in summary["totals"].keys():
                summary["totals"][k] += res.get(k, 0)

        out = PATHS["ob_metadata"] / f"validate_normalized_{symbol}.json"
        out.write_text(json.dumps(summary, indent=2))
        logger.info(f"Validation report written: {out}")


if __name__ == "__main__":
    main()