from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS, INTERVAL


def setup_logging() -> logging.Logger:
    PATHS["ohlc_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ohlc_logs"] / "ohlc_continuity.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_symbol(symbol: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    fp = PATHS["ohlc_raw"] / f"{symbol}_{INTERVAL}.parquet"
    if not fp.exists():
        logger.error(f"Missing OHLC file: {fp}")
        return None
    df = pd.read_parquet(fp)
    if "timestamp" not in df.columns:
        logger.error("timestamp column not found in OHLC file")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def group_missing_ranges(missing: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(missing) == 0:
        return []
    missing = missing.sort_values()
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if (ts - prev).total_seconds() == 60:
            prev = ts
            continue
        # break in continuity: close previous range
        duration = int((prev - start).total_seconds() / 60) + 1
        ranges.append((start, prev, duration))
        start = ts
        prev = ts
    # last range
    duration = int((prev - start).total_seconds() / 60) + 1
    ranges.append((start, prev, duration))
    return ranges


def check_continuity(symbol: str, logger: logging.Logger) -> bool:
    df = load_symbol(symbol, logger)
    if df is None or df.empty:
        return False

    ts = df["timestamp"].dt.floor("T")
    start = ts.min().floor("T")
    end = ts.max().floor("T")

    minute_index = pd.date_range(start=start, end=end, freq="T", tz="UTC")
    observed = pd.DatetimeIndex(ts.unique()).sort_values()

    missing = minute_index.difference(observed)
    duplicates = df.duplicated(subset=["timestamp"]).sum()

    missing_ranges = group_missing_ranges(missing)

    total_expected = len(minute_index)
    total_present = len(observed)
    coverage = (total_present / total_expected * 100) if total_expected > 0 else 100.0

    logger.info(f"Symbol: {symbol}")
    logger.info(f"Range: {start} -> {end} ({total_expected} minutes)")
    logger.info(f"Present minutes: {total_present} ({coverage:.2f}% coverage)")
    logger.info(f"Duplicate timestamps (exact): {duplicates}")
    logger.info(f"Missing minutes: {len(missing)}; Gap segments: {len(missing_ranges)}")

    if missing_ranges:
        # Save detailed list
        PATHS["ohlc_metadata"].mkdir(parents=True, exist_ok=True)
        missing_csv = PATHS["ohlc_metadata"] / f"missing_minutes_{symbol}.csv"
        pd.DataFrame({"missing_timestamp": missing}).to_csv(missing_csv, index=False)
        logger.info(f"Missing timestamps saved to: {missing_csv}")

        # Show up to first 5 ranges
        for i, (s, e, n) in enumerate(missing_ranges[:5], 1):
            logger.info(f"  Gap {i}: {s} -> {e} ({n} min)")
        if len(missing_ranges) > 5:
            logger.info(f"  ... and {len(missing_ranges) - 5} more gap ranges")
    else:
        logger.info("No missing minutes detected.")

    return len(missing) == 0 and duplicates == 0


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Check continuity of OHLC minute data")
    parser.add_argument("--symbol", help="Symbol to check (default: all in config)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else PAIRS

    ok_count = 0
    for symbol in symbols:
        ok = check_continuity(symbol, logger)
        ok_count += 1 if ok else 0
    logger.info(f"Continuity OK: {ok_count}/{len(symbols)} symbols")


if __name__ == "__main__":
    main()