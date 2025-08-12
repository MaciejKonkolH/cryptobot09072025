from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PAIRS, PATHS, LOGGING


def setup_logging() -> logging.Logger:
    for key in ("ob_merge_logs", "ob_merge_metadata", "ob_merged_raw"):
        PATHS[key].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_merge_logs"] / "merge_orderbook_to_wide.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_all_csv(symbol: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    files = sorted(PATHS["ob_raw_csv"].glob(f"{symbol}-bookDepth-*.csv"))
    if not files:
        logger.error(f"No raw CSV files for {symbol}")
        return None
    dfs: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            # Expect columns: timestamp, percentage, depth, notional
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {fp.name}: {e}")
    if not dfs:
        return None
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    return data


def interpolate_snapshots(s1: pd.Series, s2: pd.Series, ratio: float) -> dict:
    out = {}
    for col in s1.index:
        if col in ("timestamp", "percentage"):
            continue
        if col in s2.index and pd.api.types.is_numeric_dtype(type(s1[col])):
            try:
                v1 = float(s1[col])
                v2 = float(s2[col])
                out[col] = v1 + (v2 - v1) * ratio
            except Exception:
                pass
    return out


def create_two_snapshots_for_minute(minute_start: pd.Timestamp, minute_df: pd.DataFrame, all_sorted: pd.DataFrame) -> Optional[pd.DataFrame]:
    minute_end = minute_start + timedelta(minutes=1)
    cnt = len(minute_df)
    if cnt >= 2:
        selected = minute_df.sort_values("timestamp").iloc[[0, -1]].copy()
        return selected
    elif cnt == 1:
        single = minute_df.iloc[0]
        # find nearest after minute
        after = all_sorted[all_sorted["timestamp"] >= minute_end]
        if len(after) > 0:
            after_row = after.iloc[0]
            interp = interpolate_snapshots(single, after_row, 0.5)
            interp_ts = min(max(single["timestamp"] + (after_row["timestamp"] - single["timestamp"]) / 2, minute_start), minute_end)
            interp_row = {**interp, "timestamp": interp_ts}
            return pd.DataFrame([single.to_dict(), interp_row])
        else:
            return pd.DataFrame([single.to_dict(), single.to_dict()])
    else:
        # 0 snapshots: use nearest before and after
        before = all_sorted[all_sorted["timestamp"] < minute_start]
        after = all_sorted[all_sorted["timestamp"] >= minute_end]
        if len(before) > 0 and len(after) > 0:
            b = before.iloc[-1]
            a = after.iloc[0]
            s1 = interpolate_snapshots(b, a, 0.25)
            s2 = interpolate_snapshots(b, a, 0.75)
            s1["timestamp"] = minute_start
            s2["timestamp"] = minute_end - timedelta(milliseconds=1)
            return pd.DataFrame([s1, s2])
        else:
            return None


def transform_long_to_wide(two_snapshots_df: pd.DataFrame, minute_ts: pd.Timestamp) -> dict:
    # Expect columns: timestamp, percentage, depth, notional
    # pivot depth/notional by percentage for snapshot1 and snapshot2
    two_snapshots_df = two_snapshots_df.sort_values("timestamp")
    row = {
        "timestamp": minute_ts,
        "snapshot1_timestamp": two_snapshots_df.iloc[0]["timestamp"],
        "snapshot2_timestamp": two_snapshots_df.iloc[1]["timestamp"],
    }
    for idx, prefix in enumerate(["snapshot1_", "snapshot2_"], start=0):
        snap = two_snapshots_df.iloc[idx]
        # group by percentage if duplicates
        # here, assume one row per percentage; if multiple, take first
        # we will pivot depth/notional columns
        for col in two_snapshots_df.columns:
            if col in ("timestamp", "percentage"):
                continue
        # Build series for this snapshot
        snap_df = two_snapshots_df[two_snapshots_df["timestamp"] == snap["timestamp"]]
        depth_pivot = snap_df.pivot_table(index=None, columns="percentage", values="depth", aggfunc="first")
        notional_pivot = snap_df.pivot_table(index=None, columns="percentage", values="notional", aggfunc="first")
        if isinstance(depth_pivot, pd.Series):
            for pct, val in depth_pivot.items():
                row[f"{prefix}depth_{pct}"] = val
        if isinstance(notional_pivot, pd.Series):
            for pct, val in notional_pivot.items():
                row[f"{prefix}notional_{pct}"] = val
    return row


def merge_symbol(symbol: str, logger: logging.Logger) -> Optional[Path]:
    data = load_all_csv(symbol, logger)
    if data is None or data.empty:
        return None

    data = data.sort_values("timestamp").reset_index(drop=True)

    # Build minute bins using data range
    start_ts = data["timestamp"].min().floor("T")
    end_ts = data["timestamp"].max().ceil("T")
    minute_index = pd.date_range(start=start_ts, end=end_ts, freq="T", tz="UTC")

    out_rows: List[dict] = []
    grouped = data.groupby(pd.Grouper(key="timestamp", freq="T"))

    for minute_start in minute_index:
        minute_df = grouped.get_group(minute_start) if minute_start in grouped.groups else pd.DataFrame(columns=data.columns)
        two = create_two_snapshots_for_minute(minute_start, minute_df, data)
        if two is None or len(two) < 2:
            continue
        wide_row = transform_long_to_wide(two, minute_start)
        out_rows.append(wide_row)

    if not out_rows:
        logger.warning(f"No output rows for {symbol}")
        return None

    result_df = pd.DataFrame(out_rows)
    result_df = result_df.sort_values("timestamp").reset_index(drop=True)
    out_file = PATHS["ob_merged_raw"] / f"orderbook_wide_raw_{symbol}.feather"
    result_df.to_feather(out_file)
    logger.info(f"Saved wide raw for {symbol}: {len(result_df):,} minutes -> {out_file}")
    return out_file


def main() -> None:
    logger = setup_logging()
    ok = 0
    for symbol in PAIRS:
        ok += 1 if merge_symbol(symbol, logger) else 0
    logger.info(f"Done merge to wide. Success: {ok}/{len(PAIRS)}")


if __name__ == "__main__":
    main()