from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS


def setup_logging() -> logging.Logger:
    PATHS["ob_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_logs"] / "normalize_daily_csv.log"
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


def vectorized_two_snapshots(day_df: pd.DataFrame, day_date: datetime) -> pd.DataFrame:
    # Build minute index for the day [00:00, 24:00)
    start = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    minute_index = pd.date_range(start=start, end=end - timedelta(minutes=1), freq="min", tz="UTC")

    # Prepare per-minute first/last timestamps for minutes that have data
    df = day_df.copy()
    df["minute"] = df["timestamp"].dt.floor("min")
    grp = df.groupby("minute")["timestamp"]
    ts_first = grp.first()
    ts_last = grp.last()

    present_minutes = ts_first.index

    # Minutes with >= 2 snapshots: select first and last directly
    ge2_minutes = present_minutes[(ts_first != ts_last)]
    ts1_df = ts_first[ge2_minutes].reset_index().rename(columns={"timestamp": "timestamp", "minute": "minute"})
    ts1_df["snap"] = 1
    ts2_df = ts_last[ge2_minutes].reset_index().rename(columns={"timestamp": "timestamp", "minute": "minute"})
    ts2_df["snap"] = 2
    ts_map_ge2 = pd.concat([ts1_df, ts2_df], ignore_index=True)
    sel_ge2 = df.merge(ts_map_ge2[["timestamp", "snap"]], on="timestamp", how="inner")
    sel_ge2["minute"] = sel_ge2["timestamp"].dt.floor("min")

    # Minutes with exactly 1 snapshot: duplicate rows to second timestamp within minute (minute+45s)
    eq1_minutes = present_minutes[(ts_first == ts_last)]
    dup_rows = pd.DataFrame()
    if len(eq1_minutes) > 0:
        base1 = df[df["minute"].isin(eq1_minutes)].copy()
        base1["snap"] = 1
        dup2 = base1.copy()
        # Default duplicate at :45s, but if the single snapshot already falls at :45s,
        # place the synthetic one at :15s to ensure two UNIQUE timestamps per minute.
        dup2["timestamp"] = dup2["minute"] + pd.Timedelta(seconds=45)
        mask_45 = base1["timestamp"].dt.second == 45
        if mask_45.any():
            dup2.loc[mask_45, "timestamp"] = base1.loc[mask_45, "minute"] + pd.Timedelta(seconds=15)
        dup2["snap"] = 2
        dup_rows = pd.concat([base1, dup2], ignore_index=True)

    # Minutes with no data: copy FULL level sets from nearest snapshot timestamps (prev/next) and
    # place them at :15s (snap=1) and :45s (snap=2). If only one side exists, duplicate it to make two.
    missing_minutes = pd.Index(minute_index).difference(present_minutes)
    synth_rows = pd.DataFrame()
    if len(missing_minutes) > 0:
        # Base levels to replicate for each source snapshot timestamp
        base_levels = df[["timestamp", "percentage", "depth", "notional"]]
        # Unique snapshot timestamps available in the day
        unique_ts = pd.DataFrame({"src_ts": base_levels["timestamp"].drop_duplicates().sort_values()})

        # Build mapping minute -> nearest prev snapshot timestamp
        prev_keys = pd.DataFrame({"minute": missing_minutes}).sort_values("minute")
        prev_map = pd.merge_asof(
            prev_keys,
            unique_ts,
            left_on="minute",
            right_on="src_ts",
            direction="backward",
        )
        prev_map = prev_map[pd.notna(prev_map["src_ts"])].copy()

        # Build mapping minute -> nearest next snapshot timestamp
        next_keys = prev_keys.copy()
        next_map = pd.merge_asof(
            next_keys,
            unique_ts,
            left_on="minute",
            right_on="src_ts",
            direction="forward",
        )
        next_map = next_map[pd.notna(next_map["src_ts"])].copy()

        frames = []
        # Join to copy FULL level sets for prev side
        if not prev_map.empty:
            prev_join = prev_map.merge(base_levels, left_on="src_ts", right_on="timestamp", how="left")
            prev_join["timestamp"] = prev_join["minute"] + pd.Timedelta(seconds=15)
            prev_join["snap"] = 1
            frames.append(prev_join[["timestamp", "percentage", "depth", "notional", "snap", "minute"]])
        # Join to copy FULL level sets for next side
        if not next_map.empty:
            next_join = next_map.merge(base_levels, left_on="src_ts", right_on="timestamp", how="left")
            next_join["timestamp"] = next_join["minute"] + pd.Timedelta(seconds=45)
            next_join["snap"] = 2
            frames.append(next_join[["timestamp", "percentage", "depth", "notional", "snap", "minute"]])

        # Determine minutes covered by only prev or only next and duplicate the full set to fill the pair
        minutes_prev = set(prev_map["minute"]) if not prev_map.empty else set()
        minutes_next = set(next_map["minute"]) if not next_map.empty else set()
        only_prev = minutes_prev.difference(minutes_next)
        only_next = minutes_next.difference(minutes_prev)

        if only_prev:
            dup_prev = prev_map[prev_map["minute"].isin(list(only_prev))]
            dup_prev_join = dup_prev.merge(base_levels, left_on="src_ts", right_on="timestamp", how="left")
            dup_prev_join["timestamp"] = dup_prev_join["minute"] + pd.Timedelta(seconds=45)
            dup_prev_join["snap"] = 2
            frames.append(dup_prev_join[["timestamp", "percentage", "depth", "notional", "snap", "minute"]])
        if only_next:
            dup_next = next_map[next_map["minute"].isin(list(only_next))]
            dup_next_join = dup_next.merge(base_levels, left_on="src_ts", right_on="timestamp", how="left")
            dup_next_join["timestamp"] = dup_next_join["minute"] + pd.Timedelta(seconds=15)
            dup_next_join["snap"] = 1
            frames.append(dup_next_join[["timestamp", "percentage", "depth", "notional", "snap", "minute"]])

        synth_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Combine all
    all_rows = []
    if not sel_ge2.empty:
        all_rows.append(sel_ge2[["timestamp", "percentage", "depth", "notional", "snap", "minute"]])
    if not dup_rows.empty:
        all_rows.append(dup_rows[["timestamp", "percentage", "depth", "notional", "snap", "minute"]])
    if not synth_rows.empty:
        all_rows.append(synth_rows)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "percentage", "depth", "notional"])  # empty

    sel = pd.concat(all_rows, ignore_index=True)

    sel = sel.sort_values(["minute", "snap", "percentage"]).reset_index(drop=True)
    out = sel[["timestamp", "percentage", "depth", "notional"]].copy()
    # Format timestamp to ISO8601 with UTC
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace("\+0000", "+00:00", regex=False)
    return out


def normalize_one(symbol: str, ds: str, force: bool = False) -> Tuple[str, bool]:
    out_fp = PATHS["ob_normalized_csv"] / f"{symbol}-bookDepth-{ds}.csv"
    if out_fp.exists() and not force:
        return str(out_fp), True
    day_df = load_day(symbol, ds)
    if day_df is None or day_df.empty:
        return str(out_fp), False
    day_date = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    norm_df = vectorized_two_snapshots(day_df, day_date)
    norm_df.to_csv(out_fp, index=False)
    return str(out_fp), True


def normalize_symbol(symbol: str, logger: logging.Logger, days: List[str], workers: int = 1, force: bool = False) -> None:
    PATHS["ob_normalized_csv"].mkdir(parents=True, exist_ok=True)
    logger.info(f"Normalizing {symbol}: {len(days)} days (workers={workers})")
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(normalize_one, symbol, ds, force): ds for ds in days}
            done = 0
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    out_fp, ok = fut.result()
                    done += 1
                    if ok and (force or Path(out_fp).exists()):
                        logger.info(f"[{done}/{len(days)}] Saved normalized day: {out_fp}")
                    else:
                        logger.warning(f"[{done}/{len(days)}] Skipped/empty day: {ds}")
                except Exception as e:
                    done += 1
                    logger.error(f"[{done}/{len(days)}] Error normalizing {ds}: {e}")
    else:
        for i, ds in enumerate(days, 1):
            out_fp, ok = normalize_one(symbol, ds, force)
            if ok:
                logger.info(f"[{i}/{len(days)}] Saved normalized day: {out_fp}")
            else:
                logger.warning(f"[{i}/{len(days)}] Skipped/empty day: {ds}")


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Normalize raw orderbook CSVs to 2 snapshots per minute per day")
    parser.add_argument("--symbol", help="Symbol to process (default: all in config)")
    parser.add_argument("--days", help="Comma-separated YYYY-MM-DD list to limit processing")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (per-day)")
    parser.add_argument("--force", action="store_true", help="Force re-normalization and overwrite existing outputs")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else PAIRS

    for symbol in symbols:
        all_days = list_raw_days(symbol)
        days = all_days
        if args.days:
            sel = [d.strip() for d in args.days.split(",") if d.strip()]
            days = [d for d in all_days if d in sel]
        elif args.start or args.end:
            s = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
            e = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
            def in_range(d: str) -> bool:
                dd = datetime.strptime(d, "%Y-%m-%d")
                if s and dd < s:
                    return False
                if e and dd > e:
                    return False
                return True
            days = [d for d in all_days if in_range(d)]

        normalize_symbol(symbol, logger, days, workers=max(1, args.workers), force=bool(args.force))


if __name__ == "__main__":
    main()