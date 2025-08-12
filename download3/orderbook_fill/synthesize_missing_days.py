from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS

# We will try to reuse available range probing from orderbook_downloader
try:
    import orderbook.orderbook_downloader as obd  # type: ignore
except Exception:
    obd = None


def setup_logging() -> logging.Logger:
    PATHS["ob_fill_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_fill_logs"] / "synthesize_missing_days.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_cached_range(symbol: str) -> Optional[Tuple[datetime, datetime]]:
    cache_file = PATHS["ob_metadata"] / "available_ranges.json"
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            if symbol in cache:
                oldest = datetime.fromisoformat(cache[symbol]["oldest"])
                latest = datetime.fromisoformat(cache[symbol]["latest"])
                return oldest, latest
        except Exception:
            return None
    return None


def probe_range(symbol: str, logger: logging.Logger) -> Optional[Tuple[datetime, datetime]]:
    if obd is None:
        logger.error("Range probe unavailable")
        return None
    session = None
    try:
        import requests
        session = requests.Session()
        session.headers.update({"User-Agent": "download3-synthesize/1.0"})
        return obd.get_available_date_range(symbol, session, logger)
    except Exception as e:
        logger.error(f"Range probe failed: {e}")
        return None
    finally:
        try:
            if session is not None:
                session.close()
        except Exception:
            pass


def list_existing_dates(symbol: str) -> List[str]:
    PATHS["ob_raw_csv"].mkdir(parents=True, exist_ok=True)
    dates: List[str] = []
    for p in PATHS["ob_raw_csv"].glob(f"{symbol}-bookDepth-*.csv"):
        try:
            dates.append(p.stem.split("-bookDepth-")[-1])
        except Exception:
            continue
    return sorted(dates)


def make_date_list(start: datetime, end: datetime) -> List[str]:
    out: List[str] = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def read_day_csv(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    fp = PATHS["ob_raw_csv"] / f"{symbol}-bookDepth-{date_str}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return None


def day_medians(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Returns median depth and median notional per percentage for the day
    grp = df.groupby("percentage")
    med_depth = grp["depth"].median()
    med_notional = grp["notional"].median()
    return med_depth, med_notional


def pick_donor(symbol: str, missing_date: datetime, existing_dates: List[str]) -> Optional[str]:
    # Pick nearest existing day; prefer same weekday if tie
    if not existing_dates:
        return None
    candidates = []
    for ds in existing_dates:
        d = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dist = abs((d - missing_date).days)
        same_wd = 1 if d.weekday() == missing_date.weekday() else 0
        candidates.append((dist, -same_wd, ds))  # minimize dist, prefer same weekday (negative for sorting)
    candidates.sort()
    return candidates[0][2]


def find_neighbor(symbol: str, boundary_date: datetime, existing_dates: List[str], direction: int) -> Optional[str]:
    # direction -1 for before, +1 for after
    dates_dt = [datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc) for ds in existing_dates]
    if direction < 0:
        before = [ds for ds in dates_dt if ds < boundary_date]
        if not before:
            return None
        return before[-1].strftime("%Y-%m-%d")
    else:
        after = [ds for ds in dates_dt if ds > boundary_date]
        if not after:
            return None
        return after[0].strftime("%Y-%m-%d")


def group_consecutive(dates: List[str]) -> List[List[str]]:
    if not dates:
        return []
    dates_sorted = sorted(dates)
    seqs: List[List[str]] = []
    cur_seq = [dates_sorted[0]]
    for prev, cur in zip(dates_sorted, dates_sorted[1:]):
        prev_dt = datetime.strptime(prev, "%Y-%m-%d")
        cur_dt = datetime.strptime(cur, "%Y-%m-%d")
        if (cur_dt - prev_dt).days == 1:
            cur_seq.append(cur)
        else:
            seqs.append(cur_seq)
            cur_seq = [cur]
    seqs.append(cur_seq)
    return seqs


def compute_scale_series(med_R: pd.Series, med_B: Optional[pd.Series], med_A: Optional[pd.Series], t: float) -> Dict[float, float]:
    # Geometric interpolation of ratios per percentage
    scales: Dict[float, float] = {}
    idx = med_R.index
    # default ratios = 1
    ratioB = (med_B / med_R).reindex(idx, fill_value=1.0) if med_B is not None else pd.Series(1.0, index=idx)
    ratioA = (med_A / med_R).reindex(idx, fill_value=1.0) if med_A is not None else pd.Series(1.0, index=idx)
    # Clip to avoid zeros/negatives
    ratioB = ratioB.fillna(1.0).clip(lower=1e-9)
    ratioA = ratioA.fillna(1.0).clip(lower=1e-9)
    # Interpolate: scale = ratioB^(1-t) * ratioA^t
    vals = np.exp((1 - t) * np.log(ratioB.values) + t * np.log(ratioA.values))
    for pct, val in zip(idx, vals):
        scales[pct] = float(val)
    return scales


def synthesize_from_donor(symbol: str, target_date: str, donor_df: pd.DataFrame, scales_depth: Dict[float, float], jitter_sigma: float, logger: logging.Logger) -> bool:
    med_npd = donor_df.groupby("percentage")["npd"].median().fillna(0.0)
    target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    rng_seed = (hash((symbol, target_date)) & 0xFFFFFFFF)
    rng = np.random.RandomState(rng_seed)

    out_rows: List[Dict] = []
    for _, row in donor_df.iterrows():
        pct = row["percentage"]
        scale = scales_depth.get(pct, 1.0)
        depth_new = float(row["depth"]) * scale
        npd_val = row["npd"]
        if not pd.notna(npd_val) or npd_val <= 0:
            npd_val = float(med_npd.get(pct, 0.0))
        notional_new = depth_new * float(npd_val)

        # Deterministic jitter per row
        eps = float(np.clip(rng.normal(loc=0.0, scale=jitter_sigma), -3 * jitter_sigma, 3 * jitter_sigma))
        jitter_mult = max(0.0, 1.0 + eps)
        depth_new *= jitter_mult
        notional_new *= jitter_mult

        ts: pd.Timestamp = row["timestamp"]  # tz-aware
        tod = (ts.hour, ts.minute, ts.second, ts.microsecond)
        ts_new = target_dt.replace(hour=tod[0], minute=tod[1], second=tod[2], microsecond=tod[3])

        out_rows.append({
            "timestamp": ts_new.isoformat(),
            "percentage": pct,
            "depth": depth_new,
            "notional": notional_new,
        })

    if not out_rows:
        logger.warning(f"No rows synthesized for {target_date}")
        return False

    out_fp = PATHS["ob_raw_csv"] / f"{symbol}-bookDepth-{target_date}.csv"
    pd.DataFrame(out_rows).to_csv(out_fp, index=False)
    return True


def main() -> None:
    logger = setup_logging()

    for symbol in PAIRS:
        rng = load_cached_range(symbol) or probe_range(symbol, logger)
        if not rng:
            logger.error(f"Cannot determine available range for {symbol}")
            continue
        start, end = rng
        all_days = make_date_list(start, end)
        existing = list_existing_dates(symbol)
        missing_all = [d for d in all_days if d not in existing]
        if not missing_all:
            logger.info(f"No missing days for {symbol}")
            continue
        logger.info(f"{symbol}: {len(missing_all)} missing days -> synthesizing with sequence-aware scaling")

        # Process consecutive sequences; do not update neighbors within a sequence
        sequences = group_consecutive(missing_all)
        for seq in sequences:
            # Determine neighbors using the original existing set (exclude current sequence)
            seq_start_dt = datetime.strptime(seq[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            seq_end_dt = datetime.strptime(seq[-1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            nb_str = find_neighbor(symbol, seq_start_dt, existing, direction=-1)
            na_str = find_neighbor(symbol, seq_end_dt, existing, direction=+1)

            # Pick a fixed donor for the whole sequence (use middle day of sequence for weekday preference)
            mid_idx = len(seq) // 2
            mid_dt = datetime.strptime(seq[mid_idx], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            donor_str = pick_donor(symbol, mid_dt, existing)
            if not donor_str:
                logger.warning(f"No donor available for sequence {seq}; skipping")
                continue

            donor_df = read_day_csv(symbol, donor_str)
            if donor_df is None or donor_df.empty:
                logger.warning(f"Donor day unreadable: {donor_str}; skipping sequence {seq}")
                continue
            donor_df = donor_df.copy()
            donor_df["npd"] = donor_df.apply(lambda r: (r["notional"] / r["depth"]) if r["depth"] and r["depth"] != 0 else float("nan"), axis=1)
            med_depth_R, _ = day_medians(donor_df)

            # Neighbor medians (computed once for sequence)
            med_B = None
            med_A = None
            if nb_str:
                dfB = read_day_csv(symbol, nb_str)
                if dfB is not None and not dfB.empty:
                    med_B, _ = day_medians(dfB)
            if na_str:
                dfA = read_day_csv(symbol, na_str)
                if dfA is not None and not dfA.empty:
                    med_A, _ = day_medians(dfA)

            # Synthesize each day with interpolated scale t in (i+1)/(N+1)
            N = len(seq)
            for i, ds in enumerate(seq):
                t = (i + 1) / (N + 1) if (med_B is not None or med_A is not None) else 0.0
                scales = compute_scale_series(med_depth_R, med_B, med_A, t)
                ok = synthesize_from_donor(symbol, ds, donor_df, scales, jitter_sigma=0.01, logger=logger)
                if ok:
                    logger.info(f"Synthesized {symbol} {ds} using donor {donor_str} (t={t:.2f}, neighbors: {nb_str}, {na_str})")
                else:
                    logger.warning(f"Failed to synthesize {symbol} {ds}")

            # After sequence, update existing to include synthesized days
            existing.extend(seq)
            existing.sort()


if __name__ == "__main__":
    main()