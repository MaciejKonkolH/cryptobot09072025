from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PAIRS, PATHS, LOGGING, INTERVAL


def setup_logging() -> logging.Logger:
    for key in ("merge_logs", "merge_metadata", "merged_data"):
        PATHS[key].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["merge_logs"] / "merge_ohlc_orderbook.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_ohlc(symbol: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    fp = PATHS["ohlc_raw"] / f"{symbol}_{INTERVAL}.parquet"
    if not fp.exists():
        logger.error(f"Missing OHLC {fp}")
        return None
    df = pd.read_parquet(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_orderbook_wide(symbol: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load normalized merged orderbook and reshape to wide per-minute with two snapshots and all percentage levels.

    Output index: minute (tz-aware UTC)
    Columns (dynamic):
      - depth_{rank}_{pctKey}
      - notional_{rank}_{pctKey}
    where rank in {1,2} and pctKey is p{abs(p)} or m{abs(p)} for positive/negative percentage levels.
    """
    fp = PATHS["ob_normalized_merged"] / f"{symbol}_normalized_merged.feather"
    if not fp.exists():
        logger.error(f"Missing orderbook {fp}")
        return None
    df = pd.read_feather(fp)
    if df.empty:
        logger.error(f"Orderbook file is empty: {fp}")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["minute"] = df["timestamp"].dt.floor("min")
    # Identify first and last snapshot timestamps per minute
    ts_agg = df.groupby("minute")["timestamp"].agg(ts_1="min", ts_2="max").reset_index()
    df = df.merge(ts_agg, on="minute", how="inner")
    # Assign rank by matching exact snapshot timestamp to first/last
    df["rank"] = 0
    df.loc[df["timestamp"] == df["ts_1"], "rank"] = 1
    df.loc[df["timestamp"] == df["ts_2"], "rank"] = 2
    df = df[df["rank"].isin([1, 2])]

    # Pivot depth and notional across (rank, percentage)
    depth_p = df.pivot_table(index="minute", columns=["rank", "percentage"], values="depth", aggfunc="first")
    notional_p = df.pivot_table(index="minute", columns=["rank", "percentage"], values="notional", aggfunc="first")

    # Build flat column names
    def pct_key(val: float) -> str:
        try:
            v = float(val)
        except Exception:
            return f"pct_{str(val)}"
        if float(v).is_integer():
            v = int(v)
        sign = "p" if v >= 0 else "m"
        return f"{sign}{abs(v)}"

    depth_p.columns = [f"depth_{int(col[0])}_{pct_key(col[1])}" for col in depth_p.columns]
    notional_p.columns = [f"notional_{int(col[0])}_{pct_key(col[1])}" for col in notional_p.columns]

    # ts_1 and ts_2 per minute (from ts_agg)
    ts_df = ts_agg.set_index("minute").sort_index()
    ob_wide = pd.concat([ts_df, depth_p, notional_p], axis=1)
    # Ensure flat string columns (no MultiIndex remnants)
    ob_wide.columns = pd.Index([str(c) for c in ob_wide.columns])
    ob_wide = ob_wide.sort_index(axis=1)
    ob_wide = ob_wide.sort_index()
    return ob_wide


def align_ranges_minute(ohlc_min: pd.DataFrame, ob_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_start = max(ohlc_min.index.min(), ob_wide.index.min())
    common_end = min(ohlc_min.index.max(), ob_wide.index.max())
    ohlc_f = ohlc_min[(ohlc_min.index >= common_start) & (ohlc_min.index <= common_end)].copy()
    ob_f = ob_wide[(ob_wide.index >= common_start) & (ob_wide.index <= common_end)].copy()
    return ohlc_f, ob_f


def merge_symbol(symbol: str, logger: logging.Logger) -> Optional[Path]:
    # Load OHLC and reshape to minute index
    ohlc = load_ohlc(symbol, logger)
    if ohlc is None:
        return None
    ohlc_min = ohlc.copy()
    ohlc_min["minute"] = ohlc_min["timestamp"].dt.floor("min")
    ohlc_min = ohlc_min.set_index("minute").sort_index()
    ohlc_min = ohlc_min.drop(columns=["timestamp"])  # minute is the time key

    # Load orderbook and reshape to wide per-minute
    ob_wide = load_orderbook_wide(symbol, logger)
    if ob_wide is None:
        return None

    # Align common minute range
    ohlc_f, ob_f = align_ranges_minute(ohlc_min, ob_wide)

    # Build training3-compatible aliases for depth (snapshot1/2, ±levels)
    def add_snapshot_aliases(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Map depth_1_* to snapshot1_depth_*, depth_2_* to snapshot2_depth_*
        for rank, snap in [(1, "snapshot1"), (2, "snapshot2")]:
            for lvl in [1, 2, 3, 4, 5]:
                pcol = f"depth_{rank}_p{lvl}"
                mcol = f"depth_{rank}_m{lvl}"
                if pcol in out.columns:
                    out[f"{snap}_depth_{lvl}"] = out[pcol]
                if mcol in out.columns:
                    out[f"{snap}_depth_{-lvl}"] = out[mcol]
        # Bid/ask volumes (sum of 5 near levels) and spread from best levels
        if all(c in out.columns for c in ["snapshot1_depth_-1", "snapshot1_depth_1"]):
            out["snapshot1_bid_volume"] = sum(out.get(f"snapshot1_depth_{-i}", 0) for i in range(1, 6))
            out["snapshot1_ask_volume"] = sum(out.get(f"snapshot1_depth_{i}", 0) for i in range(1, 6))
            out["snapshot1_spread"] = out["snapshot1_depth_1"] - out["snapshot1_depth_-1"]
            out["spread"] = out["snapshot1_spread"]
        # Optional snapshot2 aggregates
        if all(c in out.columns for c in ["snapshot2_depth_-1", "snapshot2_depth_1"]):
            out["snapshot2_bid_volume"] = sum(out.get(f"snapshot2_depth_{-i}", 0) for i in range(1, 6))
            out["snapshot2_ask_volume"] = sum(out.get(f"snapshot2_depth_{i}", 0) for i in range(1, 6))
            out["snapshot2_spread"] = out["snapshot2_depth_1"] - out["snapshot2_depth_-1"]
        return out

    ob_f = add_snapshot_aliases(ob_f)

    # Join on minute index; each minute -> one row with OHLC + two snapshots (+ts_1, ts_2) and aliases
    merged = ohlc_f.join(ob_f, how="inner").reset_index().rename(columns={"minute": "timestamp"})

    out_fp = PATHS["merged_data"] / f"merged_{symbol}.parquet"
    merged.to_parquet(out_fp, index=False)
    logger.info(f"Saved merged for {symbol}: {len(merged):,} rows -> {out_fp}")
    return out_fp


def main() -> None:
    logger = setup_logging()
    ok = 0
    for symbol in PAIRS:
        ok += 1 if merge_symbol(symbol, logger) else 0
    logger.info(f"Done merge OHLC+Orderbook. Success: {ok}/{len(PAIRS)}")


if __name__ == "__main__":
    main()