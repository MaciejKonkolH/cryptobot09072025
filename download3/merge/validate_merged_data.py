from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS


def setup_logging() -> logging.Logger:
    PATHS["merge_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["merge_logs"] / "validate_merged_data.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_merged(symbol: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    fp = PATHS["merged_data"] / f"merged_{symbol}.parquet"
    if not fp.exists():
        logger.error(f"Missing merged file: {fp}")
        return None
    df = pd.read_parquet(fp)
    if "timestamp" not in df.columns:
        logger.error("Merged file has no 'timestamp' column")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def infer_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    ohlc_candidates = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    ohlc_cols = [c for c in df.columns if c in ohlc_candidates]
    # New wide schema: two snapshots per minute, many percentage levels
    ob_cols = [c for c in df.columns if c.startswith("depth_") or c.startswith("notional_")]
    other_cols = [c for c in df.columns if c not in ("timestamp", *ohlc_cols, *ob_cols)]
    return {"ohlc": ohlc_cols, "orderbook": ob_cols, "other": other_cols}


def check_continuity(df: pd.DataFrame) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    ts = df["timestamp"].dropna().sort_values().reset_index(drop=True)
    if ts.empty:
        res.update({"minutes_expected": 0, "minutes_present": 0, "missing_minutes": 0, "examples_missing": []})
        return res
    start = ts.iloc[0].floor("min")
    end = ts.iloc[-1].floor("min")
    full_range = pd.date_range(start=start, end=end, freq="min", tz="UTC")
    present_minutes = ts.dt.floor("min").drop_duplicates()
    missing = full_range.difference(present_minutes)
    res.update({
        "range_start": start.isoformat(),
        "range_end": end.isoformat(),
        "minutes_expected": int(len(full_range)),
        "minutes_present": int(len(present_minutes)),
        "missing_minutes": int(len(missing)),
        "examples_missing": [m.isoformat() for m in missing[:10]],
    })
    return res


def pick_example_full_row(df: pd.DataFrame, columns_to_check: List[str]) -> Optional[Dict[str, Any]]:
    if not columns_to_check:
        columns_to_check = [c for c in df.columns if c != "timestamp"]
    mask = ~df[columns_to_check].isnull().any(axis=1)
    if not mask.any():
        return None
    row = df.loc[mask.idxmax()]
    out = {c: (row[c].isoformat() if isinstance(row[c], pd.Timestamp) else row[c]) for c in ["timestamp", *columns_to_check] if c in row}
    return out


def validate_symbol(symbol: str, logger: logging.Logger) -> Path:
    df = load_merged(symbol, logger)
    PATHS["merge_metadata"].mkdir(parents=True, exist_ok=True)
    report_fp = PATHS["merge_metadata"] / f"validate_merged_{symbol}.json"
    if df is None or df.empty:
        report_fp.write_text(json.dumps({"symbol": symbol, "error": "merged file missing or empty"}, indent=2))
        logger.info(f"Report written: {report_fp}")
        return report_fp

    groups = infer_column_groups(df)
    # Schema checks
    required_ohlc = ["open", "high", "low", "close"]
    # Require presence of both rank groups for depth/notional (any percentage set)
    has_rank1 = any(c.startswith("depth_1_") for c in df.columns) and any(c.startswith("notional_1_") for c in df.columns)
    has_rank2 = any(c.startswith("depth_2_") for c in df.columns) and any(c.startswith("notional_2_") for c in df.columns)
    missing_required = [c for c in required_ohlc if c not in df.columns]
    if not (has_rank1 and has_rank2):
        missing_required.append("depth_*/notional_* columns for rank 1 and/or rank 2")

    # Basic shape
    totals: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columns_list": list(df.columns),
        "schema_missing_required": missing_required,
    }

    # Date stats
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    totals.update({
        "timestamp_min": ts_min.isoformat(),
        "timestamp_max": ts_max.isoformat(),
    })

    # Continuity check (minute grid)
    continuity = check_continuity(df)

    # NaN checks
    nan_per_col = df.isnull().sum().to_dict()
    rows_with_any_nan = int(df.isnull().any(axis=1).sum())
    # Focused NaN for OB wide columns and OHLC columns
    ob_cols = [c for c in df.columns if c.startswith("depth_") or c.startswith("notional_")]
    ohlc_cols = [c for c in required_ohlc if c in df.columns]
    rows_with_any_nan_in_ob = int(df[ob_cols].isnull().any(axis=1).sum()) if ob_cols else rows_with_any_nan
    rows_with_any_nan_in_ohlc = int(df[ohlc_cols].isnull().any(axis=1).sum()) if ohlc_cols else rows_with_any_nan

    # Negatives
    depth_12 = pd.concat([df.filter(regex=r"^depth_1_"), df.filter(regex=r"^depth_2_")], axis=1)
    notional_12 = pd.concat([df.filter(regex=r"^notional_1_"), df.filter(regex=r"^notional_2_")], axis=1)
    negative_depth = int((depth_12 < 0).any(axis=1).sum())
    negative_notional = int((notional_12 < 0).any(axis=1).sum())

    # Example of fully-complete row (no NaN in any column)
    example = pick_example_full_row(df, [c for c in df.columns if c != "timestamp"])

    report = {
        "symbol": symbol,
        "groups": groups,
        "totals": totals,
        "continuity": continuity,
        "nan": {
            "rows_with_any_nan": rows_with_any_nan,
            "rows_with_any_nan_in_ob": rows_with_any_nan_in_ob,
            "rows_with_any_nan_in_ohlc": rows_with_any_nan_in_ohlc,
            "per_column": nan_per_col,
        },
        "values_checks": {
            "negative_depth_rows": negative_depth,
            "negative_notional_rows": negative_notional,
        },
        "example_full_row": example,
    }

    report_fp.write_text(json.dumps(report, indent=2))
    logger.info(f"Validation report written: {report_fp}")
    # Brief console summary
    logger.info(
        f"Rows: {totals['rows']:,}, Range: {continuity.get('range_start')} -> {continuity.get('range_end')}, "
        f"Missing minutes: {continuity.get('missing_minutes')} | Rows with any NaN: {rows_with_any_nan:,}"
    )
    if example is None:
        logger.warning("No fully-complete row found (some columns contain NaN in every row)")
    return report_fp


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Validate merged OHLC+Orderbook dataset for continuity and completeness")
    parser.add_argument("--symbol", help="Symbol to validate (default: all in config)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else PAIRS
    for sym in symbols:
        validate_symbol(sym, logger)


if __name__ == "__main__":
    main()

