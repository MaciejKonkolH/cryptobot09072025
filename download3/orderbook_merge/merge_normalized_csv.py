from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, PAIRS


def setup_logging() -> logging.Logger:
    PATHS["ob_merge_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_merge_logs"] / "merge_normalized_csv.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def list_normalized_days(symbol: str) -> List[Path]:
    return sorted(PATHS["ob_normalized_csv"].glob(f"{symbol}-bookDepth-*.csv"))


def merge_symbol(symbol: str, logger: logging.Logger) -> Optional[Path]:
    files = list_normalized_days(symbol)
    if not files:
        logger.error(f"No normalized CSVs for {symbol}")
        return None
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {fp.name}: {e}")
    if not dfs:
        return None
    full = pd.concat(dfs, ignore_index=True)
    # Ensure timestamp is string (ISO) as written
    out_dir = PATHS["ob_normalized_merged"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{symbol}_normalized_merged.feather"
    # convert timestamp back to datetime for feather
    full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
    full.to_feather(out_fp)
    logger.info(f"Saved merged normalized: {out_fp} ({len(full):,} rows)")
    return out_fp


def main() -> None:
    logger = setup_logging()
    ok = 0
    for symbol in PAIRS:
        ok += 1 if merge_symbol(symbol, logger) else 0
    logger.info(f"Done merge normalized. Success: {ok}/{len(PAIRS)}")


if __name__ == "__main__":
    main()