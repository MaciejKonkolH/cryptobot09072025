from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import requests

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PATHS, LOGGING, RETRY, PAIRS

# Reuse constants and helpers from downloader when possible
try:
    from orderbook_downloader import get_available_date_range, BASE_URL  # type: ignore
except Exception:
    BASE_URL = "https://data.binance.vision/data"


def setup_logging() -> logging.Logger:
    PATHS["ob_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_logs"] / "check_missing_csv.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_cached_range(symbol: str, logger: logging.Logger) -> Optional[Tuple[datetime, datetime]]:
    cache_file = PATHS["ob_metadata"] / "available_ranges.json"
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            if symbol in cache:
                d = cache[symbol]
                oldest = datetime.fromisoformat(d["oldest"])
                latest = datetime.fromisoformat(d["latest"])
                logger.info(f"Cached available range for {symbol}: {oldest.date()} - {latest.date()}")
                return oldest, latest
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
    return None


def probe_range_if_needed(symbol: str, logger: logging.Logger) -> Optional[Tuple[datetime, datetime]]:
    # Try to import and use downloader's probe
    try:
        import orderbook_downloader as obd  # type: ignore
        session = requests.Session()
        session.headers.update({"User-Agent": "download3-check-missing/1.0"})
        return obd.get_available_date_range(symbol, session, logger)
    except Exception as e:
        logger.error(f"Range probe failed: {e}")
        return None


def list_existing_dates(symbol: str) -> List[str]:
    PATHS["ob_raw_csv"].mkdir(parents=True, exist_ok=True)
    existing = []
    for p in PATHS["ob_raw_csv"].glob(f"{symbol}-bookDepth-*.csv"):
        try:
            existing.append(p.stem.split("-bookDepth-")[-1])
        except Exception:
            continue
    return sorted(existing)


def date_range_list(start: datetime, end: datetime) -> List[str]:
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def check_online_availability(symbol: str, missing_dates: List[str], logger: logging.Logger) -> tuple[List[str], List[str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "download3-check-missing/1.0"})
    available: List[str] = []
    not_found: List[str] = []
    timeout = int(RETRY["timeout_seconds"]) if isinstance(RETRY.get("timeout_seconds"), (int, float)) else 30
    for ds in missing_dates:
        url = f"{BASE_URL}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{ds}.zip"
        try:
            r = session.head(url, timeout=timeout)
            if r.status_code == 200:
                available.append(ds)
            else:
                not_found.append(ds)
        except Exception:
            # Treat network errors as unknown/not found for now
            not_found.append(ds)
    return available, not_found


def write_report(symbol: str, total_days: int, existing: List[str], available: List[str], not_found: List[str], logger: logging.Logger) -> Path:
    PATHS["ob_metadata"].mkdir(parents=True, exist_ok=True)
    report = {
        "symbol": symbol,
        "total_days": total_days,
        "existing_days": len(existing),
        "missing_days": total_days - len(existing),
        "online_available_missing": available,
        "not_found_on_server": not_found,
    }
    out = PATHS["ob_metadata"] / f"missing_report_{symbol}.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Report written: {out}")
    return out


def main() -> None:
    logger = setup_logging()
    symbols = PAIRS
    for symbol in symbols:
        rng = load_cached_range(symbol, logger) or probe_range_if_needed(symbol, logger)
        if not rng:
            logger.error(f"Cannot determine range for {symbol}")
            continue
        start, end = rng
        all_days = date_range_list(start, end)
        existing = list_existing_dates(symbol)
        missing = [d for d in all_days if d not in existing]
        logger.info(f"{symbol}: total={len(all_days)}, existing={len(existing)}, missing={len(missing)}")
        if not missing:
            logger.info(f"No missing CSV files for {symbol}")
            continue
        # Log exact missing dates
        logger.info(f"Missing dates for {symbol}: {', '.join(missing)}")
        available, not_found = check_online_availability(symbol, missing, logger)
        write_report(symbol, len(all_days), existing, available, not_found, logger)
        # Log split lists explicitly
        if available:
            logger.info(f"Available online (can retry): {', '.join(available)}")
        if not_found:
            logger.info(f"Not found on server: {', '.join(not_found)}")


if __name__ == "__main__":
    main()