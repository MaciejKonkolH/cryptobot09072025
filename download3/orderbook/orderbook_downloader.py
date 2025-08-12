from __future__ import annotations

import os
import sys
import json
import time
import zipfile
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

from config.config import PAIRS, PATHS, LOGGING, RETRY

BASE_URL = "https://data.binance.vision/data"


def setup_logging() -> logging.Logger:
    for key in ("ob_logs", "ob_metadata", "ob_progress", "ob_raw_zip", "ob_raw_csv"):
        PATHS[key].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ob_logs"] / "orderbook_downloader.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def check_exists(session: requests.Session, url: str, timeout: int) -> bool:
    try:
        r = session.head(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def download_and_extract(session: requests.Session, url: str, zip_path: Path, csv_path: Path, timeout: int, logger: logging.Logger) -> bool:
    for attempt in range(int(RETRY["max_retries"])):
        try:
            with session.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    time.sleep(RETRY["retry_delay_seconds"])  # backoff
                    continue
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = zf.namelist()[0]
                zf.extract(csv_name, str(csv_path.parent))
                extracted = csv_path.parent / csv_name
                extracted.rename(csv_path)
            zip_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.debug(f"Download error {url}: {e}")
            time.sleep(RETRY["retry_delay_seconds"])  # backoff
    return False


def get_available_date_range(symbol: str, session: requests.Session, logger: logging.Logger) -> Optional[Tuple[datetime, datetime]]:
    cache_file = PATHS["ob_metadata"] / "available_ranges.json"
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            if symbol in cache:
                d = cache[symbol]
                oldest = datetime.fromisoformat(d["oldest"])
                latest = datetime.fromisoformat(d["latest"])
                logger.info(f"Cached available range for {symbol}: {oldest.date()} - {latest.date()}")
                return oldest, latest
        except Exception:
            pass

    logger.info(f"Probing available orderbook range for {symbol}...")

    # Find oldest
    oldest: Optional[datetime] = None
    current_year = datetime.utcnow().year
    session_timeout = int(RETRY["timeout_seconds"])
    for year in range(2019, current_year + 1):
        # probe last day of months
        for month in range(1, 13):
            if month == 12:
                last_day = 31
            else:
                nxt = datetime(year, month + 1, 1)
                last_day = (nxt - timedelta(days=1)).day
            d = datetime(year, month, last_day)
            date_str = d.strftime("%Y-%m-%d")
            url = f"{BASE_URL}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
            if check_exists(session, url, session_timeout):
                # refine to first day of that month
                for day in range(1, last_day + 1):
                    dd = datetime(year, month, day)
                    s = dd.strftime("%Y-%m-%d")
                    url2 = f"{BASE_URL}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{s}.zip"
                    if check_exists(session, url2, session_timeout):
                        oldest = dd
                        break
                if oldest:
                    break
        if oldest:
            break

    # Find latest (last 30 days)
    latest: Optional[datetime] = None
    for i in range(0, 30):
        dd = datetime.utcnow() - timedelta(days=i)
        s = dd.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{s}.zip"
        if check_exists(session, url, session_timeout):
            latest = dd
            break

    if oldest and latest:
        cache.setdefault(symbol, {})
        cache[symbol] = {"oldest": oldest.isoformat(), "latest": latest.isoformat()}
        try:
            cache_file.write_text(json.dumps(cache, indent=2))
        except Exception:
            pass
        logger.info(f"Available range for {symbol}: {oldest.date()} - {latest.date()}")
        return oldest, latest
    logger.error(f"Could not determine available range for {symbol}")
    return None


def iter_dates(start_date: datetime, end_date: datetime) -> List[datetime]:
    dates: List[datetime] = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def write_progress_line(text: str) -> None:
    sys.stdout.write("\r" + text[:180])
    sys.stdout.flush()


def download_symbol(symbol: str, logger: logging.Logger) -> bool:
    PATHS["ob_raw_zip"].mkdir(parents=True, exist_ok=True)
    PATHS["ob_raw_csv"].mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "download3-orderbook/1.0"})

    rng = get_available_date_range(symbol, session, logger)
    if not rng:
        return False
    start_date, end_date = rng

    # Existing dates
    existing = set()
    for p in PATHS["ob_raw_csv"].glob(f"{symbol}-bookDepth-*.csv"):
        try:
            date_str = p.stem.split("-bookDepth-")[-1]
            existing.add(date_str)
        except Exception:
            continue

    all_dates = iter_dates(start_date, end_date)
    total_days = len(all_dates)
    remaining_dates = [d for d in all_dates if d.strftime("%Y-%m-%d") not in existing]
    logger.info(
        f"{symbol}: total days {total_days:,}, existing {len(existing):,}, to download {len(remaining_dates):,}"
    )

    ok = 0
    failed = 0
    started = time.time()

    try:
        for idx, d in enumerate(remaining_dates, start=1):
            date_str = d.strftime("%Y-%m-%d")
            url = f"{BASE_URL}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
            zip_path = PATHS["ob_raw_zip"] / f"{symbol}-bookDepth-{date_str}.zip"
            csv_path = PATHS["ob_raw_csv"] / f"{symbol}-bookDepth-{date_str}.csv"
            success = download_and_extract(session, url, zip_path, csv_path, int(RETRY["timeout_seconds"]), logger)
            if success:
                ok += 1
            else:
                failed += 1
            elapsed = time.time() - started
            rate = (idx / elapsed) if elapsed > 0 else 0.0
            left = len(remaining_dates) - idx
            eta_sec = left / rate if rate > 0 else 0
            write_progress_line(
                f"[{symbol}] {idx}/{len(remaining_dates)} days | ok={ok} fail={failed} | rate={rate:.2f}/s | ETA={int(eta_sec)}s | {date_str}"
            )
            time.sleep(0.02)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Partial results saved.")
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()

    logger.info(f"Orderbook downloaded {symbol}: ok={ok}, fail={failed}, remaining={len(remaining_dates) - ok - failed}")
    return ok > 0


def main() -> None:
    logger = setup_logging()
    success = 0
    for symbol in PAIRS:
        success += 1 if download_symbol(symbol, logger) else 0
    logger.info(f"Done Orderbook. Success: {success}/{len(PAIRS)}")


if __name__ == "__main__":
    main()