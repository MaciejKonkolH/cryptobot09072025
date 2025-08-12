from __future__ import annotations

import sys
import time
import json
import logging
import argparse
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

# Ensure project root (download3) is on sys.path regardless of CWD
DOWNLOAD3_ROOT = Path(__file__).resolve().parents[1]
if str(DOWNLOAD3_ROOT) not in sys.path:
    sys.path.insert(0, str(DOWNLOAD3_ROOT))

try:
    import ccxt  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# Local imports
from config.config import PAIRS, INTERVAL, OHLC_HISTORY_BACK_MINUTES, PATHS, LOGGING, RETRY, ensure_directories_exist


def setup_logging() -> logging.Logger:
    ensure_directories_exist()
    PATHS["ohlc_logs"].mkdir(parents=True, exist_ok=True)
    log_file = PATHS["ohlc_logs"] / "ohlc_downloader.log"
    logging.basicConfig(
        level=getattr(logging, LOGGING["level"]),
        format=LOGGING["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def init_exchange():
    return ccxt.binanceusdm({
        "timeout": int(RETRY["timeout_seconds"]) * 1000,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def to_ccxt_symbol(pair: str) -> str:
    """Convert symbols like BTCUSDT -> BTC/USDT:USDT for binanceusdm."""
    p = pair.upper()
    if p.endswith("USDT"):
        base = p[:-4]
        return f"{base}/USDT:USDT"
    return p


def load_existing(symbol: str) -> Optional[pd.DataFrame]:
    output_file = PATHS["ohlc_raw"] / f"{symbol}_{INTERVAL}.parquet"
    if output_file.exists():
        try:
            df = pd.read_parquet(output_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return None
    return None


def fetch_range(exchange, ccxt_symbol: str, since_ms: int, interval: str, limit: int = 1000) -> list:
    for attempt in range(int(RETRY["max_retries"])):
        try:
            return exchange.fetch_ohlcv(ccxt_symbol, timeframe=interval, since=since_ms, limit=limit)
        except Exception:
            time.sleep(RETRY["retry_delay_seconds"])  # backoff simple
    return []


def parse_date(date_str: Optional[str], *, is_end: bool = False) -> Optional[datetime]:
    if not date_str:
        return None
    # Accept YYYY-MM-DD or ISO
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        # fallback: only date
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Assume naive as UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if is_end and dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        # Treat end date as inclusive whole day -> add 1 day
        dt = dt + timedelta(days=1)
    return dt


def humanize_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def write_progress_line(text: str) -> None:
    sys.stdout.write("\r" + text[:180])  # limit width
    sys.stdout.flush()


def download_symbol(symbol: str, logger: logging.Logger, *, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None, resume: bool = True) -> bool:
    ensure_directories_exist()
    PATHS["ohlc_raw"].mkdir(parents=True, exist_ok=True)
    exchange = init_exchange()

    # Determine time range
    if start_dt is None or end_dt is None:
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        end_dt = end_dt or now_utc
        default_start = end_dt - timedelta(minutes=OHLC_HISTORY_BACK_MINUTES)
        start_dt = start_dt or default_start

    # Normalize tz
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    # Resume from existing if requested and range not explicitly forcing start
    if resume and start_dt is None:
        existing = load_existing(symbol)
        if existing is not None and not existing.empty:
            start_dt = existing["timestamp"].max().to_pydatetime()

    ccxt_symbol = to_ccxt_symbol(symbol)

    # Initial planning (may be recalibrated after first data chunk if exchange skips empty history)
    base_start_dt = start_dt
    total_minutes = max(0, int((end_dt - base_start_dt).total_seconds() // 60))
    limit = 1000
    expected_chunks = math.ceil(total_minutes / limit) if total_minutes > 0 else 0

    logger.info(
        f"Downloading OHLC {symbol} ({ccxt_symbol}) from {start_dt} to {end_dt} ({INTERVAL})\n"
        f"Expected: {total_minutes:,} minutes ≈ {expected_chunks} chunks"
    )

    all_rows = []
    current_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    chunks_done = 0
    start_time = time.time()
    adjusted_logged = False

    try:
        while current_ms < end_ms:
            chunk = fetch_range(exchange, ccxt_symbol, current_ms, INTERVAL, limit=limit)
            if not chunk:
                logger.warning("Empty chunk or fetch failed; stopping.")
                break

            # Rebase progress on first available candle if exchange skipped empty part
            if chunks_done == 0 and chunk:
                first_ts = chunk[0][0]
                first_dt = datetime.fromtimestamp(first_ts / 1000.0, tz=timezone.utc)
                if first_dt > base_start_dt:
                    base_start_dt = first_dt
                    total_minutes = max(0, int((end_dt - base_start_dt).total_seconds() // 60))
                    expected_chunks = math.ceil(total_minutes / limit) if total_minutes > 0 else 0
                    if not adjusted_logged:
                        logger.info(f"Adjusted start to first available timestamp: {first_dt}")
                        adjusted_logged = True

            all_rows.extend(chunk)
            last_ts = chunk[-1][0]
            # Prevent infinite loop on stale last_ts
            if last_ts <= current_ms:
                logger.warning("Non-advancing last_ts detected; stopping to avoid loop.")
                break
            current_ms = last_ts + 1

            chunks_done += 1
            last_dt = datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc)
            progressed_min = min(total_minutes, max(0, int((last_dt - base_start_dt).total_seconds() // 60)))
            percent = (progressed_min / total_minutes * 100) if total_minutes > 0 else 100.0
            elapsed = time.time() - start_time
            speed = (progressed_min / elapsed) if elapsed > 0 else 0.0  # minutes per second
            remaining_min = max(0, total_minutes - progressed_min)
            eta = humanize_seconds(remaining_min / speed) if speed > 0 else "n/a"

            # single-line progress
            write_progress_line(
                f"[{symbol}] Chunks {chunks_done}/{expected_chunks} | {progressed_min:,}/{total_minutes:,} min ({percent:.2f}%) | "
                f"Last {last_dt} | Elapsed {humanize_seconds(elapsed)} | ETA {eta}"
            )

            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving fetched data so far...")
    finally:
        # end progress line
        sys.stdout.write("\n")
        sys.stdout.flush()

    if not all_rows:
        logger.warning(f"No data fetched for {symbol}")
        return False

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    output_file = PATHS["ohlc_raw"] / f"{symbol}_{INTERVAL}.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {symbol}: {len(df):,} rows to {output_file}")
    return True


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="download3 OHLC downloader")
    parser.add_argument("--start", help="Start datetime (YYYY-MM-DD or ISO)")
    parser.add_argument("--end", help="End datetime (inclusive day if date only)")
    parser.add_argument("--symbol", help="Single symbol to download (default from config)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else PAIRS
    start_dt = parse_date(args.start, is_end=False) if args.start else None
    end_dt = parse_date(args.end, is_end=True) if args.end else None

    ok = 0
    for symbol in symbols:
        ok += 1 if download_symbol(symbol, logger, start_dt=start_dt, end_dt=end_dt, resume=False if start_dt else True) else 0
    logger.info(f"Done OHLC. Success: {ok}/{len(symbols)}")


if __name__ == "__main__":
    main()