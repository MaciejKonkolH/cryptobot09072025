from __future__ import annotations

"""
Central configuration for download3 pipeline (OHLC + Orderbook + Merge).
All paths are relative to the project root `crypto/download3/` and created on demand.
"""
from pathlib import Path
from typing import List, Dict

# Project root for download3 (this file lives in crypto/download3/config/)
DOWNLOAD3_ROOT: Path = Path(__file__).resolve().parents[1]

# Symbols / pairs to process (futures)
PAIRS: List[str] = [
    "BTCUSDT",
    "XRPUSDT",
    "ETHUSDT",
]

# Time / data settings
INTERVAL: str = "1m"  # OHLC interval

# History backfill settings (minutes)
OHLC_HISTORY_BACK_MINUTES: int = 43200  # 30 days for feature calc alignment

# Gap filling thresholds
GAPS: Dict[str, float | int] = {
    "max_small_gap_minutes": 5,
    "max_medium_gap_minutes": 60,
    "price_change_threshold_percent": 2.0,
}

# Concurrency / performance
CONCURRENCY: Dict[str, int] = {
    "max_workers": 8,
}

# Retry / backoff
RETRY: Dict[str, int | float] = {
    "max_retries": 3,
    "retry_delay_seconds": 1,
    "timeout_seconds": 30,
}

# Paths layout
PATHS: Dict[str, Path] = {
    # OHLC
    "ohlc_raw": DOWNLOAD3_ROOT / "OHLC" / "raw",
    "ohlc_logs": DOWNLOAD3_ROOT / "OHLC" / "logs",
    "ohlc_metadata": DOWNLOAD3_ROOT / "OHLC" / "metadata",
    "ohlc_progress": DOWNLOAD3_ROOT / "OHLC" / "progress",

    # Orderbook download
    "ob_raw_zip": DOWNLOAD3_ROOT / "orderbook" / "raw_zip",
    "ob_raw_csv": DOWNLOAD3_ROOT / "orderbook" / "raw_csv",
    "ob_logs": DOWNLOAD3_ROOT / "orderbook" / "logs",
    "ob_metadata": DOWNLOAD3_ROOT / "orderbook" / "metadata",
    "ob_progress": DOWNLOAD3_ROOT / "orderbook" / "progress",

    # Normalized per-day CSV (2 snapshots per minute)
    "ob_normalized_csv": DOWNLOAD3_ROOT / "orderbook" / "normalized_csv",
    "ob_normalized_merged": DOWNLOAD3_ROOT / "orderbook" / "normalized_merged",

    # Orderbook merge (legacy wide)
    "ob_merged_raw": DOWNLOAD3_ROOT / "orderbook_merge" / "merged_raw",
    "ob_merge_logs": DOWNLOAD3_ROOT / "orderbook_merge" / "logs",
    "ob_merge_metadata": DOWNLOAD3_ROOT / "orderbook_merge" / "metadata",

    # Orderbook fill (legacy)
    "ob_completed": DOWNLOAD3_ROOT / "orderbook_fill" / "completed",
    "ob_fill_logs": DOWNLOAD3_ROOT / "orderbook_fill" / "logs",
    "ob_fill_metadata": DOWNLOAD3_ROOT / "orderbook_fill" / "metadata",

    # Final merge OHLC + orderbook
    "merged_data": DOWNLOAD3_ROOT / "merge" / "merged_data",
    "merge_logs": DOWNLOAD3_ROOT / "merge" / "logs",
    "merge_metadata": DOWNLOAD3_ROOT / "merge" / "metadata",
}

# Logging
LOGGING: Dict[str, str] = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
}


def ensure_directories_exist() -> None:
    """Create all configured directories if not present."""
    for key, path in PATHS.items():
        path.mkdir(parents=True, exist_ok=True)