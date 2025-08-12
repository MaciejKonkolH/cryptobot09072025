from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config.config import DOWNLOAD3_ROOT


STEPS = {
    "ohlc": [sys.executable, str(DOWNLOAD3_ROOT / "OHLC" / "ohlc_downloader.py")],
    "orderbook_download": [sys.executable, str(DOWNLOAD3_ROOT / "orderbook" / "orderbook_downloader.py")],
    "orderbook_merge": [sys.executable, str(DOWNLOAD3_ROOT / "orderbook_merge" / "merge_orderbook_to_wide.py")],
    "orderbook_fill": [sys.executable, str(DOWNLOAD3_ROOT / "orderbook_fill" / "fill_orderbook_gaps.py")],
    "merge_all": [sys.executable, str(DOWNLOAD3_ROOT / "merge" / "merge_ohlc_orderbook.py")],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="download3 pipeline orchestrator")
    parser.add_argument("--steps", default="all", help="Comma-separated steps or 'all'")
    args = parser.parse_args()

    step_list = list(STEPS.keys()) if args.steps == "all" else [s.strip() for s in args.steps.split(",")]

    for step in step_list:
        cmd = STEPS.get(step)
        if not cmd:
            print(f"Unknown step: {step}")
            sys.exit(1)
        print(f"\n=== Running step: {step} ===")
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"Step failed: {step}")
            sys.exit(ret)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()