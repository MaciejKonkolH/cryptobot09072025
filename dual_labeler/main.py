import sys
from pathlib import Path
import argparse
import json
import time
from typing import Tuple

import numpy as np
import pandas as pd

# Ensure project root on sys.path for absolute imports when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_labeler import config as cfg
from dual_labeler.logger import setup_logging


def load_features(symbol: str) -> pd.DataFrame:
    path = cfg.INPUT_DIR / cfg.INPUT_TEMPLATE.format(symbol=symbol)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku cech: {path}")
    df = pd.read_feather(path)
    if "timestamp" not in df.columns:
        raise ValueError("Plik cech nie zawiera kolumny 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    # Ensure OHLC present
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Brakuje kolumny '{c}' w pliku cech")
    return df


def find_first_hit(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    start_idx: int,
    tp_pct: float,
) -> Tuple[int, int]:
    """Scan forward until price hits +tp or -tp relative to close[start_idx].

    Returns a tuple (label, steps):
      - label: 0=LONG if +tp hit first, 1=SHORT if -tp hit first
      - steps: number of steps ahead where winner was found

    Tie-break when both thresholds are hit in the same bar:
      - if that bar is bullish (close > open) -> LONG
      - if bearish (close < open) -> SHORT
      - if doji (close == open) -> LONG
    """
    entry = close[start_idx]
    up = entry * (1 + tp_pct / 100.0)
    down = entry * (1 - tp_pct / 100.0)
    n = len(close)
    for j in range(start_idx + 1, n):
        idx = j
        hit_up = high[idx] >= up
        hit_down = low[idx] <= down
        if hit_up and not hit_down:
            return 0, idx - start_idx
        if hit_down and not hit_up:
            return 1, idx - start_idx
        if hit_up and hit_down:
            # tie-break by bar direction
            if close[idx] > open_[idx]:
                return 0, idx - start_idx
            elif close[idx] < open_[idx]:
                return 1, idx - start_idx
            else:
                return 0, idx - start_idx
    # If no hit until the end of data, signal unresolved by returning -1
    return -1, -1


def label_levels(df: pd.DataFrame, logger, progress_every: int = 0) -> Tuple[pd.DataFrame, int]:
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()

    labels_dict = {}
    first_unresolved_at = None

    for tp in cfg.TP_LEVELS:
        level_start = time.time()
        logger.info(f"Poziom {tp}%: start etykietowania na {n:,} wierszach")
        suffix = f"tp{str(tp).replace('.', 'p')}_sl{str(tp).replace('.', 'p')}"
        col = f"label_{suffix}"
        lab = np.zeros(n, dtype=np.int8)

        unresolved_found = False
        # Milestones at 25%, 50%, 75%, 100%
        milestones = sorted(set([
            int(n * 0.25),
            int(n * 0.50),
            int(n * 0.75),
            n,
        ]))
        milestone_ptr = 0
        for i in range(n):
            label, steps = find_first_hit(high, low, open_, close, i, tp)
            if label == -1:
                first_unresolved_at = i if first_unresolved_at is None else min(first_unresolved_at, i)
                unresolved_found = True
                break
            lab[i] = label
            # progress log at milestones only
            if milestone_ptr < len(milestones) and (i + 1) >= milestones[milestone_ptr]:
                elapsed = time.time() - level_start
                done = milestones[milestone_ptr]
                done_ratio = done / n if n > 0 else 1.0
                eta = (elapsed / (i + 1)) * (n - (i + 1)) if i + 1 > 0 else 0.0
                logger.info(
                    f"Poziom {tp}%: {done:,}/{n:,} ({done_ratio*100:.0f}%) elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m"
                )
                milestone_ptr += 1

        labels_dict[col] = lab
        if unresolved_found:
            logger.info(f"Poziom {tp}%: pierwsza nierozstrzygnięta pozycja przy indeksie {first_unresolved_at}")
        logger.info(f"Poziom {tp}%: zakończono w { (time.time()-level_start)/60:.1f}m")

    # If any level had unresolved, truncate all outputs from the earliest unresolved index
    if first_unresolved_at is not None:
        trunc_idx = first_unresolved_at
        out = df.iloc[:trunc_idx].copy()
        for col, arr in labels_dict.items():
            out[col] = arr[:trunc_idx]
    else:
        out = df.copy()
        for col, arr in labels_dict.items():
            out[col] = arr

    return out, (first_unresolved_at if first_unresolved_at is not None else n)


def run(symbol: str, max_rows: int | None = None, start_idx: int = 0):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"dual_labeler start: {symbol}")
    df = load_features(symbol)
    logger.info(f"Wejście: {len(df):,} wierszy, {len(df.columns)} kolumn")
    if start_idx or max_rows:
        end_idx = start_idx + max_rows if max_rows is not None else None
        df = df.iloc[start_idx:end_idx]
        logger.info(f"Zakres roboczy: i=[{start_idx}:{'end' if end_idx is None else end_idx}] -> {len(df):,} wierszy")

    start = time.time()
    out_df, cutoff = label_levels(df, logger)

    # Save
    out_path = cfg.OUTPUT_DIR / f"labeled_{symbol}.feather"
    out_df.reset_index().to_feather(out_path)

    # Metadata
    meta = {
        "symbol": symbol,
        "rows": int(len(out_df)),
        "cols": int(len(out_df.columns)),
        "tp_levels": list(cfg.TP_LEVELS),
        "cutoff_index": int(cutoff),
        "label_columns": [c for c in out_df.columns if c.startswith("label_")],
    }
    meta_path = cfg.METADATA_DIR / f"labeled_{symbol}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    logger.info(f"Zapisano: {out_path}")
    logger.info(f"Ucięto końcówkę od indeksu: {cutoff} (gdy zabrakło danych na rozstrzygnięcie)")
    logger.info(f"Czas: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="dual_labeler - 2-class labeling (LONG/SHORT), symmetric TP=SL, scan until hit")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL, help="Symbol np. BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=None, help="Opcjonalne ograniczenie liczby wierszy (dla testów)")
    parser.add_argument("--start-idx", type=int, default=0, help="Opcjonalny początkowy indeks (dla testów)")
    args = parser.parse_args()
    run(args.symbol, max_rows=args.max_rows, start_idx=args.start_idx)


if __name__ == "__main__":
    main()

