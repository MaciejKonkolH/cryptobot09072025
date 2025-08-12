import sys
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd

# Ensure project root on sys.path for absolute imports when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeler5 import config as cfg
from labeler5.logger import setup_logging


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
    for c in ["high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Brakuje kolumny '{c}' w pliku cech")
    return df


def label_single_level(high: np.ndarray, low: np.ndarray, close: np.ndarray, tp_pct: float, sl_pct: float,
                       future_window: int, logger) -> tuple[np.ndarray, int]:
    """Label each time step as if opening a position with TP and SL.

    Strategy:
      - For LONG: find the first event among TP (high >= entry*(1+tp)) and SL (low <= entry*(1-sl)).
      - For SHORT: find the first event among TP (low <= entry*(1-tp)) and SL (high >= entry*(1+sl)).
      - Choose direction where TP happens before SL. If both directions achieve TP-before-SL,
        choose the one with earlier TP time; ties -> NEUTRAL. Intrabar TP&SL on the same bar count as conflict
        and are treated as no clear win for that direction.
    """
    n = len(close)
    labels = np.full(n, 2, dtype=np.int8)  # 0=LONG,1=SHORT,2=NEUTRAL
    conflicts = 0

    fw = future_window

    def first_event_long(i: int, entry: float) -> tuple[str | None, int | None, bool]:
        long_tp = entry * (1 + tp_pct / 100.0)
        long_sl = entry * (1 - sl_pct / 100.0)
        for j in range(1, fw + 1):
            h = high[i + j]
            l = low[i + j]
            tp_hit = h >= long_tp
            sl_hit = l <= long_sl
            if tp_hit and not sl_hit:
                return "TP", j, False
            if sl_hit and not tp_hit:
                return "SL", j, False
            if tp_hit and sl_hit:
                # Intrabar ambiguity: TP and SL in the same bar
                return "BOTH", j, True
        return None, None, False

    def first_event_short(i: int, entry: float) -> tuple[str | None, int | None, bool]:
        short_tp = entry * (1 - tp_pct / 100.0)
        short_sl = entry * (1 + sl_pct / 100.0)
        for j in range(1, fw + 1):
            h = high[i + j]
            l = low[i + j]
            tp_hit = l <= short_tp
            sl_hit = h >= short_sl
            if tp_hit and not sl_hit:
                return "TP", j, False
            if sl_hit and not tp_hit:
                return "SL", j, False
            if tp_hit and sl_hit:
                return "BOTH", j, True
        return None, None, False

    # Drop last fw rows later
    for i in range(0, n - fw):
        entry = close[i]

        resL, tL, confL = first_event_long(i, entry)
        resS, tS, confS = first_event_short(i, entry)

        # Count intrabar conflicts
        if confL:
            conflicts += 1
        if confS:
            conflicts += 1

        long_wins = (resL == "TP")
        short_wins = (resS == "TP")

        if long_wins and not short_wins:
            labels[i] = 0
        elif short_wins and not long_wins:
            labels[i] = 1
        elif long_wins and short_wins:
            # Earlier TP decides; exact tie -> NEUTRAL and count as conflict
            if tL is not None and tS is not None:
                if tL < tS:
                    labels[i] = 0
                elif tS < tL:
                    labels[i] = 1
                else:
                    labels[i] = 2
                    conflicts += 1
            else:
                labels[i] = 2
        else:
            labels[i] = 2

    # Last fw rows removed by caller
    return labels, conflicts


def run(symbol: str):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"Labeler5 start: {symbol}")
    df = load_features(symbol)
    n = len(df)
    logger.info(f"Wejście: {n:,} wierszy, {len(df.columns)} kolumn")

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    labels_dict = {}
    total_conflicts = 0

    start = time.time()
    for idx, (tp, sl) in enumerate(cfg.TP_SL_LEVELS, 1):
        level_desc = f"TP={tp}%,SL={sl}%"
        logger.info(f"Poziom {idx}/{len(cfg.TP_SL_LEVELS)}: {level_desc}")
        lab, conflicts = label_single_level(high, low, close, tp, sl, cfg.FUTURE_WINDOW, logger)
        total_conflicts += conflicts
        suffix = f"tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"
        col = f"label_{suffix}"
        labels_dict[col] = lab
        logger.info(f"  Konflikty (intrabar/remeis): {conflicts:,}")

    # Assemble output (drop last FW rows)
    fw = cfg.FUTURE_WINDOW
    out = df.iloc[: n - fw].copy()
    for col, arr in labels_dict.items():
        out[col] = arr[: n - fw]

    # Save
    out_path = cfg.OUTPUT_DIR / f"labeled_{symbol}.feather"
    out.reset_index().to_feather(out_path)

    meta = {
        "symbol": symbol,
        "rows": int(len(out)),
        "cols": int(len(out.columns)),
        "levels": len(cfg.TP_SL_LEVELS),
        "future_window": cfg.FUTURE_WINDOW,
        "total_conflicts": int(total_conflicts),
        "label_columns": list(labels_dict.keys()),
    }
    meta_path = cfg.METADATA_DIR / f"labeled_{symbol}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    logger.info(f"Zapisano: {out_path}")
    logger.info(f"Konflikty łącznie (losowe): {total_conflicts:,}")
    logger.info(f"Czas: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Labeler5 - 3-class labeling for feature_calculator3 data")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL, help="Symbol np. BTCUSDT")
    args = parser.parse_args()
    run(args.symbol)


if __name__ == "__main__":
    main()

