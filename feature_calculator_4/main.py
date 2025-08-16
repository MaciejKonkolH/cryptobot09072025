import sys
from pathlib import Path
import argparse
import json
import pandas as pd

# Ensure project root on sys.path for absolute imports when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_calculator_4 import config as cfg
from feature_calculator_4.logger import setup_logging
from feature_calculator_4.feature_builder import compute_features


def load_merged(symbol: str) -> pd.DataFrame:
    path = cfg.MERGE_DIR / cfg.INPUT_TEMPLATE.format(symbol=symbol)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def save_outputs(symbol: str, df_feat: pd.DataFrame, logger) -> None:
    out_path = cfg.OUTPUT_DIR / f"features_{symbol}.feather"
    cfg.ensure_dirs()
    df_feat.reset_index().to_feather(out_path)
    logger.info(f"Zapisano cechy: {out_path}")

    meta = {
        "symbol": symbol,
        "n_rows": int(len(df_feat)),
        "n_cols": int(len(df_feat.columns)),
        "columns": list(df_feat.columns),
        "channel_windows": cfg.CHANNEL_WINDOWS,
    }
    meta_path = cfg.METADATA_DIR / f"features_{symbol}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Zapisano metadata: {meta_path}")


def run(symbol: str):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"FeatureCalculator4 start: {symbol}")
    df = load_merged(symbol)
    logger.info(f"Wejście: {len(df):,} wierszy, {len(df.columns)} kolumn")

    # Progress info
    if cfg.SHOW_PROGRESS:
        logger.info("Obliczanie cech (paski postępu dla kanałów)...")
    feat = compute_features(df, progress=cfg.SHOW_PROGRESS)
    # Warm-up removal: ensure windows up to 240 ready
    feat = feat.iloc[240:].copy()
    # Carry raw OHLCV for labeler compatibility
    raw_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if raw_cols:
        feat = feat.join(df[raw_cols].iloc[240:], how="left")
    feat = feat.replace([float('inf'), float('-inf')], pd.NA).fillna(0.0)
    logger.info(f"Final: {len(feat):,} wierszy, {len(feat.columns)} kolumn")

    save_outputs(symbol, feat, logger)
    logger.info("FeatureCalculator4 done")


def main():
    parser = argparse.ArgumentParser(description="Feature calculator v4 (agreed features only)")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL, help="Symbol np. BTCUSDT")
    args = parser.parse_args()
    run(args.symbol)


if __name__ == "__main__":
    main()

