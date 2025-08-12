import sys
from pathlib import Path
import argparse
import json
import pandas as pd

# Ensure project root on sys.path for absolute imports when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_calculator3 import config as cfg
from feature_calculator3.logger import setup_logging
from feature_calculator3.feature_builder import (
    compute_core_features,
    add_short_lags,
    add_time_binning,
    finalize,
)


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
    df_feat.reset_index().to_feather(out_path)
    logger.info(f"Zapisano cechy: {out_path}")

    meta = {
        "symbol": symbol,
        "n_rows": int(len(df_feat)),
        "n_cols": int(len(df_feat.columns)),
        "columns": list(df_feat.columns),
        "hybrid": cfg.USE_HYBRID,
        "short_lags": cfg.SHORT_LAGS,
        "bin_buckets": cfg.BIN_BUCKETS,
    }
    meta_path = cfg.METADATA_DIR / f"features_{symbol}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Zapisano metadata: {meta_path}")


def run(symbol: str):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"FeatureCalculator3 start: {symbol}")
    df = load_merged(symbol)
    logger.info(f"Wejście: {len(df):,} wierszy, {len(df.columns)} kolumn")

    # carry raw columns needed for labeling
    raw_cols = [c for c in cfg.RAW_COLUMNS if c in df.columns]
    raw_df = df[raw_cols].copy()

    base = compute_core_features(df)
    logger.info(f"Bazowe cechy: {len(base.columns)} kolumn")

    if cfg.USE_HYBRID:
        base = add_short_lags(base)
        base = add_time_binning(base)
        logger.info("Dodano krótkie lagi i binning czasu (hybryda)")

    # merge raw columns back for labeler compatibility
    merged = base.join(raw_df, how="left")
    final = finalize(merged)
    logger.info(f"Final: {len(final):,} wierszy, {len(final.columns)} kolumn")

    save_outputs(symbol, final, logger)
    logger.info("FeatureCalculator3 done")


def main():
    parser = argparse.ArgumentParser(description="Feature calculator v3 (hybrid)")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL, help="Symbol np. BTCUSDT")
    args = parser.parse_args()
    run(args.symbol)


if __name__ == "__main__":
    main()

