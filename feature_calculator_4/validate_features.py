import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_calculator_4 import config as cfg
from feature_calculator_4.logger import setup_logging


def load_features(symbol: str) -> pd.DataFrame:
    path = cfg.OUTPUT_DIR / f"features_{symbol}.feather"
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku cech: {path}")
    df = pd.read_feather(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def validate(df: pd.DataFrame) -> dict:
    result = {}

    # Base info
    result["rows"] = int(len(df))
    result["cols"] = int(len(df.columns))

    # NaN/Inf checks
    is_inf = np.isinf(df.select_dtypes(include=[np.number]))
    nan_counts = df.isna().sum()
    inf_counts = is_inf.sum()

    result["columns_with_nan"] = [c for c, v in nan_counts.items() if v > 0]
    result["columns_with_inf"] = [c for c, v in inf_counts.items() if v > 0]
    result["rows_with_any_nan"] = int(df.isna().any(axis=1).sum())
    result["rows_with_any_inf"] = int(is_inf.any(axis=1).sum())

    # Constant/All-NaN/Zero-Variance
    const_cols = []
    allnan_cols = []
    zero_var_cols = []
    num_df = df.select_dtypes(include=[np.number])

    for c in df.columns:
        if df[c].isna().all():
            allnan_cols.append(c)
        elif df[c].nunique(dropna=True) == 1:
            const_cols.append(c)

    for c in num_df.columns:
        s = num_df[c]
        if s.var(ddof=0) == 0:
            zero_var_cols.append(c)

    result["allnan_columns"] = allnan_cols
    result["constant_columns"] = const_cols
    result["zero_variance_columns"] = zero_var_cols

    # Timestamp continuity (1-minute steps)
    if isinstance(df.index, pd.DatetimeIndex):
        diffs = df.index.to_series().diff().dropna()
        one_min = pd.Timedelta(minutes=1)
        result["timestamp_min"] = df.index.min().isoformat()
        result["timestamp_max"] = df.index.max().isoformat()
        result["missing_minutes"] = int((diffs != one_min).sum())
    else:
        result["timestamp_min"] = None
        result["timestamp_max"] = None
        result["missing_minutes"] = None

    # Basic sanity: numeric columns finite ratio
    total_numeric = int(num_df.shape[1])
    finite_mask = np.isfinite(num_df)
    result["numeric_columns"] = total_numeric
    result["numeric_rows_with_all_finite"] = int(finite_mask.all(axis=1).sum())

    return result


def save_report(symbol: str, report: dict, logger) -> Path:
    out = cfg.METADATA_DIR / f"validate_features_{symbol}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Raport walidacji zapisany: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Walidacja pliku cech (feature_calculator_4)")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL, help="Symbol np. BTCUSDT")
    args = parser.parse_args()

    logger = setup_logging()
    cfg.ensure_dirs()

    try:
        df = load_features(args.symbol)
        logger.info(f"Wczytano: {len(df):,} wierszy, {len(df.columns)} kolumn")
        report = validate(df)
        save_report(args.symbol, report, logger)

        logger.info(
            f"NaN cols: {len(report['columns_with_nan'])}, Inf cols: {len(report['columns_with_inf'])}, "
            f"All-NaN: {len(report['allnan_columns'])}, Const: {len(report['constant_columns'])}, ZeroVar: {len(report['zero_variance_columns'])}"
        )
        if report.get("missing_minutes") is not None:
            logger.info(f"Braki minut (nieciągły timestamp): {report['missing_minutes']}")
    except Exception as e:
        logger.error(f"Błąd walidacji: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

