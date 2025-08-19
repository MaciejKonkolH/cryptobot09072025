import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from . import config as cfg
from .utils import setup_logging


logger = setup_logging()


def load_labeled(symbol: str) -> pd.DataFrame:
    path = cfg.INPUT_DIR / cfg.INPUT_TEMPLATE.format(symbol=symbol)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {path}")
    df = pd.read_feather(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')
        df = df.set_index("timestamp")
    return df


def split_scale(df: pd.DataFrame):
    # Infer label columns present in the dataframe (dual_labeler format)
    def _infer_label_columns(df_cols):
        inferred = []
        for tp, sl in cfg.TP_SL_LEVELS:
            tp_str_primary = str(tp).replace('.', 'p')
            sl_str = str(sl).replace('.', 'p')
            name = f"label_tp{tp_str_primary}_sl{sl_str}"
            if name not in df_cols:
                raise KeyError(f"Brak kolumny etykiety dla poziomu TP={tp}, SL={sl}. Szukano: {name}")
            inferred.append(name)
        return inferred

    y_cols = _infer_label_columns(df.columns)
    try:
        setattr(cfg, 'RESOLVED_LABEL_COLUMNS', y_cols)
    except Exception:
        pass

    # Detect features: numeric only, drop labels and OHLC
    drop_cols = set(y_cols + ["open", "high", "low", "close", "volume"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in numeric_cols if c not in drop_cols]

    # Feature selection
    mode = getattr(cfg, 'FEATURE_SELECTION_MODE', 'all')
    if mode in ('custom', 'custom_strict'):
        custom = getattr(cfg, 'CUSTOM_FEATURE_LIST', [])
        present = [c for c in custom if c in X_cols]
        missing = [c for c in custom if c not in X_cols]
        if mode == 'custom_strict':
            if missing:
                raise KeyError(
                    f"Brak wymaganych cech w CUSTOM_FEATURE_LIST (tryb strict). Brakuje: {missing}"
                )
            X_cols = list(custom)
            logger.info(f"Używam CUSTOM_FEATURE_LIST (strict): {len(X_cols)} cech")
        else:
            if present:
                logger.info(f"Używam CUSTOM_FEATURE_LIST: {len(present)} cech")
                X_cols = present

    X = df[X_cols]
    y = df[y_cols]

    total = len(df)
    val_ratio = float(getattr(cfg, 'VALIDATION_SPLIT', 0.15))
    test_ratio = float(getattr(cfg, 'TEST_SPLIT', 0.15))
    train_ratio = 1.0 - val_ratio - test_ratio
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)
    try:
        logger.info(
            f"Splits -> train: {train_ratio:.2%} ({train_end}), val: {val_ratio:.2%} ({val_end-train_end}), test: {test_ratio:.2%} ({total-val_end})"
        )
    except Exception:
        pass

    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    # Clean/scale
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    train_median = X_train.median()
    X_train = X_train.fillna(train_median)
    X_val = X_val.fillna(train_median)
    X_test = X_test.fillna(train_median)

    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_cols)
    X_val = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_cols)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_cols)

    return (X_train, X_val, X_test, y_train, y_val, y_test, scaler, X_cols)

