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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def split_scale(df: pd.DataFrame):
    # Detect features: numeric only, drop labels and OHLC
    drop_cols = set(cfg.LABEL_COLUMNS + ["open", "high", "low", "close", "volume"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in numeric_cols if c not in drop_cols]
    y_cols = cfg.LABEL_COLUMNS

    X = df[X_cols]
    y = df[y_cols]

    total = len(df)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    # Clean inf/NaN before scaling
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

