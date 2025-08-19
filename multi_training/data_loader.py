from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from . import config as cfg
from .utils import setup_logging


logger = setup_logging()


def _infer_label_columns(df_cols: List[str]) -> List[str]:
    inferred = []
    for tp, sl in cfg.TP_SL_LEVELS:
        tp_str_primary = str(tp).replace('.', 'p')
        sl_str = str(sl).replace('.', 'p')
        candidates = [f"label_tp{tp_str_primary}_sl{sl_str}"]
        if float(tp).is_integer():
            candidates.append(f"label_tp{int(tp)}_sl{sl_str}")
        chosen = None
        for name in candidates:
            if name in df_cols:
                chosen = name
                break
        if chosen is None:
            raise KeyError(f"Brak kolumny etykiety dla poziomu TP={tp}, SL={sl}. Próbowano: {candidates}")
        inferred.append(chosen)
    return inferred


def _load_one(symbol: str) -> pd.DataFrame:
    path = cfg.INPUT_DIR / cfg.INPUT_TEMPLATE.format(symbol=symbol)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {path}")
    df = pd.read_feather(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    # Date filter
    if cfg.DATE_FILTER_START:
        df = df[df.index >= pd.to_datetime(cfg.DATE_FILTER_START, utc=True)]
    if cfg.DATE_FILTER_END:
        df = df[df.index <= pd.to_datetime(cfg.DATE_FILTER_END, utc=True)]
    # Keep symbol column
    df = df.copy()
    df['__symbol__'] = symbol
    return df


def load_multi(symbols: List[str]) -> pd.DataFrame:
    dfs = [_load_one(sym) for sym in symbols]
    # Intersect columns across symbols to avoid missing features
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    # Ensure labels are present (from any df)
    label_cols = _infer_label_columns(list(common_cols))
    # Build final ordered column set: labels + numeric features
    # numeric only, exclude OHLC
    drop_cols = set(['open', 'high', 'low', 'close', 'volume'])
    numeric_cols = [c for c in common_cols if c not in label_cols and c not in drop_cols]
    # Apply feature selection analogous to training5
    mode = getattr(cfg, 'FEATURE_SELECTION_MODE', 'all')
    X_cols = list(numeric_cols)
    if mode == 't3_37' or getattr(cfg, 'USE_TRAINING3_FEATURE_WHITELIST', False):
        whitelist = [
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
            'price_strength', 'price_consistency_score',
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling', 'volume_price_correlation', 'volume_momentum',
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
            'market_trend_strength', 'market_trend_direction', 'market_choppiness', 'bollinger_band_width', 'market_regime',
            'volatility_regime', 'volatility_percentile', 'volatility_persistence', 'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
            'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum', 'order_flow_imbalance', 'order_flow_trend',
        ]
        X_cols = [c for c in whitelist if c in X_cols]
    elif mode in ('custom', 'custom_strict'):
        custom = getattr(cfg, 'CUSTOM_FEATURE_LIST', [])
        present = [c for c in custom if c in X_cols]
        missing = [c for c in custom if c not in X_cols]
        if mode == 'custom_strict' and missing:
            raise KeyError(f"Brak wymaganych cech (strict) w multi: {missing}")
        if present:
            X_cols = list(present)

    # Add one-hot symbol columns (not subjected to whitelist)
    if cfg.ADD_SYMBOL_ONEHOTS:
        for sym in symbols:
            col = f"sym_{sym}"
            if col not in X_cols:
                X_cols.append(col)

    # Concat aligned dfs, keep only columns needed
    keep_cols = label_cols + X_cols + ['__symbol__']
    sliced = []
    for d in dfs:
        # build one-hots if needed
        if cfg.ADD_SYMBOL_ONEHOTS:
            for sym in symbols:
                d[f"sym_{sym}"] = 1.0 if d['__symbol__'].iloc[0] == sym else 0.0
        sliced.append(d[keep_cols])
    df = pd.concat(sliced, axis=0).sort_index()

    # Split per symbol then concatenate splits to avoid leakage across symbols
    all_train = []
    all_val = []
    all_test = []
    test_symbol_series_parts = []
    for sym in symbols:
        part = df[df['__symbol__'] == sym]
        part = part.drop(columns=['__symbol__'])
        total = len(part)
        train_end = int((1.0 - cfg.VALIDATION_SPLIT - cfg.TEST_SPLIT) * total)
        val_end = int((1.0 - cfg.TEST_SPLIT) * total)
        all_train.append(part.iloc[:train_end])
        all_val.append(part.iloc[train_end:val_end])
        test_part = part.iloc[val_end:]
        all_test.append(test_part)
        test_symbol_series_parts.append(pd.Series(sym, index=test_part.index))

    train_df = pd.concat(all_train, axis=0)
    val_df = pd.concat(all_val, axis=0)
    test_df = pd.concat(all_test, axis=0)
    test_symbols = pd.concat(test_symbol_series_parts, axis=0).reindex(test_df.index)

    # Separate X/y
    X_train = train_df[X_cols]
    y_train = train_df[label_cols]
    X_val = val_df[X_cols]
    y_val = val_df[label_cols]
    X_test = test_df[X_cols]
    y_test = test_df[label_cols]

    # Clean inf/NaN before scaling using train medians only
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

    return (X_train, X_val, X_test, y_train, y_val, y_test, scaler, X_cols, label_cols, test_symbols)

