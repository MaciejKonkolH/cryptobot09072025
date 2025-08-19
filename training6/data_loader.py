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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    # Optional explicit date filter
    if getattr(cfg, 'DATE_FILTER_START', None) or getattr(cfg, 'DATE_FILTER_END', None):
        start = getattr(cfg, 'DATE_FILTER_START', None)
        end = getattr(cfg, 'DATE_FILTER_END', None)
        if start is not None:
            df = df[df.index >= pd.to_datetime(start, utc=True)]
        if end is not None:
            df = df[df.index <= pd.to_datetime(end, utc=True)]

    # Optional alignment to training3 reference file time range
    if getattr(cfg, 'ALIGN_TO_TRAINING3_RANGE', False):
        ref_path = getattr(cfg, 'TRAINING3_REF_FILE', None)
        if ref_path is not None and Path(ref_path).exists():
            ref = pd.read_feather(ref_path)
            if "timestamp" in ref.columns:
                ref_ts = pd.to_datetime(ref["timestamp"], utc=True)
                ref_min = ref_ts.min()
                ref_max = ref_ts.max()
                before = len(df)
                df = df.loc[(df.index >= ref_min) & (df.index <= ref_max)]
                logger.info(f"Docięto zakres do training3: {ref_min} .. {ref_max} (pozostawiono {len(df)}/{before})")
        else:
            logger.warning("ALIGN_TO_TRAINING3_RANGE=True, ale plik referencyjny nie istnieje.")
    return df


def split_scale(df: pd.DataFrame):
    # Infer label columns present in the dataframe, tolerant to naming variants (e.g., tp1 vs tp1p0)
    def _infer_label_columns(df_cols):
        inferred = []
        for tp, sl in cfg.TP_SL_LEVELS:
            tp_str_primary = str(tp).replace('.', 'p')
            sl_str = str(sl).replace('.', 'p')
            candidates = [
                f"label_tp{tp_str_primary}_sl{sl_str}",
            ]
            # Alternative for integer TP like 1.0 -> tp1
            if float(tp).is_integer():
                candidates.append(f"label_tp{int(tp)}_sl{sl_str}")
            # Pick the first that exists
            chosen = None
            for name in candidates:
                if name in df_cols:
                    chosen = name
                    break
            if chosen is None:
                raise KeyError(f"Brak kolumny etykiety dla poziomu TP={tp}, SL={sl}. Próbowano: {candidates}")
            inferred.append(chosen)
        return inferred

    y_cols = _infer_label_columns(df.columns)
    # Expose resolved label columns for downstream components
    try:
        setattr(cfg, 'RESOLVED_LABEL_COLUMNS', y_cols)
    except Exception:
        pass

    # Detect features: numeric only, drop labels and OHLC
    drop_cols = set(y_cols + ["open", "high", "low", "close", "volume"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in numeric_cols if c not in drop_cols]

    # Feature selection modes
    mode = getattr(cfg, 'FEATURE_SELECTION_MODE', 'all')
    if mode == 't3_37' or getattr(cfg, 'USE_TRAINING3_FEATURE_WHITELIST', False):
        # Whitelist derived from training3/config.py FEATURES mapping to labeler3 columns
        whitelist = [
            # Trend/position (examples based on available L3 columns)
            'price_trend_30m', 'price_trend_2h', 'price_trend_6h',
            'price_strength', 'price_consistency_score',
            'price_vs_ma_60', 'price_vs_ma_240', 'ma_trend', 'price_volatility_rolling',
            # Volume features
            'volume_trend_1h', 'volume_intensity', 'volume_volatility_rolling',
            'volume_price_correlation', 'volume_momentum',
            # OB (relative)
            'spread_tightness', 'depth_ratio_s1', 'depth_ratio_s2', 'depth_momentum',
            # Market regime
            'market_trend_strength', 'market_trend_direction', 'market_choppiness',
            'bollinger_band_width', 'market_regime',
            # Volatility
            'volatility_regime', 'volatility_percentile', 'volatility_persistence',
            'volatility_momentum', 'volatility_of_volatility', 'volatility_term_structure',
            # Imbalance/flow
            'volume_imbalance', 'weighted_volume_imbalance', 'volume_imbalance_trend',
            'price_pressure', 'weighted_price_pressure', 'price_pressure_momentum',
            'order_flow_imbalance', 'order_flow_trend',
        ]
        present = [c for c in whitelist if c in X_cols]
        missing = [c for c in whitelist if c not in X_cols]
        if len(present) == 0:
            logger.warning("Whitelist 37 cech z training3 nie pokrywa się z kolumnami X. Pozostawiam pełny zestaw.")
        else:
            if missing:
                logger.info(f"Brakujące z whitelisty ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
            logger.info(f"Używam whitelisty training3: {len(present)} cech")
            X_cols = present
    elif mode in ('custom', 'custom_strict'):
        custom = getattr(cfg, 'CUSTOM_FEATURE_LIST', [])
        present = [c for c in custom if c in X_cols]
        missing = [c for c in custom if c not in X_cols]
        if mode == 'custom_strict':
            if missing:
                raise KeyError(
                    f"Brak wymaganych cech w CUSTOM_FEATURE_LIST (tryb strict). Brakuje: {missing}"
                )
            # Enforce exact order as provided by the user
            X_cols = list(custom)
            logger.info(f"Używam CUSTOM_FEATURE_LIST (strict): {len(X_cols)} cech")
        else:
            if missing:
                logger.warning(f"Brakujące z CUSTOM_FEATURE_LIST ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
            if present:
                logger.info(f"Używam CUSTOM_FEATURE_LIST: {len(present)} cech")
                X_cols = present

    X = df[X_cols]
    y = df[y_cols]

    total = len(df)
    # Use configurable splits from config.py
    val_ratio = float(getattr(cfg, 'VALIDATION_SPLIT', 0.15))
    test_ratio = float(getattr(cfg, 'TEST_SPLIT', 0.15))
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"VALIDATION_SPLIT/TEST_SPLIT must be in [0,1). Got: {val_ratio}, {test_ratio}")
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0.0:
        raise ValueError(
            f"Invalid splits: train portion <= 0. Received VALIDATION_SPLIT={val_ratio}, TEST_SPLIT={test_ratio}"
        )
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

