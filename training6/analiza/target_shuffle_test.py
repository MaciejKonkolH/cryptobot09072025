import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure 'crypto' project root is in sys.path so that `import training6` works when run by path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse training6 pipeline
from training6 import config as cfg
from training6.data_loader import load_labeled, split_scale
from training6.model_builder import BinaryDirectionalXGB


def ensure_run_dir(symbol: str) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = cfg.get_report_dir(symbol) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def compute_confusion(y_true: np.ndarray, p_long: np.ndarray, p_short: np.ndarray, thr_long: float, thr_short: float) -> np.ndarray:
    pred = np.full_like(p_long, fill_value=2, dtype=int)
    long_m = (p_long >= thr_long) & (p_long > p_short)
    short_m = (p_short >= thr_short) & (p_short > p_long)
    pred[long_m] = 0
    pred[short_m] = 1
    # cm: rows=actual (0,1,2), cols=pred (0,1,2)
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y_true.astype(int), pred.astype(int)):
        if 0 <= a <= 2 and 0 <= b <= 2:
            cm[a, b] += 1
    return cm


def weighted_trade_accuracy(cm: np.ndarray) -> float:
    col_long = cm[0, 0] + cm[1, 0] + cm[2, 0]
    col_short = cm[0, 1] + cm[1, 1] + cm[2, 1]
    p_long = (cm[0, 0] / col_long) if col_long > 0 else None
    p_short = (cm[1, 1] / col_short) if col_short > 0 else None
    num = 0.0
    den = 0
    if col_long and p_long is not None:
        num += col_long * p_long
        den += col_long
    if col_short and p_short is not None:
        num += col_short * p_short
        den += col_short
    return float(num / den) if den > 0 else 0.0


def profit_threshold_for_col(label_cols: list[str], col: str) -> float:
    idx = label_cols.index(col)
    tp, sl = cfg.TP_SL_LEVELS[idx]
    return float(sl) / float(tp + sl)


def tune_thresholds_on_validation(y_val_col: pd.Series, p_long_v: np.ndarray, p_short_v: np.ndarray, profit_thr: float) -> tuple[float, float]:
    grid = [0.50, 0.55, 0.60, 0.65, 0.70]
    best = (0.50, 0.50, -1e9, 0.0)  # thr_long, thr_short, safety, wacc
    yv = y_val_col.values.astype(int)
    for tL in grid:
        for tS in grid:
            cm = compute_confusion(yv, p_long_v, p_short_v, tL, tS)
            wacc = weighted_trade_accuracy(cm)
            safety = wacc - profit_thr
            if (safety > best[2]) or (safety == best[2] and wacc > best[3]):
                best = (tL, tS, safety, wacc)
    return float(best[0]), float(best[1])


def block_shuffle_series(y: pd.Series, freq: str = 'W-MON', rng: np.random.Generator | None = None) -> pd.Series:
    rng = rng or np.random.default_rng()
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex for block shuffle")
    groups = y.groupby(y.index.to_period(freq))
    parts = []
    for _, g in groups:
        vals = g.values.copy()
        rng.shuffle(vals)
        parts.append(pd.Series(vals, index=g.index))
    return pd.concat(parts).sort_index()


def run_shuffle_test(symbol: str, n_iters: int, block_freq: str, out_dir: Path, levels_limit: int | None = None, fast_mode: bool = True):
    df = load_labeled(symbol)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)
    label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
    if levels_limit is not None:
        label_cols = label_cols[:int(levels_limit)]

    # Train on true labels
    speed_kwargs = {"n_estimators": 150, "early_stopping_rounds": 10, "verbose_eval": 0} if fast_mode else {}
    model_true = BinaryDirectionalXGB(feat_names, **speed_kwargs)
    model_true.fit(X_train, y_train, X_val, y_val)

    true_scores = {}
    thresholds_map = {}
    for col in label_cols:
        proba_v = model_true.predict_proba_level(X_val, col)
        pL_v = proba_v['P_long'].values
        pS_v = proba_v['P_short'].values
        profit_thr = profit_threshold_for_col(label_cols, col)
        tL, tS = tune_thresholds_on_validation(y_val[col], pL_v, pS_v, profit_thr)
        thresholds_map[col] = {"thr_long": tL, "thr_short": tS}

        proba_t = model_true.predict_proba_level(X_test, col)
        cm_true = compute_confusion(y_test[col].values.astype(int), proba_t['P_long'].values, proba_t['P_short'].values, tL, tS)
        true_scores[col] = weighted_trade_accuracy(cm_true)

    # Permutation distribution
    rng = np.random.default_rng()
    perm_scores = {col: [] for col in label_cols}
    for itr in range(int(n_iters)):
        print(f"[perm] {itr+1}/{n_iters}")
        # Build permuted y DataFrames (independently per set and per column)
        ytr_p = y_train.copy()
        yva_p = y_val.copy()
        yte_p = y_test.copy()
        for col in label_cols:
            ytr_p[col] = block_shuffle_series(y_train[col], freq=block_freq, rng=rng)
            yva_p[col] = block_shuffle_series(y_val[col], freq=block_freq, rng=rng)
            yte_p[col] = block_shuffle_series(y_test[col], freq=block_freq, rng=rng)

        model_p = BinaryDirectionalXGB(feat_names, **speed_kwargs)
        model_p.fit(X_train, ytr_p, X_val, yva_p)
        for col in label_cols:
            proba_v = model_p.predict_proba_level(X_val, col)
            pL_v = proba_v['P_long'].values
            pS_v = proba_v['P_short'].values
            profit_thr = profit_threshold_for_col(label_cols, col)
            tL, tS = tune_thresholds_on_validation(yva_p[col], pL_v, pS_v, profit_thr)

            proba_t = model_p.predict_proba_level(X_test, col)
            cm_p = compute_confusion(yte_p[col].values.astype(int), proba_t['P_long'].values, proba_t['P_short'].values, tL, tS)
            perm_scores[col].append(weighted_trade_accuracy(cm_p))

    # Summarize p-values
    summary = {}
    for col in label_cols:
        arr = np.array(perm_scores[col], dtype=float)
        true_val = float(true_scores[col])
        p_value = float((np.sum(arr >= true_val) + 1) / (len(arr) + 1))  # add-1 smoothing
        idx = label_cols.index(col)
        tp, sl = cfg.TP_SL_LEVELS[idx]
        summary[col] = {
            "tp": float(tp),
            "sl": float(sl),
            "true_wacc": true_val,
            "perm_mean": float(np.mean(arr)) if len(arr) else None,
            "perm_std": float(np.std(arr)) if len(arr) else None,
            "p_value": p_value,
            "n_iters": int(n_iters),
            "thresholds": thresholds_map.get(col, {}),
            "block_freq": block_freq,
        }

    with open(out_dir / f"shuffle_test_summary_{symbol}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Also save per-level samples
    for col in label_cols:
        if perm_scores[col]:
            dfp = pd.DataFrame({"perm_wacc": perm_scores[col]})
            dfp.to_csv(out_dir / f"shuffle_test_perm_{symbol}_{col}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Target-shuffle significance test (training6)")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--n-iters", type=int, default=50, help="Liczba permutacji (domyślnie 50)")
    parser.add_argument("--block-freq", default='W-MON', help="Częstotliwość bloków do tasowania (np. W-MON, D)")
    parser.add_argument("--levels-limit", type=int, default=5, help="Ile pierwszych poziomów TP/SL testować (domyślnie 5 dla szybkości)")
    parser.add_argument("--full", action='store_true', help="Wyłącz tryb szybki (wolniej, dokładniej)")
    args = parser.parse_args()

    cfg.ensure_dirs()
    out_dir = ensure_run_dir(args.symbol)
    run_shuffle_test(
        args.symbol,
        n_iters=int(args.n_iters),
        block_freq=str(args.block_freq),
        out_dir=out_dir,
        levels_limit=(None if args.levels_limit is None else int(args.levels_limit)),
        fast_mode=(not args.full),
    )


if __name__ == "__main__":
    main()

