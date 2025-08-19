import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import time

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training5 import config as cfg
from training5.data_loader import load_labeled, split_scale
from training5.model_builder import MultiOutputXGB


def ensure_run_dir(symbol: str) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = cfg.get_report_dir(symbol) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def compute_wacc_from_cm(cm: np.ndarray) -> float:
    col_long = cm[0, 0] + cm[1, 0] + cm[2, 0]
    col_short = cm[0, 1] + cm[1, 1] + cm[2, 1]
    p_long = (cm[0, 0] / col_long) if col_long > 0 else None
    p_short = (cm[1, 1] / col_short) if col_short > 0 else None
    num, den = 0.0, 0
    if col_long and p_long is not None:
        num += col_long * p_long
        den += col_long
    if col_short and p_short is not None:
        num += col_short * p_short
        den += col_short
    return float(num / den) if den > 0 else 0.0


def predict_dir_with_thresholds(proba: np.ndarray, thr_long: float, thr_short: float) -> np.ndarray:
    # proba shape: (N, 3) for classes [LONG, SHORT, NEUTRAL]
    p_long = proba[:, 0]
    p_short = proba[:, 1]
    pred = np.full(proba.shape[0], 2, dtype=int)  # default NEUTRAL
    pred[p_long >= thr_long] = 0
    # SHORT only where not already LONG
    mask_short = (pred != 0) & (p_short >= thr_short)
    pred[mask_short] = 1
    return pred


def confusion_matrix_3(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(actual.astype(int), pred.astype(int)):
        cm[a, b] += 1
    return cm


def _parse_tp_sl_from_label(label_name: str) -> tuple[float, float] | tuple[None, None]:
    try:
        # label_tp1p4_sl0p7 -> (1.4, 0.7)
        parts = label_name.split('_')
        tp_raw = parts[1]  # e.g., 'tp1p4'
        sl_raw = parts[2]  # e.g., 'sl0p7'
        tp = float(tp_raw.replace('tp', '').replace('p', '.'))
        sl = float(sl_raw.replace('sl', '').replace('p', '.'))
        return tp, sl
    except Exception:
        return None, None


def run_shuffle_test(
    symbol: str,
    n_iters: int,
    block_freq: str,
    out_dir: Path,
    fast_mode: bool = True,
    levels_filter: list[str] | None = None,
    thr_long: float = 0.5,
    thr_short: float = 0.5,
    test_window: str | None = None,
):
    df = load_labeled(symbol)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)
    label_cols_all = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
    if levels_filter:
        # Accept both space-separated and comma-separated lists
        norm = []
        for item in levels_filter:
            norm.extend([s for s in item.split(',') if s])
        levels_filter_set = set(norm)
        label_cols = [c for c in label_cols_all if c in levels_filter_set]
        if not label_cols:
            raise ValueError(f"Żaden z poziomów nie pasuje do RESOLVED_LABEL_COLUMNS. Podano: {sorted(levels_filter_set)}")
        print(f"Używam filtrowanych poziomów ({len(label_cols)}): {label_cols}")
    else:
        label_cols = list(label_cols_all)

    # Optionally limit levels in the underlying model by overriding RESOLVED_LABEL_COLUMNS during fit
    old_resolved = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', None)
    try:
        setattr(cfg, 'RESOLVED_LABEL_COLUMNS', label_cols)

        # Train on true labels
        model_true = MultiOutputXGB(feat_names)
        # fast_mode: reduce estimators and verbose
        if fast_mode:
            cfg.XGB_N_ESTIMATORS = min(200, getattr(cfg, 'XGB_N_ESTIMATORS', 400))
            cfg.XGB_EARLY_STOPPING_ROUNDS = min(10, getattr(cfg, 'XGB_EARLY_STOPPING_ROUNDS', 20))
            cfg.TRAIN_VERBOSE_EVAL = 0
        model_true.fit(X_train, y_train, X_val, y_val)
        # Populate internal probas_ for test set
        _ = model_true.predict(X_test)
    finally:
        if old_resolved is not None:
            setattr(cfg, 'RESOLVED_LABEL_COLUMNS', old_resolved)
    # Optional per-window grouping for test set
    windows = None
    if test_window is not None and isinstance(X_test.index, pd.DatetimeIndex):
        periods = X_test.index.to_period(test_window)
        windows = sorted(periods.unique())

    summary = {}
    summary_windows: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for idx, col in enumerate(label_cols):
        proba = model_true.probas_[col]  # (N,3)
        pred = predict_dir_with_thresholds(proba, thr_long=thr_long, thr_short=thr_short)
        cm = confusion_matrix_3(y_test[col].values.astype(int), pred)
        wacc = compute_wacc_from_cm(cm)
        tp_val, sl_val = _parse_tp_sl_from_label(col)
        summary[col] = {
            "tp": float(tp_val) if tp_val is not None else None,
            "sl": float(sl_val) if sl_val is not None else None,
            "true_wacc": float(wacc),
            "p_value": None,  # filled later
            "perm_mean": None,
            "perm_std": None,
            "n_iters": int(n_iters),
            "block_freq": block_freq,
            "thr_long": float(thr_long),
            "thr_short": float(thr_short),
            "test_window": test_window,
        }
        if windows is not None:
            summary_windows[col] = {}
            periods = X_test.index.to_period(test_window)  # type: ignore[arg-type]
            for w in windows:
                mask = periods == w
                pred_w = pred[mask]
                y_w = y_test[col].values.astype(int)[mask]
                cm_w = confusion_matrix_3(y_w, pred_w)
                wacc_w = compute_wacc_from_cm(cm_w)
                summary_windows[col][str(w)] = {
                    "true_wacc": float(wacc_w),
                    "perm_mean": None,
                    "perm_std": None,
                    "p_value": None,
                }

    # Permutations
    rng = np.random.default_rng()
    perm_scores = {col: [] for col in label_cols}
    perm_scores_windows: dict[str, dict[str, list[float]]] = {}
    if windows is not None:
        for col in label_cols:
            perm_scores_windows[col] = {str(w): [] for w in windows}
    print(f"[perm] plan: {n_iters} iteracji")
    start_time = time.time()
    for itr in range(int(n_iters)):
        elapsed_s = time.time() - start_time
        avg_s = elapsed_s / (itr + 1) if itr + 1 > 0 else 0.0
        remaining_s = max(0.0, avg_s * (n_iters - itr - 1))
        print(f"[perm] {itr+1}/{n_iters} | elapsed {elapsed_s/60:.1f}m | ETA ~{remaining_s/60:.1f}m", flush=True)
        ytr_p = y_train.copy()
        yva_p = y_val.copy()
        yte_p = y_test.copy()
        for col in label_cols:
            ytr_p[col] = block_shuffle_series(y_train[col], freq=block_freq, rng=rng)
            yva_p[col] = block_shuffle_series(y_val[col], freq=block_freq, rng=rng)
            yte_p[col] = block_shuffle_series(y_test[col], freq=block_freq, rng=rng)

        # Limit levels during permuted fit as well
        old_resolved_p = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', None)
        try:
            setattr(cfg, 'RESOLVED_LABEL_COLUMNS', label_cols)
            model_p = MultiOutputXGB(feat_names)
            model_p.fit(X_train, ytr_p, X_val, yva_p)
            _ = model_p.predict(X_test)
        finally:
            if old_resolved_p is not None:
                setattr(cfg, 'RESOLVED_LABEL_COLUMNS', old_resolved_p)
        for idx, col in enumerate(label_cols):
            proba = model_p.probas_[col]
            pred = predict_dir_with_thresholds(proba, thr_long=thr_long, thr_short=thr_short)
            cm = confusion_matrix_3(yte_p[col].values.astype(int), pred)
            perm_scores[col].append(compute_wacc_from_cm(cm))
            if windows is not None:
                periods = X_test.index.to_period(test_window)  # type: ignore[arg-type]
                for w in windows:
                    mask = periods == w
                    pred_w = pred[mask]
                    y_w = yte_p[col].values.astype(int)[mask]
                    cm_w = confusion_matrix_3(y_w, pred_w)
                    wacc_w = compute_wacc_from_cm(cm_w)
                    perm_scores_windows[col][str(w)].append(wacc_w)

    # Summarize p-values
    for col in label_cols:
        arr = np.array(perm_scores[col], dtype=float)
        true_val = float(summary[col]["true_wacc"])  # type: ignore
        p_value = float((np.sum(arr >= true_val) + 1) / (len(arr) + 1)) if len(arr) else None
        summary[col]["perm_mean"] = float(np.mean(arr)) if len(arr) else None
        summary[col]["perm_std"] = float(np.std(arr)) if len(arr) else None
        summary[col]["p_value"] = p_value
        if windows is not None:
            for w, vals in perm_scores_windows[col].items():
                arr_w = np.array(vals, dtype=float)
                true_w = summary_windows[col][w]["true_wacc"]  # type: ignore[index]
                p_w = float((np.sum(arr_w >= true_w) + 1) / (len(arr_w) + 1)) if len(arr_w) else None
                summary_windows[col][w]["perm_mean"] = float(np.mean(arr_w)) if len(arr_w) else None
                summary_windows[col][w]["perm_std"] = float(np.std(arr_w)) if len(arr_w) else None
                summary_windows[col][w]["p_value"] = p_w

    # Attach windows to main summary and save
    if windows is not None:
        for col in label_cols:
            summary[col]["windows"] = summary_windows.get(col, {})
    with open(out_dir / f"shuffle_test_summary_{symbol}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Target-shuffle significance test (training5, 3-class)")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--block-freq", default='W-MON')
    parser.add_argument("--full", action='store_true')
    parser.add_argument("--levels", nargs='*', help="Lista etykiet poziomów do przetestowania (np. label_tp1p4_sl0p7 ... lub jako csv)")
    parser.add_argument("--thr-long", type=float, default=0.5, help="Próg pewności dla LONG (P_long >= thr)")
    parser.add_argument("--thr-short", type=float, default=0.5, help="Próg pewności dla SHORT (P_short >= thr)")
    parser.add_argument("--test-window", type=str, default=None, help="Częstotliwość okna testowego, np. W-MON, M, D")
    args = parser.parse_args()

    cfg.ensure_dirs()
    out_dir = ensure_run_dir(args.symbol)
    run_shuffle_test(
        args.symbol,
        n_iters=int(args.n_iters),
        block_freq=str(args.block_freq),
        out_dir=out_dir,
        fast_mode=(not args.full),
        levels_filter=(args.levels or None),
        thr_long=float(args.thr_long),
        thr_short=float(args.thr_short),
        test_window=(args.test_window if args.test_window else None),
    )


if __name__ == "__main__":
    main()

