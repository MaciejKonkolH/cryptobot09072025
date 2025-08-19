import sys
from pathlib import Path
import argparse
import json

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training6 import config as cfg
from training6.utils import setup_logging
from training6.data_loader import load_labeled, split_scale
from training6.model_builder import BinaryDirectionalXGB
import joblib
from training6.report import save_markdown_report
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from datetime import datetime
import numpy as np


def run(symbol: str):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"training6 start: {symbol}")
    df = load_labeled(symbol)
    logger.info(f"Wejście: {len(df):,} wierszy, {len(df.columns)} kolumn")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)
    logger.info(f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    # Create per-run output directory under reports/{symbol}/run_{timestamp}
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = cfg.get_report_dir(symbol) / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log and persist the exact feature list used for training (into run_dir)
    try:
        logger.info(f"Features used ({len(feat_names)}): {', '.join(feat_names)}")
        fpath = run_dir / f"features_used_{symbol}_{run_ts}.txt"
        with open(fpath, 'w', encoding='utf-8') as f:
            for name in feat_names:
                f.write(f"{name}\n")
        logger.info(f"Saved features list: {fpath}")
    except Exception as e:
        logger.warning(f"Failed to save/log features list: {e}")

    model = BinaryDirectionalXGB(feat_names)
    model.fit(X_train, y_train, X_val, y_val)
    # Tune per-level decision thresholds on validation
    from training6 import report as report_mod
    # Binary directional metrics
    metrics = {}
    label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
    default_thr_long = float(getattr(cfg, 'DECISION_THRESHOLD_LONG', 0.5))
    default_thr_short = float(getattr(cfg, 'DECISION_THRESHOLD_SHORT', 0.5))
    from sklearn.metrics import confusion_matrix
    timestamp_str = run_ts
    pair_str = (symbol[:-4] + "/USDT:USDT") if symbol.upper().endswith("USDT") else symbol
    # Grid for thresholds (separate for LONG/SHORT)
    THR_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
    selected_thresholds = {}
    # Helper: compute weighted trade accuracy from CM
    def _weighted_trade_accuracy(cm):
        col_long = cm[0][0] + cm[1][0] + cm[2][0]
        col_short = cm[0][1] + cm[1][1] + cm[2][1]
        p_long = (cm[0][0] / col_long) if col_long > 0 else None
        p_short = (cm[1][1] / col_short) if col_short > 0 else None
        numerator = 0.0
        denom = 0
        if col_long and p_long is not None:
            numerator += col_long * p_long
            denom += col_long
        if col_short and p_short is not None:
            numerator += col_short * p_short
            denom += col_short
        return (numerator / denom) if denom > 0 else 0.0
    # Tune per level using validation set
    for col in label_cols:
        y_val_series = y_val[col].astype(int)
        proba_val = model.predict_proba_level(X_val, col)
        P_long_v = proba_val['P_long'].values
        P_short_v = proba_val['P_short'].values
        # Profit threshold based on TP/SL for this level
        lvl_idx = label_cols.index(col)
        tp, sl = cfg.TP_SL_LEVELS[lvl_idx]
        profit_thr = report_mod._profit_threshold(float(tp), float(sl)) / 100.0
        best = {
            'thr_long': default_thr_long,
            'thr_short': default_thr_short,
            'wacc': 0.0,
            'safety': -1e9,
        }
        for tL in THR_GRID:
            for tS in THR_GRID:
                pred_v = np.full_like(P_long_v, fill_value=2, dtype=int)
                long_m = (P_long_v >= tL) & (P_long_v > P_short_v)
                short_m = (P_short_v >= tS) & (P_short_v > P_long_v)
                pred_v[long_m] = 0
                pred_v[short_m] = 1
                cm_v = confusion_matrix(y_val_series, pred_v, labels=[0, 1, 2])
                wacc = _weighted_trade_accuracy(cm_v)
                safety = (wacc - profit_thr)
                if safety > best['safety'] or (safety == best['safety'] and wacc > best['wacc']):
                    best = {'thr_long': tL, 'thr_short': tS, 'wacc': wacc, 'safety': safety}
        selected_thresholds[col] = best
    # Save thresholds to JSON in run_dir
    try:
        with open(run_dir / f"selected_thresholds_{symbol}_{timestamp_str}.json", 'w', encoding='utf-8') as f:
            json.dump(selected_thresholds, f, indent=2)
    except Exception:
        pass
    for col in label_cols:
        y_series = y_test[col].astype(int)
        proba_df = model.predict_proba_level(X_test, col)
        P_long = proba_df['P_long'].values
        P_short = proba_df['P_short'].values
        pred_dir = np.full_like(P_long, fill_value=2, dtype=int)
        thr_long = float(selected_thresholds.get(col, {}).get('thr_long', default_thr_long))
        thr_short = float(selected_thresholds.get(col, {}).get('thr_short', default_thr_short))
        long_mask = (P_long >= thr_long) & (P_long > P_short)
        short_mask = (P_short >= thr_short) & (P_short > P_long)
        pred_dir[long_mask] = 0
        pred_dir[short_mask] = 1

        cm = confusion_matrix(y_series, pred_dir, labels=[0, 1, 2])
        rep = classification_report(y_series, pred_dir, target_names=['LONG', 'SHORT', 'NEUTRAL'], labels=[0, 1, 2], output_dict=True, zero_division=0)
        acc = accuracy_score(y_series, pred_dir)
        metrics[col] = {
            "accuracy": acc,
            "report": rep,
            "confusion_matrix": cm.tolist(),
            "confidence_results": {},
            "level_index": label_cols.index(col),
            "decision_thresholds": {"long": thr_long, "short": thr_short},
        }

        # Per-level predictions CSVs (compatible filenames)
        idx_ts = X_test.index
        rows = []
        trade_rows = []
        for i in range(len(idx_ts)):
            pL = float(P_long[i])
            pS = float(P_short[i])
            if (pL >= thr_long) and (pL > pS):
                signal = 'long'
                conf = pL
            elif (pS >= thr_short) and (pS > pL):
                signal = 'short'
                conf = pS
            else:
                signal = 'neutral'
                conf = max(pL, pS)
            rows.append({
                'timestamp': str(idx_ts[i]),
                'pair': pair_str,
                'signal': signal,
                'confidence': conf,
                'P_long': pL,
                'P_short': pS,
            })
            if signal != 'neutral':
                true_cls = int(y_series.iloc[i])
                is_correct = ((signal == 'long' and true_cls == 0) or (signal == 'short' and true_cls == 1))
                trade_rows.append({
                    'timestamp': str(idx_ts[i]),
                    'pair': pair_str,
                    'signal': signal,
                    'confidence': conf,
                    'P_long': pL,
                    'P_short': pS,
                    'true_label': ['LONG', 'SHORT', 'NEUTRAL'][true_cls],
                    'correct': bool(is_correct),
                    'result': 'WIN' if is_correct else 'LOSS',
                })

        pred_out = pd.DataFrame(rows)
        pred_csv = run_dir / f"predictions_{symbol}_{col}_{timestamp_str}.csv"
        pred_out.to_csv(pred_csv, index=False)
        if trade_rows:
            trades_out = pd.DataFrame(trade_rows)
            trades_csv = run_dir / f"predictions_trades_{symbol}_{col}_{timestamp_str}.csv"
            trades_out.to_csv(trades_csv, index=False)

        # Confidence threshold blocks (30%, 40%, 45%, 50%)
        conf_map = {}
        for thr_percent in [30.0, 40.0, 45.0, 50.0]:
            thr_conf = thr_percent / 100.0
            pred_dir_h = np.full_like(P_long, fill_value=2, dtype=int)
            long_h = (P_long >= thr_conf) & (P_long > P_short)
            short_h = (P_short >= thr_conf) & (P_short > P_long)
            pred_dir_h[long_h] = 0
            pred_dir_h[short_h] = 1
            from sklearn.metrics import confusion_matrix
            cm_h = confusion_matrix(y_series, pred_dir_h, labels=[0, 1, 2])
            rep_h = classification_report(y_series, pred_dir_h, target_names=['LONG', 'SHORT', 'NEUTRAL'], labels=[0, 1, 2], output_dict=True, zero_division=0)
            acc_h = accuracy_score(y_series, pred_dir_h)
            n_total = int(len(y_series))
            n_high = int((pred_dir_h != 2).sum())
            conf_map[thr_conf] = {
                'n_high_conf': n_high,
                'n_total': n_total,
                'percentage': (n_high / n_total * 100.0) if n_total > 0 else 0.0,
                'accuracy': acc_h,
                'classification_report': rep_h,
                'confusion_matrix': cm_h.tolist(),
            }
        metrics[col]['confidence_results'] = conf_map

    # Save artifacts (models)
    model_dir = cfg.MODELS_DIR / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    for col in label_cols:
        model.model_long[col].save_model(str(model_dir / f"model_long_{col}.json"))
        model.model_short[col].save_model(str(model_dir / f"model_short_{col}.json"))

    # Save scaler and metadata for live inference (FreqTrade compatibility)
    try:
        joblib.dump(scaler, str(model_dir / "scaler.pkl"))
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to save scaler.pkl: {e}")

    try:
        metadata = {
            "model_type": "xgboost_individual",
            "n_features": len(feat_names),
            "feature_names": list(feat_names),
            "label_columns": cfg.LABEL_COLUMNS,
            "tp_sl_levels": cfg.TP_SL_LEVELS,
        }
        with open(str(model_dir / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to save metadata.json: {e}")

    # Save metrics JSON into run_dir
    with open(run_dir / f"metrics_{symbol}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # (per-level CSV already saved above)

    # Feature importance CSV (aggregate gain across ALL per-level models)
    feat_importance = np.zeros(len(feat_names), dtype=float)
    name_to_index = {name: i for i, name in enumerate(feat_names)}
    for col in label_cols:
        for m in [model.model_long[col], model.model_short[col]]:
            imp = m.get_score(importance_type='gain')
            for k, v in imp.items():
                idxf = None
                if isinstance(k, str) and k.startswith('f') and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < len(feat_names):
                        idxf = idx
                elif k in name_to_index:
                    idxf = name_to_index[k]
                if idxf is not None:
                    feat_importance[idxf] += float(v)
    # Normalize to sum=1 (if non-zero) and save with higher precision
    total_gain = float(feat_importance.sum())
    if total_gain > 0.0:
        feat_importance = feat_importance / total_gain
    fi_df = pd.DataFrame({'feature': feat_names, 'importance': feat_importance})
    fi_df = fi_df.sort_values('importance', ascending=False)
    fi_csv = run_dir / f"feature_importance_{timestamp_str}.csv"
    fi_df.to_csv(fi_csv, index=False, float_format='%.10f')

    logger.info("training6 binary done")
    # Build markdown report similar to training4
    model_params = {
        'N_ESTIMATORS': cfg.XGB_N_ESTIMATORS,
        'LEARNING_RATE': cfg.XGB_LEARNING_RATE,
        'MAX_DEPTH': cfg.XGB_MAX_DEPTH,
        'SUBSAMPLE': cfg.XGB_SUBSAMPLE,
        'COLSAMPLE_BYTREE': cfg.XGB_COLSAMPLE_BYTREE,
        'EARLY_STOPPING_ROUNDS': cfg.XGB_EARLY_STOPPING_ROUNDS,
        'GAMMA': cfg.XGB_GAMMA,
        'RANDOM_STATE': cfg.XGB_RANDOM_STATE,
    }
    data_info = {
        'n_features': len(feat_names),
        'feature_names': list(feat_names),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'train_range': f"{X_train.index.min()} - {X_train.index.max()}",
        'test_range': f"{X_test.index.min()} - {X_test.index.max()}",
    }
    # Convert metrics to evaluation_results-like structure
    evaluation_results = {}
    for col, m in metrics.items():
        evaluation_results[col] = {
            'accuracy': m['accuracy'],
            'classification_report': m['report'],
            'confusion_matrix': m.get('confusion_matrix'),
            'confidence_results': m.get('confidence_results'),
            'level_index': m.get('level_index'),
        }
    # Attach best validation mlogloss per level (if available)
    try:
        if hasattr(model, 'best_scores') and isinstance(model.best_scores, dict):
            data_info['best_validation_mlogloss'] = model.best_scores
    except Exception:
        pass

    # Save unified markdown and JSON reports
    save_markdown_report(evaluation_results, model_params, data_info, cfg, symbol, out_dir=run_dir)
    from training6.report import save_json_report
    save_json_report(evaluation_results, model_params, data_info, cfg, symbol, out_dir=run_dir)


def main():
    parser = argparse.ArgumentParser(description="training6 - Binary Directional XGBoost (two-model)")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL)
    args = parser.parse_args()
    run(args.symbol)


if __name__ == "__main__":
    main()

