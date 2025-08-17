import sys
from pathlib import Path
import argparse
import json

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training5 import config as cfg
from training5.utils import setup_logging
from training5.data_loader import load_labeled, split_scale
from training5.model_builder import MultiOutputXGB
import joblib
from training5.report import save_markdown_report
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from datetime import datetime
import numpy as np


def run(symbol: str):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"training5 start: {symbol}")
    df = load_labeled(symbol)
    logger.info(f"Wejście: {len(df):,} wierszy, {len(df.columns)} kolumn")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)
    logger.info(f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    # Log and persist the exact feature list used for training
    try:
        logger.info(f"Features used ({len(feat_names)}): {', '.join(feat_names)}")
        rep_dir = cfg.get_report_dir(symbol)
        ts_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        fpath = rep_dir / f"features_used_{symbol}_{ts_now}.txt"
        with open(fpath, 'w', encoding='utf-8') as f:
            for name in feat_names:
                f.write(f"{name}\n")
        logger.info(f"Saved features list: {fpath}")
    except Exception as e:
        logger.warning(f"Failed to save/log features list: {e}")

    model = MultiOutputXGB(feat_names)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)
    # Basic metrics
    metrics = {}
    from sklearn.metrics import confusion_matrix
    thresholds = [0.3, 0.4, 0.45, 0.5]
    # Use resolved label columns (from data_loader) if available
    label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
    for level_idx, col in enumerate(label_cols):
        y_true_c = y_test[col]
        y_pred_c = y_pred[col]
        acc = accuracy_score(y_true_c, y_pred_c)
        rep = classification_report(
            y_true_c,
            y_pred_c,
            target_names=['LONG', 'SHORT', 'NEUTRAL'],
            labels=[0, 1, 2],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1, 2])

        # Confidence thresholds analysis
        proba = model.probas_[col]  # shape (n,3)
        maxp = np.max(proba, axis=1)
        conf_results = {}
        for thr in thresholds:
            mask = maxp >= thr
            n_high = int(mask.sum())
            if n_high == 0:
                conf_results[thr] = None
                continue
            y_true_h = y_true_c[mask]
            y_pred_h = y_pred_c[mask]
            acc_h = accuracy_score(y_true_h, y_pred_h)
            rep_h = classification_report(
                y_true_h,
                y_pred_h,
                target_names=['LONG', 'SHORT', 'NEUTRAL'],
                labels=[0, 1, 2],
                output_dict=True,
                zero_division=0,
            )
            cm_h = confusion_matrix(y_true_h, y_pred_h, labels=[0, 1, 2])
            conf_results[thr] = {
                'n_high_conf': n_high,
                'n_total': int(len(y_true_c)),
                'percentage': float(n_high) / float(len(y_true_c)) * 100.0,
                'accuracy': acc_h,
                'classification_report': rep_h,
                'confusion_matrix': cm_h.tolist(),
            }

        metrics[col] = {
            "accuracy": acc,
            "report": rep,
            "confusion_matrix": cm.tolist(),
            "confidence_results": conf_results,
            "level_index": level_idx,
        }

    # Save artifacts (models)
    model_dir = cfg.MODELS_DIR / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(model.models):
        m.save_model(str(model_dir / f"model_{i+1}.json"))

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

    # Save metrics JSON
    rep_dir = cfg.get_report_dir(symbol)
    with open(rep_dir / f"metrics_{symbol}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions CSV (thresholded, easy-to-read format)
    idx = cfg.CSV_PREDICTIONS_MODEL_INDEX
    level_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
    level_col = level_cols[idx]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    proba = model.probas_[level_col]  # shape (n,3) -> [LONG, SHORT, NEUTRAL]
    idx_ts = X_test.index
    maxp = np.max(proba, axis=1)
    pred_class = np.argmax(proba, axis=1)
    # Threshold 0.5 for consistency with strategy/backtest signals
    thr = 0.5

    def _to_pair_str(sym: str) -> str:
        if sym.upper().endswith("USDT"):
            base = sym[:-4]
            return f"{base}/USDT:USDT"
        return sym

    pair_str = _to_pair_str(symbol)
    rows = []
    trade_rows = []  # only LONG/SHORT, no NEUTRAL, with correctness
    for i in range(proba.shape[0]):
        if maxp[i] >= thr and pred_class[i] in (0, 1):
            signal = 'long' if pred_class[i] == 0 else 'short'
        else:
            signal = 'neutral'
        rows.append({
            'timestamp': str(idx_ts[i]),
            'pair': pair_str,
            'signal': signal,
            'confidence': float(maxp[i]),
            'prob_SHORT': float(proba[i, 1]),
            'prob_LONG': float(proba[i, 0]),
            'prob_NEUTRAL': float(proba[i, 2]),
        })
        # Trades-only file with correctness info
        if signal != 'neutral':
            true_cls = int(y_test[level_col].iloc[i])
            is_correct = int(pred_class[i]) == true_cls
            trade_rows.append({
                'timestamp': str(idx_ts[i]),
                'pair': pair_str,
                'signal': signal,
                'confidence': float(maxp[i]),
                'prob_SHORT': float(proba[i, 1]),
                'prob_LONG': float(proba[i, 0]),
                'prob_NEUTRAL': float(proba[i, 2]),
                'true_label': ['LONG', 'SHORT', 'NEUTRAL'][true_cls],
                'correct': bool(is_correct),
                'result': 'WIN' if is_correct else 'LOSS',
            })

    pred_out = pd.DataFrame(rows)
    pred_csv = rep_dir / f"predictions_{symbol}_{level_col}_{timestamp_str}.csv"
    pred_out.to_csv(pred_csv, index=False)

    # Save trades-only CSV (LONG/SHORT predictions, no NEUTRAL) with correctness
    if trade_rows:
        trades_out = pd.DataFrame(trade_rows)
        trades_csv = rep_dir / f"predictions_trades_{symbol}_{level_col}_{timestamp_str}.csv"
        trades_out.to_csv(trades_csv, index=False)

    # Feature importance CSV (aggregate gain across models)
    # Handle both index-based keys ('f0', 'f1', ...) and explicit feature-name keys
    feat_importance = np.zeros(len(feat_names), dtype=float)
    name_to_index = {name: i for i, name in enumerate(feat_names)}
    for m in model.models:
        imp = m.get_score(importance_type='gain')  # dict {feature_key: gain}
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
    fi_csv = rep_dir / f"feature_importance_{timestamp_str}.csv"
    fi_df.to_csv(fi_csv, index=False, float_format='%.10f')

    logger.info("training5 done")
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
    save_markdown_report(evaluation_results, model_params, data_info, cfg, symbol)
    from training5.report import save_json_report
    save_json_report(evaluation_results, model_params, data_info, cfg, symbol)


def main():
    parser = argparse.ArgumentParser(description="training5 - Multi-Output XGBoost")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL)
    args = parser.parse_args()
    run(args.symbol)


if __name__ == "__main__":
    main()

