import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_training import config as cfg
from dual_training.utils import setup_logging
from dual_training.data_loader import load_labeled, split_scale
from dual_training.model_builder import BinaryPerLevelXGB
from dual_training.report import save_markdown_report, save_json_report


def run(symbol: str):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"dual_training start: {symbol}")
    df = load_labeled(symbol)
    logger.info(f"Wejście: {len(df):,} wierszy, {len(df.columns)} kolumn")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)
    logger.info(f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    # Per-run output directory
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = cfg.get_report_dir(symbol) / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save feature list
    try:
        fpath = run_dir / f"features_used_{symbol}_{run_ts}.txt"
        with open(fpath, 'w', encoding='utf-8') as f:
            for name in feat_names:
                f.write(f"{name}\n")
        logger.info(f"Saved features list: {fpath}")
    except Exception as e:
        logger.warning(f"Failed to save/log features list: {e}")

    model = BinaryPerLevelXGB(feat_names)
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate per level; compute binary metrics and confidence blocks to mimic training5 style
    prob_pos = model.predict_proba(X_test)
    evaluation_results = {}
    label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
    for level_idx, col in enumerate(label_cols):
        y_true_bin = (y_test[col].values == 0).astype(int)
        p = prob_pos[col].values
        try:
            auc = float(roc_auc_score(y_true_bin, p))
        except Exception:
            auc = float('nan')
        # Base predictions at 0.5
        y_pred_bin = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y_true_bin, y_pred_bin))
        # 2x2 confusion matrix in LONG/SHORT space (rows=actual [LONG,SHORT], cols=pred [LONG,SHORT])
        cm2 = confusion_matrix(y_true_bin, y_pred_bin, labels=[1, 0]).tolist()
        # Classification report over two classes (order aligned with LONG first)
        rep2 = classification_report(y_true_bin, y_pred_bin, target_names=['LONG', 'SHORT'], labels=[1, 0], output_dict=True, zero_division=0)
        # Average profit per trade (percent) for base predictions at 0.5
        tp_pct = float(cfg.TP_SL_LEVELS[level_idx][0]) if level_idx < len(cfg.TP_SL_LEVELS) else 0.0
        total_trades = cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1]
        correct_trades = cm2[0][0] + cm2[1][1]
        incorrect_trades = cm2[0][1] + cm2[1][0]
        avg_profit_pct_base = (tp_pct * (correct_trades - incorrect_trades) / total_trades) if total_trades > 0 else 0.0
        # Confidence blocks
        thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.66, 0.69, 0.72]
        conf_results = {}
        for thr in thresholds:
            mask = p >= thr
            n_high = int(mask.sum())
            if n_high == 0:
                conf_results[thr] = None
                continue
            yt = y_true_bin[mask]
            yp = (p[mask] >= 0.5).astype(int)
            acc_h = float(accuracy_score(yt, yp))
            rep_h = classification_report(yt, yp, target_names=['LONG', 'SHORT'], labels=[1, 0], output_dict=True, zero_division=0)
            cm_h = confusion_matrix(yt, yp, labels=[1, 0]).tolist()
            # average profit per trade at this threshold (percent)
            tot_h = cm_h[0][0] + cm_h[0][1] + cm_h[1][0] + cm_h[1][1]
            corr_h = cm_h[0][0] + cm_h[1][1]
            incorr_h = cm_h[0][1] + cm_h[1][0]
            avg_profit_pct_h = (tp_pct * (corr_h - incorr_h) / tot_h) if tot_h > 0 else 0.0
            conf_results[thr] = {
                'n_high_conf': n_high,
                'n_total': int(len(y_true_bin)),
                'percentage': float(n_high) / float(len(y_true_bin)) * 100.0,
                'accuracy': acc_h,
                'classification_report': rep_h,
                'confusion_matrix': cm_h,
                'avg_profit_pct': avg_profit_pct_h,
            }

        best_info = model.best_scores.get(col, {}) if hasattr(model, 'best_scores') else {}
        evaluation_results[col] = {
            'level_index': level_idx,
            'auc': auc,
            'best_logloss': best_info.get('best_logloss'),
            'best_iteration': best_info.get('best_iteration'),
            'accuracy': acc,
            'classification_report': rep2,
            'confusion_matrix': cm2,
            'confidence_results': conf_results,
            'avg_profit_pct_base': avg_profit_pct_base,
        }

    # Save models
    model_dir = cfg.MODELS_DIR / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(model.models):
        m.save_model(str(model_dir / f"model_{i+1}.json"))
    try:
        joblib.dump(scaler, str(model_dir / "scaler.pkl"))
    except Exception as e:
        logger.error(f"Failed to save scaler.pkl: {e}")

    # Save reports
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
    save_markdown_report(evaluation_results, model_params, data_info, cfg, symbol, out_dir=run_dir)
    save_json_report(evaluation_results, model_params, data_info, cfg, symbol, out_dir=run_dir)


def main():
    parser = argparse.ArgumentParser(description="dual_training - Binary XGBoost per TP=SL level (compatible with dual_labeler)")
    parser.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL)
    args = parser.parse_args()
    run(args.symbol)


if __name__ == "__main__":
    main()

