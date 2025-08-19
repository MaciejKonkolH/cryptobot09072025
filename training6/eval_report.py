import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import xgboost as xgb
import numpy as np
import pandas as pd

# Allow script run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training6 import config as cfg
from training6.utils import setup_logging
from training6.data_loader import load_labeled, split_scale
from training6.report import save_markdown_report, save_json_report


def load_models(feature_names):
    models = []
    model_dir = cfg.MODES_DIR if hasattr(cfg, 'MODES_DIR') else cfg.MODELS_DIR
    # assume files model_1.json .. model_15.json are present in symbol subdir
    return models


def run(symbol: str, args=None):
    logger = setup_logging()
    cfg.ensure_dirs()
    df = load_labeled(symbol)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)

    # Load boosters
    boosters = []
    model_dir = (cfg.MODELS_DIR / symbol)
    for i in range(1, len(cfg.LABEL_COLUMNS) + 1):
        fp = model_dir / f"model_{i}.json"
        b = xgb.Booster()
        b.load_model(str(fp))
        boosters.append(b)

    # Predict probabilities
    dtest = xgb.DMatrix(X_test, feature_names=feat_names)
    probas = {}
    preds = {}
    for i, col in enumerate(cfg.LABEL_COLUMNS):
        proba = boosters[i].predict(dtest)
        probas[col] = proba
        preds[col] = np.argmax(proba, axis=1)
    y_pred = pd.DataFrame(preds, index=X_test.index)

    # Fast path: only export trades JSON for a single case
    if args is not None and getattr(args, 'save_trades', False) and getattr(args, 'trades_only', False):
        # Resolve level by tp/sl if provided, else by level_idx
        level_idx = None
        if args.tp is not None and args.sl is not None:
            target = (float(args.tp), float(args.sl))
            best_i, best_err = None, float('inf')
            for i, (tp_lvl, sl_lvl) in enumerate(cfg.TP_SL_LEVELS):
                err = abs(tp_lvl - target[0]) + abs(sl_lvl - target[1])
                if err < best_err:
                    best_err, best_i = err, i
            level_idx = best_i
        elif args.level_idx is not None:
            level_idx = int(args.level_idx)
        else:
            raise SystemExit("When --save_trades is set, provide --tp and --sl, or --level_idx")

        if level_idx < 0 or level_idx >= len(cfg.LABEL_COLUMNS):
            raise SystemExit("Invalid level index for trades export")

        col = cfg.LABEL_COLUMNS[level_idx]
        proba = probas[col]
        thr = float(args.conf_thr)
        if thr > 1.0:
            thr = thr / 100.0
        maxp = np.max(proba, axis=1)
        pred_class = np.argmax(proba, axis=1)
        mask_conf = maxp >= thr
        mask_trade = (pred_class == 0) | (pred_class == 1)
        mask = mask_conf & mask_trade

        idx_ts = X_test.index
        true_labels = y_test[col].values
        trades = []
        for i in np.where(mask)[0]:
            ts = idx_ts[i]
            pred_cls = int(pred_class[i])
            true_cls = int(true_labels[i])
            result = 'WIN' if pred_cls == true_cls else 'LOSS'
            trades.append({
                'timestamp': str(ts),
                'signal': 'LONG' if pred_cls == 0 else 'SHORT',
                'probability': float(maxp[i]),
                'true_label': ['LONG', 'SHORT', 'NEUTRAL'][true_cls],
                'result': result,
            })

        rep_dir = cfg.get_report_dir(symbol)
        tp_val, sl_val = cfg.TP_SL_LEVELS[level_idx]
        ts_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = rep_dir / f"trades_{symbol}_tp{tp_val}_sl{sl_val}_thr{int(thr*100)}_{ts_now}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': symbol,
                'tp': tp_val,
                'sl': sl_val,
                'confidence_threshold': thr,
                'level_index': level_idx,
                'n_trades': len(trades),
                'trades': trades,
            }, f, ensure_ascii=False)
        logger = setup_logging()
        logger.info(f"Saved trades JSON: {out_path}")
        return

    # Build metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    thresholds = [0.3, 0.4, 0.45, 0.5]
    metrics = {}
    for level_idx, col in enumerate(cfg.LABEL_COLUMNS):
        y_true_c = y_test[col]
        y_pred_c = y_pred[col]
        acc = accuracy_score(y_true_c, y_pred_c)
        rep = classification_report(
            y_true_c, y_pred_c,
            target_names=['LONG', 'SHORT', 'NEUTRAL'], labels=[0, 1, 2],
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1, 2])
        # Confidence
        proba = probas[col]
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
                y_true_h, y_pred_h,
                target_names=['LONG', 'SHORT', 'NEUTRAL'], labels=[0, 1, 2],
                output_dict=True, zero_division=0,
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
            'accuracy': acc,
            'report': rep,
            'confusion_matrix': cm.tolist(),
            'confidence_results': conf_results,
            'level_index': level_idx,
        }

    # Report
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
    evaluation_results = {}
    for col, m in metrics.items():
        evaluation_results[col] = {
            'accuracy': m['accuracy'],
            'classification_report': m['report'],
            'confusion_matrix': m['confusion_matrix'],
            'confidence_results': m['confidence_results'],
            'level_index': m['level_index'],
        }

    save_markdown_report(evaluation_results, model_params, data_info, cfg, symbol)
    save_json_report(evaluation_results, model_params, data_info, cfg, symbol)

    # Optional: save trades JSON for a single specified TP/SL and confidence threshold
    if args is not None and getattr(args, 'save_trades', False):
        # Resolve level by tp/sl if provided, else by level_idx
        level_idx = None
        if args.tp is not None and args.sl is not None:
            # find closest matching level
            target = (float(args.tp), float(args.sl))
            best_i, best_err = None, float('inf')
            for i, (tp_lvl, sl_lvl) in enumerate(cfg.TP_SL_LEVELS):
                err = abs(tp_lvl - target[0]) + abs(sl_lvl - target[1])
                if err < best_err:
                    best_err, best_i = err, i
            level_idx = best_i
        elif args.level_idx is not None:
            level_idx = int(args.level_idx)
        else:
            raise SystemExit("When --save_trades is set, provide --tp and --sl, or --level_idx")

        if level_idx < 0 or level_idx >= len(cfg.LABEL_COLUMNS):
            raise SystemExit("Invalid level index for trades export")

        col = cfg.LABEL_COLUMNS[level_idx]
        proba = probas[col]
        preds_arr = y_pred[col].values
        # threshold: accept both [0,1] or percent
        thr = float(args.conf_thr)
        if thr > 1.0:
            thr = thr / 100.0
        maxp = np.max(proba, axis=1)
        pred_class = np.argmax(proba, axis=1)
        mask_conf = maxp >= thr
        mask_trade = (pred_class == 0) | (pred_class == 1)
        mask = mask_conf & mask_trade

        # Prepare JSON rows
        idx_ts = X_test.index
        true_labels = y_test[col].values
        trades = []
        for i in np.where(mask)[0]:
            ts = idx_ts[i]
            pred_cls = int(pred_class[i])
            true_cls = int(true_labels[i])
            result = 'WIN' if pred_cls == true_cls else 'LOSS'
            trades.append({
                'timestamp': str(ts),
                'signal': 'LONG' if pred_cls == 0 else 'SHORT',
                'probability': float(maxp[i]),
                'true_label': ['LONG', 'SHORT', 'NEUTRAL'][true_cls],
                'result': result,
            })

        rep_dir = cfg.get_report_dir(symbol)
        tp_val, sl_val = cfg.TP_SL_LEVELS[level_idx]
        ts_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = rep_dir / f"trades_{symbol}_tp{tp_val}_sl{sl_val}_thr{int(thr*100)}_{ts_now}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': symbol,
                'tp': tp_val,
                'sl': sl_val,
                'confidence_threshold': thr,
                'level_index': level_idx,
                'n_trades': len(trades),
                'trades': trades,
            }, f, ensure_ascii=False)
        logger.info(f"Saved trades JSON: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate report from saved models (no retraining)')
    parser.add_argument('--symbol', default=cfg.DEFAULT_SYMBOL)
    parser.add_argument('--save_trades', action='store_true', help='Export trades JSON for a single TP/SL and threshold')
    parser.add_argument('--trades_only', action='store_true', help='Skip report generation; export only trades JSON')
    parser.add_argument('--tp', type=float, default=None, help='TP percent (e.g., 0.6) to select level')
    parser.add_argument('--sl', type=float, default=None, help='SL percent (e.g., 0.2) to select level')
    parser.add_argument('--level_idx', type=int, default=None, help='Alternative to --tp/--sl: explicit level index (0-based)')
    parser.add_argument('--conf_thr', type=float, default=0.5, help='Confidence threshold (0-1 or percent, e.g., 0.5 or 50)')
    args = parser.parse_args()
    run(args.symbol, args)


if __name__ == '__main__':
    main()

