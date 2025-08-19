import sys
from pathlib import Path
import argparse
import json

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_training import config as cfg
from multi_training.utils import setup_logging
from multi_training.data_loader import load_multi
from multi_training.model_builder import MultiSymbolXGB
from multi_training.report import save_json_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


def run(symbols: list[str]):
    logger = setup_logging()
    cfg.ensure_dirs()

    logger.info(f"multi_training start: {symbols}")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names, label_cols, test_symbols = load_multi(symbols)
    logger.info(f"Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}; features={len(feat_names)}")

    # Per-run output directory
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = cfg.get_report_dir() / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save features list and symbols
    with open(run_dir / f"features_used_MULTI_{run_ts}.txt", 'w', encoding='utf-8') as f:
        for name in feat_names:
            f.write(f"{name}\n")
    with open(run_dir / f"symbols_{run_ts}.txt", 'w', encoding='utf-8') as f:
        for s in symbols:
            f.write(f"{s}\n")

    model = MultiSymbolXGB(feat_names, label_cols)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)

    # Metrics per level
    metrics = {}
    thresholds = [0.3, 0.4, 0.45, 0.5]
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

        # Confidence thresholds
        proba = model.probas_[col]
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

    # Save models and scaler
    model_dir = cfg.MODELS_DIR / "MULTI"
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(model.models):
        m.save_model(str(model_dir / f"model_{i+1}.json"))
    joblib.dump(scaler, str(model_dir / "scaler.pkl"))

    # Save predictions per level (full and trades-only) with pair info
    def _to_pair_str(sym: str) -> str:
        return f"{sym[:-4]}/USDT:USDT" if sym.upper().endswith("USDT") else sym

    thr = 0.5
    pair_series = test_symbols.map(_to_pair_str)
    for level_col in label_cols:
        proba = model.probas_[level_col]
        idx_ts = X_test.index
        maxp = np.max(proba, axis=1)
        pred_class = np.argmax(proba, axis=1)

        rows = []
        trade_rows = []
        for i in range(proba.shape[0]):
            if maxp[i] >= thr and pred_class[i] in (0, 1):
                signal = 'long' if pred_class[i] == 0 else 'short'
            else:
                signal = 'neutral'
            rows.append({
                'timestamp': str(idx_ts[i]),
                'pair': str(pair_series.iloc[i]),
                'signal': signal,
                'confidence': float(maxp[i]),
                'prob_SHORT': float(proba[i, 1]),
                'prob_LONG': float(proba[i, 0]),
                'prob_NEUTRAL': float(proba[i, 2]),
            })
            if signal != 'neutral':
                true_cls = int(y_test[level_col].iloc[i])
                is_correct = int(pred_class[i]) == true_cls
                trade_rows.append({
                    'timestamp': str(idx_ts[i]),
                    'pair': str(pair_series.iloc[i]),
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
        pred_csv = run_dir / f"predictions_MULTI_{level_col}_{run_ts}.csv"
        pred_out.to_csv(pred_csv, index=False)
        if trade_rows:
            trades_out = pd.DataFrame(trade_rows)
            trades_csv = run_dir / f"predictions_trades_MULTI_{level_col}_{run_ts}.csv"
            trades_out.to_csv(trades_csv, index=False)

    # Save feature importance
    feat_importance = np.zeros(len(feat_names), dtype=float)
    name_to_index = {name: i for i, name in enumerate(feat_names)}
    for m in model.models:
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
    total_gain = float(feat_importance.sum())
    if total_gain > 0.0:
        feat_importance = feat_importance / total_gain
    fi_df = pd.DataFrame({'feature': feat_names, 'importance': feat_importance}).sort_values('importance', ascending=False)
    fi_df.to_csv(run_dir / f"feature_importance_{run_ts}.csv", index=False, float_format='%.10f')

    # Build report
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
        'symbols': symbols,
        'n_features': len(feat_names),
        'feature_names': list(feat_names),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'train_range': f"{X_train.index.min()} - {X_train.index.max()}",
        'test_range': f"{X_test.index.min()} - {X_test.index.max()}",
        'best_validation_mlogloss': getattr(model, 'best_scores', {}),
    }
    save_json_report(metrics, model_params, data_info, out_dir=run_dir)


def main():
    parser = argparse.ArgumentParser(description="multi_training - one XGBoost over multiple symbols")
    parser.add_argument("--symbols", nargs="*", default=cfg.SYMBOLS, help="List of symbols, e.g., BTCUSDT ETHUSDT XRPUSDT")
    args = parser.parse_args()
    run(args.symbols)


if __name__ == "__main__":
    main()

