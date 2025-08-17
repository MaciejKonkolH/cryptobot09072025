import xgboost as xgb
import numpy as np
import pandas as pd
from . import config as cfg
from .utils import setup_logging


logger = setup_logging()


class MultiOutputXGB:
    def __init__(self, feature_names):
        self.models = []
        self.feature_names = feature_names
        # Per-level training diagnostics
        self.training_curves = {}
        self.best_scores = {}

    def _params(self):
        return {
            'max_depth': cfg.XGB_MAX_DEPTH,
            'learning_rate': cfg.XGB_LEARNING_RATE,
            'subsample': cfg.XGB_SUBSAMPLE,
            'colsample_bytree': cfg.XGB_COLSAMPLE_BYTREE,
            'gamma': cfg.XGB_GAMMA,
            'random_state': cfg.XGB_RANDOM_STATE,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'reg_alpha': cfg.XGB_REG_ALPHA,
            'reg_lambda': cfg.XGB_REG_LAMBDA,
            'min_child_weight': cfg.XGB_MIN_CHILD_WEIGHT,
        }

    def fit(self, X_train, y_train, X_val, y_val):
        self.models = []
        params = self._params()
        dval_cache = xgb.DMatrix(X_val, label=None, feature_names=self.feature_names)

        # Use resolved label columns if provided by data_loader (to handle naming differences)
        label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
        for i, col in enumerate(label_cols, 1):
            logger.info(f"Trening poziomu {i}/{len(cfg.LABEL_COLUMNS)}: {col}")
            # Optional class weights to handle class imbalance (0=LONG,1=SHORT,2=NEUTRAL)
            if getattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING', False):
                weights_map = getattr(cfg, 'CLASS_WEIGHTS', {0: 1.0, 1: 1.0, 2: 1.0})
                ytr = y_train[col].values
                yva = y_val[col].values
                wtr = [float(weights_map.get(int(lbl), 1.0)) for lbl in ytr]
                # Apply weights ONLY on training set (align with training3 behavior). Validation without weights.
                dtrain = xgb.DMatrix(X_train, label=ytr, weight=wtr, feature_names=self.feature_names)
                dval = xgb.DMatrix(X_val, label=yva, feature_names=self.feature_names)
            else:
                dtrain = xgb.DMatrix(X_train, label=y_train[col], feature_names=self.feature_names)
                dval = xgb.DMatrix(X_val, label=y_val[col], feature_names=self.feature_names)

            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=cfg.XGB_N_ESTIMATORS,
                evals=[(dval, 'validation')],
                early_stopping_rounds=cfg.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=cfg.TRAIN_VERBOSE_EVAL,
                evals_result=evals_result,
            )
            self.models.append(model)

            # Store curves and best metrics
            self.training_curves[col] = evals_result
            # Prefer Booster.best_score / best_iteration when available
            best_score = getattr(model, 'best_score', None)
            best_iter = getattr(model, 'best_iteration', None)
            if best_score is None or best_iter is None:
                try:
                    val_curve = evals_result.get('validation', {}).get('mlogloss', [])
                    if val_curve:
                        best_score = float(min(val_curve))
                        best_iter = int(val_curve.index(best_score))
                except Exception:
                    pass
            # Fallback to best_ntree_limit if best_iteration missing
            if best_iter is None:
                best_iter = getattr(model, 'best_ntree_limit', None)
            try:
                logger.info(f"Najlepszy validation-mlogloss ({col}): {best_score} @ iter {best_iter}")
            except Exception:
                pass
            self.best_scores[col] = {
                'best_mlogloss': float(best_score) if best_score is not None else None,
                'best_iteration': int(best_iter) if best_iter is not None else None,
            }

    def predict(self, X) -> pd.DataFrame:
        preds = {}
        self.probas_ = {}
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
        for i, col in enumerate(label_cols):
            proba = self.models[i].predict(dtest)
            self.probas_[col] = proba
            preds[col] = np.argmax(proba, axis=1)
        return pd.DataFrame(preds, index=getattr(X, 'index', None))

