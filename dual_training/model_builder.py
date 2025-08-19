import xgboost as xgb
import numpy as np
import pandas as pd
from . import config as cfg
from .utils import setup_logging


logger = setup_logging()


class BinaryPerLevelXGB:
    def __init__(self, feature_names):
        self.models = []
        self.feature_names = feature_names
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
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'reg_alpha': cfg.XGB_REG_ALPHA,
            'reg_lambda': cfg.XGB_REG_LAMBDA,
            'min_child_weight': cfg.XGB_MIN_CHILD_WEIGHT,
        }

    def fit(self, X_train, y_train, X_val, y_val):
        self.models = []
        params = self._params()
        dval_cache = xgb.DMatrix(X_val, label=None, feature_names=self.feature_names)
        label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
        for i, col in enumerate(label_cols, 1):
            logger.info(f"Trening poziomu {i}/{len(label_cols)}: {col} (binary)")
            ytr = (y_train[col].values == 0).astype(int)  # LONG as positive class
            yva = (y_val[col].values == 0).astype(int)
            dtrain = xgb.DMatrix(X_train, label=ytr, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=yva, feature_names=self.feature_names)

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
            self.training_curves[col] = evals_result
            best_score = getattr(model, 'best_score', None)
            best_iter = getattr(model, 'best_iteration', None)
            self.best_scores[col] = {
                'best_logloss': float(best_score) if best_score is not None else None,
                'best_iteration': int(best_iter) if best_iter is not None else None,
            }

    def predict_proba(self, X) -> pd.DataFrame:
        preds = {}
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
        for i, col in enumerate(label_cols):
            p_pos = self.models[i].predict(dtest)
            preds[col] = p_pos
        return pd.DataFrame(preds, index=getattr(X, 'index', None))

