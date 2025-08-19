import xgboost as xgb
import numpy as np
import pandas as pd
from . import config as cfg
from .utils import setup_logging


logger = setup_logging()


class MultiSymbolXGB:
    def __init__(self, feature_names, label_columns):
        self.models = []
        self.feature_names = feature_names
        self.label_columns = label_columns
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

        for i, col in enumerate(self.label_columns, 1):
            logger.info(f"Trening poziomu {i}/{len(self.label_columns)}: {col}")
            # Build DMatrices with optional class weights on training only
            if getattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING', False):
                weights_map = getattr(cfg, 'CLASS_WEIGHTS', {0: 1.0, 1: 1.0, 2: 1.0})
                ytr = y_train[col].values
                yva = y_val[col].values
                wtr = [float(weights_map.get(int(lbl), 1.0)) for lbl in ytr]
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
            best_score = getattr(model, 'best_score', None)
            best_iter = getattr(model, 'best_iteration', None)
            try:
                logger.info(f"Najlepszy validation-mlogloss ({col}): {best_score} @ iter {best_iter}")
            except Exception:
                pass
            self.best_scores[col] = {
                'best_mlogloss': float(best_score) if best_score is not None else None,
                'best_iteration': int(best_iter) if best_iter is not None else None,
            }

    def predict(self, X) -> pd.DataFrame:
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        preds = {}
        self.probas_ = {}
        for i, col in enumerate(self.label_columns):
            proba = self.models[i].predict(dtest)
            self.probas_[col] = proba
            preds[col] = np.argmax(proba, axis=1)
        return pd.DataFrame(preds, index=getattr(X, 'index', None))

