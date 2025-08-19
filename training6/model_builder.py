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


class BinaryDirectionalXGB:
    """Two independent binary models per TP/SL level: LONG vs rest and SHORT vs rest.

    - Keeps NEUTRAL as negative examples with a small sample weight.
    - Provides calibrated probabilities P_long and P_short for each level.
    """

    def __init__(self, feature_names, n_estimators: int | None = None, early_stopping_rounds: int | None = None, verbose_eval: int | None = None, param_overrides: dict | None = None):
        self.feature_names = feature_names
        self.model_long = {}
        self.model_short = {}
        self.training_info_ = {}
        self.best_scores = {}
        # Optional overrides for speed/experiments
        self._n_estimators = int(n_estimators) if n_estimators is not None else None
        self._early_stopping_rounds = int(early_stopping_rounds) if early_stopping_rounds is not None else None
        self._verbose_eval = int(verbose_eval) if verbose_eval is not None else None
        self._param_overrides = dict(param_overrides) if param_overrides is not None else None

    def _params(self):
        params = {
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
        if self._param_overrides:
            params.update(self._param_overrides)
        return params

    def _build_binary_targets(self, y_df, pos_class: int):
        """Map 3-class labels to binary target and sample weights.

        pos_class=0 -> LONG model; pos_class=1 -> SHORT model.
        """
        y = y_df.values.astype(int)
        # target: 1 for pos_class, 0 otherwise
        target = (y == pos_class).astype(float)
        # weights: positive=1.0, opposite=1.0, neutral=low
        neutral_mask = (y == 2)
        opp_mask = (y == (1 - pos_class))
        pos_mask = (y == pos_class)
        weights = (
            pos_mask * cfg.BINARY_POSITIVE_WEIGHT +
            opp_mask * cfg.BINARY_NEGATIVE_WEIGHT +
            neutral_mask * cfg.BINARY_NEUTRAL_WEIGHT
        ).astype(float)
        return target.ravel(), weights.ravel()

    def fit(self, X_train, y_train, X_val, y_val):
        params = self._params()
        label_cols = getattr(cfg, 'RESOLVED_LABEL_COLUMNS', cfg.LABEL_COLUMNS)
        self.training_info_ = {}
        self.best_scores = {}
        for col in label_cols:
            try:
                lvl_idx = label_cols.index(col)
                tp, sl = cfg.TP_SL_LEVELS[lvl_idx]
                logger.info(f"Trening poziomu {lvl_idx+1}/{len(label_cols)} ({col}) -> TP={tp}%, SL={sl}% [LONG]")
            except Exception:
                logger.info(f"Trening poziomu ({col}) [LONG]")
            # LONG model for this level
            ytr_long, wtr_long = self._build_binary_targets(y_train[col], pos_class=0)
            yva_long, wva_long = self._build_binary_targets(y_val[col], pos_class=0)
            dtrain_long = xgb.DMatrix(X_train, label=ytr_long, weight=wtr_long, feature_names=self.feature_names)
            dval_long = xgb.DMatrix(X_val, label=yva_long, weight=wva_long, feature_names=self.feature_names)
            evals_result_long = {}
            m_long = xgb.train(
                params,
                dtrain_long,
                num_boost_round=self._n_estimators or cfg.XGB_N_ESTIMATORS,
                evals=[(dval_long, 'validation')],
                early_stopping_rounds=self._early_stopping_rounds or cfg.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=self._verbose_eval if self._verbose_eval is not None else cfg.TRAIN_VERBOSE_EVAL,
                evals_result=evals_result_long,
            )
            self.model_long[col] = m_long

            # SHORT model for this level
            try:
                logger.info(f"Trening poziomu {lvl_idx+1}/{len(label_cols)} ({col}) -> TP={tp}%, SL={sl}% [SHORT]")
            except Exception:
                logger.info(f"Trening poziomu ({col}) [SHORT]")
            ytr_short, wtr_short = self._build_binary_targets(y_train[col], pos_class=1)
            yva_short, wva_short = self._build_binary_targets(y_val[col], pos_class=1)
            dtrain_short = xgb.DMatrix(X_train, label=ytr_short, weight=wtr_short, feature_names=self.feature_names)
            dval_short = xgb.DMatrix(X_val, label=yva_short, weight=wva_short, feature_names=self.feature_names)
            evals_result_short = {}
            m_short = xgb.train(
                params,
                dtrain_short,
                num_boost_round=self._n_estimators or cfg.XGB_N_ESTIMATORS,
                evals=[(dval_short, 'validation')],
                early_stopping_rounds=self._early_stopping_rounds or cfg.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=self._verbose_eval if self._verbose_eval is not None else cfg.TRAIN_VERBOSE_EVAL,
                evals_result=evals_result_short,
            )
            self.model_short[col] = m_short

            self.training_info_[col] = {'long': evals_result_long, 'short': evals_result_short}

    def predict_proba_level(self, X, col: str) -> pd.DataFrame:
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        p_long = self.model_long[col].predict(dtest)
        p_short = self.model_short[col].predict(dtest)
        return pd.DataFrame({'P_long': p_long, 'P_short': p_short}, index=getattr(X, 'index', None))

