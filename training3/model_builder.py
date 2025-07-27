"""
Budowanie i trening modelu XGBoost Multi-Output używając native API.
"""
import xgboost as xgb
import numpy as np
import pandas as pd
import logging
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from training3 import config as cfg
from tqdm import tqdm


logger = logging.getLogger(__name__)

class MultiOutputXGBoost:
    """
    Klasa do trenowania osobnych modeli XGBoost dla 5 poziomów TP/SL.
    Używa native XGBoost API z early stopping i prawdopodobieństwami.
    """
    def __init__(self):
        self.models = []  # lista 5 modeli XGBoost (native)
        self.feature_importances = None

    def build_model(self):
        # Parametry XGBoost dla native API
        xgb_params = {
            'max_depth': cfg.XGB_MAX_DEPTH,
            'learning_rate': cfg.XGB_LEARNING_RATE,
            'subsample': cfg.XGB_SUBSAMPLE,
            'colsample_bytree': cfg.XGB_COLSAMPLE_BYTREE,
            'gamma': cfg.XGB_GAMMA,
            'random_state': cfg.XGB_RANDOM_STATE,
            'verbosity': 0,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss'
        }
        return xgb_params

    def train_model(self, X_train, y_train, X_val, y_val):
        logger.info("Rozpoczynanie treningu osobnych modeli XGBoost (native API) dla każdego poziomu TP/SL...")
        self.models = []
        xgb_params = self.build_model()
        start_time = time.time()
        
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            level_desc = cfg.TP_SL_LEVELS_DESC[i]
            logger.info(f"=== TRENING POZIOMU {i+1}/5 ({level_desc}) ===")
            y_level = y_train[label_col]
            
            # Balansowanie klas dla tego poziomu
            if cfg.ENABLE_CLASS_BALANCING:
                X_bal, y_bal = self._balance_single_level(X_train, y_level)
                logger.info(f"  Po balansowaniu: {len(X_bal)} próbek")
            else:
                X_bal, y_bal = X_train, y_level
                logger.info(f"  Bez balansowania: {len(X_bal)} próbek")
                
                # Loguj rozkład klas dla tego poziomu
                class_counts = pd.Series(y_bal).value_counts().sort_index()
                logger.info(f"  Rozkład klas: LONG={class_counts.get(0, 0):,}, SHORT={class_counts.get(1, 0):,}, NEUTRAL={class_counts.get(2, 0):,}")
            
            # Przygotuj dane dla native API
            dtrain = xgb.DMatrix(X_bal, label=y_bal)
            dval = xgb.DMatrix(X_val, label=y_val[label_col])
            
            logger.info(f"  Rozpoczynanie treningu modelu {i+1}...")
            
            # Trening z early stopping
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=cfg.XGB_N_ESTIMATORS,
                evals=[(dval, 'validation')],
                early_stopping_rounds=cfg.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=True
            )
            
            # Loguj informacje o treningu
            best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else cfg.XGB_N_ESTIMATORS
            best_score = model.best_score if hasattr(model, 'best_score') else "N/A"
            logger.info(f"  Model {i+1} wytrenowany. Najlepsza iteracja: {best_iteration}, Best Score: {best_score}")
            
            # Sprawdź czy early stopping zadziałało
            if hasattr(model, 'best_iteration') and model.best_iteration < cfg.XGB_N_ESTIMATORS:
                logger.info(f"  [OK] Early stopping zadziałało! Model zatrzymany na iteracji {model.best_iteration}")
            else:
                logger.info(f"  [INFO] Model trenował się do końca ({cfg.XGB_N_ESTIMATORS} iteracji)")
            
            self.models.append(model)
        
        total_time = time.time() - start_time
        logger.info(f"Całkowity czas treningu: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Predykcje walidacyjne
        y_val_pred = self.predict(X_val)
        self._log_validation_metrics(y_val, y_val_pred)
        logger.info("Trening wszystkich modeli zakończony pomyślnie.")

    def _balance_single_level(self, X, y):
        # Prosta implementacja: undersampling klasy większościowej
        from sklearn.utils import resample
        df = pd.DataFrame(X)
        df['label'] = y.values if hasattr(y, 'values') else y
        min_count = df['label'].value_counts().min()
        dfs = [resample(df[df['label']==cls], replace=False, n_samples=min_count, random_state=42) for cls in df['label'].unique()]
        df_bal = pd.concat(dfs)
        X_bal = df_bal.drop('label', axis=1).values
        y_bal = df_bal['label'].values
        return X_bal, y_bal

    def _calculate_single_level_weights(self, y):
        # Wagi odwrotnie proporcjonalne do liczności klas
        from collections import Counter
        counts = Counter(y)
        total = sum(counts.values())
        weights = {cls: total/counts[cls] for cls in counts}
        return np.array([weights[cls] for cls in y])

    def predict(self, X):
        """
        Zwraca DataFrame z predykcjami dla każdego poziomu.
        Używa native API - zwraca prawdopodobieństwa.
        """
        preds = {}
        for i, model in enumerate(self.models):
            dtest = xgb.DMatrix(X)
            # Native API zwraca prawdopodobieństwa
            probabilities = model.predict(dtest)
            # Konwertuj na klasy (argmax)
            predictions = np.argmax(probabilities.reshape(-1, 3), axis=1)
            preds[cfg.LABEL_COLUMNS[i]] = predictions
        return pd.DataFrame(preds, index=getattr(X, 'index', None))

    def predict_proba(self, X):
        """
        Zwraca prawdopodobieństwa dla każdego poziomu.
        Native API - zwraca surowe prawdopodobieństwa.
        """
        probas = {}
        for i, model in enumerate(self.models):
            dtest = xgb.DMatrix(X)
            # Native API zwraca prawdopodobieństwa
            probabilities = model.predict(dtest)
            # Reshape do formatu [n_samples, n_classes]
            probas[cfg.LABEL_COLUMNS[i]] = probabilities.reshape(-1, 3)
        return probas

    def get_feature_importance(self):
        # Zwraca średnią ważność cech ze wszystkich modeli
        importances = [model.get_score(importance_type='gain') for model in self.models]
        # Konwertuj słowniki na array
        avg_importances = np.zeros(len(cfg.FEATURES))
        
        for imp_dict in importances:
            for feat_name, importance in imp_dict.items():
                try:
                    # Native API używa 'f0', 'f1', etc.
                    if feat_name.startswith('f'):
                        feat_idx = int(feat_name[1:])
                        if feat_idx < len(cfg.FEATURES):
                            avg_importances[feat_idx] += importance
                except (ValueError, IndexError):
                    # Ignoruj nieprawidłowe nazwy cech
                    continue
        
        avg_importances /= len(self.models)
        return avg_importances

    def save_model(self, filepath):
        import joblib
        for i, model in enumerate(self.models):
            model.save_model(f"{filepath}_level{i+1}.json")
        logger.info(f"Modele zapisane do {filepath}_level*.json")

    def load_model(self, filepath):
        import joblib
        self.models = []
        for i in range(5):
            model = xgb.Booster()
            model.load_model(f"{filepath}_level{i+1}.json")
            self.models.append(model)
        logger.info(f"Modele wczytane z {filepath}_level*.json")

    def _log_validation_metrics(self, y_val_true, y_val_pred):
        logger.info("Metryki walidacyjne:")
        for i, label_col in enumerate(cfg.LABEL_COLUMNS):
            level_desc = cfg.TP_SL_LEVELS_DESC[i]
            accuracy = accuracy_score(y_val_true[label_col], y_val_pred[label_col])
            cm = confusion_matrix(y_val_true[label_col], y_val_pred[label_col])
            class_report = classification_report(
                y_val_true[label_col], 
                y_val_pred[label_col],
                target_names=['LONG', 'SHORT', 'NEUTRAL'],
                output_dict=True,
                zero_division=0
            )
            logger.info(f"  {level_desc}:")
            logger.info(f"    Accuracy: {accuracy:.4f}")
            logger.info(f"    LONG: P={class_report['LONG']['precision']:.3f}, R={class_report['LONG']['recall']:.3f}, F1={class_report['LONG']['f1-score']:.3f}")
            logger.info(f"    SHORT: P={class_report['SHORT']['precision']:.3f}, R={class_report['SHORT']['recall']:.3f}, F1={class_report['SHORT']['f1-score']:.3f}")
            logger.info(f"    NEUTRAL: P={class_report['NEUTRAL']['precision']:.3f}, R={class_report['NEUTRAL']['recall']:.3f}, F1={class_report['NEUTRAL']['f1-score']:.3f}")
            logger.info("    Confusion Matrix:")
            logger.info("                Predicted")
            logger.info("    Actual    LONG  SHORT  NEUTRAL")
            logger.info(f"    LONG     {cm[0][0]:4d}  {cm[0][1]:5d}  {cm[0][2]:7d}")
            logger.info(f"    SHORT    {cm[1][0]:4d}  {cm[1][1]:5d}  {cm[1][2]:7d}")
            logger.info(f"    NEUTRAL  {cm[2][0]:4d}  {cm[2][1]:5d}  {cm[2][2]:7d}") 