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
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


class XGBoostWrapper:
    """Wrapper dla native XGBoost model kompatybilny z sklearn"""
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        self.classes_ = np.array([0, 1, 2])  # LONG, SHORT, NEUTRAL
    
    def predict_proba(self, X):
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        probs = self.xgb_model.predict(dtest)
        return probs.reshape(-1, 3)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)



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
        
        # Dodaj wagi klas jeśli włączone
        if hasattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING') and cfg.ENABLE_CLASS_WEIGHTS_IN_TRAINING:
            # Dla multi-class, używamy sample_weight zamiast scale_pos_weight
            # scale_pos_weight działa tylko dla binary classification
            logger.info(f"Wagi klas WŁĄCZONE: {cfg.CLASS_WEIGHTS}")
        else:
            logger.info("Wagi klas WYŁĄCZONE - standardowe trening")
            
        return xgb_params

    def train_model(self, X_train, y_train, X_val, y_val):
        logger.info("Rozpoczynanie treningu osobnych modeli XGBoost (native API) dla każdego poziomu TP/SL...")
        
        # Ustaw nazwy cech
        self.feature_names = X_train.columns.tolist()
        logger.info(f"Nazwy cech ustawione: {len(self.feature_names)} cech")
        
        self.models = []
        xgb_params = self.build_model()
        start_time = time.time()
        
        self.level_logloss = []
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
            
            # Przygotuj dane dla native API z wagami klas
            if hasattr(cfg, 'ENABLE_CLASS_WEIGHTS_IN_TRAINING') and cfg.ENABLE_CLASS_WEIGHTS_IN_TRAINING:
                # Oblicz wagi dla każdej próbki na podstawie jej klasy
                sample_weights = np.array([cfg.CLASS_WEIGHTS[int(y)] for y in y_bal])
                dtrain = xgb.DMatrix(X_bal, label=y_bal, weight=sample_weights, feature_names=self.feature_names)
                logger.info(f"  Wagi próbek: LONG={cfg.CLASS_WEIGHTS[0]}, SHORT={cfg.CLASS_WEIGHTS[1]}, NEUTRAL={cfg.CLASS_WEIGHTS[2]}")
            else:
                dtrain = xgb.DMatrix(X_bal, label=y_bal, feature_names=self.feature_names)
            
            # Walidacja bez wag (standardowe)
            dval = xgb.DMatrix(X_val, label=y_val[label_col], feature_names=self.feature_names)
            
            logger.info(f"  Rozpoczynanie treningu modelu {i+1}...")
            
            # Trening z early stopping
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=cfg.XGB_N_ESTIMATORS,
                evals=[(dval, 'validation')],
                early_stopping_rounds=cfg.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=50
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

        # Zapisz podsumowanie logloss per poziom do raportu (latest + timestamp)
        try:
            rows = []
            for i, model in enumerate(self.models):
                level_desc = cfg.TP_SL_LEVELS_DESC[i]
                best_it = getattr(model, 'best_iteration', None)
                best_sc = getattr(model, 'best_score', None)
                rows.append((i+1, level_desc, best_it, best_sc))
            import csv
            cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            for fname in [f'logloss_summary_{ts}.csv', 'logloss_summary_latest.csv']:
                outp = Path(cfg.REPORT_DIR) / fname
                with open(outp, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['level_index', 'level_desc', 'best_iteration', 'best_logloss'])
                    w.writerows(rows)
            logger.info("Zapisano podsumowanie logloss: logloss_summary_latest.csv")
        except Exception as e:
            logger.warning(f"Nie udało się zapisać podsumowania logloss: {e}")
        
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
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
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
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            # Native API zwraca prawdopodobieństwa
            probabilities = model.predict(dtest)
            # Reshape do formatu [n_samples, n_classes]
            probas[cfg.LABEL_COLUMNS[i]] = probabilities.reshape(-1, 3)
        return probas

    def get_feature_importance(self):
        # Zwraca średnią ważność cech ze wszystkich modeli
        importances = [model.get_score(importance_type='gain') for model in self.models]
        # Konwertuj słowniki na array - użyj dynamicznej liczby cech
        if hasattr(self, 'feature_names'):
            n_features = len(self.feature_names)
        else:
            n_features = len(cfg.FEATURES)
            
        avg_importances = np.zeros(n_features)
        
        for imp_dict in importances:
            for feat_name, importance in imp_dict.items():
                try:
                    # Native API używa 'f0', 'f1', etc.
                    if feat_name.startswith('f'):
                        feat_idx = int(feat_name[1:])
                        if feat_idx < n_features:
                            avg_importances[feat_idx] += importance
                except (ValueError, IndexError):
                    # Ignoruj nieprawidłowe nazwy cech
                    continue
        
        avg_importances /= len(self.models)
        return avg_importances

    def save_model(self, filepath):
        """
        Zapisuje każdy model osobno w formacie JSON dla łatwego użycia w FreqTrade.
        Każdy model ma jasną nazwę z poziomem TP/SL.
        """
        import os
        import re
        
        # Utwórz katalog jeśli nie istnieje
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Zapisz każdy model osobno
        for i, model in enumerate(self.models):
            level_desc = cfg.TP_SL_LEVELS_DESC[i]
            
            # Wyciągnij TP/SL z opisu
            match = re.search(r'TP: ([\d.]+)%, SL: ([\d.]+)%', level_desc)
            if match:
                tp = match.group(1).replace('.', 'p')  # 0.6 -> 0p6
                sl = match.group(2).replace('.', 'p')  # 0.3 -> 0p3
                model_filename = f"model_tp{tp}_sl{sl}.json"
            else:
                model_filename = f"model_level{i+1}.json"
            
            # Pełna ścieżka do modelu
            model_path = os.path.join(os.path.dirname(filepath), model_filename)
            
            # Zapisz model w formacie JSON
            model.save_model(model_path)
            logger.info(f"Model {i+1} ({level_desc}) zapisany: {model_path}")
        
        logger.info(f"Wszystkie modele zapisane osobno w katalogu: {os.path.dirname(filepath)}")
        
        # Zapisz informacje o modelach do pliku index
        models_info = []
        for i, level_desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            match = re.search(r'TP: ([\d.]+)%, SL: ([\d.]+)%', level_desc)
            if match:
                tp = match.group(1).replace('.', 'p')
                sl = match.group(2).replace('.', 'p')
                model_filename = f"model_tp{tp}_sl{sl}.json"
            else:
                model_filename = f"model_level{i+1}.json"
            
            models_info.append({
                'index': i,
                'filename': model_filename,
                'description': level_desc,
                'tp_sl': match.groups() if match else None
            })
        
        # Zapisz index modeli
        import json
        index_path = os.path.join(os.path.dirname(filepath), 'models_index.json')
        with open(index_path, 'w') as f:
            json.dump(models_info, f, indent=2)
        logger.info(f"Index modeli zapisany: {index_path}")

    def load_model(self, filepath):
        """Wczytuje modele z nowego formatu (osobne pliki JSON)."""
        import os
        import re
        
        self.models = []
        
        # Wczytaj modele z katalogu
        model_dir = os.path.dirname(filepath)
        
        for i, level_desc in enumerate(cfg.TP_SL_LEVELS_DESC):
            model = xgb.Booster()
            
            # Wyciągnij TP/SL z opisu
            match = re.search(r'TP: ([\d.]+)%, SL: ([\d.]+)%', level_desc)
            if match:
                tp = match.group(1).replace('.', 'p')  # 0.6 -> 0p6
                sl = match.group(2).replace('.', 'p')  # 0.3 -> 0p3
                model_filename = f"model_tp{tp}_sl{sl}.json"
            else:
                model_filename = f"model_level{i+1}.json"
            
            model_path = os.path.join(model_dir, model_filename)
            
            try:
                model.load_model(model_path)
                logger.info(f"Model {i+1} ({level_desc}) wczytany z: {model_path}")
            except Exception as e:
                logger.error(f"Błąd wczytywania modelu {i+1}: {e}")
                raise
            
            self.models.append(model)
        
        logger.info(f"Wszystkie modele wczytane pomyślnie z katalogu: {model_dir}")

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