"""
Głęboka analiza przyczyny bias SHORT > LONG w XGBoost.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler

# Dodaj ścieżkę do modułu
sys.path.append(str(Path(__file__).parent))
import config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_classes(probabilities):
    """Bezpieczna funkcja do konwersji prawdopodobieństw na klasy."""
    if len(probabilities.shape) == 1:
        # multi:softmax zwraca klasy bezpośrednio
        return probabilities.astype(int)
    else:
        # multi:softprob zwraca prawdopodobieństwa
        return np.argmax(probabilities, axis=1)

def deep_xgboost_bias_analysis():
    """Głęboka analiza przyczyny bias w XGBoost."""
    
    logger.info("=== GŁĘBOKA ANALIZA PRZYCZYNY BIAS XGBOOST ===")
    
    # Wczytaj dane
    try:
        df = pd.read_feather(cfg.INPUT_FILE_PATH)
        logger.info(f"Dane wczytane: {len(df):,} wierszy")
    except Exception as e:
        logger.error(f"Błąd wczytywania danych: {e}")
        return
    
    # Ustaw indeks na timestamp jeśli istnieje
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info("Ustawiono timestamp jako indeks")
    
    # Wybierz cechy i etykiety
    X = df[cfg.FEATURES]
    y = df[cfg.LABEL_COLUMNS]
    
    # Usuń wiersze z brakującymi danymi
    mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
    X = X[mask]
    y = y[mask]
    
    # Chronologiczny podział danych
    total_samples = len(X)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size+val_size]
    X_test = X.iloc[train_size+val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size+val_size]
    y_test = y.iloc[train_size+val_size:]
    
    # Analiza dla pierwszego poziomu
    label_col = cfg.LABEL_COLUMNS[0]
    level_desc = cfg.TP_SL_LEVELS_DESC[0]
    
    logger.info(f"\n--- ANALIZA POZIOMU: {level_desc} ---")
    
    # Sprawdź rozkład klas
    train_counts = y_train[label_col].value_counts().sort_index()
    logger.info("Rozkład klas w train:")
    logger.info(f"  LONG={train_counts.get(0, 0):,}, SHORT={train_counts.get(1, 0):,}, NEUTRAL={train_counts.get(2, 0):,}")
    
    # Test 1: Sprawdź czy problem jest w objective function
    logger.info(f"\n--- TEST 1: ANALIZA OBJECTIVE FUNCTION ---")
    
    # Sprawdź różne objective functions
    objectives = ['multi:softprob', 'multi:softmax']
    
    for objective in objectives:
        logger.info(f"\nTestowanie objective: {objective}")
        
        xgb_params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': 42,
            'verbosity': 0,
            'objective': objective,
            'num_class': 3,
            'eval_metric': 'mlogloss'
        }
        
        # Skaluj dane
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Trenuj model
        y_train_level = y_train[label_col]
        y_val_level = y_val[label_col]
        
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train_level)
        dval = xgb.DMatrix(X_val_scaled, label=y_val_level)
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=50,
            evals=[(dval, 'validation')],
            verbose_eval=False
        )
        
        # Predykcje
        probabilities = model.predict(dval)
        predictions = predict_classes(probabilities)
        
        # Analiza bias
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        long_pred = pred_counts.get(0, 0)
        short_pred = pred_counts.get(1, 0)
        
        logger.info(f"  Predykcje: LONG={long_pred}, SHORT={short_pred}")
        if short_pred > long_pred:
            logger.warning(f"  ⚠️ SHORT > LONG bias: {short_pred/long_pred:.2f}x")
        else:
            logger.info(f"  ✅ LONG >= SHORT")
    
    # Test 2: Sprawdź czy problem jest w eval_metric
    logger.info(f"\n--- TEST 2: ANALIZA EVAL METRIC ---")
    
    eval_metrics = ['mlogloss', 'merror']
    
    for metric in eval_metrics:
        logger.info(f"\nTestowanie eval_metric: {metric}")
        
        xgb_params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': 42,
            'verbosity': 0,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': metric
        }
        
        # Skaluj dane
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Trenuj model
        y_train_level = y_train[label_col]
        y_val_level = y_val[label_col]
        
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train_level)
        dval = xgb.DMatrix(X_val_scaled, label=y_val_level)
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=50,
            evals=[(dval, 'validation')],
            verbose_eval=False
        )
        
        # Predykcje
        probabilities = model.predict(dval)
        predictions = predict_classes(probabilities)
        
        # Analiza bias
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        long_pred = pred_counts.get(0, 0)
        short_pred = pred_counts.get(1, 0)
        
        logger.info(f"  Predykcje: LONG={long_pred}, SHORT={short_pred}")
        if short_pred > long_pred:
            logger.warning(f"  ⚠️ SHORT > LONG bias: {short_pred/long_pred:.2f}x")
        else:
            logger.info(f"  ✅ LONG >= SHORT")
    
    # Test 3: Sprawdź czy problem jest w sample weights
    logger.info(f"\n--- TEST 3: ANALIZA SAMPLE WEIGHTS ---")
    
    # Oblicz sample weights dla klas mniejszościowych
    class_counts = y_train_level.value_counts().sort_index()
    total_samples = len(y_train_level)
    
    # Wagi odwrotnie proporcjonalne do liczby próbek
    sample_weights = np.ones(len(y_train_level))
    for class_idx in [0, 1]:  # LONG, SHORT
        class_mask = y_train_level == class_idx
        sample_weights[class_mask] = total_samples / (len(class_counts) * class_counts[class_idx])
    
    logger.info(f"Sample weights: LONG={sample_weights[y_train_level == 0][0]:.2f}, SHORT={sample_weights[y_train_level == 1][0]:.2f}")
    
    xgb_params = {
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss'
    }
    
    # Skaluj dane
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Trenuj model z sample weights
    y_train_level = y_train[label_col]
    y_val_level = y_val[label_col]
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_level, weight=sample_weights)
    dval = xgb.DMatrix(X_val_scaled, label=y_val_level)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=50,
        evals=[(dval, 'validation')],
        verbose_eval=False
    )
    
    # Predykcje
    probabilities = model.predict(dval)
    predictions = predict_classes(probabilities)
    
    # Analiza bias
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    long_pred = pred_counts.get(0, 0)
    short_pred = pred_counts.get(1, 0)
    
    logger.info(f"Z sample weights:")
    logger.info(f"  Predykcje: LONG={long_pred}, SHORT={short_pred}")
    if short_pred > long_pred:
        logger.warning(f"  ⚠️ SHORT > LONG bias: {short_pred/long_pred:.2f}x")
    else:
        logger.info(f"  ✅ LONG >= SHORT")
    
    # Test 4: Sprawdź czy problem jest w feature importance
    logger.info(f"\n--- TEST 4: ANALIZA FEATURE IMPORTANCE ---")
    
    # Sprawdź czy niektóre cechy faworyzują SHORT
    feature_importance = model.get_score(importance_type='gain')
    
    # Mapuj feature indices na nazwy
    feature_map = {}
    for i, feature_name in enumerate(cfg.FEATURES):
        feature_map[f'f{i}'] = feature_name
    
    logger.info("Top 10 cech z największą ważnością:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feat_idx, importance) in enumerate(sorted_features[:10]):
        feat_name = feature_map.get(feat_idx, feat_idx)
        logger.info(f"  {i+1}. {feat_name}: {importance:.4f}")
    
    # Test 5: Sprawdź czy problem jest w danych (SHORT vs LONG samples)
    logger.info(f"\n--- TEST 5: ANALIZA DANYCH SHORT vs LONG ---")
    
    # Sprawdź czy SHORT i LONG mają różne charakterystyki cech
    long_mask = y_train_level == 0
    short_mask = y_train_level == 1
    
    logger.info(f"Liczba próbek: LONG={np.sum(long_mask):,}, SHORT={np.sum(short_mask):,}")
    
    # Sprawdź różnice w cechach między LONG a SHORT
    logger.info("Różnice w cechach (LONG vs SHORT):")
    for i, feature in enumerate(cfg.FEATURES[:10]):  # Pierwsze 10 cech
        long_mean = X_train[feature][long_mask].mean()
        short_mean = X_train[feature][short_mask].mean()
        diff = abs(long_mean - short_mean)
        
        logger.info(f"  {feature}: LONG={long_mean:.4f}, SHORT={short_mean:.4f}, diff={diff:.4f}")
        
        if diff > 0.1:  # Znacząca różnica
            logger.warning(f"    ⚠️ Znacząca różnica w {feature}")
    
    # Test 6: Sprawdź czy problem jest w random_state
    logger.info(f"\n--- TEST 6: ANALIZA RANDOM STATE ---")
    
    random_states = [42, 123, 456, 789, 999]
    
    for rs in random_states:
        logger.info(f"\nTestowanie random_state: {rs}")
        
        xgb_params['random_state'] = rs
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=50,
            evals=[(dval, 'validation')],
            verbose_eval=False
        )
        
        # Predykcje
        probabilities = model.predict(dval)
        predictions = predict_classes(probabilities)
        
        # Analiza bias
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        long_pred = pred_counts.get(0, 0)
        short_pred = pred_counts.get(1, 0)
        
        logger.info(f"  Predykcje: LONG={long_pred}, SHORT={short_pred}")
        if short_pred > long_pred:
            logger.warning(f"  ⚠️ SHORT > LONG bias: {short_pred/long_pred:.2f}x")
        else:
            logger.info(f"  ✅ LONG >= SHORT")
    
    # Podsumowanie
    logger.info(f"\n--- PODSUMOWANIE GŁĘBOKIEJ ANALIZY ---")
    logger.info("Sprawdzone potencjalne przyczyny bias:")
    logger.info("  1. Objective function - różne typy")
    logger.info("  2. Eval metric - różne metryki")
    logger.info("  3. Sample weights - wagi dla klas mniejszościowych")
    logger.info("  4. Feature importance - analiza cech")
    logger.info("  5. Dane SHORT vs LONG - różnice w cechach")
    logger.info("  6. Random state - różne seedy")
    
    logger.info("\nWNIOSEK: Sprawdź wyniki powyżej aby znaleźć konkretną przyczynę.")

if __name__ == "__main__":
    deep_xgboost_bias_analysis() 