"""
Udowodnienie hipotezy o przyczynie bias SHORT > LONG w XGBoost.
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

def prove_bias_hypothesis():
    """Udowadnia hipotezę o przyczynie bias SHORT > LONG."""
    
    logger.info("=== UDOWODNIENIE HIPOTEZY BIAS SHORT > LONG ===")
    
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
    
    # Test 1: Sprawdź czy SHORT ma bardziej "wyraźne" sygnały
    logger.info(f"\n--- TEST 1: ANALIZA WYRAŹNOŚCI SYGNAŁÓW ---")
    
    y_train_level = y_train[label_col]
    long_mask = y_train_level == 0
    short_mask = y_train_level == 1
    
    # Sprawdź wariancję cech dla LONG vs SHORT
    logger.info("Wariancja cech (LONG vs SHORT):")
    for i, feature in enumerate(cfg.FEATURES[:15]):  # Pierwsze 15 cech
        long_var = X_train[feature][long_mask].var()
        short_var = X_train[feature][short_mask].var()
        
        logger.info(f"  {feature}: LONG_var={long_var:.6f}, SHORT_var={short_var:.6f}")
        
        if short_var > long_var * 1.2:  # SHORT ma 20% większą wariancję
            logger.warning(f"    ⚠️ SHORT ma większą wariancję w {feature}")
    
    # Test 2: Sprawdź czy SHORT ma bardziej "ekstremalne" wartości
    logger.info(f"\n--- TEST 2: ANALIZA EKSTREMALNOŚCI WARTOŚCI ---")
    
    logger.info("Ekstremalne wartości (LONG vs SHORT):")
    for i, feature in enumerate(cfg.FEATURES[:10]):  # Pierwsze 10 cech
        long_min = X_train[feature][long_mask].min()
        long_max = X_train[feature][long_mask].max()
        short_min = X_train[feature][short_mask].min()
        short_max = X_train[feature][short_mask].max()
        
        long_range = long_max - long_min
        short_range = short_max - short_min
        
        logger.info(f"  {feature}:")
        logger.info(f"    LONG: min={long_min:.4f}, max={long_max:.4f}, range={long_range:.4f}")
        logger.info(f"    SHORT: min={short_min:.4f}, max={short_max:.4f}, range={short_range:.4f}")
        
        if short_range > long_range * 1.1:  # SHORT ma 10% większy zakres
            logger.warning(f"    ⚠️ SHORT ma większy zakres w {feature}")
    
    # Test 3: Sprawdź czy SHORT ma bardziej "jednoznaczne" wzorce
    logger.info(f"\n--- TEST 3: ANALIZA JEDNOZNACZNOŚCI WZORCÓW ---")
    
    # Sprawdź korelacje między cechami dla LONG vs SHORT
    logger.info("Korelacje między cechami (LONG vs SHORT):")
    
    # Wybierz kilka kluczowych cech
    key_features = ['rsi_14', 'macd_hist', 'bb_position', 'price_to_ma_60', 'price_to_ma_240']
    
    for i, feat1 in enumerate(key_features):
        for j, feat2 in enumerate(key_features[i+1:], i+1):
            long_corr = X_train[feat1][long_mask].corr(X_train[feat2][long_mask])
            short_corr = X_train[feat1][short_mask].corr(X_train[feat2][short_mask])
            
            logger.info(f"  {feat1} vs {feat2}: LONG_corr={long_corr:.3f}, SHORT_corr={short_corr:.3f}")
            
            if abs(short_corr) > abs(long_corr) * 1.2:  # SHORT ma 20% silniejszą korelację
                logger.warning(f"    ⚠️ SHORT ma silniejszą korelację między {feat1} i {feat2}")
    
    # Test 4: Sprawdź czy SHORT ma bardziej "separable" wzorce
    logger.info(f"\n--- TEST 4: ANALIZA SEPAROWALNOŚCI WZORCÓW ---")
    
    # Sprawdź czy SHORT jest łatwiejszy do odróżnienia od NEUTRAL niż LONG
    neutral_mask = y_train_level == 2
    
    logger.info("Separowalność od NEUTRAL (LONG vs SHORT):")
    for i, feature in enumerate(cfg.FEATURES[:10]):  # Pierwsze 10 cech
        long_mean = X_train[feature][long_mask].mean()
        short_mean = X_train[feature][short_mask].mean()
        neutral_mean = X_train[feature][neutral_mask].mean()
        
        long_neutral_diff = abs(long_mean - neutral_mean)
        short_neutral_diff = abs(short_mean - neutral_mean)
        
        logger.info(f"  {feature}:")
        logger.info(f"    LONG-NEUTRAL diff: {long_neutral_diff:.4f}")
        logger.info(f"    SHORT-NEUTRAL diff: {short_neutral_diff:.4f}")
        
        if short_neutral_diff > long_neutral_diff * 1.2:  # SHORT 20% bardziej różni się od NEUTRAL
            logger.warning(f"    ⚠️ SHORT bardziej różni się od NEUTRAL w {feature}")
    
    # Test 5: Sprawdź czy SHORT ma bardziej "konsystentne" wzorce
    logger.info(f"\n--- TEST 5: ANALIZA KONSYSTENCJI WZORCÓW ---")
    
    # Sprawdź czy SHORT ma bardziej konsystentne wzorce (mniejsza wariancja w ramach klasy)
    logger.info("Konsystencja wzorców (LONG vs SHORT):")
    for i, feature in enumerate(cfg.FEATURES[:10]):  # Pierwsze 10 cech
        long_std = X_train[feature][long_mask].std()
        short_std = X_train[feature][short_mask].std()
        
        # Mniejsza wariancja = bardziej konsystentny wzorzec
        if short_std < long_std * 0.8:  # SHORT ma 20% mniejszą wariancję
            logger.warning(f"    ⚠️ SHORT ma bardziej konsystentny wzorzec w {feature} (std: {short_std:.4f} vs {long_std:.4f})")
    
    # Test 6: Sprawdź czy SHORT ma bardziej "wyraźne" decyzyjne granice
    logger.info(f"\n--- TEST 6: ANALIZA GRANIC DECYZYJNYCH ---")
    
    # Trenuj prosty model i sprawdź granice decyzyjne
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
    
    # Oblicz sample weights dla klas mniejszościowych
    class_counts = y_train_level.value_counts().sort_index()
    total_samples = len(y_train_level)
    
    sample_weights = np.ones(len(y_train_level))
    for class_idx in [0, 1]:  # LONG, SHORT
        class_mask = y_train_level == class_idx
        sample_weights[class_mask] = total_samples / (len(class_counts) * class_counts[class_idx])
    
    # Trenuj model
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
    
    # Sprawdź feature importance
    feature_importance = model.get_score(importance_type='gain')
    
    # Mapuj feature indices na nazwy
    feature_map = {}
    for i, feature_name in enumerate(cfg.FEATURES):
        feature_map[f'f{i}'] = feature_name
    
    logger.info("Feature importance (top 15):")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feat_idx, importance) in enumerate(sorted_features[:15]):
        feat_name = feature_map.get(feat_idx, feat_idx)
        logger.info(f"  {i+1}. {feat_name}: {importance:.4f}")
    
    # Test 7: Sprawdź czy SHORT ma bardziej "wyraźne" prawdopodobieństwa
    logger.info(f"\n--- TEST 7: ANALIZA PRAWDOPODOBIEŃSTW ---")
    
    # Predykcje
    probabilities = model.predict(dval)
    predictions = np.argmax(probabilities, axis=1)
    
    # Sprawdź pewność predykcji dla LONG vs SHORT
    y_val_level = y_val[label_col]
    
    long_val_mask = y_val_level == 0
    short_val_mask = y_val_level == 1
    
    if np.sum(long_val_mask) > 0 and np.sum(short_val_mask) > 0:
        # Prawdopodobieństwa dla prawdziwych LONG
        long_probs = probabilities[long_val_mask]
        long_conf = np.max(long_probs, axis=1)
        
        # Prawdopodobieństwa dla prawdziwych SHORT
        short_probs = probabilities[short_val_mask]
        short_conf = np.max(short_probs, axis=1)
        
        logger.info(f"Pewność predykcji:")
        logger.info(f"  LONG: mean_conf={long_conf.mean():.4f}, std_conf={long_conf.std():.4f}")
        logger.info(f"  SHORT: mean_conf={short_conf.mean():.4f}, std_conf={short_conf.std():.4f}")
        
        if short_conf.mean() > long_conf.mean():
            logger.warning(f"  ⚠️ SHORT ma wyższą pewność predykcji")
    
    # Podsumowanie dowodów
    logger.info(f"\n--- PODSUMOWANIE DOWODÓW ---")
    logger.info("Sprawdzone aspekty:")
    logger.info("  1. Wariancja cech - czy SHORT ma większą wariancję")
    logger.info("  2. Ekstremalność wartości - czy SHORT ma większy zakres")
    logger.info("  3. Jednoznaczność wzorców - czy SHORT ma silniejsze korelacje")
    logger.info("  4. Separowalność od NEUTRAL - czy SHORT łatwiej odróżnić")
    logger.info("  5. Konsystencja wzorców - czy SHORT ma bardziej stabilne wzorce")
    logger.info("  6. Granice decyzyjne - analiza feature importance")
    logger.info("  7. Prawdopodobieństwa - czy SHORT ma wyższą pewność")
    
    logger.info("\nWNIOSEK: Sprawdź wyniki powyżej aby udowodnić hipotezę.")

if __name__ == "__main__":
    prove_bias_hypothesis() 