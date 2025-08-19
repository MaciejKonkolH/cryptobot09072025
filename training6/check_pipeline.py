import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training5 import config as cfg
from training5.data_loader import load_labeled, split_scale


def main(symbol: str):
    df = load_labeled(symbol)
    print(f"Loaded labeled: {len(df):,} rows, {len(df.columns)} cols")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names = split_scale(df)
    print(f"X shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"y shapes: train={y_train.shape}, val={y_val.shape}, test={y_test.shape}")
    print(f"n_features: {len(feat_names)}; first 5: {feat_names[:5]}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else cfg.DEFAULT_SYMBOL
    main(sym)

