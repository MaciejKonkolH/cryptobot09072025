from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import joblib
import json
import xgboost as xgb


class Training3ArtifactLoader:
    """
    Loads training3 artifacts:
    - XGBoost Booster (JSON format) for a specific TP/SL level
    - RobustScaler (joblib)
    - metadata_{tp_sl}.json containing feature_names
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)

    def load(self, model_filename: str, metadata_filename: str, scaler_filename: str) -> Tuple[xgb.Booster, object, List[str]]:
        model_path = self.models_dir / model_filename
        meta_path = self.models_dir / metadata_filename
        scaler_path = self.models_dir / scaler_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        model = xgb.Booster()
        model.load_model(str(model_path))

        scaler = joblib.load(str(scaler_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names")
        if not feature_names or not isinstance(feature_names, list):
            raise ValueError("Invalid metadata: missing feature_names list")

        return model, scaler, feature_names

