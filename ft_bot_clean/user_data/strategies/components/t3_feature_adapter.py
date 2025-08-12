from __future__ import annotations

from typing import List
import pandas as pd


class Training3FeatureAdapter:
    """
    Ensures the incoming dataframe is adapted to exactly the feature set
    used during training3 (order and presence).
    Missing columns are filled with 0.0; extra columns are ignored.
    """

    def prepare_features(self, dataframe: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        if dataframe is None or dataframe.empty:
            return pd.DataFrame(columns=feature_names, index=dataframe.index if dataframe is not None else None)

        df = dataframe.copy()
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder and keep only needed columns
        df_feat = df[feature_names].copy()
        return df_feat

