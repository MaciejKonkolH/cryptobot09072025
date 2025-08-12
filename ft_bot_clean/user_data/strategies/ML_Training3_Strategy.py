from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, List
import sys

import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

# Ensure local strategy helpers importable
_STRAT_DIR = Path(__file__).resolve().parent
if str(_STRAT_DIR) not in sys.path:
    sys.path.insert(0, str(_STRAT_DIR))

try:
    from utils.t3_model_loader import Training3ArtifactLoader
    from components.t3_feature_adapter import Training3FeatureAdapter
    from components.t3_signal_generator import Training3SignalGenerator
except Exception:
    # Allow import errors during static analysis
    Training3ArtifactLoader = None  # type: ignore
    Training3FeatureAdapter = None  # type: ignore
    Training3SignalGenerator = None  # type: ignore


class ML_Training3_Strategy(IStrategy):
    timeframe = "1m"
    can_short: bool = True

    # Defaults (overridable by JSON config)
    model_filename: str = "model_tp0p8_sl0p3.json"
    metadata_filename: str = "metadata_tp0p8_sl0p3.json"
    scaler_filename: str = "scaler.pkl"
    confidence_threshold_long: float = 0.40
    confidence_threshold_short: float = 0.40

    # Internal lazy state
    _initialized: bool = False
    _artifact_loader: Optional[Training3ArtifactLoader] = None
    _feature_adapter: Optional[Training3FeatureAdapter] = None
    _signal_generator: Optional[Training3SignalGenerator] = None
    _feature_names: Optional[List[str]] = None
    _model = None
    _scaler = None
    _models_dir: Optional[Path] = None
    _features_df: Optional[pd.DataFrame] = None

    def informative_pairs(self):
        return []

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # Load config JSON if available
        cfg_path = _STRAT_DIR / "configs" / "training3_strategy_config.json"
        cfg_data = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                cfg_data = {}

        # Resolve models dir (robust relative to repo root 'crypto')
        base_root = _STRAT_DIR.resolve().parents[2]  # .../crypto
        models_dir_cfg = cfg_data.get("models_dir")
        if models_dir_cfg:
            p = Path(models_dir_cfg)
            # Normalize relative paths like "crypto/training3/output/models" to avoid double "crypto"
            if not p.is_absolute():
                parts = p.parts
                if len(parts) > 0 and parts[0].lower() == 'crypto':
                    p = Path(*parts[1:])
                self._models_dir = base_root / p
            else:
                self._models_dir = p
        else:
            self._models_dir = base_root / "training3" / "output" / "models"

        # Override filenames and thresholds
        self.model_filename = cfg_data.get("model_filename", self.model_filename)
        self.metadata_filename = cfg_data.get("metadata_filename", self.metadata_filename)
        self.scaler_filename = cfg_data.get("scaler_filename", self.scaler_filename)
        # Read thresholds with priority: main config ml_config -> strategy config -> defaults
        try:
            from freqtrade.configuration import Configuration
            # Load main config once to read ml_config section
            main_cfg = Configuration.from_files([str((_STRAT_DIR / '..' / 'config.json').resolve())]).get_config()
            ml_cfg = main_cfg.get('ml_config', {}) if isinstance(main_cfg, dict) else {}
        except Exception:
            ml_cfg = {}
        self.confidence_threshold_long = float(ml_cfg.get('confidence_threshold_long', cfg_data.get("ml_long_threshold", self.confidence_threshold_long)))
        self.confidence_threshold_short = float(ml_cfg.get('confidence_threshold_short', cfg_data.get("ml_short_threshold", self.confidence_threshold_short)))

        # Instantiate helpers
        self._artifact_loader = Training3ArtifactLoader(self._models_dir)
        self._feature_adapter = Training3FeatureAdapter()
        self._signal_generator = Training3SignalGenerator(
            long_threshold=self.confidence_threshold_long,
            short_threshold=self.confidence_threshold_short,
        )

        # Load artifacts
        self._model, self._scaler, self._feature_names = self._artifact_loader.load(
            model_filename=self.model_filename,
            metadata_filename=self.metadata_filename,
            scaler_filename=self.scaler_filename,
        )

        # Strictly load precomputed features from labeler3 output for exact feature alignment
        project_root = _STRAT_DIR.resolve().parents[2]
        default_feat_path = project_root / "labeler3" / "output" / "ohlc_orderbook_labeled_3class_fw120m_15levels.feather"
        feat_path_cfg = cfg_data.get("features_path")
        feat_path = Path(feat_path_cfg) if feat_path_cfg else default_feat_path
        if not feat_path.is_absolute():
            feat_path = project_root / feat_path
        if not feat_path.exists():
            raise FileNotFoundError(f"Features file not found: {feat_path}")
        df_feat = pd.read_feather(feat_path)
        if "timestamp" in df_feat.columns:
            df_feat["date"] = pd.to_datetime(df_feat["timestamp"], utc=True)
        elif "date" in df_feat.columns:
            df_feat["date"] = pd.to_datetime(df_feat["date"], utc=True)
        else:
            raise ValueError("Features file must contain 'timestamp' or 'date' column.")
        df_feat = df_feat.set_index("date").sort_index()
        need_cols = list(self._feature_names or [])
        if not need_cols:
            raise RuntimeError("Empty feature_names from metadata.")
        missing = [c for c in need_cols if c not in df_feat.columns]
        if missing:
            raise ValueError(f"Missing required feature columns ({len(missing)}): {missing[:10]}...")
        feat_sel = df_feat[need_cols].copy()
        if feat_sel.isna().any().any():
            raise ValueError("NaN detected in selected features.")
        if not np.isfinite(feat_sel.to_numpy()).all():
            raise ValueError("Inf or non-finite detected in selected features.")
        self._features_df = feat_sel

        self._initialized = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        df = dataframe.copy()
        df["enter_long"] = False
        df["enter_short"] = False

        if df.empty:
            return df

        # Lazy init of artifacts
        try:
            self._ensure_initialized()
        except Exception:
            # On any loading error, keep neutral signals
            return df

        # Build features: require external features aligned by timestamp; no fallback allowed
        try:
            if self._features_df is None or not (self._feature_names or []):
                raise RuntimeError("Features are not initialized. Provide a valid features file with exact training columns.")
            if "date" not in df.columns:
                raise RuntimeError("Input dataframe missing 'date' column for timestamp alignment.")
            idx = pd.to_datetime(df["date"], utc=True)
            X = self._features_df.reindex(idx)
            if X.isna().any().any():
                missing_rows = int(X.isna().any(axis=1).sum())
                raise RuntimeError(f"Missing feature rows for {missing_rows} timestamps in batch. Aborting.")

            # Get signals with probabilities for CSV logging
            sigs = self._signal_generator.generate_signals_with_proba_for_batch(self._model, self._scaler, X)
            if sigs and len(sigs) == len(df):
                # Assign entries by positional index
                for i, row in enumerate(sigs):
                    if row.get('signal') == 'long':
                        df.iat[i, df.columns.get_loc('enter_long')] = True
                    elif row.get('signal') == 'short':
                        df.iat[i, df.columns.get_loc('enter_short')] = True
                # Save CSV of predictions for verification
                try:
                    out_rows = []
                    for i, row in enumerate(sigs):
                        ts = df.loc[df.index[i], 'date'] if 'date' in df.columns else None
                        out_rows.append({
                            'timestamp': str(ts) if ts is not None else str(idx[i]),
                            'pair': metadata.get('pair') if isinstance(metadata, dict) else '',
                            'signal': row.get('signal'),
                            'confidence': float(row.get('confidence', 0.0)),
                            'prob_SHORT': float(row.get('prob_SHORT', 0.0)),
                            'prob_LONG': float(row.get('prob_LONG', 0.0)),
                            'prob_NEUTRAL': float(row.get('prob_NEUTRAL', 0.0)),
                        })
                    rep_dir = _STRAT_DIR / 'predictions'
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    tsname = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
                    pairstr = (metadata.get('pair') if isinstance(metadata, dict) else 'PAIR').replace('/', '').replace(':', '')
                    out_path = rep_dir / f'predictions_{pairstr}_{tsname}.csv'
                    pd.DataFrame(out_rows).to_csv(out_path, index=False)
                except Exception:
                    pass
        except Exception:
            raise

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        df = dataframe.copy()
        df["exit_long"] = False
        df["exit_short"] = False
        return df

