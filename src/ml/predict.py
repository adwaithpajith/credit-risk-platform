"""Inference: load the saved model and score new applicants."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from src.data.preprocessor import align_to_manifest
from src.ml.train import MANIFEST_PATH, MODEL_PATH
from src.utils.config import config
from src.utils.helpers import risk_band
from src.utils.logger import get_logger

logger = get_logger("predict")


@dataclass
class ScoredApplicant:
    probability: float
    risk_band: str
    threshold_used: float


class RiskScorer:
    """Thin, cached wrapper around the trained LightGBM booster. Instantiate
    once (the UI does this via st.cache_resource) -- loading the model
    from disk on every prediction would be wasteful."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run "
                "`python -m src.ml.train` first (see README 'Quick start')."
            )
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.manifest = joblib.load(MANIFEST_PATH)
        logger.info("Loaded model with decision threshold=%.3f", self.threshold)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of raw applicant rows (same schema as the raw
        application table). Returns a DataFrame of probability + band."""
        X = align_to_manifest(df, self.manifest)
        proba = self.model.predict(X)
        bands = [
            risk_band(p, config.model.low_risk_band_max, config.model.medium_risk_band_max)
            for p in proba
        ]
        out = df.copy()
        out["DEFAULT_PROBABILITY"] = proba
        out["RISK_BAND"] = bands
        out["PREDICTED_DEFAULT"] = (proba >= self.threshold).astype(int)
        return out

    def score_one(self, applicant: dict) -> ScoredApplicant:
        df = pd.DataFrame([applicant])
        scored = self.score_dataframe(df).iloc[0]
        return ScoredApplicant(
            probability=float(scored["DEFAULT_PROBABILITY"]),
            risk_band=scored["RISK_BAND"],
            threshold_used=self.threshold,
        )

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expose the exact matrix the model scores on (post feature-
        engineering, pre-prediction) -- used by explain.py for SHAP."""
        return align_to_manifest(df, self.manifest)
