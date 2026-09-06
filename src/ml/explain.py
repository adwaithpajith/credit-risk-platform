"""
Explainable AI layer using SHAP (TreeExplainer -- exact and fast for
gradient-boosted trees, no sampling approximation needed).

Two levels of explanation are produced, matching what a real credit team
needs:
  - Global: which features drive risk across the whole portfolio (for
    model governance / regulator review).
  - Local: for one specific applicant, which features pushed their score
    up or down, in plain business language (for an adverse-action notice
    or a credit officer's file note).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.ml.predict import RiskScorer
from src.utils.logger import get_logger

logger = get_logger("explain")

# Maps engineered/raw feature names to a short, non-technical phrase --
# used to turn SHAP output into sentences a non-technical user can read,
# per the assignment's Part 4 requirement.
_FEATURE_PLAIN_LANGUAGE = {
    "EXT_SOURCE_MEAN": "overall external credit score",
    "EXT_SOURCE_1": "external credit bureau score #1",
    "EXT_SOURCE_2": "external credit bureau score #2",
    "EXT_SOURCE_3": "external credit bureau score #3",
    "CREDIT_TO_INCOME_RATIO": "loan amount relative to income",
    "ANNUITY_TO_INCOME_RATIO": "monthly payment burden relative to income",
    "AGE_YEARS": "applicant age",
    "EMPLOYED_YEARS": "length of current employment",
    "DAYS_EMPLOYED": "employment tenure",
    "AMT_CREDIT": "requested loan amount",
    "AMT_INCOME_TOTAL": "declared annual income",
    "BUREAU_ACTIVE_RATIO": "share of active prior credits",
    "BUREAU_HAS_OVERDUE": "history of overdue payments with other lenders",
    "DEF_30_CNT_SOCIAL_CIRCLE": "defaults observed in applicant's social circle",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income source type",
    "REGION_RATING_CLIENT": "regional risk rating",
}


def plain_language(feature: str) -> str:
    return _FEATURE_PLAIN_LANGUAGE.get(feature, feature.replace("_", " ").lower())


@dataclass
class LocalExplanation:
    probability: float
    risk_band: str
    top_factors: list[dict]  # [{feature, value, shap_value, direction, plain_text}, ...]


class RiskExplainer:
    def __init__(self, scorer: RiskScorer) -> None:
        self.scorer = scorer
        self._explainer = shap.TreeExplainer(scorer.model)

    def global_importance(self, X: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
        """Mean |SHAP value| per feature across a sample of the portfolio --
        the standard global feature-importance view."""
        sample = X.sample(min(sample_size, len(X)), random_state=42) if len(X) > sample_size else X
        shap_values = self._explainer.shap_values(sample)
        if isinstance(shap_values, list):  # some SHAP versions return [class0, class1]
            shap_values = shap_values[1]
        importance = pd.DataFrame({
            "feature": sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance["plain_language"] = importance["feature"].map(plain_language)
        return importance

    def explain_one(self, applicant_row: pd.DataFrame, top_k: int = 5) -> LocalExplanation:
        """Explain a single applicant's prediction. `applicant_row` must be
        a 1-row DataFrame in raw schema (same as score_one's input)."""
        X = self.scorer.get_feature_matrix(applicant_row)
        proba = float(self.scorer.model.predict(X)[0])

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_row = shap_values[0]

        contributions = []
        for feat, val, sv in zip(X.columns, X.iloc[0], shap_row):
            contributions.append({
                "feature": feat,
                "value": val,
                "shap_value": float(sv),
                "direction": "increases risk" if sv > 0 else "decreases risk",
                "plain_text": (
                    f"{plain_language(feat)} ({val}) "
                    f"{'increases' if sv > 0 else 'decreases'} default risk"
                ),
            })
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        from src.utils.helpers import risk_band
        from src.utils.config import config
        band = risk_band(proba, config.model.low_risk_band_max, config.model.medium_risk_band_max)

        return LocalExplanation(
            probability=proba, risk_band=band, top_factors=contributions[:top_k]
        )
