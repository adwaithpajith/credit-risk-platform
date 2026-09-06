"""
Cleaning, feature engineering, and encoding.

Design choice: we deliberately do NOT one-hot encode categoricals or scale
numerics here. LightGBM (our model, see src/ml/train.py) splits on raw
values and handles native missing values, so one-hot encoding would only
inflate dimensionality and blur SHAP attributions across dummy columns
without buying accuracy. Categoricals are cast to pandas `category` dtype
and passed to LightGBM directly. This is a deliberate, defensible modeling
decision, not a shortcut -- documented again in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessor")

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# Sentinel used by Home Credit for "not applicable" employment duration
# (pensioners / unemployed). Left as-is it would be read as 1000 years
# employed, which corrupts DAYS_EMPLOYED as a numeric feature.
_DAYS_EMPLOYED_ANOMALY = 365243


@dataclass
class FeatureManifest:
    """Records exactly what the preprocessor did, so predict.py can apply
    the identical transformation to new/unseen applicants at inference
    time, and so the UI/README can display an honest feature list."""
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)
    category_levels: dict[str, list[str]] = field(default_factory=dict)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven feature engineering. Each feature below encodes a
    genuine underwriting heuristic (affordability ratios, tenure, life
    stage) rather than being a blind polynomial/interaction dump -- this
    is what separates a credit-risk model from a generic Kaggle template."""
    df = df.copy()

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(_DAYS_EMPLOYED_ANOMALY, np.nan)

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE_RATIO"] = (
        df["EMPLOYED_YEARS"] / df["AGE_YEARS"].replace(0, np.nan)
    )

    df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    )
    df["ANNUITY_TO_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MISSING_COUNT"] = df[ext_cols].isna().sum(axis=1)

    if "BUREAU_CREDIT_COUNT" in df.columns:
        df["BUREAU_OVERDUE_PER_CREDIT"] = (
            df["BUREAU_AMT_OVERDUE_TOTAL"] / df["BUREAU_CREDIT_COUNT"].replace(0, np.nan)
        )

    return df


def _infer_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {TARGET_COL, ID_COL}
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_model_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, FeatureManifest]:
    """Full pipeline: engineer features -> infer dtypes -> cast categoricals.
    Returns (X, y_or_None, manifest). Works identically whether TARGET is
    present (training) or absent (scoring new applicants)."""
    df = engineer_features(df)

    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    numeric_cols, categorical_cols = _infer_feature_types(df)

    manifest = FeatureManifest(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
    )

    X = df[numeric_cols + categorical_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category")
        manifest.category_levels[col] = list(X[col].cat.categories)

    manifest.engineered_features = [
        c for c in [
            "AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYED_TO_AGE_RATIO",
            "CREDIT_TO_INCOME_RATIO", "ANNUITY_TO_INCOME_RATIO",
            "CREDIT_TO_GOODS_RATIO", "INCOME_PER_FAMILY_MEMBER",
            "ANNUITY_TO_CREDIT_RATIO", "EXT_SOURCE_MEAN", "EXT_SOURCE_MAX",
            "EXT_SOURCE_MIN", "EXT_SOURCE_MISSING_COUNT",
            "BUREAU_OVERDUE_PER_CREDIT",
        ] if c in X.columns
    ]

    logger.info(
        "Built model frame: %d rows, %d numeric + %d categorical features (%d engineered)",
        len(X), len(numeric_cols), len(categorical_cols), len(manifest.engineered_features),
    )
    return X, y, manifest


def align_to_manifest(df: pd.DataFrame, manifest: FeatureManifest) -> pd.DataFrame:
    """Apply an already-fitted manifest to new data at inference time so
    train/serve feature skew can't silently creep in. Missing columns are
    added as all-NaN (LightGBM handles NaN natively); category levels
    unseen at training time are mapped to NaN rather than crashing."""
    df = engineer_features(df)
    all_cols = manifest.numeric_features + manifest.categorical_features

    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    X = df[all_cols].copy()
    for col in manifest.numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in manifest.categorical_features:
        known = manifest.category_levels.get(col, [])
        cat_dtype = pd.CategoricalDtype(categories=known)
        # Values outside the training-time vocabulary must become NaN, not
        # raise -- an applicant with a category never seen during training
        # is a normal, expected event at inference time, not an error.
        # `.where(...)` masks unknowns to NaN *before* the categorical cast,
        # which is the pandas-3-safe way to do this (a plain `.astype()`
        # with unseen values present is deprecated and will start raising).
        X[col] = X[col].where(X[col].isin(known)).astype(cat_dtype)
    return X
