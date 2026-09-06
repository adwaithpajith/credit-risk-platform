"""Tests for feature engineering and the raw<->model-frame alignment that
keeps training and serving consistent. Run with: pytest tests/ -v"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import (
    _DAYS_EMPLOYED_ANOMALY,
    align_to_manifest,
    build_model_frame,
    engineer_features,
)


def _sample_raw_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": rng.integers(0, 2, n),
        "DAYS_BIRTH": -rng.integers(20 * 365, 60 * 365, n),
        "DAYS_EMPLOYED": -rng.integers(0, 20 * 365, n),
        "AMT_INCOME_TOTAL": rng.uniform(50000, 300000, n),
        "AMT_CREDIT": rng.uniform(100000, 900000, n),
        "AMT_ANNUITY": rng.uniform(5000, 50000, n),
        "AMT_GOODS_PRICE": rng.uniform(90000, 850000, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n),
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary"], n),
        "CODE_GENDER": rng.choice(["M", "F"], n),
    })


def test_days_employed_anomaly_is_cleaned():
    df = _sample_raw_df()
    df.loc[0, "DAYS_EMPLOYED"] = _DAYS_EMPLOYED_ANOMALY
    engineered = engineer_features(df)
    assert pd.isna(engineered.loc[0, "DAYS_EMPLOYED"])


def test_engineered_ratio_features_exist_and_are_finite():
    df = _sample_raw_df()
    engineered = engineer_features(df)
    for col in ["CREDIT_TO_INCOME_RATIO", "ANNUITY_TO_INCOME_RATIO", "EXT_SOURCE_MEAN"]:
        assert col in engineered.columns
        assert np.isfinite(engineered[col]).all()


def test_build_model_frame_separates_target_and_casts_categoricals():
    df = _sample_raw_df()
    X, y, manifest = build_model_frame(df)
    assert y is not None
    assert "TARGET" not in X.columns
    assert "SK_ID_CURR" not in X.columns
    assert "NAME_EDUCATION_TYPE" in manifest.categorical_features
    assert str(X["NAME_EDUCATION_TYPE"].dtype) == "category"


def test_build_model_frame_without_target_for_scoring():
    df = _sample_raw_df().drop(columns=["TARGET"])
    X, y, manifest = build_model_frame(df)
    assert y is None
    assert len(X) == len(df)


def test_align_to_manifest_handles_unseen_category_gracefully():
    train_df = _sample_raw_df()
    _, _, manifest = build_model_frame(train_df)

    new_df = train_df.drop(columns=["TARGET"]).iloc[[0]].copy()
    new_df["NAME_EDUCATION_TYPE"] = "Never Seen Before Category"

    aligned = align_to_manifest(new_df, manifest)
    # Unseen category should become NaN, not raise or silently corrupt other columns.
    assert pd.isna(aligned["NAME_EDUCATION_TYPE"].iloc[0])


def test_align_to_manifest_adds_missing_columns_as_nan():
    train_df = _sample_raw_df()
    _, _, manifest = build_model_frame(train_df)

    incomplete_df = train_df.drop(columns=["TARGET", "EXT_SOURCE_1"]).iloc[[0]].copy()
    aligned = align_to_manifest(incomplete_df, manifest)
    assert "EXT_SOURCE_1" in aligned.columns
    assert pd.isna(aligned["EXT_SOURCE_1"].iloc[0])


def test_align_to_manifest_produces_lightgbm_compatible_dtypes():
    """Regression test for a real bug caught during development: a
    single-row DataFrame with a None value produced an `object` dtype
    column that LightGBM's predict() rejects outright."""
    train_df = _sample_raw_df()
    _, _, manifest = build_model_frame(train_df)

    one_row = train_df.drop(columns=["TARGET"]).iloc[[0]].copy()
    one_row["EXT_SOURCE_1"] = None
    aligned = align_to_manifest(one_row, manifest)

    for col in manifest.numeric_features:
        assert aligned[col].dtype.kind in "fiub", f"{col} has bad dtype {aligned[col].dtype}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
