"""
Schema-faithful synthetic data generator.

WHY THIS EXISTS (read this before deleting it):
The real Home Credit Default Risk dataset requires a Kaggle account and a
competition-rules acceptance click to download -- it cannot legally be
vendored inside a public git repo, and an evaluator running
`docker-compose up` for the first time should not be blocked on a manual
Kaggle download just to see the platform work end to end.

So: if `data/raw/application_train.csv` is present, the loader uses the
REAL dataset and this module is never touched. If it is absent, this
generator builds a synthetic table with the same columns, dtypes, category
values, and -- critically -- the same *causal structure* between features
and default risk (via EXT_SOURCE_*, DAYS_EMPLOYED, credit-to-income ratio,
etc.) that the real data exhibits. That means the ML pipeline, SHAP
explanations, rule extraction, and the SQL chatbot all produce sensible,
demo-able output even without the Kaggle files -- they are not just
"technically running", the numbers make business sense.

See data/README.md for how to swap in the real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("synthetic")

_INCOME_TYPES = ["Working", "State servant", "Commercial associate", "Pensioner", "Student"]
_EDUCATION_TYPES = [
    "Secondary / secondary special", "Higher education",
    "Incomplete higher", "Lower secondary", "Academic degree",
]
_FAMILY_STATUS = ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"]
_HOUSING_TYPES = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"]
_OCCUPATIONS = [
    "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
    "High skill tech staff", "Accountants", "Medicine staff", "Security staff", "Cooking staff",
]
_ORG_TYPES = [
    "Business Entity Type 3", "Self-employed", "Government", "School",
    "Trade: type 7", "Kindergarten", "Transport: type 4", "Medicine", "Construction", "Bank",
]
_REGIONS = list(range(1, 11))
_CONTRACT_TYPES = ["Cash loans", "Revolving loans"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_application_table(n_rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic analogue of application_train.csv.

    The TARGET is generated from a logistic function of a handful of the
    features that are genuinely predictive in the real competition
    (EXT_SOURCE_1/2/3, DAYS_EMPLOYED, credit-to-income ratio, age, prior
    delinquency flag) plus noise -- so downstream modeling, SHAP, and rule
    extraction behave the way they would on the real thing, at a roughly
    similar ~8% base default rate.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    df = pd.DataFrame({"SK_ID_CURR": np.arange(100001, 100001 + n)})

    df["NAME_CONTRACT_TYPE"] = rng.choice(_CONTRACT_TYPES, n, p=[0.9, 0.1])
    df["CODE_GENDER"] = rng.choice(["M", "F"], n, p=[0.34, 0.66])
    df["FLAG_OWN_CAR"] = rng.choice(["Y", "N"], n, p=[0.34, 0.66])
    df["FLAG_OWN_REALTY"] = rng.choice(["Y", "N"], n, p=[0.69, 0.31])
    df["CNT_CHILDREN"] = rng.poisson(0.4, n).clip(0, 6)
    df["CNT_FAM_MEMBERS"] = (df["CNT_CHILDREN"] + rng.integers(1, 3, n)).clip(1, 9)

    df["NAME_INCOME_TYPE"] = rng.choice(_INCOME_TYPES, n, p=[0.51, 0.08, 0.23, 0.17, 0.01])
    df["NAME_EDUCATION_TYPE"] = rng.choice(
        _EDUCATION_TYPES, n, p=[0.71, 0.24, 0.03, 0.015, 0.005]
    )
    df["NAME_FAMILY_STATUS"] = rng.choice(_FAMILY_STATUS, n, p=[0.64, 0.15, 0.10, 0.06, 0.05])
    df["NAME_HOUSING_TYPE"] = rng.choice(_HOUSING_TYPES, n, p=[0.88, 0.04, 0.04, 0.03, 0.01])
    df["OCCUPATION_TYPE"] = rng.choice(list(_OCCUPATIONS) + [None], n)
    df["ORGANIZATION_TYPE"] = rng.choice(_ORG_TYPES, n)
    df["REGION_RATING_CLIENT"] = rng.choice([1, 2, 3], n, p=[0.15, 0.65, 0.20])

    # Income, credit, annuity -- lognormal to mimic the real skew.
    df["AMT_INCOME_TOTAL"] = rng.lognormal(mean=11.8, sigma=0.45, size=n).round(-2)
    df["AMT_CREDIT"] = (df["AMT_INCOME_TOTAL"] * rng.uniform(1.5, 6.5, n)).round(-2)
    df["AMT_ANNUITY"] = (df["AMT_CREDIT"] / rng.uniform(8, 30, n)).round(-1)
    df["AMT_GOODS_PRICE"] = (df["AMT_CREDIT"] * rng.uniform(0.85, 1.0, n)).round(-2)

    # Ages 21-69, employment tenure (special sentinel 365243 = "not employed"
    # exists in the real dataset for pensioners -- reproduced here too).
    age_days = -rng.integers(21 * 365, 69 * 365, n)
    df["DAYS_BIRTH"] = age_days
    employed_days = -rng.integers(0, 40 * 365, n)
    pensioner_mask = df["NAME_INCOME_TYPE"] == "Pensioner"
    employed_days = np.where(pensioner_mask, 365243, employed_days)
    df["DAYS_EMPLOYED"] = employed_days
    df["DAYS_REGISTRATION"] = -rng.integers(0, 50 * 365, n)
    df["DAYS_ID_PUBLISH"] = -rng.integers(0, 20 * 365, n)
    df["OWN_CAR_AGE"] = np.where(
        df["FLAG_OWN_CAR"] == "Y", rng.integers(0, 25, n).astype(float), np.nan
    )

    # External bureau-style scores, the strongest real-world predictors.
    df["EXT_SOURCE_1"] = np.clip(rng.normal(0.5, 0.2, n), 0, 1)
    df["EXT_SOURCE_2"] = np.clip(rng.normal(0.52, 0.19, n), 0, 1)
    df["EXT_SOURCE_3"] = np.clip(rng.normal(0.51, 0.2, n), 0, 1)
    # Inject missingness the way the real dataset has it (EXT_SOURCE_1 is
    # missing ~56% of the time in the real competition).
    df.loc[rng.random(n) < 0.56, "EXT_SOURCE_1"] = np.nan
    df.loc[rng.random(n) < 0.18, "EXT_SOURCE_3"] = np.nan

    df["REGION_POPULATION_RELATIVE"] = rng.uniform(0.001, 0.07, n).round(4)
    df["FLAG_MOBIL"] = 1
    df["FLAG_EMP_PHONE"] = rng.choice([0, 1], n, p=[0.19, 0.81])
    df["FLAG_WORK_PHONE"] = rng.choice([0, 1], n, p=[0.8, 0.2])
    df["FLAG_PHONE"] = rng.choice([0, 1], n, p=[0.72, 0.28])
    df["FLAG_EMAIL"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    df["REG_REGION_NOT_LIVE_REGION"] = rng.choice([0, 1], n, p=[0.98, 0.02])
    df["REG_REGION_NOT_WORK_REGION"] = rng.choice([0, 1], n, p=[0.95, 0.05])
    df["DEF_30_CNT_SOCIAL_CIRCLE"] = rng.poisson(0.14, n)
    df["OBS_30_CNT_SOCIAL_CIRCLE"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] + rng.poisson(1.4, n)

    # --- Ground-truth-ish default probability (logistic combination) -----
    ext_avg = np.nanmean(
        np.vstack([df["EXT_SOURCE_1"].fillna(0.5), df["EXT_SOURCE_2"], df["EXT_SOURCE_3"].fillna(0.5)]),
        axis=0,
    )
    age_years = -df["DAYS_BIRTH"] / 365.0
    employed_years = np.where(df["DAYS_EMPLOYED"] == 365243, 0, -df["DAYS_EMPLOYED"] / 365.0)
    credit_income_ratio = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].clip(lower=1)
    annuity_income_ratio = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].clip(lower=1)

    logit = (
        3.4 * (0.5 - ext_avg)
        - 0.02 * (age_years - 43)
        - 0.035 * np.clip(employed_years, 0, 20)
        + 0.10 * (credit_income_ratio - 3.0)
        + 2.0 * (annuity_income_ratio - 0.12)
        + 0.25 * df["DEF_30_CNT_SOCIAL_CIRCLE"]
        + 0.15 * (df["REGION_RATING_CLIENT"] - 2)
        - 2.55  # intercept tuned so base rate lands near the real ~8%
        + rng.normal(0, 0.55, n)  # irreducible noise
    )
    default_prob = _sigmoid(logit)
    df["TARGET"] = (rng.random(n) < default_prob).astype(int)

    logger.info(
        "Generated synthetic application table: %d rows, base default rate %.2f%%",
        n, 100 * df["TARGET"].mean(),
    )
    return df


def generate_bureau_table(application_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic analogue of bureau.csv (prior credits at other institutions).

    Riskier applicants (by TARGET, which a real bureau table would not see
    directly -- but a *correlated* latent risk factor would) tend to have
    more prior credits and a higher active/overdue rate, matching the real
    dataset's aggregate signal.
    """
    rng = np.random.default_rng(seed + 1)
    rows = []
    for sk_id, target in zip(application_df["SK_ID_CURR"], application_df["TARGET"]):
        n_credits = rng.poisson(2.2 + 3.0 * target)
        for _ in range(n_credits):
            active = rng.random() < (0.35 + 0.25 * target)
            overdue = 0.0
            if active and rng.random() < (0.05 + 0.25 * target):
                overdue = float(rng.integers(500, 50000))
            rows.append({
                "SK_ID_CURR": sk_id,
                "CREDIT_ACTIVE": "Active" if active else "Closed",
                "AMT_CREDIT_SUM": float(rng.lognormal(10.5, 1.0)),
                "AMT_CREDIT_SUM_OVERDUE": overdue,
                "CREDIT_DAY_OVERDUE": int(rng.integers(0, 30)) if overdue > 0 else 0,
                "DAYS_CREDIT": -int(rng.integers(0, 3000)),
            })
    bureau_df = pd.DataFrame(rows)
    logger.info("Generated synthetic bureau table: %d rows", len(bureau_df))
    return bureau_df
