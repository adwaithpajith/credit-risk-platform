"""
Loads and joins the raw dataset tables.

Resolution order for the main application table:
  1. data/raw/application_train.csv (+ application_test.csv if present)
     -- the real Home Credit Default Risk files, if the user downloaded
     them from Kaggle (see data/README.md).
  2. Synthetic fallback (src.data.synthetic) -- schema-faithful, so every
     downstream module works unmodified either way.

Same pattern for bureau.csv. This module is the ONLY place that should
know whether we're on real or synthetic data -- everything downstream
just sees a DataFrame with the expected columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import synthetic
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("loader")

# Only these bureau.csv columns feed build_bureau_features() below -- the
# real file also carries CREDIT_CURRENCY, DAYS_CREDIT_ENDDATE, CREDIT_TYPE,
# etc. that we don't use. Reading just what's needed via `usecols` roughly
# halves peak memory on a 1.7M-row file, which matters on memory-
# constrained environments (a free Colab/Kaggle instance, a small CI
# runner) without costing anything on a well-resourced machine.
_BUREAU_USECOLS = [
    "SK_ID_CURR", "CREDIT_ACTIVE", "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT",
]


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """In-place-style numeric downcast (float64->float32, int64->smallest
    int type that fits). On the real 307k-row/122-column application
    table this typically cuts memory by roughly half with zero effect on
    model quality -- LightGBM is perfectly happy with float32, and the
    original Kaggle files store nothing that needs float64 precision."""
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


@dataclass
class LoadedData:
    application: pd.DataFrame
    bureau: pd.DataFrame | None
    source: str  # "kaggle" or "synthetic" -- surfaced in the UI/README so
                 # nobody mistakes demo output for the real competition score.


def _find_raw_file(filename: str) -> Path | None:
    path = config.paths.raw_dir / filename
    return path if path.exists() else None


def load_raw_tables() -> LoadedData:
    app_path = _find_raw_file("application_train.csv")

    if app_path is not None:
        logger.info("Loading real dataset from %s", app_path)
        application = pd.read_csv(app_path)
        application = _downcast_numeric(application)
        bureau_path = _find_raw_file("bureau.csv")
        if bureau_path is not None:
            bureau = pd.read_csv(bureau_path, usecols=_BUREAU_USECOLS)
            bureau = _downcast_numeric(bureau)
        else:
            bureau = None
        source = "kaggle"
    else:
        expected_path = config.paths.raw_dir / "application_train.csv"
        if not config.data.use_synthetic_fallback:
            raise FileNotFoundError(
                f"Real dataset not found at {expected_path}.\n\n"
                "This platform is built to run on the actual Home Credit Default Risk "
                "dataset from Kaggle:\n"
                "  https://www.kaggle.com/competitions/home-credit-default-risk/data\n\n"
                "Fix: download application_train.csv (and optionally bureau.csv) and place "
                "them in data/raw/, i.e.:\n"
                f"  {expected_path}\n"
                f"  {config.paths.raw_dir / 'bureau.csv'}  (optional, enables bureau features)\n\n"
                "See data/README.md for exact steps. (A synthetic dev fixture exists in "
                "src/data/synthetic.py for internal testing only -- opt into it with "
                "USE_SYNTHETIC_FALLBACK=true if you explicitly want it, but it is OFF by "
                "default because this assignment requires the real dataset.)"
            )
        logger.warning(
            "application_train.csv not found in data/raw/. "
            "USE_SYNTHETIC_FALLBACK=true, so falling back to synthetic data for this run."
        )
        application = synthetic.generate_application_table(
            n_rows=config.data.synthetic_rows, seed=config.data.synthetic_seed
        )
        bureau = synthetic.generate_bureau_table(application, seed=config.data.synthetic_seed)
        source = "synthetic"

    return LoadedData(application=application, bureau=bureau, source=source)


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the credit-bureau table (one row per prior credit) into
    one row per applicant. This is the classic Home Credit feature-
    engineering trick -- most of the competition's signal beyond
    EXT_SOURCE_* comes from bureau aggregates, not the main table alone.
    """
    if bureau is None or bureau.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_CURR", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    ).reset_index()

    agg["BUREAU_ACTIVE_RATIO"] = (
        agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_CREDIT_COUNT"].clip(lower=1)
    )
    agg["BUREAU_HAS_OVERDUE"] = (agg["BUREAU_AMT_OVERDUE_TOTAL"] > 0).astype(int)
    return agg


def load_full_dataset() -> tuple[pd.DataFrame, str]:
    """Convenience entry point used by training, EDA, and the DB loader:
    returns one flat, joined DataFrame plus the data source label."""
    raw = load_raw_tables()
    bureau_features = build_bureau_features(raw.bureau)
    df = raw.application.merge(bureau_features, on="SK_ID_CURR", how="left")

    bureau_cols = [c for c in bureau_features.columns if c != "SK_ID_CURR"]
    df[bureau_cols] = df[bureau_cols].fillna(0)

    logger.info(
        "Loaded full dataset: %d rows, %d columns, source=%s",
        len(df), df.shape[1], raw.source,
    )
    return df, raw.source
