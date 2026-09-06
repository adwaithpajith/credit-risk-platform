"""
Populates the analytical database (sql/schema.sql) that the talk-to-data
chatbot queries. Runs against whatever DATABASE_URL points to -- SQLite by
default for local dev, Postgres inside docker-compose.

This is intentionally a separate step from training: the model artifact
(models/*.joblib) is the source of truth for scoring, and this script's
job is only to materialize a queryable snapshot of applicants + their
scores for the NL-to-SQL layer. Re-run it whenever the model or data
changes.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from src.data.loader import load_full_dataset
from src.ml.predict import RiskScorer
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("db_loader")

_COLUMN_MAP = {
    "SK_ID_CURR": "sk_id_curr", "TARGET": "target",
    "NAME_CONTRACT_TYPE": "name_contract_type", "CODE_GENDER": "code_gender",
    "FLAG_OWN_CAR": "flag_own_car", "FLAG_OWN_REALTY": "flag_own_realty",
    "CNT_CHILDREN": "cnt_children", "CNT_FAM_MEMBERS": "cnt_fam_members",
    "NAME_INCOME_TYPE": "name_income_type", "NAME_EDUCATION_TYPE": "name_education_type",
    "NAME_FAMILY_STATUS": "name_family_status", "NAME_HOUSING_TYPE": "name_housing_type",
    "OCCUPATION_TYPE": "occupation_type", "ORGANIZATION_TYPE": "organization_type",
    "REGION_RATING_CLIENT": "region_rating_client", "AMT_INCOME_TOTAL": "amt_income_total",
    "AMT_CREDIT": "amt_credit", "AMT_ANNUITY": "amt_annuity",
    "AMT_GOODS_PRICE": "amt_goods_price", "AGE_YEARS": "age_years",
    "EMPLOYED_YEARS": "employed_years", "EXT_SOURCE_MEAN": "ext_source_mean",
    "CREDIT_TO_INCOME_RATIO": "credit_to_income_ratio",
    "ANNUITY_TO_INCOME_RATIO": "annuity_to_income_ratio",
    "BUREAU_CREDIT_COUNT": "bureau_credit_count", "BUREAU_ACTIVE_COUNT": "bureau_active_count",
    "BUREAU_HAS_OVERDUE": "bureau_has_overdue",
    "DEFAULT_PROBABILITY": "default_probability", "RISK_BAND": "risk_band",
}


def build_analytical_table() -> pd.DataFrame:
    """Load raw data, engineer features, score with the trained model, and
    return one flat table matching sql/schema.sql's column names."""
    df, source = load_full_dataset()
    logger.info("Scoring %d applicants for the analytical table (source=%s)", len(df), source)

    scorer = RiskScorer()
    features_needed = df.drop(columns=["TARGET"], errors="ignore")
    scored = scorer.score_dataframe(features_needed)
    if "TARGET" in df.columns:
        scored["TARGET"] = df["TARGET"].values

    from src.data.preprocessor import engineer_features
    scored = engineer_features(scored)

    available_cols = {src: dst for src, dst in _COLUMN_MAP.items() if src in scored.columns}
    table = scored[list(available_cols.keys())].rename(columns=available_cols)
    return table


def load_into_database(table: pd.DataFrame | None = None) -> int:
    if table is None:
        table = build_analytical_table()

    engine = create_engine(config.db.database_url)
    schema_sql = (config.paths.sql_dir / "schema.sql").read_text()

    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

    table.to_sql("applicants", engine, if_exists="append", index=False)
    logger.info("Loaded %d rows into 'applicants' table at %s", len(table), config.db.database_url)
    return len(table)


if __name__ == "__main__":
    n = load_into_database()
    print(f"Loaded {n} rows into the analytical database.")
