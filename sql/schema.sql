-- Credit Risk Intelligence Platform -- analytical schema
-- Loaded by src/data/db_loader.py from the (real-or-synthetic) application
-- + bureau tables, plus the model's own scored output, so the talk-to-data
-- chatbot can answer both raw-data questions ("average income by
-- education") and model-output questions ("how many high risk applicants
-- in region 3") from one place.

DROP TABLE IF EXISTS applicants;
CREATE TABLE applicants (
    sk_id_curr               BIGINT PRIMARY KEY,
    target                   INTEGER,               -- 1 = defaulted, NULL for unlabeled/scoring-only rows
    name_contract_type       VARCHAR(32),
    code_gender               VARCHAR(4),
    flag_own_car             VARCHAR(4),
    flag_own_realty          VARCHAR(4),
    cnt_children             INTEGER,
    cnt_fam_members          INTEGER,
    name_income_type         VARCHAR(64),
    name_education_type      VARCHAR(64),
    name_family_status       VARCHAR(64),
    name_housing_type        VARCHAR(64),
    occupation_type          VARCHAR(64),
    organization_type        VARCHAR(64),
    region_rating_client     INTEGER,
    amt_income_total         DOUBLE PRECISION,
    amt_credit               DOUBLE PRECISION,
    amt_annuity              DOUBLE PRECISION,
    amt_goods_price          DOUBLE PRECISION,
    age_years                DOUBLE PRECISION,
    employed_years           DOUBLE PRECISION,
    ext_source_mean          DOUBLE PRECISION,
    credit_to_income_ratio   DOUBLE PRECISION,
    annuity_to_income_ratio  DOUBLE PRECISION,
    bureau_credit_count      DOUBLE PRECISION,
    bureau_active_count      DOUBLE PRECISION,
    bureau_has_overdue       DOUBLE PRECISION,
    default_probability      DOUBLE PRECISION,       -- populated after scoring
    risk_band                VARCHAR(16)              -- 'Low' | 'Medium' | 'High'
);

CREATE INDEX IF NOT EXISTS idx_applicants_risk_band ON applicants (risk_band);
CREATE INDEX IF NOT EXISTS idx_applicants_education ON applicants (name_education_type);
CREATE INDEX IF NOT EXISTS idx_applicants_income_type ON applicants (name_income_type);
CREATE INDEX IF NOT EXISTS idx_applicants_region ON applicants (region_rating_client);
