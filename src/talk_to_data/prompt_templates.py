"""
Versioned prompt templates.

Token-optimization approach (see README for the full write-up):
  1. We inject a hand-written, condensed schema description (~150 tokens)
     instead of a full CREATE TABLE dump or, worse, sample rows -- the LLM
     does not need real applicant data to write SQL, and shipping real
     rows into the prompt would also leak PII into LLM provider logs.
  2. Few-shot examples are capped at 6 and chosen to cover the *shapes* of
     query the chatbot must support (aggregation, grouping, filtering,
     ratio computation, top-N) rather than every possible question --
     generalization comes from the shapes, not from coverage of literal
     questions.
  3. Prompts are versioned (PROMPT_VERSION) so that if we tighten the
     system prompt later (e.g. after seeing a hallucination in the wild)
     we can A/B compare against the previous version's logged
     question/SQL pairs instead of guessing whether a change helped.
  4. See nl_to_sql.py for the response CACHE, which is the biggest lever:
     repeated/similar questions cost zero additional tokens.
"""
from __future__ import annotations

PROMPT_VERSION = "v1"

SCHEMA_DESCRIPTION = """
Table: applicants  (one row per loan applicant)
Columns:
  sk_id_curr               BIGINT   -- applicant id
  target                    INT     -- 1 = defaulted historically, 0 = repaid, NULL = not yet resolved
  name_contract_type        TEXT    -- 'Cash loans' | 'Revolving loans'
  code_gender                TEXT    -- 'M' | 'F'
  flag_own_car, flag_own_realty  TEXT  -- 'Y' | 'N'
  cnt_children, cnt_fam_members  INT
  name_income_type          TEXT    -- e.g. 'Working', 'Pensioner', 'Commercial associate'
  name_education_type       TEXT    -- e.g. 'Higher education', 'Secondary / secondary special'
  name_family_status        TEXT
  name_housing_type         TEXT
  occupation_type           TEXT
  organization_type         TEXT
  region_rating_client      INT     -- 1 (best) to 3 (worst) regional risk rating
  amt_income_total          FLOAT   -- annual declared income
  amt_credit                FLOAT   -- requested loan amount
  amt_annuity               FLOAT   -- monthly payment
  amt_goods_price           FLOAT
  age_years                 FLOAT
  employed_years            FLOAT
  ext_source_mean           FLOAT   -- averaged external bureau score, 0 (risky) to 1 (safe)
  credit_to_income_ratio    FLOAT
  annuity_to_income_ratio   FLOAT
  bureau_credit_count       FLOAT   -- number of prior credits at other institutions
  bureau_active_count       FLOAT   -- number of those still active
  bureau_has_overdue        FLOAT   -- 1 if any prior credit is overdue
  default_probability       FLOAT   -- the ML model's predicted probability of default, 0-1
  risk_band                 TEXT    -- the ML model's output: 'Low' | 'Medium' | 'High'
""".strip()

SYSTEM_PROMPT = f"""You are a SQL analyst for a bank's credit risk platform. You translate a \
business user's natural-language question into a single read-only SQL SELECT query against \
the schema below, then the query result is returned to you to phrase as a short, plain-English \
business answer.

{SCHEMA_DESCRIPTION}

Rules (violating any of these makes your output unusable):
- Output ONLY the SQL query in a ```sql code block. No explanation in that step.
- SELECT statements only. Never write INSERT/UPDATE/DELETE/DROP/ALTER or any DDL/DML.
- Only reference the `applicants` table and the columns listed above -- never invent a column.
- Prefer aggregate queries (AVG/COUNT/SUM/GROUP BY) over dumping raw rows when the question
  asks for a statistic, trend, or comparison.
- Always include a LIMIT clause (100 is a reasonable default for row-level queries).
- If the question cannot be answered from this schema, output exactly: NO_QUERY_POSSIBLE
"""

FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "question": "What's the average income of applicants with higher education?",
        "sql": (
            "SELECT AVG(amt_income_total) AS avg_income FROM applicants "
            "WHERE name_education_type = 'Higher education' LIMIT 100"
        ),
    },
    {
        "question": "How many applicants are classified as high risk?",
        "sql": "SELECT COUNT(*) AS high_risk_count FROM applicants WHERE risk_band = 'High' LIMIT 100",
    },
    {
        "question": "What is the default rate by income type?",
        "sql": (
            "SELECT name_income_type, AVG(target) AS default_rate, COUNT(*) AS n "
            "FROM applicants WHERE target IS NOT NULL "
            "GROUP BY name_income_type ORDER BY default_rate DESC LIMIT 100"
        ),
    },
    {
        "question": "Show me the top 5 occupation types by average requested credit amount.",
        "sql": (
            "SELECT occupation_type, AVG(amt_credit) AS avg_credit FROM applicants "
            "GROUP BY occupation_type ORDER BY avg_credit DESC LIMIT 5"
        ),
    },
    {
        "question": "Which region rating has the highest share of high-risk applicants?",
        "sql": (
            "SELECT region_rating_client, "
            "AVG(CASE WHEN risk_band = 'High' THEN 1.0 ELSE 0.0 END) AS high_risk_share, "
            "COUNT(*) AS n FROM applicants "
            "GROUP BY region_rating_client ORDER BY high_risk_share DESC LIMIT 100"
        ),
    },
    {
        "question": "Compare average annuity-to-income ratio between male and female applicants.",
        "sql": (
            "SELECT code_gender, AVG(annuity_to_income_ratio) AS avg_ratio FROM applicants "
            "GROUP BY code_gender LIMIT 100"
        ),
    },
]


def build_sql_generation_prompt(question: str) -> list[dict]:
    """Returns a messages list (system + few-shot as prior turns + the
    real question) ready to hand to the LLM client."""
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": f"```sql\n{ex['sql']}\n```"})
    messages.append({"role": "user", "content": question})
    return messages


ANSWER_PHRASING_PROMPT = """You are a credit-risk analyst explaining a query result to a non-technical business user.

Rules:
1. Answer the user's question directly in 1-2 concise sentences.
2. You MUST include the exact numeric value(s) shown in the query result.
3. Never omit, replace, round away, or invent a numeric value from the result.
4. If the result contains a rate or proportion between 0 and 1, express it as a percentage and include the underlying value accurately.
5. Do not mention SQL, tables, or column names. Use plain business language.
6. If the result is empty, say so plainly and suggest the user rephrase.
7. Return only the business answer, with no preamble.

Question: {question}

Query result (CSV, first rows):
{result_csv}"""
