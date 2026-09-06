"""
SQL safety guardrails for LLM-generated queries.

An LLM turning natural language into SQL is a prompt-injection and
data-integrity risk: a malicious or confused prompt could produce
`DROP TABLE`, `UPDATE ... SET`, or a query against a table that doesn't
exist. This module is the actual enforcement layer -- the LLM's output is
NEVER executed until it passes every check here. This is what "hallucination
control" and "SQL validation" mean in production, not just prompting the
model to "please only write SELECT statements".

Layers of defense:
  1. Parse with sqlglot -- reject anything that isn't a single, well-formed
     SELECT statement (blocks DDL/DML outright, not just by keyword-matching).
  2. Table/column allow-list -- only the `applicants` table and its known
     columns may be referenced.
  3. Forbidden keyword scan as a belt-and-braces backstop for constructs
     sqlglot might parse but that we still don't want (e.g. PRAGMA, ATTACH).
  4. Mandatory LIMIT injection -- caps result size regardless of what the
     LLM wrote, so a runaway query can't flood the chat UI or blow the
     token budget of the follow-up "explain these results" call.
  5. Execution happens in a read-only transaction as a final backstop even
     if every check above somehow had a gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("sql_guard")

ALLOWED_TABLES = {"applicants"}
ALLOWED_COLUMNS = {
    "sk_id_curr", "target", "name_contract_type", "code_gender", "flag_own_car",
    "flag_own_realty", "cnt_children", "cnt_fam_members", "name_income_type",
    "name_education_type", "name_family_status", "name_housing_type",
    "occupation_type", "organization_type", "region_rating_client",
    "amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
    "age_years", "employed_years", "ext_source_mean", "credit_to_income_ratio",
    "annuity_to_income_ratio", "bureau_credit_count", "bureau_active_count",
    "bureau_has_overdue", "default_probability", "risk_band",
}
FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "attach", "detach", "pragma", "vacuum", "replace",
    "exec", "execute", "call",
}


@dataclass
class ValidationResult:
    is_valid: bool
    safe_sql: str | None
    error: str | None


def validate_sql(raw_sql: str, dialect: str = "postgres") -> ValidationResult:
    sql = raw_sql.strip().rstrip(";")

    lowered = sql.lower()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in lowered.split() or f" {kw} " in f" {lowered} ":
            return ValidationResult(False, None, f"Forbidden keyword detected: '{kw}'")

    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # sqlglot raises its own ParseError subclasses
        return ValidationResult(False, None, f"SQL failed to parse: {exc}")

    if len(parsed) != 1:
        return ValidationResult(False, None, "Only a single SQL statement is allowed.")

    statement = parsed[0]
    if not isinstance(statement, exp.Select):
        return ValidationResult(
            False, None,
            f"Only SELECT statements are allowed, got: {type(statement).__name__}",
        )

    tables = {t.name.lower() for t in statement.find_all(exp.Table)}
    disallowed_tables = tables - ALLOWED_TABLES
    if disallowed_tables:
        return ValidationResult(
            False, None, f"Query references disallowed table(s): {disallowed_tables}"
        )
    if not tables:
        return ValidationResult(False, None, "Query does not reference the applicants table.")

    # Aliases defined in the SELECT list (e.g. `AVG(target) AS default_rate`)
    # are legitimate to reference in ORDER BY / HAVING and are not real
    # table columns -- exclude them before checking against the allow-list,
    # or a query as ordinary as `ORDER BY default_rate` gets false-flagged.
    select_aliases = {
        a.alias.lower()
        for a in statement.expressions
        if isinstance(a, exp.Alias) and a.alias
    }

    columns = {c.name.lower() for c in statement.find_all(exp.Column)}
    disallowed_columns = columns - ALLOWED_COLUMNS - select_aliases - {"*"}
    if disallowed_columns:
        return ValidationResult(
            False, None, f"Query references unknown column(s): {disallowed_columns}"
        )

    # Enforce a row cap regardless of what the model wrote.
    limit_expr = statement.args.get("limit")
    row_cap = config.db.query_row_limit
    if limit_expr is None:
        statement = statement.limit(row_cap)
    else:
        try:
            requested = int(limit_expr.expression.this)
            if requested > row_cap:
                statement = statement.limit(row_cap)
        except (AttributeError, ValueError):
            statement = statement.limit(row_cap)

    safe_sql = statement.sql(dialect=dialect)
    logger.info("SQL validated OK: %s", safe_sql)
    return ValidationResult(True, safe_sql, None)
