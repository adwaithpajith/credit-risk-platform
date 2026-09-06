"""Executes pre-validated SQL against the analytical database and formats
results for both the LLM (compact) and the UI (a DataFrame/markdown table).
NEVER import this module to run SQL that hasn't gone through
sql_guard.validate_sql() first -- see nl_to_sql.py for the only sanctioned
call path."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("query_runner")

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        # SQLite connections aren't thread-safe by default, which matters
        # because Streamlit can call into this from more than one thread.
        connect_args = (
            {"check_same_thread": False}
            if config.db.database_url.startswith("sqlite")
            else {}
        )
        _engine = create_engine(
            config.db.database_url, pool_pre_ping=True, connect_args=connect_args
        )
    return _engine


@dataclass
class QueryResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    truncated: bool

    def to_markdown(self, max_rows: int = 20) -> str:
        if self.dataframe.empty:
            return "_No rows returned._"
        preview = self.dataframe.head(max_rows)
        return preview.to_markdown(index=False)

    def to_compact_text(self, max_rows: int = 15) -> str:
        """A short, token-efficient textual summary handed to the LLM for
        the follow-up "explain this in plain English" step -- we deliberately
        do NOT dump the full DataFrame into the prompt (see nl_to_sql.py's
        token-optimization notes)."""
        if self.dataframe.empty:
            return "No rows returned."
        preview = self.dataframe.head(max_rows)
        return preview.to_csv(index=False)


def run_query(safe_sql: str) -> QueryResult:
    """Execute SQL that has ALREADY passed sql_guard.validate_sql(). Runs
    in a read-only-intent transaction (rolled back, never committed) as a
    final backstop in case anything upstream had a gap."""
    engine = get_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            result = conn.execute(text(safe_sql))
            rows = result.fetchall()
            columns = list(result.keys())
        finally:
            trans.rollback()  # belt-and-braces: never persist side effects

    df = pd.DataFrame(rows, columns=columns)
    truncated = len(df) >= config.db.query_row_limit
    logger.info("Query returned %d rows (truncated=%s)", len(df), truncated)
    return QueryResult(sql=safe_sql, dataframe=df, row_count=len(df), truncated=truncated)
