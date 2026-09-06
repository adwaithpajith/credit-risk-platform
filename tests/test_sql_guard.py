"""Tests for the SQL safety guardrails -- the most security-critical piece
of the talk-to-data chatbot. Run with: pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.talk_to_data.sql_guard import validate_sql


def test_valid_select_passes():
    result = validate_sql("SELECT AVG(amt_income_total) FROM applicants")
    assert result.is_valid
    assert "applicants" in result.safe_sql.lower()


def test_valid_group_by_with_alias_passes():
    sql = (
        "SELECT name_income_type, AVG(target) as default_rate FROM applicants "
        "GROUP BY name_income_type ORDER BY default_rate DESC"
    )
    result = validate_sql(sql)
    assert result.is_valid, result.error


def test_drop_table_is_blocked():
    result = validate_sql("DROP TABLE applicants")
    assert not result.is_valid


def test_stacked_statement_injection_is_blocked():
    result = validate_sql("SELECT * FROM applicants; DROP TABLE applicants;")
    assert not result.is_valid


def test_update_statement_is_blocked():
    result = validate_sql("UPDATE applicants SET target = 0")
    assert not result.is_valid


def test_unknown_table_is_blocked():
    result = validate_sql("SELECT * FROM users")
    assert not result.is_valid


def test_unknown_column_is_blocked():
    result = validate_sql("SELECT ssn FROM applicants")
    assert not result.is_valid


def test_missing_limit_is_auto_injected():
    result = validate_sql("SELECT * FROM applicants")
    assert result.is_valid
    assert "limit" in result.safe_sql.lower()


def test_oversized_limit_is_capped():
    result = validate_sql("SELECT * FROM applicants LIMIT 999999999")
    assert result.is_valid
    assert "999999999" not in result.safe_sql


def test_subquery_on_allowed_table_passes():
    sql = (
        "SELECT * FROM applicants WHERE sk_id_curr IN "
        "(SELECT sk_id_curr FROM applicants WHERE risk_band = 'High')"
    )
    result = validate_sql(sql)
    assert result.is_valid, result.error


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
