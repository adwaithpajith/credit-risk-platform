from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.talk_to_data import prompt_templates as pt
from src.talk_to_data.query_runner import QueryResult, run_query
from src.talk_to_data.sql_guard import validate_sql
from src.utils.config import config
from src.utils.helpers import stable_hash
from src.utils.logger import get_logger

logger = get_logger("nl_to_sql")

CACHE_PATH = config.paths.data_dir / "nl_to_sql_cache.json"


@dataclass
class ChatAnswer:
    question: str
    sql: str | None
    result: QueryResult | None
    answer_text: str
    source: str  # "gemini" | "fallback_pattern" | "cache" | "blocked" | "no_query"
    error: str | None = None


# --------------------------------------------------------------------- #
# Response cache
# --------------------------------------------------------------------- #

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


# --------------------------------------------------------------------- #
# Keyword-pattern fallback library
#
# These queries work without an API key and provide deterministic,
# hallucination-free answers for common business questions.
# --------------------------------------------------------------------- #

_FALLBACK_PATTERNS: list[tuple[list[str], str]] = [
    (
        ["default rate", "income type"],
        "SELECT name_income_type, AVG(target) AS default_rate, COUNT(*) AS n "
        "FROM applicants WHERE target IS NOT NULL GROUP BY name_income_type "
        "ORDER BY default_rate DESC LIMIT 100",
    ),
    (
        ["default rate", "education"],
        "SELECT name_education_type, AVG(target) AS default_rate, COUNT(*) AS n "
        "FROM applicants WHERE target IS NOT NULL GROUP BY name_education_type "
        "ORDER BY default_rate DESC LIMIT 100",
    ),
    (
        ["how many", "high risk"],
        "SELECT COUNT(*) AS high_risk_count "
        "FROM applicants WHERE risk_band = 'High' LIMIT 100",
    ),
    (
        ["how many", "medium risk"],
        "SELECT COUNT(*) AS medium_risk_count "
        "FROM applicants WHERE risk_band = 'Medium' LIMIT 100",
    ),
    (
        ["average income", "education"],
        "SELECT name_education_type, "
        "AVG(amt_income_total) AS avg_income "
        "FROM applicants "
        "GROUP BY name_education_type "
        "ORDER BY avg_income DESC LIMIT 100",
    ),
    (
        ["top", "occupation"],
        "SELECT occupation_type, "
        "AVG(amt_credit) AS avg_credit "
        "FROM applicants "
        "WHERE occupation_type IS NOT NULL "
        "GROUP BY occupation_type "
        "ORDER BY avg_credit DESC LIMIT 5",
    ),
    (
        ["region", "risk"],
        "SELECT region_rating_client, "
        "AVG(CASE WHEN risk_band = 'High' THEN 1.0 ELSE 0.0 END) "
        "AS high_risk_share, "
        "COUNT(*) AS n "
        "FROM applicants "
        "GROUP BY region_rating_client "
        "ORDER BY high_risk_share DESC LIMIT 100",
    ),
    (
        ["gender", "annuity"],
        "SELECT code_gender, "
        "AVG(annuity_to_income_ratio) AS avg_ratio "
        "FROM applicants "
        "GROUP BY code_gender LIMIT 100",
    ),
    (
        ["risk band", "distribution"],
        "SELECT risk_band, COUNT(*) AS n "
        "FROM applicants "
        "GROUP BY risk_band "
        "ORDER BY n DESC LIMIT 100",
    ),
    (
        ["average credit", "amount"],
        "SELECT AVG(amt_credit) AS avg_credit_amount "
        "FROM applicants LIMIT 100",
    ),
    (
        ["how many", "low risk"],
        "SELECT COUNT(*) AS low_risk_count "
        "FROM applicants WHERE risk_band = 'Low' LIMIT 100",
    ),
    (
        ["default rate", "gender"],
        "SELECT code_gender, AVG(target) AS default_rate, COUNT(*) AS n "
        "FROM applicants WHERE target IS NOT NULL "
        "GROUP BY code_gender ORDER BY default_rate DESC LIMIT 100",
    ),
]


def _match_fallback_pattern(question: str) -> str | None:
    q = question.lower()

    for keywords, sql in _FALLBACK_PATTERNS:
        if all(keyword in q for keyword in keywords):
            return sql

    return None


# --------------------------------------------------------------------- #
# Gemini LLM backend
# --------------------------------------------------------------------- #

def _extract_sql_block(text: str) -> str | None:
    """
    Extract SQL from Gemini's response.

    Supports:
      ```sql
      SELECT ...
      ```

    and plain SELECT responses.
    """

    match = re.search(
        r"```sql\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    if "NO_QUERY_POSSIBLE" in text:
        return None

    stripped = text.strip()

    if stripped.lower().startswith("select"):
        return stripped

    return None


def _call_gemini(
    system: str,
    messages: list[dict],
    max_tokens: int,
) -> str:
    """
    Call Google's Gemini API.

    The google-genai package is imported lazily so that the application
    can still run using deterministic fallback patterns when no API key
    is configured.
    """

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=config.llm.gemini_api_key
    )

    contents = []

    for message in messages:
        role = (
            "model"
            if message["role"] == "assistant"
            else "user"
        )

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(
                        text=message["content"]
                    )
                ],
            )
        )

    response = client.models.generate_content(
        model=config.llm.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=config.llm.temperature,
        ),
    )

    return response.text or ""


def generate_sql_via_llm(question: str) -> str | None:
    """
    Generate SQL from a natural-language question using Gemini.
    """

    messages = pt.build_sql_generation_prompt(question)

    raw = _call_gemini(
        system=pt.SYSTEM_PROMPT,
        messages=messages,
        max_tokens=config.llm.max_tokens,
    )

    return _extract_sql_block(raw)


def phrase_answer_via_llm(
    question: str,
    result: QueryResult,
) -> str:
    """
    Use Gemini to turn SQL results into a concise business answer.
    """

    prompt = pt.ANSWER_PHRASING_PROMPT.format(
        question=question,
        result_csv=result.to_compact_text(),
    )

    return _call_gemini(
        system=(
            "You explain data results clearly and concisely "
            "for business users. Do not invent numbers or facts."
        ),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=200,
    ).strip()


# --------------------------------------------------------------------- #
# Deterministic answer generation
# --------------------------------------------------------------------- #

def _phrase_answer_without_llm(
    question: str,
    result: QueryResult,
) -> str:
    """
    Deterministic business-readable phrasing used when Gemini is
    unavailable or its answer fails grounding validation.
    """

    if result.dataframe.empty:
        return "No matching applicants were found for that question."

    df = result.dataframe

    def format_value(column: str, value) -> str:
        column_lower = column.lower()

        # Convert proportions/rates to business-friendly percentages.
        if (
            "rate" in column_lower
            or "percentage" in column_lower
            or "share" in column_lower
        ):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)

                if 0 <= numeric <= 1:
                    return f"{numeric * 100:.2f}%"

        # Format ordinary numeric values cleanly.
        if isinstance(value, float):
            return f"{value:,.2f}"

        if isinstance(value, int):
            return f"{value:,}"

        return str(value)

    def friendly_label(column: str) -> str:
        labels = {
            "default_rate": "Default rate",
            "high_risk_share": "High-risk share",
            "avg_income": "Average income",
            "avg_credit_amount": "Average credit amount",
            "avg_credit": "Average credit",
            "low_risk_count": "Low-risk applicants",
            "medium_risk_count": "Medium-risk applicants",
            "high_risk_count": "High-risk applicants",
            "n": "Applicants",
            "code_gender": "Gender",
            "name_education_type": "Education",
            "name_income_type": "Income type",
            "occupation_type": "Occupation",
            "region_rating_client": "Region rating",
        }

        return labels.get(
            column,
            column.replace("_", " ").capitalize(),
        )

    if df.shape == (1, 1):
        column = str(df.columns[0])
        value = df.iloc[0, 0]

        return f"{friendly_label(column)}: {format_value(column, value)}"

    lines = [
        f"Here are the top results for your question "
        f"({len(df)} row(s) shown):"
    ]

    for _, row in df.head(5).iterrows():
        parts = []

        for column in df.columns:
            value = row[column]
            parts.append(
                f"{friendly_label(str(column))}: "
                f"{format_value(str(column), value)}"
            )

        lines.append(" - " + ", ".join(parts))

    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------- #


def _answer_is_grounded(
    answer: str,
    result: QueryResult,
) -> bool:
    """
    Check whether an LLM-generated answer faithfully represents
    the important numeric value returned by the database.
    """

    if result.dataframe.empty:
        return True

    if not answer or not answer.strip():
        return False

    if result.dataframe.shape == (1, 1):
        value = result.dataframe.iloc[0, 0]
        column = str(result.dataframe.columns[0]).lower()

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value_float = float(value)

            # Rates/proportions should be communicated as percentages.
            if "rate" in column or "percentage" in column:
                percentage = (
                    value_float * 100
                    if 0 <= value_float <= 1
                    else value_float
                )

                expected_values = [
                    f"{percentage:.2f}%",
                    f"{percentage:.1f}%",
                    f"{percentage:.3f}%",
                ]

                answer_lower = answer.lower()

                return any(
                    expected.lower() in answer_lower
                    for expected in expected_values
                )

            # For ordinary numeric results, require a sufficiently
            # precise representation rather than a short substring.
            expected_values = [
                f"{value_float:.6f}",
                f"{value_float:.4f}",
                f"{value_float:.3f}",
                f"{value_float:.2f}",
            ]

            return any(
                expected in answer
                for expected in expected_values
            )

    return True

def _safe_phrase_answer(
    question: str,
    result: QueryResult,
) -> str:
    """
    Get a business-readable answer from Gemini, but fall back to
    deterministic formatting if the answer is empty or not grounded
    in the database result.
    """

    if not config.llm.has_api_key:
        return _phrase_answer_without_llm(question, result)

    try:
        answer = phrase_answer_via_llm(
            question,
            result,
        )

        if _answer_is_grounded(answer, result):
            return answer

        logger.warning(
            "Gemini answer failed grounding check; "
            "using deterministic phrasing."
        )

    except Exception as exc:
        logger.warning(
            "Gemini answer phrasing failed (%s); "
            "using deterministic phrasing.",
            exc,
        )

    return _phrase_answer_without_llm(question, result)


def ask(question: str) -> ChatAnswer:
    """
    Process a natural-language question end-to-end.

    Flow:

        User question
              ↓
        Cache lookup
              ↓
        Gemini SQL generation
              ↓
        Fallback patterns if Gemini unavailable
              ↓
        SQL safety validation
              ↓
        Database execution
              ↓
        Gemini answer phrasing
              ↓
        ChatAnswer
    """

    question = question.strip()

    if not question:
        return ChatAnswer(
            question="",
            sql=None,
            result=None,
            answer_text="Please enter a question.",
            source="no_query",
        )

    # --------------------------------------------------------------- #
    # Cache lookup
    # --------------------------------------------------------------- #

    cache_key = stable_hash(
        pt.PROMPT_VERSION,
        question.lower(),
    )

    if config.llm.enable_cache:
        cache = _load_cache()

        if cache_key in cache:
            entry = cache[cache_key]

            logger.info(
                "Cache hit for question: %r",
                question,
            )

            validation = validate_sql(
                entry["sql"]
            )

            if validation.is_valid:
                result = run_query(
                    validation.safe_sql
                )

                if config.llm.has_api_key:
                    try:
                        answer_text = _safe_phrase_answer(
                            question,
                            result,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Gemini answer phrasing failed on cache hit: %s",
                            exc,
                        )

                        answer_text = _phrase_answer_without_llm(
                            question,
                            result,
                        )
                else:
                    answer_text = _phrase_answer_without_llm(
                        question,
                        result,
                    )

                return ChatAnswer(
                    question=question,
                    sql=validation.safe_sql,
                    result=result,
                    answer_text=answer_text,
                    source="cache",
                )

    # --------------------------------------------------------------- #
    # Gemini SQL generation
    # --------------------------------------------------------------- #

    raw_sql = None
    source = "gemini"

    if config.llm.has_api_key:
        try:
            raw_sql = generate_sql_via_llm(
                question
            )

        except Exception as exc:
            logger.warning(
                "Gemini SQL generation failed (%s); "
                "falling back to pattern matching.",
                exc,
            )

            raw_sql = None

    # --------------------------------------------------------------- #
    # Deterministic fallback
    # --------------------------------------------------------------- #

    if raw_sql is None:
        raw_sql = _match_fallback_pattern(
            question
        )

        source = "fallback_pattern"

    # --------------------------------------------------------------- #
    # No matching query
    # --------------------------------------------------------------- #

    if raw_sql is None:
        return ChatAnswer(
            question=question,
            sql=None,
            result=None,
            answer_text=(
                "I couldn't map that question to the applicant data. "
                "Try asking about income, credit amount, risk bands, "
                "default rates, or a specific segment "
                "(education, occupation, region, gender)."
            ),
            source="no_query",
        )

    # --------------------------------------------------------------- #
    # SQL safety validation
    # --------------------------------------------------------------- #

    validation = validate_sql(
        raw_sql
    )

    if not validation.is_valid:
        logger.warning(
            "Generated SQL rejected by guard: %s | sql=%s",
            validation.error,
            raw_sql,
        )

        return ChatAnswer(
            question=question,
            sql=raw_sql,
            result=None,
            answer_text=(
                "I generated a query but it failed a safety check "
                f"({validation.error}). Please rephrase your question."
            ),
            source="blocked",
            error=validation.error,
        )

    # --------------------------------------------------------------- #
    # Execute SQL
    # --------------------------------------------------------------- #

    result = run_query(
        validation.safe_sql
    )

    # --------------------------------------------------------------- #
    # Generate business-readable answer
    # --------------------------------------------------------------- #

    if source == "gemini" and config.llm.has_api_key:
        try:
            answer_text = _safe_phrase_answer(
                question,
                result,
            )

        except Exception as exc:
            logger.warning(
                "Gemini answer phrasing failed (%s); "
                "using deterministic phrasing.",
                exc,
            )

            answer_text = _phrase_answer_without_llm(
                question,
                result,
            )

    else:
        answer_text = _phrase_answer_without_llm(
            question,
            result,
        )

    # --------------------------------------------------------------- #
    # Save SQL to cache
    # --------------------------------------------------------------- #

    if config.llm.enable_cache:
        cache = _load_cache()

        cache[cache_key] = {
            "question": question,
            "sql": validation.safe_sql,
        }

        _save_cache(cache)

    return ChatAnswer(
        question=question,
        sql=validation.safe_sql,
        result=result,
        answer_text=answer_text,
        source=source,
    )