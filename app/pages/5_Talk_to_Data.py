"""Talk to Data page: NL -> SQL chatbot over the applicant portfolio."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.talk_to_data.nl_to_sql import ask
from src.utils.config import config
from src.utils.theme import apply_theme, page_header, status_badge

st.set_page_config(page_title="Talk to Data", layout="wide")
apply_theme()
page_header("Talk to Data")

if config.llm.has_api_key:
    st.markdown(status_badge("Live LLM", "ok"), unsafe_allow_html=True)
    st.caption(f"LLM backend: **{config.llm.provider}** ({config.llm.active_model_name})")
else:
    st.markdown(status_badge("Fallback mode", "info"), unsafe_allow_html=True)
    st.caption(
        "No LLM API key configured -- running in **built-in query pattern** mode. "
        "Set GEMINI_API_KEY in your .env for free-form questions (Gemini has a free tier)."
    )

with st.expander("Example questions you can ask"):
    st.markdown(
        """
- What is the default rate by income type?
- How many applicants are high risk?
- What is the average income by education?
- Show me the top occupation types by average credit amount
- Which region has the highest risk?
- Compare gender by annuity ratio
- What is the distribution of risk band?
"""
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sql"):
            with st.expander("Show generated SQL"):
                st.code(turn["sql"], language="sql")
        if turn.get("table") is not None and not turn["table"].empty:
            st.dataframe(turn["table"], use_container_width=True)

question = st.chat_input("Ask a question about the applicant portfolio...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(question)
        st.write(answer.answer_text)
        table = None
        if answer.sql:
            with st.expander("Show generated SQL"):
                st.code(answer.sql, language="sql")
        if answer.result is not None:
            table = answer.result.dataframe
            st.dataframe(table, use_container_width=True)
        st.caption(f"Source: {answer.source}")

    st.session_state.chat_history.append({
        "question": question, "answer": answer.answer_text,
        "sql": answer.sql, "table": table,
    })
