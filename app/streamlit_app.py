"""
Main Streamlit entry point -- Home page. Individual sections (EDA, Risk
Prediction, Explainability, Business Rules, Talk to Data) live in
app/pages/ as Streamlit's native multi-page mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.data.loader import load_full_dataset
from src.ml.train import METRICS_PATH, MODEL_PATH
from src.utils.config import config
from src.utils.theme import apply_theme, page_header, status_badge

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    layout="wide",
)
apply_theme()

page_header(
    "AI-Powered Credit Risk Intelligence Platform",
    subtitle="NeoStats AI Engineer Assignment -- Home Credit Default Risk",
)

st.markdown(
    """
Use the sidebar to navigate:

- **EDA** -- dataset summary, data quality, and key business insights
- **Risk Prediction** -- score a new applicant and get a risk band
- **Explainability** -- SHAP-based reasons behind any prediction
- **Business Rules** -- ML model distilled into auditable IF/THEN rules
- **Talk to Data** -- ask questions about the applicant portfolio in plain English
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

try:
    df, source = load_full_dataset()
    col1.metric("Applicants loaded", f"{len(df):,}")
    col1.caption(f"Data source: **{source}**" + (" (see data/README.md for real Kaggle data)" if source == "synthetic" else ""))
except Exception as exc:  # noqa: BLE001
    col1.error(f"Could not load data: {exc}")

if MODEL_PATH.exists():
    col2.metric("Model status", "Trained")
    col2.markdown(status_badge("Ready", "ok"), unsafe_allow_html=True)
    if METRICS_PATH.exists():
        import json
        metrics = json.loads(METRICS_PATH.read_text())
        col2.caption(f"Test PR-AUC: **{metrics['test_pr_auc']:.3f}** | Test ROC-AUC: **{metrics['test_roc_auc']:.3f}**")
else:
    col2.metric("Model status", "Not trained")
    col2.markdown(status_badge("Action needed", "warn"), unsafe_allow_html=True)
    col2.caption("Run `python -m src.ml.train` (see README).")

col3.metric("LLM configured", "Yes" if config.llm.has_api_key else "No")
col3.markdown(
    status_badge("Live LLM", "ok") if config.llm.has_api_key else status_badge("Fallback mode", "info"),
    unsafe_allow_html=True,
)
col3.caption(
    "Talk-to-Data will use the LLM." if config.llm.has_api_key
    else "Talk-to-Data uses built-in query patterns until an API key is set in .env."
)

st.divider()
st.subheader("Architecture at a glance")
st.markdown(
    """
```
Raw data (Kaggle CSV or synthetic fallback)
        |
   src/data/  -> loader.py, preprocessor.py  (clean, engineer features)
        |
   src/ml/    -> train.py (LightGBM + CV + cost-optimal threshold)
              -> explain.py (SHAP)
              -> rules.py (surrogate decision tree -> business rules)
        |
   src/data/db_loader.py -> scores + loads applicants into SQL DB
        |
   src/talk_to_data/ -> prompt_templates.py, sql_guard.py, nl_to_sql.py, query_runner.py
        |
   app/  (this Streamlit UI) <---- ties every module above into one product
```
    """
)
