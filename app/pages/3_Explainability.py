"""Explainability page: SHAP global importance + per-applicant local explanation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.loader import load_full_dataset
from src.ml.explain import RiskExplainer
from src.ml.predict import RiskScorer
from src.utils.theme import apply_theme, page_header, risk_band_html, style_fig

st.set_page_config(page_title="Explainability", layout="wide")
apply_theme()
page_header("Explainable AI (SHAP)")


@st.cache_resource(show_spinner="Loading model and explainer...")
def _load():
    scorer = RiskScorer()
    explainer = RiskExplainer(scorer)
    return scorer, explainer


@st.cache_data(show_spinner="Loading data...")
def _load_data():
    df, source = load_full_dataset()
    return df, source


scorer, explainer = _load()
df, source = _load_data()

tab1, tab2 = st.tabs(["Global: what drives risk portfolio-wide", "Local: explain one applicant"])

with tab1:
    st.write(
        "Mean absolute SHAP value per feature across a sample of the portfolio -- "
        "this is the model's own view of what matters, for governance and audit review."
    )
    X = scorer.get_feature_matrix(df.drop(columns=["TARGET"], errors="ignore"))
    with st.spinner("Computing SHAP values..."):
        importance = explainer.global_importance(X, sample_size=1500)
    fig = px.bar(
        importance.head(15), x="mean_abs_shap", y="plain_language", orientation="h",
        title="Top 15 features driving default risk (global)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.dataframe(importance.head(20), use_container_width=True)

with tab2:
    if "last_applicant" in st.session_state:
        st.success("Using the applicant you scored on the Risk Prediction page.")
        applicant_df = pd.DataFrame([st.session_state["last_applicant"]])
    else:
        st.warning("No applicant scored yet -- showing a random applicant from the dataset instead.")
        applicant_df = df.drop(columns=["TARGET"], errors="ignore").sample(1, random_state=1)

    local = explainer.explain_one(applicant_df, top_k=8)

    c1, c2 = st.columns(2)
    c1.metric("Predicted default probability", f"{local.probability:.1%}")
    with c2:
        st.caption("RISK BAND")
        st.markdown(risk_band_html(local.risk_band), unsafe_allow_html=True)

    st.subheader("Top factors behind this prediction")
    factor_df = pd.DataFrame(local.top_factors)
    fig = px.bar(
        factor_df.sort_values("shap_value"), x="shap_value", y="plain_text", orientation="h",
        color=factor_df.sort_values("shap_value")["shap_value"] > 0,
        color_discrete_map={True: "#F0616B", False: "#4ADE80"},
        title="Contribution to this applicant's risk score (red = increases risk, green = decreases)",
    )
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="SHAP value (impact on log-odds)")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Plain-English reasons")
    for f in local.top_factors:
        st.markdown(f"- {f['plain_text']}")
