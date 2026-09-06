"""EDA page: dataset summary, data quality, and business insights."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.loader import load_full_dataset
from src.data.preprocessor import engineer_features
from src.utils.theme import apply_theme, page_header, style_fig

st.set_page_config(page_title="EDA", layout="wide")
apply_theme()
page_header("Data Understanding & Exploratory Analysis")


@st.cache_data(show_spinner="Loading dataset...")
def _load():
    df, source = load_full_dataset()
    return engineer_features(df), source


df, source = _load()
if source == "synthetic":
    st.info(
        "Running on **synthetic fallback data** (schema-faithful stand-in for the real "
        "Kaggle dataset -- see data/README.md to plug in the real files)."
    )

st.subheader("Dataset summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Default rate", f"{df['TARGET'].mean():.2%}" if "TARGET" in df.columns else "n/a")
c4.metric("Missing cells", f"{df.isna().sum().sum():,}")

st.subheader("Feature categorization")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
cat1, cat2 = st.columns(2)
cat1.write(f"**Numeric features** ({len(numeric_cols)})")
cat1.write(", ".join(numeric_cols))
cat2.write(f"**Categorical features** ({len(categorical_cols)})")
cat2.write(", ".join(categorical_cols))

st.subheader("Data quality: missing values")
missing = (
    df.isna().mean().sort_values(ascending=False).reset_index()
    .rename(columns={"index": "column", 0: "missing_fraction"})
)
missing.columns = ["column", "missing_fraction"]
missing = missing[missing["missing_fraction"] > 0].head(15)
if not missing.empty:
    fig = px.bar(missing, x="missing_fraction", y="column", orientation="h",
                 title="Top columns by missing-value fraction")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_fig(fig), use_container_width=True)
else:
    st.write("No missing values detected.")

st.divider()
st.subheader("Business insights")

insights = st.tabs([
    "1. Default rate by income type", "2. External score vs default",
    "3. Credit-to-income vs default", "4. Age vs default",
    "5. Bureau history vs default",
])

with insights[0]:
    g = df.groupby("NAME_INCOME_TYPE")["TARGET"].agg(["mean", "count"]).reset_index()
    g.columns = ["income_type", "default_rate", "n"]
    fig = px.bar(g.sort_values("default_rate", ascending=False), x="income_type", y="default_rate",
                 hover_data=["n"], title="Default rate by income type")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption(
        "**Insight:** Income type is a strong, policy-actionable segmentation variable -- "
        "some income types default several times more often than others, which should feed "
        "directly into underwriting tiers."
    )

with insights[1]:
    fig = px.histogram(df, x="EXT_SOURCE_MEAN", color=df["TARGET"].map({0: "Repaid", 1: "Defaulted"}),
                        barmode="overlay", nbins=40, title="External bureau score distribution by outcome")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption(
        "**Insight:** Defaulters are heavily concentrated at low external-score values, "
        "confirming EXT_SOURCE_* as the single most bankable signal in this dataset -- "
        "consistent with SHAP's global importance ranking on the Explainability page."
    )

with insights[2]:
    df_clip = df[df["CREDIT_TO_INCOME_RATIO"] < df["CREDIT_TO_INCOME_RATIO"].quantile(0.98)]
    fig = px.box(df_clip, x=df_clip["TARGET"].map({0: "Repaid", 1: "Defaulted"}),
                 y="CREDIT_TO_INCOME_RATIO", title="Credit-to-income ratio by outcome")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption(
        "**Insight:** Defaulters skew toward higher credit-to-income ratios -- applicants "
        "borrowing a large multiple of their income are measurably riskier, supporting an "
        "affordability cap in credit policy."
    )

with insights[3]:
    fig = px.histogram(df, x="AGE_YEARS", color=df["TARGET"].map({0: "Repaid", 1: "Defaulted"}),
                        barmode="overlay", nbins=30, title="Applicant age distribution by outcome")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption(
        "**Insight:** Younger applicants default more often than older ones -- age (as a "
        "proxy for credit history length and income stability) carries real signal, though "
        "it must be used carefully alongside fair-lending compliance review."
    )

with insights[4]:
    if "BUREAU_ACTIVE_COUNT" in df.columns:
        g = df.groupby("BUREAU_HAS_OVERDUE")["TARGET"].mean().reset_index()
        g["BUREAU_HAS_OVERDUE"] = g["BUREAU_HAS_OVERDUE"].map({0: "No prior overdue", 1: "Has prior overdue"})
        fig = px.bar(g, x="BUREAU_HAS_OVERDUE", y="TARGET", title="Default rate by prior bureau overdue history")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption(
            "**Insight:** A history of overdue payments at OTHER lenders is one of the "
            "strongest single predictors available -- exactly the kind of cross-institution "
            "signal a credit bureau feed is meant to provide, and it dominates the business "
            "rules on the Business Rules page."
        )
    else:
        st.write("Bureau data not available in this run.")

st.divider()
st.subheader("Raw feature preview")
st.dataframe(df.head(50), use_container_width=True)
