"""Business Rules page: model distilled into auditable IF/THEN rules."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.data.loader import load_full_dataset
from src.ml.predict import RiskScorer
from src.ml.rules import derive_rules, load_rules, save_rules
from src.utils.theme import apply_theme, page_header, risk_band_html

st.set_page_config(page_title="Business Rules", layout="wide")
apply_theme()
page_header("Business-Readable Decision Rules")

st.markdown(
    """
These rules are **distilled from the trained ML model**, not hand-written -- a shallow
decision tree is fit to reproduce the model's own Low/Medium/High risk-band output, then
converted into plain-English IF/THEN statements. Each rule reports:
- **Coverage (support):** how many applicants it applies to
- **Agreement:** what fraction of those applicants the rule correctly matches to the
  model's actual output (i.e. rule fidelity, not ground-truth accuracy)

Use these to explain credit policy to non-technical stakeholders or auditors without
exposing raw model internals.
"""
)

rules = load_rules()

if not rules or st.button("Re-derive rules from current model"):
    with st.spinner("Fitting surrogate decision tree..."):
        df, _ = load_full_dataset()
        scorer = RiskScorer()
        scored = scorer.score_dataframe(df.drop(columns=["TARGET"], errors="ignore"))
        X = scorer.get_feature_matrix(df.drop(columns=["TARGET"], errors="ignore"))
        rule_objs = derive_rules(X, scored["RISK_BAND"])
        save_rules(rule_objs)
        rules = load_rules()

band_filter = st.multiselect("Filter by predicted band", ["Low", "Medium", "High"],
                              default=["Low", "Medium", "High"])
rules_df = pd.DataFrame(rules)
if not rules_df.empty:
    filtered = rules_df[rules_df["predicted_band"].isin(band_filter)]
    filtered = filtered.sort_values(["predicted_band", "support"], ascending=[True, False])

    for band in ["High", "Medium", "Low"]:
        band_rules = filtered[filtered["predicted_band"] == band]
        if band_rules.empty:
            continue
        st.markdown(
            f'{risk_band_html(band, size="0.9rem")} &nbsp; **{len(band_rules)} rule(s)**',
            unsafe_allow_html=True,
        )
        for _, r in band_rules.iterrows():
            st.markdown(f"- {r['plain_english']}")
        st.write("")
else:
    st.warning("No rules available yet -- click the button above to derive them.")
