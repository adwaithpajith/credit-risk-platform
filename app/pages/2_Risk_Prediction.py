"""Risk Prediction page: score a new applicant interactively."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.ml.predict import RiskScorer
from src.utils.theme import apply_theme, page_header, risk_band_html

st.set_page_config(page_title="Risk Prediction", layout="wide")
apply_theme()
page_header("Loan Default Risk Prediction")


@st.cache_resource(show_spinner="Loading model...")
def _load_scorer():
    return RiskScorer()


try:
    scorer = _load_scorer()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Model decision threshold (cost-optimized): **{scorer.threshold:.3f}**")

with st.form("applicant_form"):
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)

    with c1:
        income_type = st.selectbox(
            "Income type",
            ["Working", "State servant", "Commercial associate", "Pensioner", "Student"],
        )
        education = st.selectbox(
            "Education",
            ["Secondary / secondary special", "Higher education", "Incomplete higher",
             "Lower secondary", "Academic degree"],
        )
        family_status = st.selectbox(
            "Family status",
            ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"],
        )
        gender = st.selectbox("Gender", ["F", "M"])
        own_car = st.selectbox("Owns a car?", ["N", "Y"])
        own_realty = st.selectbox("Owns real estate?", ["Y", "N"])

    with c2:
        age = st.slider("Age (years)", 21, 75, 38)
        employed_years = st.slider("Years employed", 0, 45, 6)
        income = st.number_input("Annual income", min_value=0, value=180000, step=5000)
        credit = st.number_input("Requested credit amount", min_value=0, value=600000, step=10000)
        annuity = st.number_input("Monthly annuity", min_value=0, value=28000, step=1000)
        goods_price = st.number_input("Goods price", min_value=0, value=580000, step=10000)

    with c3:
        ext_1 = st.slider("External score 1 (bureau)", 0.0, 1.0, 0.5)
        ext_2 = st.slider("External score 2 (bureau)", 0.0, 1.0, 0.5)
        ext_3 = st.slider("External score 3 (bureau)", 0.0, 1.0, 0.5)
        region_rating = st.selectbox("Region risk rating (1=best, 3=worst)", [1, 2, 3], index=1)
        bureau_credit_count = st.slider("Prior credits at other lenders", 0, 20, 2)
        bureau_active_count = st.slider("...of which still active", 0, 20, 1)
        bureau_has_overdue = st.selectbox("Any overdue prior credit?", ["No", "Yes"]) == "Yes"

    submitted = st.form_submit_button("Score applicant", type="primary")

if submitted:
    applicant = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Core staff",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "REGION_RATING_CLIENT": region_rating,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_BIRTH": -int(age * 365.25),
        "DAYS_EMPLOYED": -int(employed_years * 365.25),
        "DAYS_REGISTRATION": -1000,
        "DAYS_ID_PUBLISH": -500,
        "OWN_CAR_AGE": 5.0 if own_car == "Y" else None,
        "EXT_SOURCE_1": ext_1,
        "EXT_SOURCE_2": ext_2,
        "EXT_SOURCE_3": ext_3,
        "REGION_POPULATION_RELATIVE": 0.02,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 1,
        "BUREAU_CREDIT_COUNT": bureau_credit_count,
        "BUREAU_ACTIVE_COUNT": bureau_active_count,
        "BUREAU_AMT_CREDIT_SUM_TOTAL": bureau_credit_count * 150000.0,
        "BUREAU_AMT_CREDIT_SUM_MEAN": 150000.0,
        "BUREAU_AMT_OVERDUE_TOTAL": 5000.0 if bureau_has_overdue else 0.0,
        "BUREAU_MAX_DAYS_OVERDUE": 15 if bureau_has_overdue else 0,
        "BUREAU_DAYS_CREDIT_MEAN": -800.0,
        "BUREAU_ACTIVE_RATIO": bureau_active_count / max(bureau_credit_count, 1),
        "BUREAU_HAS_OVERDUE": int(bureau_has_overdue),
    }

    result = scorer.score_one(applicant)
    st.session_state["last_applicant"] = applicant
    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.divider()
    st.subheader("Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{result.probability:.1%}")
    with c2:
        st.caption("RISK BAND")
        st.markdown(risk_band_html(result.risk_band), unsafe_allow_html=True)
    decision = "Decline / manual review" if result.probability >= result.threshold_used else "Approve"
    c3.metric("Suggested decision", decision)

    st.info(
        "Head to the **Explainability** page to see exactly which factors drove this "
        "specific prediction (SHAP), or **Business Rules** for the auditable rule that "
        "matches this risk band."
    )
