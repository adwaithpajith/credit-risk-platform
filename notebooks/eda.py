# %% [markdown]
# # Home Credit Default Risk -- Exploratory Data Analysis
#
# This notebook is the source of truth for the EDA shown in the Streamlit
# app's "EDA" page (`app/pages/1_EDA.py`) -- the same `src/data` modules
# power both, so the notebook and the app can never drift out of sync.
#
# Convert to `.ipynb` with: `jupytext --to notebook eda.py`
# (or just run this file directly with `python notebooks/eda.py`)

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_full_dataset
from src.data.preprocessor import engineer_features

pd.set_option("display.max_columns", 60)

# %% [markdown]
# ## 1. Load data
#
# Requires the real dataset at `data/raw/application_train.csv` (see
# `data/README.md`). `USE_SYNTHETIC_FALLBACK=true` can be set as an env
# var to run this notebook against the synthetic dev fixture instead, for
# quick iteration without the full download.

# %%
df, source = load_full_dataset()
df = engineer_features(df)
print(f"Data source: {source}")
print(f"Shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Dataset summary & data quality

# %%
print("Base default rate:", f"{df['TARGET'].mean():.2%}")
print("\nDtype breakdown:")
print(df.dtypes.value_counts())

# %%
missing = df.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]
print(f"{len(missing)} columns have missing values.")
missing.head(15).plot(kind="barh", title="Top columns by missing fraction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Feature categorization

# %%
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
print(f"Numeric features ({len(numeric_cols)}):", numeric_cols)
print(f"\nCategorical features ({len(categorical_cols)}):", categorical_cols)

# %% [markdown]
# ## 4. Business insight #1 -- Default rate by income type
#
# **Insight:** income type is a strong, policy-actionable segmentation
# variable for underwriting tiers.

# %%
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False).plot(
    kind="bar", title="Default rate by income type"
)
plt.ylabel("Default rate")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Business insight #2 -- External bureau score vs default
#
# **Insight:** EXT_SOURCE_* is the single most bankable signal in this
# dataset -- defaulters concentrate heavily at low scores.

# %%
for target_val, label in [(0, "Repaid"), (1, "Defaulted")]:
    df.loc[df["TARGET"] == target_val, "EXT_SOURCE_MEAN"].hist(
        bins=40, alpha=0.6, label=label
    )
plt.legend()
plt.title("External score distribution by outcome")
plt.xlabel("EXT_SOURCE_MEAN")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Business insight #3 -- Credit-to-income ratio vs default
#
# **Insight:** defaulters skew toward higher credit-to-income ratios --
# supports an affordability cap in credit policy.

# %%
clip_q = df["CREDIT_TO_INCOME_RATIO"].quantile(0.98)
df_clip = df[df["CREDIT_TO_INCOME_RATIO"] < clip_q]
df_clip.boxplot(column="CREDIT_TO_INCOME_RATIO", by="TARGET")
plt.title("Credit-to-income ratio by outcome")
plt.suptitle("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Business insight #4 -- Age vs default
#
# **Insight:** younger applicants default more often -- age (as a proxy
# for credit history length and income stability) carries real signal.

# %%
for target_val, label in [(0, "Repaid"), (1, "Defaulted")]:
    df.loc[df["TARGET"] == target_val, "AGE_YEARS"].hist(bins=30, alpha=0.6, label=label)
plt.legend()
plt.title("Applicant age distribution by outcome")
plt.xlabel("Age (years)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Business insight #5 -- Prior bureau overdue history vs default
#
# **Insight:** a history of overdue payments at OTHER lenders is one of
# the strongest single predictors available -- exactly the cross-
# institution signal a credit bureau feed exists to provide.

# %%
if "BUREAU_HAS_OVERDUE" in df.columns:
    df.groupby("BUREAU_HAS_OVERDUE")["TARGET"].mean().rename(
        index={0: "No prior overdue", 1: "Has prior overdue"}
    ).plot(kind="bar", title="Default rate by prior bureau overdue history")
    plt.ylabel("Default rate")
    plt.tight_layout()
    plt.show()
else:
    print("bureau.csv not loaded -- skip (see data/README.md to enable bureau features).")

# %% [markdown]
# ## Summary of findings
#
# 1. Income type is a strong, actionable underwriting segmentation variable.
# 2. External bureau scores (EXT_SOURCE_1/2/3) are the dominant predictive
#    signal -- confirmed again by SHAP global importance in the app.
# 3. Credit-to-income ratio separates repayers from defaulters and
#    supports an affordability-based policy cap.
# 4. Age carries real signal, net of other factors -- must be used
#    carefully alongside fair-lending review.
# 5. Prior bureau overdue history is one of the single strongest
#    predictors -- it dominates the auto-derived business rules.
#
# See `src/ml/train.py` for how these findings translate into the modeling
# approach (LightGBM + cost-sensitive threshold), and the Streamlit app's
# Explainability / Business Rules pages for how the trained model
# operationalizes them.
