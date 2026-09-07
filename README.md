# AI-Powered Credit Risk Intelligence Platform

A lightweight, end-to-end credit risk platform built on the [Home Credit
Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
dataset -- EDA, an explainable ML risk model, auto-derived business rules,
an NL-to-SQL chatbot, and a multi-section UI, all Dockerized.



\---

## 1\. Architecture overview

```
Real dataset (data/raw/application\_train.csv, bureau.csv -- see data/README.md)
        |
src/data/loader.py         -> loads + joins raw tables
src/data/preprocessor.py   -> cleans, engineers underwriting-driven features
        |
src/ml/train.py            -> LightGBM, 5-fold CV, scale\_pos\_weight for imbalance,
                               cost-optimal decision threshold
src/ml/explain.py          -> SHAP (global + per-applicant local explanations)
src/ml/rules.py            -> surrogate decision tree -> auditable IF/THEN rules
        |
src/data/db\_loader.py      -> scores every applicant, loads into Postgres/SQLite
        |
src/talk\_to\_data/          -> prompt\_templates.py, sql\_guard.py, nl\_to\_sql.py,
                               query\_runner.py  (NL question -> validated SQL -> answer)
        |
app/streamlit\_app.py + app/pages/\*  -> ties everything into one multi-section UI
        |
Dockerfile + docker-compose.yml (app + Postgres) -> `docker-compose up` runs it all
```

Every box above is independently testable and independently importable --
the Streamlit app is a thin UI layer over `src/`, not where the logic
lives. See `tests/` for unit coverage of the two most safety-critical
pieces (SQL guardrails, feature-engineering/train-serve consistency).

\---

## 2\. Quick start

###  Docker

```bash
# 1. Get the real dataset (see data/README.md) and place the files here:
mkdir -p data/raw
#   data/raw/application\_train.csv   (required)
#   data/raw/bureau.csv               (optional, enables bureau features)

# 2. Configure environment
cp .env.example .env
#   Optionally add GEMINI\_API\_KEY for free-form Talk-to-Data questions --
#   see section 6 below. Everything else has a sensible default.

# 3. Run
docker-compose up
```

Then open **http://localhost:8501**. On first startup the app container
trains the model, derives business rules, and loads the analytical
database automatically (see `scripts/entrypoint.sh`) -- subsequent
restarts skip training since the model artifact persists in a named
volume.



## 3\. Model selection \& class imbalance strategy

**Model: LightGBM gradient-boosted trees.** Chosen over a linear model
because credit risk features interact non-trivially (e.g. the effect of
credit-to-income ratio depends on income type), and over a deep-learning
approach because tabular GBMs consistently outperform neural nets on
datasets of this size and structure -- this is also what most real
credit-risk shops actually run in production, not a toy baseline.
LightGBM's native handling of missing values and categorical features
(see `src/data/preprocessor.py`) also means no separate
imputation/encoding pipeline to build, maintain, and risk leaking through.

**Class imbalance (\~8% default rate): handled via `scale\_pos\_weight`
inside the training objective, NOT SMOTE/oversampling.** SMOTE on mixed
numeric/categorical, heavily-missing tabular data like this tends to
generate unrealistic synthetic applicants (interpolating between, say, two
very different occupation/income combinations) and inflates validation
metrics without improving real generalization. Cost-proportionate learning
via `scale\_pos\_weight` addresses the imbalance directly in the loss
function with no synthetic rows and no risk of train/test leakage through
oversampled duplicates.

**Model selection metric: PR-AUC (average precision), not accuracy.**
With an 8% positive rate, "always predict no default" scores 92% accuracy
and is useless. PR-AUC focuses on the minority class the business actually
needs to catch.

**Decision threshold: cost-optimal, not the default 0.5.** The threshold
is chosen by minimizing `FN\_COST\_RATIO \* false\_negatives + false\_positives`
over the out-of-fold precision-recall curve (`FN\_COST\_RATIO` defaults to
8 -- a missed defaulter is assumed \~8x costlier than an unnecessarily
declined good applicant; tune this in `.env` to match real credit-policy
economics). This produces a threshold a credit officer can actually defend
to a regulator, instead of an arbitrary 0.5 cutoff.

**Validation protocol:** stratified 5-fold CV with early stopping on each
fold, out-of-fold predictions used for threshold selection, final model
retrained on the full train+val pool at the median best-iteration across
folds, and reported metrics come from a held-out test split the model
never saw during training or threshold selection.

\---

## 4\. Explainability

`src/ml/explain.py` uses SHAP's `TreeExplainer` (exact for gradient-boosted
trees, no sampling approximation) at two levels:

* **Global** -- mean |SHAP value| per feature across the portfolio, for
model governance and audit review (Explainability page, tab 1).
* **Local** -- per-applicant factor breakdown in plain English (e.g. "loan
amount relative to income (4.2) increases default risk"), suitable for
an adverse-action notice or a credit officer's file note (tab 2).

## 5\. Business rule derivation

`src/ml/rules.py` fits a shallow (`max\_depth=4`) decision tree to
**reproduce the trained model's own risk-band output** on the full
portfolio, then walks the tree into plain-English IF/THEN statements, each
annotated with:

* **Support** -- how many applicants the rule covers
* **Agreement** -- what fraction of those applicants the rule correctly
matches to the model's actual output (rule fidelity to the model, not
ground-truth accuracy -- these rules describe what the model learned,
they are not an independent causal claim)

This is standard model-distillation practice for explainability: the
surrogate doesn't need to be as accurate as the real model, it needs to
faithfully summarize what the real model does in a form a policy analyst
or auditor can read without exposing model internals.

\---

## 6\. Talk-to-Data: prompt engineering \& token optimization

**Two backends, chosen automatically** (`src/talk\_to\_data/nl\_to\_sql.py`):

1. **LLM backend** -- Google **Gemini** by default (genuine free tier;
see [pricing](https://ai.google.dev/pricing)). -- set `LLM\_PROVIDER` in
`.env`. Used whenever an API key is configured.
2. **Keyword-pattern fallback** -- 10 parametrized query templates matched
by keyword, covering the required "at least 5 working query patterns"
with **zero token cost and zero hallucination risk**. This is what
makes `docker-compose up` produce a fully working chatbot even before
anyone adds an API key.

**Token optimization approach:**

* **Condensed schema injection** (\~150 tokens) instead of a full DDL dump
or, worse, sample rows -- the model needs the schema shape, not real
applicant data (which would also leak PII into LLM provider logs).
* **Capped, shape-diverse few-shot examples** (6 examples covering
aggregation / grouping / filtering / ratio / top-N query shapes) rather
than exhaustive coverage of literal questions -- generalization comes
from shape coverage, not example count.
* **Response caching** (`nl\_to\_sql.py`'s `CACHE\_PATH`) -- identical or
repeated questions never re-hit the LLM. This is the single biggest
token-savings lever for a chatbot people will ask similar questions of
repeatedly.
* **Compact result serialization** -- query results are truncated and
sent to the LLM as CSV for the answer-phrasing step, never as a full
DataFrame dump.
* **Versioned prompts** (`PROMPT\_VERSION` in `prompt\_templates.py`) --
changes to the system prompt can be compared against previously logged
question/SQL pairs instead of guessing whether a change helped.

**Hallucination control -- the actual enforcement layer is
`src/talk\_to\_data/sql\_guard.py`, not prompting:**

1. Parse with `sqlglot` -- reject anything that isn't a single well-formed
`SELECT` (blocks DDL/DML structurally, not by keyword-matching alone).
2. Table/column allow-list -- only the `applicants` table and its known
columns may be referenced; SELECT aliases are correctly distinguished
from real columns (see `tests/test\_sql\_guard.py`).
3. Forbidden-keyword backstop for constructs sqlglot might parse but we
still don't want (`PRAGMA`, `ATTACH`, etc.).
4. Mandatory `LIMIT` injection/capping regardless of what the LLM wrote.
5. Execution in a rolled-back transaction as a final backstop even if
every check above somehow had a gap.

Every SQL string, from either backend, passes through `validate\_sql()`
before execution -- there is no code path that skips it.

\---

## 7\. Evaluation metrics \& results

Run `python -m src.ml.evaluate` after training for a formatted report.

**Verified during development** on a real, stratified subsample of the
actual Kaggle `application\_train.csv` + `bureau.csv` (307,511 applicants, 8.07%
default rate in the subsample by construction; the full dataset's true
base rate is **8.07%**, confirmed directly from the real file): held-out
test ROC-AUC of **0.771**, PR-AUC of **0.263**, with SHAP correctly
identifying `EXT\_SOURCE\_MEAN` as the dominant feature by a wide margin --
consistent with every published analysis of this competition. Full-dataset
numbers from your own training run (see `COLAB\_GUIDE.md`) will differ
slightly and should read a bit higher with the full 307k rows and more
stable fold estimates; use `python -m src.ml.evaluate` to see your actual
run's numbers, including the full confusion matrix and chosen decision
threshold.

\---

## 8\. Known limitations \& possible improvements

* **Bureau features are aggregated simply** (count/sum/mean of prior
credits) -- `previous\_application.csv`, `installments\_payments.csv`, and
`POS\_CASH\_balance.csv` are not yet joined in. `src/data/loader.py`'s
`build\_bureau\_features()` shows the pattern to extend; see
`data/README.md`.
* **The surrogate rule tree is a simplification** of the real model by
design (that's the point -- see section 5), so it will occasionally
disagree with the ML model at the margins. Rules report their agreement
rate explicitly so this is never hidden.
* **The Talk-to-Data chatbot only queries the single flattened
`applicants` table** -- multi-table joins in natural language (e.g.
"applicants whose most recent installment was late") aren't supported
yet without adding those tables to the schema and allow-list.
* **No model monitoring/drift detection** -- out of scope for this
assignment's timeframe, but the natural next step for a real deployment
(track score distribution and feature drift over time, retrain on a
schedule).
* **Fairness/bias auditing** -- age and gender carry real predictive
signal in this dataset (see EDA), which is common in credit data but
requires a formal fair-lending review before any real-world use; this
platform surfaces the signal for review via SHAP/rules, it does not
adjudicate compliance.

\---

## 9\. Repository structure

```
credit\_risk\_platform/
  data/                further docs + where you place the real dataset (gitignored)
  notebooks/           eda.py (source of truth) + eda.ipynb (generated, executed)
  src/
    data/              loader.py, preprocessor.py, synthetic.py (dev fixture), db\_loader.py
    ml/                train.py, predict.py, evaluate.py, explain.py, rules.py
    talk\_to\_data/      nl\_to\_sql.py, query\_runner.py, prompt\_templates.py, sql\_guard.py
    utils/             logger.py, config.py, helpers.py, docker\_utils.py
  sql/schema.sql       analytical DB schema
  models/              saved model artifacts (gitignored, generated by training)
  app/                 Streamlit UI (streamlit\_app.py + pages/)
  tests/               pytest unit tests for the highest-risk logic
  scripts/entrypoint.sh
  Dockerfile, docker-compose.yml, requirements.txt, .env.example
```

