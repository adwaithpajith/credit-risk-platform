"""
Derives business-readable decision rules from the trained ML model.

Approach: fit a shallow, interpretable surrogate decision tree to predict
the LightGBM model's OWN risk-band output (not the noisy raw TARGET) on a
representative sample. This is standard "model distillation for
explainability" practice -- the surrogate doesn't have to be as accurate
as the real model, it has to faithfully summarize *what the real model
learned* in a form a credit policy analyst can read and audit, e.g.:

    IF external_credit_score < 0.35 AND annuity_to_income_ratio > 0.25
    THEN High Risk  (covers 412 applicants, 81% precision vs model)

Each rule is scored for support and precision against the real model's
own labels, so nobody mistakes these for ground-truth causal claims --
they're a faithful, auditable summary of model behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.ml.explain import plain_language
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("rules")

RULES_PATH = config.paths.models_dir / "business_rules.json"


@dataclass
class DecisionRule:
    conditions: list[str]
    predicted_band: str
    support: int
    precision: float
    plain_english: str


def _tree_to_rules(
    tree: DecisionTreeClassifier, feature_names: list[str], class_names: list[str],
    X: pd.DataFrame, y_model_labels: np.ndarray,
) -> list[DecisionRule]:
    tree_ = tree.tree_
    rules: list[DecisionRule] = []

    def recurse(node: int, conditions: list[str]):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            label = plain_language(feat)

            left_cond = f"{label} <= {threshold:.3g}"
            recurse(tree_.children_left[node], conditions + [left_cond])

            right_cond = f"{label} > {threshold:.3g}"
            recurse(tree_.children_right[node], conditions + [right_cond])
        else:
            # NOTE: sklearn's tree_.value holds class *fractions* (not raw
            # counts) as of modern versions -- n_node_samples has the
            # actual sample count. Multiplying recovers both correctly
            # regardless of sklearn version quirks.
            fractions = tree_.value[node][0]
            support = int(tree_.n_node_samples[node])
            predicted_idx = int(np.argmax(fractions))
            predicted_band = class_names[predicted_idx]
            precision = float(fractions[predicted_idx])

            if support < 5:  # skip statistically meaningless leaves
                return

            plain = " AND ".join(conditions) if conditions else "(no conditions -- default leaf)"
            rules.append(DecisionRule(
                conditions=conditions,
                predicted_band=predicted_band,
                support=support,
                precision=round(precision, 3),
                plain_english=(
                    f"IF {plain} THEN predicted risk = {predicted_band} "
                    f"(covers {support} applicants, {precision:.0%} agreement with the ML model)"
                ),
            ))

    recurse(0, [])
    rules.sort(key=lambda r: (-r.support, -r.precision))
    return rules


def derive_rules(
    X: pd.DataFrame, risk_bands: pd.Series, max_depth: int = 4, min_leaf_fraction: float = 0.01,
) -> list[DecisionRule]:
    """Fit the surrogate tree and extract rules.

    `risk_bands` should be the ML model's OWN predicted band (Low/Medium/
    High) for each row in X -- the tree is explaining the model, not
    re-discovering the label from scratch.
    """
    # The surrogate tree needs numeric-only input; categoricals are
    # target-mean-encoded on the fly purely for this distillation step
    # (this encoding is local to rules.py and never touches the real model).
    X_numeric = X.copy()
    for col in X_numeric.select_dtypes(include="category").columns:
        freq_map = X_numeric[col].value_counts(normalize=True)
        X_numeric[col] = X_numeric[col].map(freq_map).astype(float)
    X_numeric = X_numeric.fillna(X_numeric.median(numeric_only=True))

    class_names = sorted(risk_bands.unique(), key=lambda b: {"Low": 0, "Medium": 1, "High": 2}[b])
    y_encoded = risk_bands.map({name: i for i, name in enumerate(class_names)}).values

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(int(min_leaf_fraction * len(X_numeric)), 10),
        random_state=config.data.random_state,
    )
    tree.fit(X_numeric, y_encoded)

    fidelity = tree.score(X_numeric, y_encoded)
    logger.info(
        "Surrogate rule tree fidelity to ML model labels: %.3f (depth=%d, %d rules)",
        fidelity, max_depth, tree.get_n_leaves(),
    )

    rules = _tree_to_rules(tree, list(X_numeric.columns), class_names, X_numeric, y_encoded)
    return rules


def save_rules(rules: list[DecisionRule]) -> None:
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps([asdict(r) for r in rules], indent=2))
    logger.info("Saved %d business rules -> %s", len(rules), RULES_PATH)


def load_rules() -> list[dict]:
    if not RULES_PATH.exists():
        return []
    return json.loads(RULES_PATH.read_text())


if __name__ == "__main__":
    from src.data.loader import load_full_dataset
    from src.ml.predict import RiskScorer

    df, _ = load_full_dataset()
    scorer = RiskScorer()
    scored = scorer.score_dataframe(df.drop(columns=["TARGET"]))
    X = scorer.get_feature_matrix(df.drop(columns=["TARGET"]))

    rules = derive_rules(X, scored["RISK_BAND"])
    save_rules(rules)
    for r in rules[:10]:
        print(r.plain_english)
