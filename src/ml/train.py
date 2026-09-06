"""
Training pipeline for the default-risk model.

Key design decisions (also documented in the README under "Model selection
and imbalance strategy"):

1. Model: LightGBM gradient-boosted trees.
   - Handles missing values and mixed numeric/categorical features
     natively (see preprocessor.py) -- no imputation/encoding pipeline to
     maintain or leak through.
   - Fast enough to do proper 5-fold CV + early stopping inside a take-home
     time budget while still being production-grade (this is what most
     real credit-risk shops actually ship, not a toy logistic regression).

2. Class imbalance: NOT solved with naive oversampling/SMOTE.
   Home Credit's ~8% default rate is handled via `scale_pos_weight` inside
   the loss function (cost-proportionate learning) instead of duplicating
   or synthesizing minority rows. SMOTE on this kind of mixed numeric/
   categorical, heavily-missing tabular data tends to generate
   unrealistic synthetic applicants and inflate validation AUC without
   improving real generalization -- scale_pos_weight avoids that failure
   mode entirely while directly addressing the imbalance in the objective.

3. Evaluation metric for model selection: PR-AUC (average precision), not
   accuracy or plain ROC-AUC. With an 8% positive rate, a model that
   predicts "never default" gets 92% accuracy and is useless; PR-AUC
   focuses on the minority class the business actually cares about.

4. Decision threshold: NOT the default 0.5. It is chosen by minimizing
   expected business cost using FN_COST_RATIO (how much worse a missed
   defaulter is than a wrongly-declined good applicant), via the
   precision-recall curve on a held-out validation fold. This is what
   "explainable AI to interpret predictions" is *for* in a bank -- a
   threshold a credit officer can defend to a regulator.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.loader import load_full_dataset
from src.data.preprocessor import build_model_frame
from src.utils.config import config
from src.utils.helpers import timed
from src.utils.logger import get_logger

logger = get_logger("train")

MODEL_PATH = config.paths.models_dir / "lgbm_credit_risk.joblib"
METRICS_PATH = config.paths.models_dir / "metrics.json"
MANIFEST_PATH = config.paths.models_dir / "feature_manifest.joblib"


def _lgbm_params(scale_pos_weight: float) -> dict:
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": config.model.learning_rate,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": config.data.random_state,
    }


def _select_cost_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Sweep the precision-recall curve and pick the threshold that
    minimizes expected cost = FN_COST_RATIO * false_negatives + false_positives,
    normalized by sample count. Returns the threshold plus the confusion
    counts at that operating point for transparency in the README/UI."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)  # align lengths with precision/recall

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    fn_cost_ratio = config.model.fn_cost_ratio

    best = {"threshold": 0.5, "cost": float("inf")}
    for p, r, t in zip(precision, recall, thresholds):
        tp = r * n_pos
        fn = n_pos - tp
        # precision = tp / (tp + fp)  =>  fp = tp * (1 - p) / p  (guard p=0)
        fp = tp * (1 - p) / p if p > 0 else n_neg
        cost = fn_cost_ratio * fn + fp
        if cost < best["cost"]:
            best = {
                "threshold": float(t),
                "cost": float(cost),
                "precision": float(p),
                "recall": float(r),
                "estimated_fn": float(fn),
                "estimated_fp": float(fp),
            }
    return best


def train_model(save: bool = True) -> dict:
    df, source = load_full_dataset()
    X, y, manifest = build_model_frame(df)
    y = y.values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.data.test_size,
        stratify=y, random_state=config.data.random_state,
    )

    cat_features = manifest.categorical_features
    scale_pos_weight = (len(y_trainval) - y_trainval.sum()) / max(y_trainval.sum(), 1)
    logger.info(
        "Train/val pool: %d rows, base rate %.2f%%, scale_pos_weight=%.2f",
        len(y_trainval), 100 * y_trainval.mean(), scale_pos_weight,
    )

    skf = StratifiedKFold(
        n_splits=config.model.n_folds, shuffle=True, random_state=config.data.random_state
    )
    fold_metrics, oof_proba = [], np.zeros(len(y_trainval))
    best_iterations = []

    with timed("cross-validated training"):
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
            X_tr, X_val = X_trainval.iloc[tr_idx], X_trainval.iloc[val_idx]
            y_tr, y_val = y_trainval[tr_idx], y_trainval[val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
            val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_set)

            booster = lgb.train(
                _lgbm_params(scale_pos_weight),
                train_set,
                num_boost_round=config.model.num_boost_round,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(config.model.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = booster.predict(X_val, num_iteration=booster.best_iteration)
            oof_proba[val_idx] = preds
            best_iterations.append(booster.best_iteration)

            fold_pr_auc = average_precision_score(y_val, preds)
            fold_roc_auc = roc_auc_score(y_val, preds)
            fold_metrics.append({"fold": fold, "pr_auc": fold_pr_auc, "roc_auc": fold_roc_auc})
            logger.info(
                "Fold %d/%d -- PR-AUC=%.4f  ROC-AUC=%.4f  best_iter=%d",
                fold, config.model.n_folds, fold_pr_auc, fold_roc_auc, booster.best_iteration,
            )

    oof_pr_auc = average_precision_score(y_trainval, oof_proba)
    oof_roc_auc = roc_auc_score(y_trainval, oof_proba)
    logger.info("Out-of-fold -- PR-AUC=%.4f  ROC-AUC=%.4f", oof_pr_auc, oof_roc_auc)

    threshold_info = _select_cost_optimal_threshold(y_trainval, oof_proba)
    logger.info(
        "Cost-optimal threshold=%.3f (fn_cost_ratio=%.1f) -> precision=%.3f recall=%.3f",
        threshold_info["threshold"], config.model.fn_cost_ratio,
        threshold_info["precision"], threshold_info["recall"],
    )

    # Final model: retrain on the FULL train+val pool at the median
    # best_iteration found during CV, then evaluate once, honestly, on the
    # untouched test split.
    final_rounds = int(np.median(best_iterations))
    full_train_set = lgb.Dataset(X_trainval, label=y_trainval, categorical_feature=cat_features)
    final_model = lgb.train(
        _lgbm_params(scale_pos_weight), full_train_set, num_boost_round=final_rounds,
    )

    test_proba = final_model.predict(X_test)
    test_pr_auc = average_precision_score(y_test, test_proba)
    test_roc_auc = roc_auc_score(y_test, test_proba)
    test_preds = (test_proba >= threshold_info["threshold"]).astype(int)

    tp = int(((test_preds == 1) & (y_test == 1)).sum())
    fp = int(((test_preds == 1) & (y_test == 0)).sum())
    fn = int(((test_preds == 0) & (y_test == 1)).sum())
    tn = int(((test_preds == 0) & (y_test == 0)).sum())

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "n_train_val": int(len(y_trainval)),
        "n_test": int(len(y_test)),
        "base_default_rate": float(y.mean()),
        "cv_fold_metrics": fold_metrics,
        "oof_pr_auc": float(oof_pr_auc),
        "oof_roc_auc": float(oof_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "test_roc_auc": float(test_roc_auc),
        "decision_threshold": threshold_info,
        "test_confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "final_num_trees": final_rounds,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
    }

    if save:
        config.paths.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": final_model, "threshold": threshold_info["threshold"]}, MODEL_PATH
        )
        joblib.dump(manifest, MANIFEST_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        logger.info("Saved model -> %s", MODEL_PATH)
        logger.info("Saved metrics -> %s", METRICS_PATH)

    return metrics


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(
        {k: v for k, v in result.items() if k != "feature_names"}, indent=2
    ))
