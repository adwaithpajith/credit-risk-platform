"""Standalone evaluation report -- reads the metrics.json produced by
train.py and renders a human-readable summary. Kept separate from train.py
so re-inspecting results doesn't require re-training."""
from __future__ import annotations

import json

from src.ml.train import METRICS_PATH
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No metrics found at {METRICS_PATH}. Run training first.")
    return json.loads(METRICS_PATH.read_text())


def print_report() -> None:
    m = load_metrics()
    cm = m["test_confusion_matrix"]
    thr = m["decision_threshold"]

    print("=" * 64)
    print("CREDIT RISK MODEL -- EVALUATION REPORT")
    print("=" * 64)
    print(f"Trained at (UTC):        {m['trained_at_utc']}")
    print(f"Data source:             {m['data_source']}")
    print(f"Train/val rows:          {m['n_train_val']}")
    print(f"Held-out test rows:      {m['n_test']}")
    print(f"Base default rate:       {m['base_default_rate']:.2%}")
    print("-" * 64)
    print(f"Out-of-fold PR-AUC:      {m['oof_pr_auc']:.4f}")
    print(f"Out-of-fold ROC-AUC:     {m['oof_roc_auc']:.4f}")
    print(f"Held-out test PR-AUC:    {m['test_pr_auc']:.4f}")
    print(f"Held-out test ROC-AUC:   {m['test_roc_auc']:.4f}")
    print("-" * 64)
    print(f"Decision threshold:      {thr['threshold']:.3f}  "
          f"(cost-optimal, FN weighted heavier than FP)")
    print(f"  -> precision @ thresh: {thr['precision']:.3f}")
    print(f"  -> recall @ thresh:    {thr['recall']:.3f}")
    print("-" * 64)
    print("Confusion matrix on held-out test set:")
    print(f"  True Positive (correctly flagged defaulters): {cm['tp']}")
    print(f"  False Negative (missed defaulters):            {cm['fn']}")
    print(f"  False Positive (good applicants declined):     {cm['fp']}")
    print(f"  True Negative (correctly approved):            {cm['tn']}")
    print("=" * 64)


if __name__ == "__main__":
    print_report()
