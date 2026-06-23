"""Evaluation and feature-importance reporting for the XGBoost expert model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float | None:
    """Compute ROC AUC, returning ``None`` when it is undefined.

    AUC is undefined when only one class is present in ``y_true``.
    """
    if len(np.unique(y_true)) < 2:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        return None


def evaluate_split(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    n_classes: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the model on one split and return metrics plus probabilities.

    Args:
        model: A fitted XGBoost classifier.
        X: Feature matrix for the split.
        y: True labels for the split.
        split_name: Name used in the printed report (e.g. ``"test"``).
        n_classes: Number of target classes.

    Returns:
        ``(metrics, probabilities)`` where ``metrics`` is a JSON-serialisable
        dict and ``probabilities`` is an ``[n_samples, n_classes]`` array.
    """
    proba = model.predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    average = "binary" if n_classes == 2 else "macro"

    accuracy = float(accuracy_score(y, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average=average, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    auc = _safe_auc(np.asarray(y), proba, n_classes)
    cm = confusion_matrix(y, y_pred)

    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": int(len(y)),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }

    auc_str = f"{auc:.4f}" if auc is not None else "n/a"
    print(f"\n[{split_name.upper()}] "
          f"acc={accuracy:.4f} | precision={precision:.4f} | "
          f"recall={recall:.4f} | f1={f1:.4f} | auc={auc_str}")
    return metrics, proba


def print_classification_report(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> None:
    """Print a detailed per-class classification report for the test split."""
    y_pred = model.predict(X)
    target_names = (
        ["Negative (0)", "Positive (1)"]
        if n_classes == 2
        else [f"Class {i}" for i in range(n_classes)]
    )
    print("\nDetailed classification report (test split):")
    print(
        classification_report(
            y, y_pred, target_names=target_names, digits=4, zero_division=0
        )
    )


def summarize_feature_importance(
    model: XGBClassifier, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Return feature importances sorted descending as a list of records.

    Args:
        model: Fitted XGBoost classifier.
        feature_names: Column names matching the training matrix order.

    Returns:
        List of ``{"feature", "importance", "rank"}`` dicts, most important
        first.
    """
    importances = model.feature_importances_
    paired = sorted(
        zip(feature_names, importances), key=lambda kv: kv[1], reverse=True
    )
    return [
        {"rank": i + 1, "feature": name, "importance": float(value)}
        for i, (name, value) in enumerate(paired)
    ]
