"""Evaluation helpers for the TF-IDF + Random Forest baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


def evaluate(model: Pipeline, test_texts: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Print and return accuracy, precision, recall, F1, and confusion matrix."""
    y_pred = model.predict(test_texts)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "model": "TF-IDF + Random Forest",
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("Model    : TF-IDF + Random Forest")
    print(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    print(f"Precision: {precision:.4f}  (macro avg)")
    print(f"Recall   : {recall:.4f}  (macro avg)")
    print(f"F1-score : {f1:.4f}  (macro avg)")
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("  Labels : [Negative, Positive]")
    print(f"  {cm}")
    print("=" * 60)

    return results
