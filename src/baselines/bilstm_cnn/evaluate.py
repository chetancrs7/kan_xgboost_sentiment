"""Evaluation helpers for the BiLSTM-CNN baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.baselines.bilstm_cnn.model import BATCH_SIZE, ArrayLike


def evaluate(model: Any, x_test: ArrayLike, y_test: ArrayLike) -> dict[str, Any]:
    """Print and return accuracy, precision, recall, F1, and confusion matrix."""
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

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
        "model": "BiLSTM-CNN",
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("Model    : BiLSTM-CNN")
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
