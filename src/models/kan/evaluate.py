"""Evaluation, probability export, and entropy-confidence reporting for the KAN model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """Softmax class logits into a ``[n_samples, n_classes]`` probability array."""
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy (nats) of a softmax distribution.

    ``H(p) = -sum_c p_c * log(p_c)``. Ranges from 0 (fully confident, all
    mass on one class) to ``log(n_classes)`` (uniform / maximally uncertain).
    """
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def confidence_from_entropy(probs: np.ndarray) -> np.ndarray:
    """Normalised entropy-based confidence in ``[0, 1]``.

    Entropy is normalised by ``log(n_classes)`` (its maximum value) and
    inverted, so higher entropy (more uniform / uncertain predictions) maps
    to lower confidence and a fully peaked distribution maps to confidence 1.
    """
    n_classes = probs.shape[1]
    max_entropy = np.log(n_classes)
    entropy = shannon_entropy(probs)
    if max_entropy == 0:
        return np.ones_like(entropy)
    return 1.0 - (entropy / max_entropy)


def _safe_auc(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float | None:
    """Compute ROC AUC, returning ``None`` when it is undefined (single-class y_true)."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def evaluate_split(
    probs: np.ndarray, y_true: np.ndarray, split_name: str, n_classes: int
) -> dict[str, Any]:
    """Compute classification metrics for one split's predicted probabilities.

    Args:
        probs: ``[n_samples, n_classes]`` softmax probabilities.
        y_true: ``[n_samples]`` integer ground-truth labels.
        split_name: Name used in the printed summary (e.g. ``"validation"``).
        n_classes: Number of target classes.

    Returns:
        A JSON-serialisable metrics dict.
    """
    y_pred = np.argmax(probs, axis=1)
    average = "binary" if n_classes == 2 else "macro"

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = _safe_auc(np.asarray(y_true), probs, n_classes)
    cm = confusion_matrix(y_true, y_pred)
    confidence = confidence_from_entropy(probs)

    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "auc": auc,
        "mean_confidence": float(np.mean(confidence)),
        "confusion_matrix": cm.tolist(),
    }

    auc_str = f"{auc:.4f}" if auc is not None else "n/a"
    print(
        f"[{split_name.upper()}] acc={accuracy:.4f} | precision={precision:.4f} | "
        f"recall={recall:.4f} | f1={f1:.4f} | auc={auc_str} | "
        f"mean_confidence={metrics['mean_confidence']:.4f}"
    )
    return metrics
