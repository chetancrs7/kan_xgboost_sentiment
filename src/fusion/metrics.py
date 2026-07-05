"""Evaluation metrics and calibration diagnostics for the fusion network.

Provides the full metric suite required for the fusion evaluation report:
accuracy, precision, recall, macro/weighted F1, ROC-AUC, confusion matrix,
expected calibration error, plus publication-style calibration-curve and
reliability-diagram figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    labels: np.ndarray, probs: np.ndarray, prefix: str = ""
) -> dict[str, object]:
    """Compute the standard fusion evaluation metrics.

    Args:
        labels: True integer labels ``[n]``.
        probs: Predicted probabilities ``[n, C]``.
        prefix: Optional key prefix (e.g. ``"kan_"``) for side-by-side reports.

    Returns:
        Metric name -> value dict (confusion matrix as a nested list).
    """
    preds = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    if n_classes == 2:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    else:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    return {
        f"{prefix}accuracy": float(accuracy_score(labels, preds)),
        f"{prefix}precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        f"{prefix}recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        f"{prefix}macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        f"{prefix}weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        f"{prefix}roc_auc": float(roc_auc),
        f"{prefix}confusion_matrix": confusion_matrix(labels, preds).tolist(),
        f"{prefix}expected_calibration_error": expected_calibration_error(labels, probs),
    }


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """Confidence-binned ECE: mean |accuracy - confidence| weighted by bin mass."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: Path | str,
    class_names: tuple[str, ...] = ("negative", "positive"),
    title: str = "Fusion Confusion Matrix",
) -> None:
    """Save an annotated confusion-matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(
    labels: np.ndarray,
    prob_positive_by_model: dict[str, np.ndarray],
    output_path: Path | str,
    n_bins: int = 10,
) -> None:
    """Save a positive-class calibration curve for one or more models.

    Args:
        labels: True binary labels.
        prob_positive_by_model: Model name -> positive-class probability array,
            e.g. ``{"KAN": ..., "XGBoost": ..., "Fusion": ...}``.
        output_path: Destination PNG path.
        n_bins: Number of probability bins.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    for model_name, prob_pos in prob_positive_by_model.items():
        frac_pos, mean_pred = calibration_curve(
            labels, prob_pos, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted positive-class probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: Path | str,
    n_bins: int = 10,
    title: str = "Fusion Reliability Diagram",
) -> None:
    """Save a confidence-binned reliability diagram with the ECE annotated.

    Bars show per-bin accuracy against the identity line; the hatched overlay
    marks the gap between confidence and accuracy (the calibration error each
    bin contributes).
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    accuracies, confidences, weights = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        accuracies.append(correct[mask].mean() if mask.any() else 0.0)
        confidences.append(confidence[mask].mean() if mask.any() else 0.0)
        weights.append(mask.mean())
    accuracies, confidences = np.asarray(accuracies), np.asarray(confidences)

    ece = expected_calibration_error(labels, probs, n_bins=n_bins)
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.bar(
        centers, accuracies, width=width * 0.95,
        label="Per-bin accuracy", color="#4878d0", edgecolor="black", linewidth=0.5,
    )
    gap = confidences - accuracies
    ax.bar(
        centers, gap, bottom=accuracies, width=width * 0.95,
        label="Confidence-accuracy gap", color="#d65f5f", alpha=0.55,
        hatch="//", edgecolor="black", linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} (ECE = {ece:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_alpha_distribution(
    alpha: np.ndarray,
    output_path: Path | str,
    title: str = "Gate Distribution: trust in KAN (alpha)",
) -> None:
    """Save a histogram of the learned gate values (fusion interpretability)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(alpha, bins=40, range=(0, 1), color="#4878d0", edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="equal trust")
    ax.axvline(alpha.mean(), color="#d65f5f", linewidth=1.5, label=f"mean = {alpha.mean():.3f}")
    ax.set_xlabel("alpha  (1 = trust KAN, 0 = trust XGBoost)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
