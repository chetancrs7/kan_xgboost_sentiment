"""Visualization helpers for the TF-IDF + KNN baseline.

Saves evaluation figures for a trained pipeline: confusion-matrix heatmap,
ROC curve, precision-recall curve, and the k-tuning curve recorded on the
validation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline

MODEL_LABEL = "TF-IDF + KNN"
CLASS_NAMES: tuple[str, ...] = ("Negative", "Positive")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | str,
    class_names: tuple[str, ...] = CLASS_NAMES,
    title: str = f"{MODEL_LABEL} Confusion Matrix",
) -> None:
    """Save an annotated confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
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


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path | str,
    title: str = f"{MODEL_LABEL} ROC Curve",
) -> None:
    """Save the ROC curve with the AUC annotated (positive-class scores)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color="#4878d0", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path | str,
    title: str = f"{MODEL_LABEL} Precision-Recall Curve",
) -> None:
    """Save the precision-recall curve with average precision annotated."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(recall, precision, color="#4878d0", linewidth=2, label=f"PR (AP = {ap:.4f})")
    ax.axhline(
        float(np.mean(y_true)), color="k", linestyle="--", linewidth=1,
        label="Positive-class prevalence",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tuning_curve(
    tuning_history: list[dict[str, Any]],
    output_path: Path | str,
    scoring: str = "f1_macro",
) -> None:
    """Save the validation score of every (k, weights) candidate tried.

    Args:
        tuning_history: Records from ``tune_knn`` with keys
            ``k``, ``weights``, and ``score``.
        output_path: Destination PNG path.
        scoring: Metric name for the y-axis label.
    """
    df = pd.DataFrame(tuning_history)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for weight, group in df.groupby("weights"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["score"], marker="o", label=f"weights={weight}")
    best = df.loc[df["score"].idxmax()]
    ax.scatter(
        [best["k"]], [best["score"]],
        s=140, facecolors="none", edgecolors="#d65f5f", linewidths=2,
        zorder=5, label=f"best (k={int(best['k'])}, {best['score']:.4f})",
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel(f"Validation {scoring}")
    ax.set_title(f"{MODEL_LABEL} Hyperparameter Tuning")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_plots(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    output_dir: Path | str,
    tuning_history: list[dict[str, Any]] | None = None,
) -> None:
    """Save all evaluation figures for the trained pipeline to ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One predict_proba call covers every figure (KNN inference is costly).
    proba = model.predict_proba(X_test)
    y_true = np.asarray(y_test)
    y_pred = model.classes_[np.argmax(proba, axis=1)]
    y_score = proba[:, -1]  # classes_ is sorted, so the last column is positive

    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_score, out_dir / "roc_curve.png")
    plot_precision_recall_curve(y_true, y_score, out_dir / "precision_recall_curve.png")
    if tuning_history:
        plot_tuning_curve(tuning_history, out_dir / "tuning_curve.png")

    print(f"[INFO] Saved evaluation plots to '{out_dir}'.")
