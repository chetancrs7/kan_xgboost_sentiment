"""Visualization helpers for the TF-IDF + Random Forest baseline.

Saves evaluation figures for a trained pipeline: confusion-matrix heatmap,
ROC curve, precision-recall curve, the validation score of every tuning
candidate, and the most important TF-IDF features.
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

MODEL_LABEL = "TF-IDF + Random Forest"
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


def plot_tuning_results(
    tuning_history: list[dict[str, Any]],
    output_path: Path | str,
    scoring: str = "f1_macro",
) -> None:
    """Save the validation score of every forest configuration tried.

    Args:
        tuning_history: Records from ``tune_model`` with keys
            ``n_estimators``, ``max_depth``, ``min_samples_leaf``, ``score``.
        output_path: Destination PNG path.
        scoring: Metric name for the x-axis label.
    """
    df = pd.DataFrame(tuning_history).sort_values("score")
    labels = [
        f"trees={int(row.n_estimators)}, depth={row.max_depth or 'None'}, "
        f"leaf={int(row.min_samples_leaf)}"
        for row in df.itertuples()
    ]
    colors = ["#4878d0"] * len(df)
    colors[-1] = "#d65f5f"  # best candidate is last after the sort

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(df) + 1.8))
    ax.barh(range(len(df)), df["score"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(df)), labels)
    for i, score in enumerate(df["score"]):
        ax.text(score, i, f" {score:.4f}", va="center", fontsize=8)
    ax.set_xlabel(f"Validation {scoring}")
    ax.set_title(f"{MODEL_LABEL} Hyperparameter Tuning")
    ax.set_xlim(0, min(1.0, float(df['score'].max()) * 1.15))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importances(
    model: Pipeline,
    output_path: Path | str,
    top_n: int = 25,
) -> None:
    """Save the TF-IDF features with the highest forest impurity importance."""
    feature_names = np.asarray(model.named_steps["tfidf"].get_feature_names_out())
    importances = model.named_steps["rf"].feature_importances_
    top_n = min(top_n, len(feature_names))
    top = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(7.5, 0.28 * top_n + 1.5))
    ax.barh(
        range(top_n), importances[top],
        color="#4878d0", edgecolor="black", linewidth=0.4,
    )
    ax.set_yticks(range(top_n), feature_names[top])
    ax.set_xlabel("Impurity-based feature importance")
    ax.set_title(f"{MODEL_LABEL}: Top {top_n} Features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_plots(
    model: Pipeline,
    test_texts: pd.Series,
    y_test: pd.Series,
    output_dir: Path | str,
    tuning_history: list[dict[str, Any]] | None = None,
) -> None:
    """Save all evaluation figures for the trained pipeline to ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One predict_proba call covers every figure.
    proba = model.predict_proba(test_texts)
    y_true = np.asarray(y_test)
    y_pred = model.classes_[np.argmax(proba, axis=1)]
    y_score = proba[:, -1]  # classes_ is sorted, so the last column is positive

    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_score, out_dir / "roc_curve.png")
    plot_precision_recall_curve(y_true, y_score, out_dir / "precision_recall_curve.png")
    plot_feature_importances(model, out_dir / "feature_importances.png")
    if tuning_history:
        plot_tuning_results(tuning_history, out_dir / "tuning_results.png")

    print(f"[INFO] Saved evaluation plots to '{out_dir}'.")
