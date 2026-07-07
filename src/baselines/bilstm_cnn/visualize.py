"""Visualization helpers for the BiLSTM-CNN baseline.

Saves evaluation figures for a trained model: confusion-matrix heatmap,
ROC curve, precision-recall curve, and the training/validation loss and
accuracy curves from the final fit.

Kept free of TensorFlow imports so the plotting code can be reused or
tested without loading the framework; the model only needs a Keras-style
``predict`` method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

MODEL_LABEL = "BiLSTM-CNN"
CLASS_NAMES: tuple[str, ...] = ("Negative", "Positive")
BATCH_SIZE: int = 64


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


def plot_training_history(
    history: dict[str, list[float]],
    output_path: Path | str,
    title: str = f"{MODEL_LABEL} Training History",
) -> None:
    """Save side-by-side loss and accuracy curves from a Keras history dict.

    Args:
        history: ``History.history`` mapping, e.g. keys ``loss``,
            ``val_loss``, ``accuracy``, ``val_accuracy``.
        output_path: Destination PNG path.
        title: Overall figure title.
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history.get("loss", []), marker="o", label="train")
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history.get("accuracy", []), marker="o", label="train")
    if "val_accuracy" in history:
        ax_acc.plot(epochs, history["val_accuracy"], marker="o", label="validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_plots(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path | str,
    history: dict[str, list[float]] | None = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Save all evaluation figures for the trained model to ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probabilities = np.asarray(model.predict(x_test, batch_size=batch_size, verbose=0))
    y_true = np.asarray(y_test)
    y_pred = np.argmax(probabilities, axis=1)

    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    if probabilities.shape[1] == 2:
        y_score = probabilities[:, 1]
        plot_roc_curve(y_true, y_score, out_dir / "roc_curve.png")
        plot_precision_recall_curve(
            y_true, y_score, out_dir / "precision_recall_curve.png"
        )
    else:
        print("[WARN] Skipping ROC/PR curves: model is not binary.")
    if history:
        plot_training_history(history, out_dir / "training_history.png")

    print(f"[INFO] Saved evaluation plots to '{out_dir}'.")
