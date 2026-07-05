"""Reliability metrics and calibration figures for the uncertainty layer.

Metrics: Expected Calibration Error (reused from ``src.fusion.metrics``),
Maximum Calibration Error, Brier score, and Negative Log Likelihood --
computed side by side for the fused model and each frozen expert.

Figures (single-PNG, multi-panel, matching the deliverable names):

* ``calibration_plots.png``  calibration curve, reliability diagram,
  confidence histogram, and the risk-coverage curve of the uncertainty score.
* ``uncertainty_distribution.png``  score histogram with the tier thresholds,
  plus the score distribution split by prediction correctness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from src.fusion.metrics import expected_calibration_error

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def maximum_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """Largest |accuracy - confidence| gap over the non-empty confidence bins."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            mce = max(mce, abs(correct[mask].mean() - confidence[mask].mean()))
    return float(mce)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot labels."""
    n, n_classes = probs.shape
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log probability assigned to the true class."""
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probs, _EPS, 1.0)).mean())


def compute_reliability_metrics(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """The full reliability suite (ECE, MCE, Brier, NLL) for one model."""
    return {
        "expected_calibration_error": expected_calibration_error(labels, probs, n_bins),
        "maximum_calibration_error": maximum_calibration_error(labels, probs, n_bins),
        "brier_score": brier_score(labels, probs),
        "negative_log_likelihood": negative_log_likelihood(labels, probs),
        "accuracy": float((probs.argmax(axis=1) == labels).mean()),
    }


def risk_coverage_curve(
    correct: np.ndarray, uncertainty_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: error rate on the most-certain fraction.

    Samples are sorted by ascending uncertainty; ``risk[i]`` is the error rate
    when only the ``coverage[i]`` most-certain fraction is answered.
    """
    order = np.argsort(uncertainty_score, kind="stable")
    errors = 1.0 - correct[order].astype(np.float64)
    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return coverage, risk


# --------------------------------------------------------------------------- #
# Reliability-diagram panel helper (shared axis-level drawing)
# --------------------------------------------------------------------------- #
def _draw_reliability_diagram(
    ax: plt.Axes, labels: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> None:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    accuracies, confidences = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        accuracies.append(correct[mask].mean() if mask.any() else 0.0)
        confidences.append(confidence[mask].mean() if mask.any() else 0.0)
    accuracies = np.asarray(accuracies)
    gap = np.asarray(confidences) - accuracies

    width = 1.0 / n_bins
    ax.bar(
        centers, accuracies, width=width * 0.95,
        label="Per-bin accuracy", color="#4878d0", edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        centers, gap, bottom=accuracies, width=width * 0.95,
        label="Confidence-accuracy gap", color="#d65f5f", alpha=0.55,
        hatch="//", edgecolor="black", linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ece = expected_calibration_error(labels, probs, n_bins=n_bins)
    mce = maximum_calibration_error(labels, probs, n_bins=n_bins)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE={ece:.4f}, MCE={mce:.4f})")
    ax.legend(loc="upper left", fontsize=8)


# --------------------------------------------------------------------------- #
# Deliverable figures
# --------------------------------------------------------------------------- #
def plot_calibration_plots(
    labels: np.ndarray,
    prob_positive_by_model: dict[str, np.ndarray],
    fused_probs: np.ndarray,
    uncertainty_score: np.ndarray,
    output_path: Path | str,
    n_bins: int = 10,
) -> None:
    """Save the 2x2 calibration panel (``calibration_plots.png``).

    Args:
        labels: True binary labels ``[n]``.
        prob_positive_by_model: Model name -> positive-class probabilities,
            e.g. ``{"KAN": ..., "XGBoost": ..., "Fusion": ...}``.
        fused_probs: Fused probabilities ``[n, C]`` (reliability + histogram).
        uncertainty_score: Combined uncertainty score ``[n]`` (risk-coverage).
        output_path: Destination PNG.
        n_bins: Probability/confidence bins for the calibration panels.
    """
    correct = fused_probs.argmax(axis=1) == labels
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (0, 0) calibration curves
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    for model_name, prob_pos in prob_positive_by_model.items():
        frac_pos, mean_pred = calibration_curve(
            labels, prob_pos, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted positive-class probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # (0, 1) reliability diagram of the fused model
    _draw_reliability_diagram(axes[0, 1], labels, fused_probs, n_bins=n_bins)

    # (1, 0) fused-confidence histogram, split by correctness
    ax = axes[1, 0]
    confidence = fused_probs.max(axis=1)
    ax.hist(
        [confidence[correct], confidence[~correct]],
        bins=20, stacked=True, color=["#4878d0", "#d65f5f"],
        label=["Correct", "Incorrect"], edgecolor="black", linewidth=0.3,
    )
    ax.set_xlabel("Fused confidence (max probability)")
    ax.set_ylabel("Samples")
    ax.set_title("Confidence Distribution")
    ax.legend(fontsize=8)

    # (1, 1) risk-coverage curve of the combined uncertainty score
    ax = axes[1, 1]
    coverage, risk = risk_coverage_curve(correct, uncertainty_score)
    ax.plot(coverage, risk, color="#4878d0", linewidth=1.5, label="Uncertainty-ordered")
    ax.axhline(1.0 - correct.mean(), color="#d65f5f", linestyle="--", linewidth=1,
               label=f"Full-coverage risk = {1.0 - correct.mean():.3f}")
    ax.set_xlabel("Coverage (fraction of samples answered)")
    ax.set_ylabel("Risk (error rate on answered samples)")
    ax.set_title("Risk-Coverage Curve")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Fusion Calibration Diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_uncertainty_distribution(
    uncertainty_score: np.ndarray,
    correct: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    tiers: np.ndarray,
    output_path: Path | str,
) -> None:
    """Save the uncertainty-score panel (``uncertainty_distribution.png``)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: score histogram with tier thresholds and counts
    ax = axes[0]
    ax.hist(uncertainty_score, bins=40, range=(0, 1), color="#4878d0",
            edgecolor="black", linewidth=0.4)
    ax.axvline(low_threshold, color="#e8a33d", linestyle="--", linewidth=1.5,
               label=f"low/medium = {low_threshold:.2f}")
    ax.axvline(high_threshold, color="#d65f5f", linestyle="--", linewidth=1.5,
               label=f"medium/high = {high_threshold:.2f}")
    counts = {tier: int((tiers == tier).sum()) for tier in ("low", "medium", "high")}
    ax.set_xlabel("Combined uncertainty score")
    ax.set_ylabel("Samples")
    ax.set_title(
        f"Uncertainty Distribution (low={counts['low']}, "
        f"medium={counts['medium']}, high={counts['high']})"
    )
    ax.legend(fontsize=8)

    # Right: score distribution by correctness of the fused prediction
    ax = axes[1]
    ax.hist(
        [uncertainty_score[correct.astype(bool)], uncertainty_score[~correct.astype(bool)]],
        bins=30, range=(0, 1), density=True, color=["#4878d0", "#d65f5f"],
        label=["Correct", "Incorrect"], alpha=0.75,
        edgecolor="black", linewidth=0.3,
    )
    ax.set_xlabel("Combined uncertainty score")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty by Prediction Correctness")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
