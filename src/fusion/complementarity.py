"""Complementarity metrics for the KAN / XGBoost hybrid.

These metrics answer the question the residual-aware training is designed around:
*does the semantic KAN expert recover the cases the structural XGBoost expert
misses, without breaking the ones XGBoost already gets right?* They are reported
alongside the usual accuracy/F1 so specialisation can be judged directly.

Definitions (all on a single split, with hard predictions):

``kan_rescue_rate``
    Among samples XGBoost gets **wrong**, the fraction the KAN gets **right**.
    High = the KAN is rescuing structural failures.

``xgb_preservation_rate``
    Among samples XGBoost gets **right**, the fraction the KAN *also* gets
    right. High = the KAN did not damage XGBoost's easy wins (the flip side of
    over-specialisation).

``disagreement_accuracy``
    Accuracy of the chosen ``resolver`` predictions on the subset where the two
    experts **disagree** — the region where the fusion gate actually has to
    make a decision.

``double_fault_rate``
    Fraction of samples where **both** experts are wrong. This is the
    irreducible error floor no gate can fix.

``oracle_ensemble_accuracy``
    Accuracy of a perfect selector that always trusts whichever expert is
    right when at least one is (``1 - double_fault_rate``). The ceiling any
    fusion of these two experts can reach.
"""

from __future__ import annotations

import numpy as np


def _preds(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs).argmax(axis=1)


def complementarity_metrics(
    kan_probs: np.ndarray,
    xgb_probs: np.ndarray,
    labels: np.ndarray,
    fused_probs: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full complementarity metric suite for one split.

    Args:
        kan_probs: KAN probabilities ``[n, C]``.
        xgb_probs: XGBoost probabilities ``[n, C]``.
        labels: Integer labels ``[n]``.
        fused_probs: Optional fused/ensemble probabilities ``[n, C]``. When
            given, ``disagreement_accuracy`` and the reported preservation /
            rescue of the *system* are measured on the fused prediction;
            otherwise they are measured on the KAN alone.

    Returns:
        A JSON-serialisable dict of complementarity metrics.
    """
    labels = np.asarray(labels, dtype=np.int64)
    kan_pred = _preds(kan_probs)
    xgb_pred = _preds(xgb_probs)

    kan_correct = kan_pred == labels
    xgb_correct = xgb_pred == labels

    resolver_pred = _preds(fused_probs) if fused_probs is not None else kan_pred
    resolver_correct = resolver_pred == labels

    xgb_wrong = ~xgb_correct
    xgb_right = xgb_correct
    disagree = kan_pred != xgb_pred
    both_wrong = xgb_wrong & (~kan_correct)
    either_right = kan_correct | xgb_correct

    def _rate(numer_mask: np.ndarray, denom_mask: np.ndarray) -> float:
        denom = int(denom_mask.sum())
        if denom == 0:
            return float("nan")
        return float((numer_mask & denom_mask).sum() / denom)

    metrics = {
        # KAN recovers XGBoost's mistakes.
        "kan_rescue_rate": _rate(kan_correct, xgb_wrong),
        # System (fused, or KAN) recovers XGBoost's mistakes.
        "system_rescue_rate": _rate(resolver_correct, xgb_wrong),
        # KAN keeps XGBoost's easy wins.
        "xgb_preservation_rate": _rate(kan_correct, xgb_right),
        # System keeps XGBoost's easy wins.
        "system_preservation_rate": _rate(resolver_correct, xgb_right),
        # Where the experts disagree, how often is the resolver right.
        "disagreement_accuracy": _rate(resolver_correct, disagree),
        "disagreement_fraction": float(disagree.mean()),
        # Error floor and ceiling.
        "double_fault_rate": float(both_wrong.mean()),
        "oracle_ensemble_accuracy": float(either_right.mean()),
        # Context for the rates above.
        "kan_accuracy": float(kan_correct.mean()),
        "xgb_accuracy": float(xgb_correct.mean()),
        "n_samples": int(len(labels)),
    }
    if fused_probs is not None:
        metrics["fused_accuracy"] = float(resolver_correct.mean())
    return metrics


def format_complementarity(metrics: dict[str, float]) -> str:
    """Render a complementarity metrics dict as an aligned text block."""
    order = [
        "n_samples",
        "kan_accuracy",
        "xgb_accuracy",
        "fused_accuracy",
        "kan_rescue_rate",
        "system_rescue_rate",
        "xgb_preservation_rate",
        "system_preservation_rate",
        "disagreement_fraction",
        "disagreement_accuracy",
        "double_fault_rate",
        "oracle_ensemble_accuracy",
    ]
    lines = ["Complementarity metrics", "-" * 40]
    for key in order:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, int) or key == "n_samples":
                lines.append(f"  {key:<26} {int(value):,}")
            else:
                lines.append(f"  {key:<26} {value:.4f}")
    return "\n".join(lines)
