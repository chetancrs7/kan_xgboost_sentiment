"""Threshold-driven decision logic of the uncertainty-aware inference layer.

Given each sample's uncertainty tier, produce the final decision record:

* **low** - return the fused prediction normally.
* **medium** - return the fused prediction with a confidence warning.
* **high** - trigger the configured fallback mode:

  1. ``FUSED_LOW_CONFIDENCE`` - keep the fused prediction, explicitly marked
     low confidence.
  2. ``MOST_CONFIDENT_EXPERT`` - defer to whichever expert has the higher
     softmax confidence on this sample.
  3. ``NEUTRAL`` - abstain: predict Neutral (label ``-1``) with an
     uncertainty flag.
  4. ``HUMAN_REVIEW`` - keep the fused prediction but flag the sample for
     human review.
"""

from __future__ import annotations

import numpy as np

from src.uncertainty.config import FallbackMode
from src.uncertainty.scorer import TIER_HIGH, TIER_LOW, TIER_MEDIUM

#: Sentinel class index for the abstain / Neutral outcome of fallback mode 3.
NEUTRAL_LABEL = -1

#: Human-readable decision labels written to ``confidence_scores.csv``.
DECISION_ACCEPT = "accept"
DECISION_WARN = "accept_with_warning"
DECISION_FALLBACK = {
    FallbackMode.FUSED_LOW_CONFIDENCE: "fallback_fused_low_confidence",
    FallbackMode.MOST_CONFIDENT_EXPERT: "fallback_most_confident_expert",
    FallbackMode.NEUTRAL: "fallback_neutral",
    FallbackMode.HUMAN_REVIEW: "fallback_human_review",
}


def apply_fallback(
    tiers: np.ndarray,
    fused_probs: np.ndarray,
    kan_probs: np.ndarray,
    xgb_probs: np.ndarray,
    mode: FallbackMode,
) -> dict[str, np.ndarray]:
    """Resolve every sample's final prediction from its uncertainty tier.

    Args:
        tiers: Per-sample tier labels (``low`` / ``medium`` / ``high``).
        fused_probs: Fused probabilities ``[n, C]``.
        kan_probs / xgb_probs: Expert probabilities ``[n, C]``.
        mode: Fallback strategy for the high-uncertainty tier.

    Returns:
        Column name -> ``[n]`` array dict, ready for the CSV export:
        ``final_pred`` (``-1`` = Neutral/abstain), ``decision``,
        ``prediction_source``, and the ``confidence_warning`` /
        ``uncertainty_flag`` / ``needs_human_review`` indicator columns.
    """
    mode = FallbackMode(mode)
    n = len(tiers)
    fused_pred = fused_probs.argmax(axis=1)
    kan_pred = kan_probs.argmax(axis=1)
    xgb_pred = xgb_probs.argmax(axis=1)

    final_pred = fused_pred.copy()
    decision = np.full(n, DECISION_ACCEPT, dtype=object)
    source = np.full(n, "fusion", dtype=object)
    confidence_warning = np.zeros(n, dtype=bool)
    uncertainty_flag = np.zeros(n, dtype=bool)
    needs_human_review = np.zeros(n, dtype=bool)

    medium = tiers == TIER_MEDIUM
    decision[medium] = DECISION_WARN
    confidence_warning[medium] = True

    high = tiers == TIER_HIGH
    decision[high] = DECISION_FALLBACK[mode]
    uncertainty_flag[high] = True
    if mode is FallbackMode.FUSED_LOW_CONFIDENCE:
        confidence_warning[high] = True
    elif mode is FallbackMode.MOST_CONFIDENT_EXPERT:
        kan_wins = kan_probs.max(axis=1) >= xgb_probs.max(axis=1)
        final_pred[high] = np.where(kan_wins, kan_pred, xgb_pred)[high]
        source[high] = np.where(kan_wins, "kan", "xgb")[high]
    elif mode is FallbackMode.NEUTRAL:
        final_pred[high] = NEUTRAL_LABEL
        source[high] = "neutral"
    elif mode is FallbackMode.HUMAN_REVIEW:
        needs_human_review[high] = True
        source[high] = "human_review"

    # Sanity: every sample must be assigned to exactly one tier.
    unassigned = ~(medium | high | (tiers == TIER_LOW))
    if unassigned.any():
        raise ValueError(f"Unknown tier labels: {np.unique(tiers[unassigned])}")

    return {
        "final_pred": final_pred,
        "decision": decision,
        "prediction_source": source,
        "confidence_warning": confidence_warning,
        "uncertainty_flag": uncertainty_flag,
        "needs_human_review": needs_human_review,
    }


def fallback_summary(
    decisions: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, object]:
    """Aggregate accuracy/coverage statistics of the resolved decisions.

    ``Neutral`` abstentions and human-review flags are excluded from the
    automated-accuracy numerator *and* denominator: those samples were
    deliberately not answered automatically.
    """
    final = decisions["final_pred"]
    answered = (final != NEUTRAL_LABEL) & ~decisions["needs_human_review"]
    n = len(final)

    counts: dict[str, int] = {}
    for label in np.unique(decisions["decision"]):
        counts[str(label)] = int((decisions["decision"] == label).sum())

    return {
        "n_samples": n,
        "decision_counts": counts,
        "n_answered_automatically": int(answered.sum()),
        "automation_coverage": float(answered.mean()),
        "accuracy_answered": (
            float((final[answered] == labels[answered]).mean()) if answered.any() else None
        ),
        "n_neutral_abstained": int((final == NEUTRAL_LABEL).sum()),
        "n_flagged_for_human_review": int(decisions["needs_human_review"].sum()),
    }
