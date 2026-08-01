"""Uncertainty-aware fallback for the Residual-Aware Adaptive Fusion Network.

The fusion gate always returns a convex combination of the two experts. That
is the right default, but on genuinely hard inputs -- ambiguous sentiment,
sarcasm the experts read oppositely, or text far outside the training
distribution -- emitting a confident fused label is exactly where catastrophic
errors come from. This module wraps the gate output in a decision layer that
*detects* those cases and *acts* on them instead of blindly trusting the mix.

Signals monitored (all per sample)
----------------------------------
* **Fused entropy** - Shannon entropy of the fused distribution; high entropy
  means the ensemble itself is unsure.
* **Fused confidence** - the maximum fused class probability.
* **Expert disagreement** - total-variation distance between the KAN and
  XGBoost distributions; the experts pulling in opposite directions is a
  classic hard-case marker.
* **Embedding-space outlier score** - a diagonal-Gaussian Mahalanobis distance
  of the DeBERTa embedding to the training distribution, i.e. an out-of-
  distribution / novelty proxy (:class:`OutlierDetector`).

Actions (in priority order)
---------------------------
1. ``ABSTAIN_REVIEW`` - extreme entropy or a strong OOD score: return a neutral
   prediction (the training class prior) with reduced confidence and raise a
   ``review`` flag for a human.
2. ``ROUTE_KAN`` / ``ROUTE_XGB`` - the experts disagree strongly: hand the
   decision to whichever expert has the lower *expected residual* on this
   sample (the same leak-free residual estimate the gate is trained on),
   rather than averaging a right answer with a wrong one.
3. ``CONSERVATIVE`` - moderately uncertain: keep the fused label but
   temperature-soften the probabilities so downstream consumers see honest,
   de-rated confidence.
4. ``KEEP`` - confident and in-distribution: pass the fused prediction through
   untouched.

Thresholds are **calibrated from the gate-holdout split**, not hand-picked:
each threshold is a percentile of that split's own signal distribution, so the
fallback fires on the tails regardless of the absolute scale of any signal.
Everything needed to reproduce the decision (outlier statistics, thresholds,
class prior) serialises into the fusion checkpoint via :meth:`state_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.fusion.fusion_dataset import shannon_entropy

_EPS = 1e-12

# Action labels (kept as plain strings so they land readable in the CSV export).
KEEP = "keep"
ROUTE_KAN = "route_kan"
ROUTE_XGB = "route_xgb"
CONSERVATIVE = "conservative"
ABSTAIN_REVIEW = "abstain_review"
ACTIONS: tuple[str, ...] = (KEEP, ROUTE_KAN, ROUTE_XGB, CONSERVATIVE, ABSTAIN_REVIEW)


# --------------------------------------------------------------------------- #
# Embedding-space outlier / novelty detector
# --------------------------------------------------------------------------- #
class OutlierDetector:
    """Diagonal-Gaussian Mahalanobis novelty score over embeddings.

    Fitting stores the per-dimension mean and variance of the *training*
    embeddings. The score of a new embedding is the root-mean-square
    standardised deviation across dimensions::

        score(x) = sqrt( mean_d ( (x_d - mu_d)^2 / var_d ) )

    which is a full Mahalanobis distance under a diagonal covariance, divided
    by ``sqrt(D)`` so the scale is dimension-independent and thresholds
    transfer across embedding sizes. Larger = more out-of-distribution.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "OutlierDetector":
        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {emb.shape}.")
        self.mean_ = emb.mean(axis=0)
        var = emb.var(axis=0)
        var[var < _EPS] = 1.0  # constant dims contribute nothing, don't divide by ~0
        self.var_ = var
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("OutlierDetector.score called before fit().")
        emb = np.asarray(embeddings, dtype=np.float64)
        z2 = (emb - self.mean_) ** 2 / self.var_
        return np.sqrt(z2.mean(axis=1))

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "var": None if self.var_ is None else self.var_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "OutlierDetector":
        det = cls()
        det.mean_ = np.asarray(state["mean"], dtype=np.float64)
        det.var_ = np.asarray(state["var"], dtype=np.float64)
        return det


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FallbackConfig:
    """Percentile calibration settings and the absolute thresholds they yield.

    The ``*_pct`` fields are the holdout percentiles used by
    :meth:`UncertaintyFallback.fit`; the plain-named fields are the resulting
    absolute thresholds (populated during calibration and serialised).
    ``conservative_temperature`` (> 1) softens probabilities in the
    ``CONSERVATIVE`` branch.
    """

    # Calibration percentiles (0-100) of the gate-holdout signal distributions.
    entropy_high_pct: float = 85.0
    entropy_abstain_pct: float = 98.0
    confidence_low_pct: float = 10.0
    disagreement_high_pct: float = 90.0
    ood_high_pct: float = 90.0
    ood_abstain_pct: float = 99.0
    conservative_temperature: float = 2.0
    # When False (default) the fallback is *annotation-only*: it computes the
    # per-sample action + review flag but keeps the fused prediction unchanged,
    # so it never lowers accuracy — it just marks risky samples. When True it
    # additionally overrides predictions (abstain->prior, route->one expert,
    # conservative->softened) for a conservative deployment.
    apply_overrides: bool = False

    # Absolute thresholds, filled in by calibration.
    entropy_high: float = float("inf")
    entropy_abstain: float = float("inf")
    confidence_low: float = 0.0
    disagreement_high: float = float("inf")
    ood_high: float = float("inf")
    ood_abstain: float = float("inf")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "FallbackConfig":
        known = {k: data[k] for k in asdict(cls()).keys() if k in data}
        return cls(**known)


@dataclass
class FallbackResult:
    """Per-sample fallback decisions and the (possibly overridden) outputs."""

    final_probs: np.ndarray          # [n, C] after any override
    base_probs: np.ndarray           # [n, C] original fused probabilities
    actions: np.ndarray              # [n] str, one of ACTIONS
    review: np.ndarray               # [n] bool, flagged for human review
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def preds(self) -> np.ndarray:
        return self.final_probs.argmax(axis=1)

    def action_counts(self) -> dict[str, int]:
        return {a: int((self.actions == a).sum()) for a in ACTIONS}


# --------------------------------------------------------------------------- #
# Fallback decision layer
# --------------------------------------------------------------------------- #
class UncertaintyFallback:
    """Detect uncertain fusion outputs and apply the fallback policy."""

    def __init__(
        self,
        config: FallbackConfig | None = None,
        outlier: OutlierDetector | None = None,
        class_prior: np.ndarray | None = None,
    ) -> None:
        self.config = config or FallbackConfig()
        self.outlier = outlier or OutlierDetector()
        self.class_prior = class_prior
        self._fitted = class_prior is not None and outlier is not None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_embeddings: np.ndarray,
        holdout_fused_probs: np.ndarray,
        holdout_kan_probs: np.ndarray,
        holdout_xgb_probs: np.ndarray,
        holdout_embeddings: np.ndarray,
        holdout_labels: np.ndarray,
    ) -> "UncertaintyFallback":
        """Fit the outlier detector and calibrate thresholds from the holdout.

        ``train_embeddings`` define the in-distribution reference; every
        threshold is a percentile of the *holdout* signal it governs, so the
        fallback fires on that split's tails. The class prior used for
        abstentions is the holdout label distribution.
        """
        self.outlier.fit(train_embeddings)

        n_classes = holdout_fused_probs.shape[1]
        prior = np.bincount(
            np.asarray(holdout_labels, dtype=np.int64), minlength=n_classes
        ).astype(np.float64)
        self.class_prior = prior / max(prior.sum(), 1.0)

        signals = self.compute_signals(
            holdout_fused_probs, holdout_kan_probs, holdout_xgb_probs, holdout_embeddings
        )
        c = self.config
        c.entropy_high = float(np.percentile(signals["entropy"], c.entropy_high_pct))
        c.entropy_abstain = float(np.percentile(signals["entropy"], c.entropy_abstain_pct))
        c.confidence_low = float(np.percentile(signals["confidence"], c.confidence_low_pct))
        c.disagreement_high = float(
            np.percentile(signals["disagreement"], c.disagreement_high_pct)
        )
        c.ood_high = float(np.percentile(signals["ood"], c.ood_high_pct))
        c.ood_abstain = float(np.percentile(signals["ood"], c.ood_abstain_pct))
        self._fitted = True
        return self

    # -------------------------------------------------------------- signals
    def compute_signals(
        self,
        fused_probs: np.ndarray,
        kan_probs: np.ndarray,
        xgb_probs: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute the four per-sample uncertainty signals."""
        fused = np.asarray(fused_probs, dtype=np.float64)
        return {
            "entropy": shannon_entropy(fused),
            "confidence": fused.max(axis=1),
            # Total-variation distance in [0, 1]: 0 = experts agree exactly.
            "disagreement": 0.5 * np.abs(
                np.asarray(kan_probs) - np.asarray(xgb_probs)
            ).sum(axis=1),
            "ood": self.outlier.score(embeddings),
        }

    # ---------------------------------------------------------------- apply
    def apply(
        self,
        fused_probs: np.ndarray,
        kan_probs: np.ndarray,
        xgb_probs: np.ndarray,
        embeddings: np.ndarray,
        kan_residual: np.ndarray | None = None,
        xgb_residual: np.ndarray | None = None,
    ) -> FallbackResult:
        """Run the fallback policy, returning overridden outputs + decisions.

        ``kan_residual`` / ``xgb_residual`` are the experts' expected residuals
        used to pick a target when routing; when omitted, ``1 - confidence`` of
        each expert is used as a proxy.
        """
        if not self._fitted:
            raise RuntimeError("UncertaintyFallback.apply called before fit().")

        fused = np.asarray(fused_probs, dtype=np.float64)
        kan = np.asarray(kan_probs, dtype=np.float64)
        xgb = np.asarray(xgb_probs, dtype=np.float64)
        n = len(fused)
        c = self.config

        signals = self.compute_signals(fused, kan, xgb, embeddings)
        entropy, confidence = signals["entropy"], signals["confidence"]
        disagreement, ood = signals["disagreement"], signals["ood"]

        if kan_residual is None:
            kan_residual = 1.0 - kan.max(axis=1)
        if xgb_residual is None:
            xgb_residual = 1.0 - xgb.max(axis=1)
        kan_residual = np.asarray(kan_residual, dtype=np.float64)
        xgb_residual = np.asarray(xgb_residual, dtype=np.float64)

        final = fused.copy()
        actions = np.full(n, KEEP, dtype=object)
        review = np.zeros(n, dtype=bool)
        override = c.apply_overrides  # when False, only annotate (never change preds)

        prior = (
            self.class_prior
            if self.class_prior is not None
            else np.full(fused.shape[1], 1.0 / fused.shape[1])
        )

        for i in range(n):
            # 1. Abstain: the ensemble is lost, or the input is far OOD.
            if entropy[i] >= c.entropy_abstain or ood[i] >= c.ood_abstain:
                actions[i] = ABSTAIN_REVIEW
                review[i] = True
                if override:
                    final[i] = prior
                continue
            # 2. Route: strong expert disagreement -> trust the lower-residual one.
            if disagreement[i] >= c.disagreement_high:
                if kan_residual[i] <= xgb_residual[i]:
                    actions[i] = ROUTE_KAN
                    if override:
                        final[i] = kan[i]
                else:
                    actions[i] = ROUTE_XGB
                    if override:
                        final[i] = xgb[i]
                continue
            # 3. Conservative: moderately uncertain -> de-rate confidence.
            if (
                confidence[i] <= c.confidence_low
                or entropy[i] >= c.entropy_high
                or ood[i] >= c.ood_high
            ):
                actions[i] = CONSERVATIVE
                if override:
                    final[i] = _temperature_soften(fused[i], c.conservative_temperature)
                continue
            # 4. Keep (default).

        return FallbackResult(
            final_probs=final,
            base_probs=fused,
            actions=actions,
            review=review,
            signals=signals,
        )

    # ------------------------------------------------------- serialisation
    def state_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "outlier": self.outlier.state_dict(),
            "class_prior": None if self.class_prior is None else self.class_prior.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "UncertaintyFallback":
        obj = cls(
            config=FallbackConfig.from_dict(state["config"]),
            outlier=OutlierDetector.from_state_dict(state["outlier"]),
            class_prior=np.asarray(state["class_prior"], dtype=np.float64),
        )
        obj._fitted = True
        return obj


def _temperature_soften(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Raise entropy by tempering: ``softmax(log p / T)`` for ``T > 1``."""
    if temperature <= 1.0:
        return probs
    logits = np.log(np.clip(probs, _EPS, 1.0)) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fallback_report(labels: np.ndarray, result: FallbackResult) -> dict[str, object]:
    """Summarise fallback behaviour and its effect on accuracy.

    ``base_accuracy`` is the fused model's true accuracy and is the number to
    report as *the fusion's accuracy* — the fallback is a reliability layer, not
    an accuracy booster. In the default annotation-only mode ``final_accuracy``
    equals ``base_accuracy`` (predictions are untouched; only review flags are
    added). The confident/flagged split shows the flags are meaningful:
    ``accuracy_on_covered`` (non-flagged samples) should exceed
    ``accuracy_on_reviewed`` (the samples sent for human review).
    """
    labels = np.asarray(labels, dtype=np.int64)
    base_preds = result.base_probs.argmax(axis=1)
    final_preds = result.preds
    review = result.review
    covered = ~review  # samples the system is confident enough to stand behind

    counts = result.action_counts()
    n = len(labels)
    report: dict[str, object] = {
        "n_samples": int(n),
        "annotation_only": bool(np.array_equal(base_preds, final_preds)),
        "action_counts": counts,
        "action_fractions": {a: counts[a] / n for a in ACTIONS},
        "review_rate": float(review.mean()),
        "coverage": float(covered.mean()),
        # The model's real accuracy (report this as the fusion accuracy).
        "base_accuracy": float((base_preds == labels).mean()),
        # Equals base_accuracy in annotation-only mode; differs only if overrides applied.
        "final_accuracy": float((final_preds == labels).mean()),
        # Reliability: confident samples should beat flagged ones.
        "accuracy_on_covered": (
            float((final_preds[covered] == labels[covered]).mean()) if covered.any() else None
        ),
        "accuracy_on_reviewed": (
            float((final_preds[review] == labels[review]).mean()) if review.any() else None
        ),
    }
    # Kept for continuity with earlier runs (== accuracy_on_covered).
    report["selective_accuracy"] = report["accuracy_on_covered"] or 0.0
    return report


def plot_fallback_actions(result: FallbackResult, output_path: Path | str) -> None:
    """Save a bar chart of how often each fallback action fired."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = result.action_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4878d0", "#6acc64", "#ee854a", "#d5bb67", "#d65f5f"]
    ax.bar(list(counts.keys()), list(counts.values()), color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Samples")
    ax.set_title("Uncertainty-Aware Fallback: action distribution")
    for i, (_, v) in enumerate(counts.items()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
