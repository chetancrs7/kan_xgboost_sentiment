"""Configuration for the uncertainty-aware inference layer.

Everything tunable about the uncertainty module lives in one dataclass:
the per-measure combination weights, the low/medium/high score thresholds,
the fallback mode applied to high-uncertainty samples, and the Monte Carlo
dropout / Local Outlier Factor hyper-parameters. A config can be built from
CLI arguments or loaded from a JSON file, and it is echoed verbatim into
``uncertainty_report.json`` so every run is self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from src.fusion.utils import load_json


class FallbackMode(IntEnum):
    """What to do with a high-uncertainty sample."""

    FUSED_LOW_CONFIDENCE = 1  # keep the fused prediction, mark it low confidence
    MOST_CONFIDENT_EXPERT = 2  # defer to whichever expert is more confident
    NEUTRAL = 3  # abstain: return Neutral with an uncertainty flag
    HUMAN_REVIEW = 4  # flag the sample for human review


#: Measures combined into the final score, all oriented "higher = more
#: uncertain". Weights are normalised to sum to 1 at scoring time, so only
#: their relative magnitudes matter.
DEFAULT_WEIGHTS: dict[str, float] = {
    # Prediction uncertainty (fused distribution)
    "entropy": 0.10,
    "one_minus_msp": 0.10,
    "one_minus_margin": 0.05,
    # Expert disagreement
    "disagreement": 0.10,
    "js_divergence": 0.10,
    "prob_distance": 0.05,
    # Embedding uncertainty (distance from the training distribution)
    "embedding_norm_dev": 0.05,
    "mahalanobis": 0.10,
    "lof": 0.10,
    # Model uncertainty (MC dropout on the fusion gate)
    "mc_alpha_std": 0.10,
    "mc_pred_variance": 0.15,
}


@dataclass
class UncertaintyConfig:
    """Tunables of the uncertainty-aware inference layer.

    Attributes:
        weights: Relative weight of each uncertainty measure in the combined
            score. Keys must match the measure names produced by
            ``src.uncertainty.measures``.
        low_threshold: Scores at or below this are *low* uncertainty
            (prediction returned normally).
        high_threshold: Scores above this are *high* uncertainty (fallback
            logic triggers). Scores in between are *medium* (prediction
            returned with a confidence warning).
        fallback_mode: Strategy applied to high-uncertainty samples.
        mc_passes: Number of stochastic forward passes for MC dropout.
        mc_seed: RNG seed for the dropout masks (reproducible passes).
        lof_neighbors: ``n_neighbors`` of the Local Outlier Factor model.
        norm_lower_pct / norm_upper_pct: Percentiles of the reference split
            used as the [0, 1] normalisation range for each measure.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    low_threshold: float = 0.35
    high_threshold: float = 0.60
    fallback_mode: FallbackMode = FallbackMode.MOST_CONFIDENT_EXPERT
    mc_passes: int = 30
    mc_seed: int = 42
    lof_neighbors: int = 20
    norm_lower_pct: float = 1.0
    norm_upper_pct: float = 99.0

    def __post_init__(self) -> None:
        self.fallback_mode = FallbackMode(self.fallback_mode)
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 <= low < high <= 1, got "
                f"low={self.low_threshold}, high={self.high_threshold}."
            )
        if not self.weights:
            raise ValueError("weights must not be empty.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative.")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one weight must be positive.")
        if self.mc_passes < 2:
            raise ValueError("mc_passes must be >= 2 to estimate a variance.")
        if not 0.0 <= self.norm_lower_pct < self.norm_upper_pct <= 100.0:
            raise ValueError("Normalisation percentiles must satisfy 0 <= lo < hi <= 100.")

    @property
    def normalized_weights(self) -> dict[str, float]:
        """Weights rescaled to sum to exactly 1."""
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly representation (embedded in the run report)."""
        return {
            "weights": dict(self.weights),
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "fallback_mode": int(self.fallback_mode),
            "fallback_mode_name": self.fallback_mode.name,
            "mc_passes": self.mc_passes,
            "mc_seed": self.mc_seed,
            "lof_neighbors": self.lof_neighbors,
            "norm_lower_pct": self.norm_lower_pct,
            "norm_upper_pct": self.norm_upper_pct,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "UncertaintyConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_json(cls, path: Path | str) -> "UncertaintyConfig":
        return cls.from_dict(load_json(path))
