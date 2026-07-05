"""Weighted-normalised combination of the uncertainty measures.

The individual measures live on wildly different scales (entropy in [0, 1],
Mahalanobis distances in the tens), so a direct weighted sum would be
dominated by whichever measure happens to have the largest numbers. Instead:

1. :class:`MeasureNormalizer` maps every measure to [0, 1] using robust
   percentile bounds *fitted on a reference split* (the fusion validation
   split), so the test scores are comparable across runs and measures.
2. :class:`UncertaintyScorer` combines the normalised measures with the
   configured weights into a single score in [0, 1], then buckets each sample
   into the low / medium / high uncertainty tiers via the configured
   thresholds.
"""

from __future__ import annotations

import numpy as np

from src.uncertainty.config import UncertaintyConfig

TIER_LOW = "low"
TIER_MEDIUM = "medium"
TIER_HIGH = "high"


class MeasureNormalizer:
    """Percentile-based min-max normaliser, one range per measure.

    Percentile bounds (rather than the raw min/max) keep a single extreme
    outlier in the reference split from squashing everyone else to ~0.
    Out-of-range values on new data are clipped to [0, 1].
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.bounds_: dict[str, tuple[float, float]] | None = None

    def fit(self, measures: dict[str, np.ndarray]) -> "MeasureNormalizer":
        """Learn per-measure [lo, hi] ranges from the reference split."""
        self.bounds_ = {}
        for name, values in measures.items():
            lo = float(np.percentile(values, self.lower_pct))
            hi = float(np.percentile(values, self.upper_pct))
            self.bounds_[name] = (lo, hi)
        return self

    def transform(self, measures: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map each measure into [0, 1] using the fitted ranges."""
        if self.bounds_ is None:
            raise RuntimeError("MeasureNormalizer.transform called before fit().")
        missing = set(self.bounds_) - set(measures)
        if missing:
            raise ValueError(f"Measures missing at transform time: {sorted(missing)}")
        out: dict[str, np.ndarray] = {}
        for name, (lo, hi) in self.bounds_.items():
            values = np.asarray(measures[name], dtype=np.float64)
            if hi - lo < 1e-12:  # constant measure on the reference split
                out[name] = np.zeros_like(values)
            else:
                out[name] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        return out

    def state_dict(self) -> dict[str, object]:
        return {
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "bounds": {k: list(v) for k, v in (self.bounds_ or {}).items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "MeasureNormalizer":
        normalizer = cls(state["lower_pct"], state["upper_pct"])
        normalizer.bounds_ = {k: (v[0], v[1]) for k, v in state["bounds"].items()}
        return normalizer


class UncertaintyScorer:
    """Combine normalised measures into one score and assign tiers."""

    def __init__(self, config: UncertaintyConfig) -> None:
        self.config = config
        self.normalizer = MeasureNormalizer(
            lower_pct=config.norm_lower_pct, upper_pct=config.norm_upper_pct
        )

    def fit(self, reference_measures: dict[str, np.ndarray]) -> "UncertaintyScorer":
        """Fit the normalisation ranges on the reference (validation) split."""
        missing = set(self.config.weights) - set(reference_measures)
        if missing:
            raise ValueError(
                f"Config weights reference measures that were not computed: {sorted(missing)}"
            )
        self.normalizer.fit(
            {name: reference_measures[name] for name in self.config.weights}
        )
        return self

    def score(
        self, measures: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Weighted sum of the normalised measures.

        Returns:
            ``(score, normalized)``: the combined score ``[n]`` in [0, 1] and
            the per-measure normalised values used to build it.
        """
        normalized = self.normalizer.transform(measures)
        weights = self.config.normalized_weights
        score = np.zeros(len(next(iter(normalized.values()))), dtype=np.float64)
        for name, weight in weights.items():
            score += weight * normalized[name]
        return np.clip(score, 0.0, 1.0), normalized

    def assign_tiers(self, score: np.ndarray) -> np.ndarray:
        """Bucket each score into low / medium / high uncertainty."""
        tiers = np.full(len(score), TIER_MEDIUM, dtype=object)
        tiers[score <= self.config.low_threshold] = TIER_LOW
        tiers[score > self.config.high_threshold] = TIER_HIGH
        return tiers
