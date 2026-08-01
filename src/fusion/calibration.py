"""Temperature scaling for expert probability calibration.

The fusion gate decides *who to trust* mostly from the experts' confidence and
disagreement signals (SHAP confirms `xgb_confidence`, `confidence_diff`, entropy
dominate). If an expert is over/under-confident, those signals are misleading and
the gate makes worse calls. Temperature scaling fixes that with a single learned
scalar ``T`` per expert::

    p_calibrated = softmax( log(p) / T )

``T > 1`` softens an over-confident expert; ``T < 1`` sharpens an under-confident
one. It is monotonic, so an expert's own hard predictions never change — only its
confidences are re-scaled, which (a) feeds the gate honest signals and (b) shifts
the confidence-weighted convex combination ``alpha*p_kan + (1-alpha)*p_xgb``.

``T`` is fit by minimising negative log-likelihood on a held-out split, so it is a
proper calibration, not a free parameter tuned on the test set.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12


class TemperatureScaler:
    """Single-parameter temperature calibrator for a probability matrix."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        bounds: tuple[float, float] = (0.05, 20.0),
    ) -> "TemperatureScaler":
        """Learn the temperature that minimises NLL of ``probs`` against ``labels``."""
        logp = np.log(np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0))
        labels = np.asarray(labels, dtype=np.int64)
        rows = np.arange(len(labels))

        def nll(temp: float) -> float:
            scaled = logp / temp
            scaled -= scaled.max(axis=1, keepdims=True)
            p = np.exp(scaled)
            p /= p.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(p[rows, labels], _EPS, 1.0))))

        result = minimize_scalar(nll, bounds=bounds, method="bounded")
        self.temperature = float(result.x)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply the learned temperature, returning re-normalised probabilities."""
        logp = np.log(np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0))
        scaled = logp / self.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def state_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> "TemperatureScaler":
        return cls(temperature=float(state["temperature"]))


def fit_expert_calibrators(
    kan_probs: np.ndarray,
    xgb_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[TemperatureScaler, TemperatureScaler]:
    """Fit one temperature per expert on a held-out split's predictions."""
    return (
        TemperatureScaler().fit(kan_probs, labels),
        TemperatureScaler().fit(xgb_probs, labels),
    )
