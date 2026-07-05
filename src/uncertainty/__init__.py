"""Uncertainty-aware inference layer for the adaptive fusion network.

Operates strictly downstream of the fusion gate: given the fused prediction
and both experts' outputs, quantifies how trustworthy each prediction is and
routes high-uncertainty samples through configurable fallback logic.

Modules
-------
``config``       tunables: measure weights, tier thresholds, fallback mode
``measures``     prediction / disagreement / embedding / MC-dropout measures
``scorer``       percentile normalisation + weighted combination + tiers
``fallback``     low/medium/high decision logic and the four fallback modes
``reliability``  ECE, MCE, Brier, NLL, reliability diagrams, risk-coverage
``pipeline``     end-to-end run + artifact export (CSV, JSON reports, figures)
"""

from src.uncertainty.config import FallbackMode, UncertaintyConfig
from src.uncertainty.measures import EmbeddingUncertaintyEstimator, compute_measures
from src.uncertainty.scorer import MeasureNormalizer, UncertaintyScorer
from src.uncertainty.fallback import apply_fallback
from src.uncertainty.pipeline import run_uncertainty_analysis

__all__ = [
    "FallbackMode",
    "UncertaintyConfig",
    "EmbeddingUncertaintyEstimator",
    "compute_measures",
    "MeasureNormalizer",
    "UncertaintyScorer",
    "apply_fallback",
    "run_uncertainty_analysis",
]
