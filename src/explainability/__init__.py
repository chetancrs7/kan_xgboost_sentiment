"""SHAP-based explainability for the hybrid sentiment ensemble.

Two explainers, one Shapley framework so results are directly comparable:

* :mod:`src.explainability.shap_xgboost` -- exact ``TreeExplainer`` Shapley
  values for the XGBoost structural expert.
* :mod:`src.explainability.shap_fusion` -- model-agnostic ``KernelExplainer``
  attributions for the fusion gate's trust value ``alpha`` (which meta-features
  make the ensemble lean on the KAN vs. the XGBoost expert).

Why SHAP rather than LIME: the XGBoost expert is a tree ensemble, where
``TreeExplainer`` computes *exact* Shapley values in low-order polynomial time
(LIME only fits an unstable local linear surrogate); and Shapley values give a
single additive attribution scheme that applies to both the tree expert and the
neural gate, so their feature importances live on the same footing.
"""

from __future__ import annotations


def require_shap():
    """Import and return the ``shap`` module, with an actionable error if absent."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The 'shap' package is required for explainability but is not "
            "installed. Install it with:\n\n    pip install shap\n"
        ) from exc
    return shap
