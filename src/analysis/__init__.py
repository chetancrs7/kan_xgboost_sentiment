"""Interpretability and visualization tools for the KAN spline layers.

Public API:

* :mod:`src.analysis.spline_extractor` — read-only access to knots, basis
  functions, spline coefficients, forward activations, and importance
  measures of the four ``AdaptiveSplineKANLayer`` stages.
* :mod:`src.analysis.spline_visualizer` — publication-style (grayscale,
  300 DPI) figures; ``generate_all_visualizations`` produces the full
  report under ``outputs/kan_spline_visualizations/``.
* :mod:`src.analysis.tensorboard_logger` — failure-proof optional
  TensorBoard logging of spline diagnostics during training.
* ``python -m src.analysis.run_spline_analysis`` — standalone CLI that
  analyses a saved checkpoint without retraining.
"""

from src.analysis.spline_extractor import (
    LAYER_NAMES,
    LAYER_TITLES,
    EdgeCurve,
    basis_functions,
    edge_curve,
    edge_strengths,
    forward_capture,
    interior_knots,
    extended_knots,
    iter_kan_layers,
    layer_importances,
    select_top_k_dims,
    spline_coefficients,
    top_edges,
)
from src.analysis.spline_visualizer import generate_all_visualizations
from src.analysis.tensorboard_logger import SplineTensorBoardLogger

__all__ = [
    "LAYER_NAMES",
    "LAYER_TITLES",
    "EdgeCurve",
    "basis_functions",
    "edge_curve",
    "edge_strengths",
    "extended_knots",
    "forward_capture",
    "generate_all_visualizations",
    "interior_knots",
    "iter_kan_layers",
    "layer_importances",
    "select_top_k_dims",
    "spline_coefficients",
    "SplineTensorBoardLogger",
    "top_edges",
]
