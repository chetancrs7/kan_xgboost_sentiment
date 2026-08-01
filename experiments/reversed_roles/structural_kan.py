"""KAN variant for the reversed experiment: bounded layer-3 kernel.

Why this exists
---------------
The production :class:`~src.models.kan.model.KANSentimentClassifier` uses an
**unbounded thin-plate spline** for layer 3 (``SentimentDistortionLayer``,
kernel ``r^2 log|r|``). That is a good fit for *warping smooth DeBERTa embedding
manifolds*, its intended input. But in the reversed experiment the KAN is fed
**structural/linguistic count features**, which are spiky and heavy-tailed. On
that input the thin-plate layer's activation grows without bound during training
and overflows to **NaN** within the first epoch — reproducibly, across learning
rates, seeds, gradient-clip norms, input clipping, and a quantile transform.
Layer-wise tracing shows layers 1-2 (B-spline) and layer 4 (Gaussian RBF, which
is bounded in ``(0, 1]``) stay perfectly stable in the *same* run; only the
thin-plate layer 3 climbs (e.g. abs-max 9 -> 19 -> 26 -> ... -> inf).

The fix is to swap **only layer 3** for a bounded kernel, reusing the exact same
layer classes the rest of the network already uses. Everything else — the
B-spline compression/interaction layers, the RBF hotspot classifier, the
``forward`` / ``update_grids`` / ``refine_grids`` contract — is inherited
unchanged, so this remains a drop-in model for the existing
``src.models.kan.train.train_model`` loop.

Nothing in ``src/`` is modified: this subclass imports the production layers and
overrides one attribute.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kan.layers import (  # noqa: E402
    AdaptiveSplineKANLayer,
    GaussianRBFKANLayer,
)
from src.models.kan.model import DEFAULT_HIDDEN_DIMS, KANSentimentClassifier  # noqa: E402

#: Allowed bounded kernels for the layer-3 replacement.
DISTORTION_KERNELS = ("bspline", "rbf")


class StructuralKANClassifier(KANSentimentClassifier):
    """KAN whose layer-3 distortion kernel is bounded (B-spline or Gaussian RBF).

    Identical to :class:`KANSentimentClassifier` except that
    ``sentiment_distortion`` (layer 3) is replaced with a numerically bounded
    kernel so training is stable on spiky structural features.

    Args:
        distortion_kernel: ``"bspline"`` (default) reuses the same
            ``AdaptiveSplineKANLayer`` as layers 1-2 — a learnable, bounded
            spline "distortion" layer that keeps the all-spline character of the
            network. ``"rbf"`` uses a Gaussian RBF (localised, bounded in
            ``(0, 1]``). Both are stable here; the unbounded thin-plate original
            is not.
        All other args match :class:`KANSentimentClassifier`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = DEFAULT_HIDDEN_DIMS,
        n_classes: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        distortion_kernel: str = "bspline",
    ) -> None:
        super().__init__(
            input_dim, hidden_dims, n_classes, grid_size, spline_order, grid_range
        )
        if distortion_kernel not in DISTORTION_KERNELS:
            raise ValueError(
                f"distortion_kernel must be one of {DISTORTION_KERNELS}, "
                f"got {distortion_kernel!r}."
            )
        _, context_dim, distortion_dim = hidden_dims
        layer_cls = (
            AdaptiveSplineKANLayer if distortion_kernel == "bspline" else GaussianRBFKANLayer
        )
        # Replace ONLY layer 3 (unbounded thin-plate) with a bounded kernel.
        self.sentiment_distortion = layer_cls(
            context_dim, distortion_dim, grid_size, spline_order, grid_range
        )
        self.distortion_kernel = distortion_kernel
        # Rebuild the ordered tuple the parent's forward/update_grids/refine use.
        self.layers = (
            self.semantic_compression,
            self.context_interaction,
            self.sentiment_distortion,
            self.emotion_hotspot,
        )
