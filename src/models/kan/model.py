"""KAN sentiment classifier: four adaptive-spline layers over DeBERTa embeddings.

The four layers below are chained in a fixed sequence -- no extra linear
projections, normalisation, or dropout layers are inserted between them, so
the architecture matches exactly the four named stages this model is built
around, each using the spline family suited to its role:

    embedding -> SemanticCompressionLayer  (B-spline)
              -> ContextInteractionLayer   (B-spline)
              -> SentimentDistortionLayer  (thin-plate spline)
              -> EmotionHotspotLayer       (Gaussian RBF spline) -> logits
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Make the project root importable so `layers` can be imported through its
# package path regardless of how this module itself was loaded.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.kan.layers import (  # noqa: E402
    ContextInteractionLayer,
    EmotionHotspotLayer,
    SemanticCompressionLayer,
    SentimentDistortionLayer,
)

DEFAULT_HIDDEN_DIMS: tuple[int, int, int] = (256, 128, 64)


class KANSentimentClassifier(nn.Module):
    """Four-layer Kolmogorov-Arnold Network for sentiment classification.

    Args:
        input_dim: Dimensionality of the pretrained embeddings (768 for
            DeBERTa-v3-base).
        hidden_dims: Output widths of layers 1-3, i.e.
            ``(semantic_dim, context_dim, distortion_dim)``.
        n_classes: Number of sentiment classes (final layer's output width).
        grid_size: Grid resolution shared by all four layers -- B-spline
            interval count for layers 1-2 and radial-centre count for the
            thin-plate / RBF layers 3-4.
        spline_order: B-spline polynomial degree for layers 1-2 (ignored by
            the radial layers 3-4, which have no polynomial order).
        grid_range: Initial spline/centre grid range before adaptive refitting.
        aux_failure_head: When ``True``, attach a small auxiliary MLP that
            predicts the probability the XGBoost structural expert *fails* on a
            sample, using only the DeBERTa semantic representation (the layer
            input). It is an interpretable competence signal for the fusion
            gate and never alters the sentiment ``forward`` output. Default
            ``False`` keeps the module byte-identical to the original — no extra
            parameters, so existing checkpoints load unchanged.
        aux_hidden: Hidden width of the auxiliary head (only when enabled).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, int, int] = DEFAULT_HIDDEN_DIMS,
        n_classes: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        aux_failure_head: bool = False,
        aux_hidden: int = 64,
    ) -> None:
        super().__init__()
        semantic_dim, context_dim, distortion_dim = hidden_dims

        self.semantic_compression = SemanticCompressionLayer(
            input_dim, semantic_dim, grid_size, spline_order, grid_range
        )
        self.context_interaction = ContextInteractionLayer(
            semantic_dim, context_dim, grid_size, spline_order, grid_range
        )
        self.sentiment_distortion = SentimentDistortionLayer(
            context_dim, distortion_dim, grid_size, spline_order, grid_range
        )
        self.emotion_hotspot = EmotionHotspotLayer(
            distortion_dim, n_classes, grid_size, spline_order, grid_range
        )

        self.layers: tuple = (
            self.semantic_compression,
            self.context_interaction,
            self.sentiment_distortion,
            self.emotion_hotspot,
        )

        # Optional auxiliary "XGBoost competence" head. Kept OUT of ``self.layers``
        # so grid adaptation/refinement/spline-visualisation ignore it, and only
        # instantiated when requested so the default model's state_dict is
        # unchanged (backward-compatible checkpoints).
        self.aux_failure_head = bool(aux_failure_head)
        self.aux_head: nn.Module | None = None
        if self.aux_failure_head:
            self.aux_head = nn.Sequential(
                nn.Linear(input_dim, aux_hidden),
                nn.GELU(),
                nn.Linear(aux_hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the four KAN layers in sequence and return class logits.

        The auxiliary competence head (if any) is intentionally *not* invoked
        here, so every existing caller keeps receiving class logits only. Use
        :meth:`aux_failure_logit` to read the competence signal.
        """
        x = self.semantic_compression(x)
        x = self.context_interaction(x)
        x = self.sentiment_distortion(x)
        logits = self.emotion_hotspot(x)
        return logits

    def aux_failure_logit(self, x: torch.Tensor) -> torch.Tensor | None:
        """Predict the XGBoost-failure logit from the semantic embedding ``x``.

        Returns a ``[batch]`` logit tensor when the auxiliary head is enabled,
        else ``None``. ``torch.sigmoid`` of this logit is the estimated
        probability that the structural expert misclassifies the sample.
        """
        if self.aux_head is None:
            return None
        return self.aux_head(x).squeeze(-1)

    @torch.no_grad()
    def update_grids(self, x: torch.Tensor) -> None:
        """Refit every layer's adaptive spline grid to a batch of real data.

        Propagates ``x`` through the stack, calling each layer's
        ``adapt_grid`` on the activations it actually receives before
        computing that layer's output. Intended to be called periodically
        during training (see ``train.py``), not just once at init, so spline
        resolution keeps tracking the evolving activation distribution.
        """
        was_training = self.training
        self.eval()
        current = x
        for layer in self.layers:
            layer.adapt_grid(current)
            current = layer(current)
        if was_training:
            self.train()

    @torch.no_grad()
    def refine_grids(self, new_grid_size: int, x: torch.Tensor) -> int:
        """Grow every layer's spline resolution to ``new_grid_size``.

        Propagates ``x`` so each layer is refined on the activations it
        actually receives, then adapts the (now finer) grids to the data
        quantiles. This is "spline complexity increases in high-density
        regions" made literal: the extra knots/centres are both *added* and
        *placed* where the data concentrates. Because refinement replaces
        parameters, callers must rebuild the optimiser afterwards (see
        ``train.py``).

        Each layer's ``refine`` is self-guarding: a layer that would produce
        non-finite weights keeps its current grid instead. Returns the number
        of layers that were actually refined (0 means the model is unchanged).
        """
        was_training = self.training
        self.eval()
        current = x
        refined = 0
        for layer in self.layers:
            if layer.refine(new_grid_size, current):
                refined += 1
            current = layer(current)
        # Re-place the freshly added knots/centres at the data quantiles.
        self.update_grids(x)
        if was_training:
            self.train()
        return refined
