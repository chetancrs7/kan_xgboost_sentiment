"""KAN sentiment classifier: four adaptive-spline layers over DeBERTa embeddings.

The four layers below are chained in a fixed sequence -- no extra linear
projections, normalisation, or dropout layers are inserted between them, so
the architecture matches exactly the four named stages this model is built
around:

    embedding -> SemanticCompressionLayer -> ContextInteractionLayer
              -> SentimentDistortionLayer -> EmotionHotspotLayer -> logits
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
        grid_size: B-spline grid interval count, shared by all four layers.
        spline_order: B-spline polynomial degree, shared by all four layers.
        grid_range: Initial spline grid range before adaptive refitting.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, int, int] = DEFAULT_HIDDEN_DIMS,
        n_classes: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the four KAN layers in sequence and return class logits."""
        x = self.semantic_compression(x)
        x = self.context_interaction(x)
        x = self.sentiment_distortion(x)
        logits = self.emotion_hotspot(x)
        return logits

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
