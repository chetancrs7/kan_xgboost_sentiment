"""KAN sentiment classifier: four adaptive-spline layers over DeBERTa embeddings.

The four layers below are chained in a fixed sequence -- no extra linear
projections, normalisation, or dropout layers are inserted between them, so
the architecture matches exactly the four named stages this model is built
around:

    embedding -> SemanticCompressionLayer -> ContextInteractionLayer
              -> SentimentDistortionLayer -> EmotionHotspotLayer -> logits

Each stage's spline family is configurable through ``spline_types`` (see
``src/models/kan/splines.py`` for the registry). The default -- B-splines
everywhere, no separate output head -- reproduces the original architecture
exactly, including state-dict layout, so existing checkpoints keep loading.

The research-motivated *mixed-spline* configuration assigns each stage the
basis suited to its role:

    semantic_compression   bspline        stable smooth embedding compression
    context_interaction    cubic_bspline  subtle contextual shifts (negation,
                                          contrast, modifiers)
    sentiment_distortion   thin_plate     complex nonlinear sentiment warping
                                          (mixed sentiment, sarcasm)
    emotion_hotspot        rbf            localised emotional-region detection
    sentiment_classifier   linear         low-variance output classifier
                                          (``output_type="linear"``)

With ``output_type`` set, ``hidden_dims`` takes a fourth entry (the emotion-
hotspot width) and a separate head maps it to the class logits; with
``output_type=None`` the emotion-hotspot layer emits the logits directly, as
in the original model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Make the project root importable so `layers` can be imported through its
# package path regardless of how this module itself was loaded.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.kan.splines import build_spline_layer  # noqa: E402

DEFAULT_HIDDEN_DIMS: tuple[int, int, int] = (256, 128, 64)
DEFAULT_SPLINE_TYPES: tuple[str, str, str, str] = ("bspline",) * 4

#: The four named KAN stages, in forward order (mirrored by
#: ``src/analysis/spline_extractor.LAYER_NAMES``).
STAGE_NAMES: tuple[str, ...] = (
    "semantic_compression",
    "context_interaction",
    "sentiment_distortion",
    "emotion_hotspot",
)


def _normalise_output_type(output_type: str | None) -> str | None:
    if output_type is None:
        return None
    key = str(output_type).strip().lower()
    return None if key in ("", "none") else key


class KANSentimentClassifier(nn.Module):
    """Four-stage Kolmogorov-Arnold Network for sentiment classification.

    Args:
        input_dim: Dimensionality of the pretrained embeddings (768 for
            DeBERTa-v3-base).
        hidden_dims: Output widths of the hidden stages. Three entries
            ``(semantic, context, distortion)`` when ``output_type`` is
            ``None``; four entries ``(semantic, context, distortion,
            emotion)`` when a separate output head is requested.
        n_classes: Number of sentiment classes.
        grid_size: Default B-spline grid interval count for all stages.
        spline_order: Default B-spline polynomial degree for all stages.
        grid_range: Initial spline grid / centre range before adaptive
            refitting.
        spline_types: Spline family per stage, in forward order (see
            ``splines.SPLINE_REGISTRY``). Default: B-splines everywhere.
        spline_params: Optional per-stage overrides of the spline kwargs,
            e.g. ``[{}, {"grid_size": 8}, {}, {"num_centers": 10}]``.
        output_type: ``None`` (emotion-hotspot layer emits logits, original
            behaviour) or a registry name -- typically ``"linear"`` -- for a
            dedicated low-capacity classification head.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        n_classes: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        spline_types: tuple[str, ...] | None = None,
        spline_params: tuple[dict, ...] | None = None,
        output_type: str | None = None,
    ) -> None:
        super().__init__()
        spline_types = tuple(spline_types) if spline_types else DEFAULT_SPLINE_TYPES
        if len(spline_types) != 4:
            raise ValueError(
                f"spline_types needs one entry per KAN stage (4), got {len(spline_types)}: "
                f"{spline_types}"
            )
        spline_params = tuple(spline_params) if spline_params else ({},) * 4
        if len(spline_params) != 4:
            raise ValueError(f"spline_params needs 4 entries, got {len(spline_params)}.")
        output_type = _normalise_output_type(output_type)

        hidden_dims = tuple(int(d) for d in hidden_dims)
        expected = 4 if output_type else 3
        if len(hidden_dims) != expected:
            raise ValueError(
                f"hidden_dims needs {expected} entries with output_type={output_type!r} "
                f"(got {hidden_dims}). The fourth entry is the emotion-hotspot width."
            )

        self.spline_types = spline_types
        self.output_type = output_type

        # Stage widths: with a separate head the emotion-hotspot layer keeps
        # its own width; without one it maps straight to the class logits.
        stage_dims = (input_dim,) + hidden_dims + (() if output_type else (n_classes,))
        for name, spline_type, params, in_dim, out_dim in zip(
            STAGE_NAMES, spline_types, spline_params, stage_dims[:-1], stage_dims[1:]
        ):
            kwargs = {
                "grid_size": grid_size,
                "spline_order": spline_order,
                "grid_range": grid_range,
                **dict(params or {}),
            }
            setattr(self, name, build_spline_layer(spline_type, in_dim, out_dim, **kwargs))

        stack = [getattr(self, name) for name in STAGE_NAMES]
        if output_type:
            self.sentiment_classifier = build_spline_layer(
                output_type,
                hidden_dims[-1],
                n_classes,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
            stack.append(self.sentiment_classifier)
        self.layers: tuple = tuple(stack)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the KAN stages (and optional output head) in sequence."""
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def update_grids(self, x: torch.Tensor) -> None:
        """Refit every layer's adaptive spline grid to a batch of real data.

        Propagates ``x`` through the stack, calling each layer's
        ``adapt_grid`` on the activations it actually receives before
        computing that layer's output (a no-op for layers without knots or
        centres, e.g. a linear head). Intended to be called periodically
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
