"""Adaptive spline KAN layers for the sentiment classifier.

Kolmogorov-Arnold Networks replace a linear layer's fixed weight matrix with
a matrix of *learnable univariate functions*. Concretely, an edge from input
neuron ``i`` to output neuron ``j`` computes

    phi_ij(x_i) = scale_base * base_fun(x_i) + scale_sp * spline_ij(x_i)

where ``spline_ij`` is a B-spline built from a fixed knot ``grid`` and a
trainable coefficient vector ``coef`` (see ``kan.KANLayer`` /
``kan.spline.coef2curve`` in the pykan library). Every output neuron sums the
``in_dim`` edge functions feeding into it:

    y_j = sum_i phi_ij(x_i)

Two things make the spline basis "adaptive" rather than a static feature
transform:

1. ``coef`` is a plain ``nn.Parameter`` -- gradients from the downstream
   cross-entropy loss flow directly into the spline coefficients, so the
   shape of every edge function is learned during backprop exactly like any
   other weight.
2. The spline ``grid`` (the knot positions the B-splines are defined over)
   can be *refit* to the empirical range/density of the activations that
   actually flow through the layer via ``update_grid_from_samples``. This
   repo calls that periodically during training (see ``train.py``) so the
   spline resolution tracks the evolving activation distribution instead of
   staying fixed at initialisation.

``AdaptiveSplineKANLayer`` is a thin wrapper around ``kan.KANLayer`` that
exposes just the pieces this project needs (the summed output ``y``, and a
``adapt_grid`` convenience method). The four public classes below are all
this same building block; they exist as separate names -- rather than one
generic layer reused four times -- because each occupies a distinct semantic
role in the architecture and may be tuned independently (spline resolution,
output width) as the model evolves.
"""

from __future__ import annotations

import torch
from kan import KANLayer
from torch import nn


class AdaptiveSplineKANLayer(nn.Module):
    """One Kolmogorov-Arnold layer: learnable per-edge B-spline functions.

    Args:
        in_dim: Input feature dimensionality.
        out_dim: Output feature dimensionality.
        grid_size: Number of spline grid intervals (more intervals -> finer,
            more locally-adaptive splines).
        spline_order: Degree ``k`` of the B-spline basis polynomials.
        grid_range: ``(min, max)`` initial range the spline grid covers,
            later refined per-layer by ``adapt_grid``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kan_layer = KANLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            num=grid_size,
            k=spline_order,
            grid_range=list(grid_range),
            sp_trainable=True,  # spline coefficients are learnable
            sb_trainable=True,  # base-function scale is learnable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # kan.KANLayer.forward returns (y, preacts, postacts, postspline);
        # only the summed edge output `y` is needed for a feed-forward stack.
        y, _preacts, _postacts, _postspline = self.kan_layer(x)
        return y

    @torch.no_grad()
    def adapt_grid(self, x: torch.Tensor) -> None:
        """Refit the B-spline knot grid to a batch of real activations.

        This is what makes the spline basis *adaptive* beyond ordinary
        gradient descent on ``coef``: the knots themselves move to match
        wherever this layer's input actually lives, so the fixed-degree
        B-splines get their resolution concentrated where the data is.
        """
        self.kan_layer.update_grid_from_samples(x)


class SemanticCompressionLayer(AdaptiveSplineKANLayer):
    """Layer 1: compresses raw DeBERTa embeddings into a compact semantic code.

    Takes the full-width pretrained embedding (e.g. 768-dim DeBERTa-v3
    hidden state) and reduces it to ``out_dim`` learned semantic components,
    each a nonlinear spline combination of the input dimensions rather than a
    plain linear projection.
    """


class ContextInteractionLayer(AdaptiveSplineKANLayer):
    """Layer 2: models interactions between the compressed semantic features.

    Every output unit is a spline-weighted mixture of *all* semantic
    components from layer 1, letting the network capture feature
    co-occurrence patterns (e.g. negation interacting with intensifiers)
    that a single compressed dimension cannot express alone.
    """


class SentimentDistortionLayer(AdaptiveSplineKANLayer):
    """Layer 3: warps contextual features into sentiment-salient directions.

    "Distortion" reflects the KAN framing: rather than a linear rotation,
    this layer applies learned nonlinear splines that can stretch, compress,
    or fold the context representation so sentiment-relevant structure
    (polarity, intensity) becomes more linearly separable for the final
    classification layer.
    """


class EmotionHotspotLayer(AdaptiveSplineKANLayer):
    """Layer 4: final classification layer producing per-class logits.

    "Hotspot" refers to the class-specific activation peaks this layer's
    splines learn to carve out of the distorted sentiment representation --
    its ``out_dim`` equals the number of sentiment classes, and its raw
    output is used directly as logits for cross-entropy / softmax.
    """
