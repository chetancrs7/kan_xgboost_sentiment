"""Read-only extraction utilities for the KAN model's adaptive B-splines.

This module knows how to open up an ``AdaptiveSplineKANLayer`` (a thin
wrapper around ``kan.KANLayer``, see ``src/models/kan/layers.py``) and pull
out the pieces that make a KAN interpretable:

* **knots / grid** -- ``kan_layer.grid`` has shape ``(in_dim, G + 2k + 1)``:
  the ``G + 1`` *interior* knots that partition each input dimension into
  ``G`` spline intervals, extended by ``k`` extra knots on each side (needed
  so degree-``k`` B-splines are well defined at the boundary). After
  ``update_grid_from_samples`` the interior knots are no longer uniform --
  they follow the empirical distribution of that dimension's activations.
* **spline coefficients** -- ``kan_layer.coef`` has shape
  ``(in_dim, out_dim, G + k)``: one trainable coefficient vector per *edge*
  ``(input i -> output j)``. These are the weights of the B-spline basis and
  are what gradient descent shapes during training.
* **edge activation functions** -- the full learned function on an edge is

      phi_ij(x) = mask_ij * (scale_base_ij * silu(x) + scale_sp_ij * spline_ij(x))

  ``spline_ij(x) = sum_b coef[i, j, b] * B_b(x)`` where ``B_b`` are the
  B-spline basis functions over dimension ``i``'s knot grid.
* **forward activations** -- the inputs each layer actually receives, used
  both to relate spline shapes to real data and to score dimension
  importance.

Everything here is **read-only**: functions detach to NumPy and never
mutate the model, so they are safe to call at any point during or after
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from kan.spline import B_batch, coef2curve

# The four named layers of KANSentimentClassifier, in forward order.
LAYER_NAMES: tuple[str, ...] = (
    "semantic_compression",
    "context_interaction",
    "sentiment_distortion",
    "emotion_hotspot",
)

# Human-readable figure titles explaining each layer's semantic role.
LAYER_TITLES: dict[str, str] = {
    "semantic_compression": "Layer 1 — Semantic Compression (embedding → semantic code)",
    "context_interaction": "Layer 2 — Context Interaction (feature co-occurrence)",
    "sentiment_distortion": "Layer 3 — Sentiment Distortion (polarity warping)",
    "emotion_hotspot": "Layer 4 — Emotion Hotspot (class logits)",
}

# Sample-size caps that keep analysis memory bounded: a single KANLayer
# forward materialises a (batch, in_dim, out_dim) tensor, which for the
# 768 -> 256 first layer is ~200 MB at batch 256.
MAX_FORWARD_SAMPLES = 256
MAX_GRADIENT_SAMPLES = 128


def iter_kan_layers(model) -> Iterator[tuple[str, torch.nn.Module, torch.nn.Module | None]]:
    """Yield ``(name, wrapper, pykan_layer)`` for each of the four KAN layers.

    ``wrapper`` is the project's ``AdaptiveSplineKANLayer``; ``pykan_layer``
    is the underlying ``kan.KANLayer`` holding ``grid`` / ``coef`` / scales.

    Mixed-spline models (see ``src/models/kan/splines.py``) may use non-
    B-spline layers (RBF, thin-plate, linear head); those have no pykan layer
    and yield ``None`` in its place, keeping positions aligned with
    ``LAYER_NAMES`` -- consumers must skip ``None`` entries. B-spline-specific
    figures are simply not produced for such layers.
    """
    for name in LAYER_NAMES:
        wrapper = getattr(model, name)
        yield name, wrapper, getattr(wrapper, "kan_layer", None)


# ---------------------------------------------------------------------------
# Static components: knots, coefficients, edge strengths
# ---------------------------------------------------------------------------

def extended_knots(kan_layer) -> np.ndarray:
    """Full extended knot grid, shape ``(in_dim, G + 2k + 1)``.

    Includes the ``k`` boundary-extension knots on each side that pykan adds
    so that degree-``k`` B-splines have full support at the grid edges.
    """
    return kan_layer.grid.detach().cpu().numpy()


def interior_knots(kan_layer) -> np.ndarray:
    """Interior (data-covering) knots, shape ``(in_dim, G + 1)``.

    These are the knots that matter for interpretation: after grid
    adaptation their spacing mirrors the density of the activations flowing
    through each input dimension (tight spacing = high data density = more
    spline resolution spent there).
    """
    k = kan_layer.k
    return kan_layer.grid.detach()[:, k:-k].cpu().numpy()


def spline_coefficients(kan_layer) -> np.ndarray:
    """Trainable B-spline coefficients, shape ``(in_dim, out_dim, G + k)``."""
    return kan_layer.coef.detach().cpu().numpy()


def edge_strengths(kan_layer) -> np.ndarray:
    """Importance of every edge function, shape ``(in_dim, out_dim)``.

    Strength is ``|scale_sp * mask| * ||coef||_2``: how much the learned
    spline (as opposed to the fixed SiLU residual) can move the output. An
    edge with near-zero strength contributes essentially nothing nonlinear,
    so ranking by this value finds the edges where the KAN actually learned
    a transformation worth plotting.
    """
    with torch.no_grad():
        strength = (kan_layer.scale_sp * kan_layer.mask).abs() * kan_layer.coef.norm(dim=2)
    return strength.cpu().numpy()


def top_edges(kan_layer, k: int = 6) -> list[tuple[int, int]]:
    """Top-``k`` edges ``(in_index, out_index)`` ranked by `edge_strengths`."""
    strengths = edge_strengths(kan_layer)
    k = min(k, strengths.size)
    flat = np.argsort(strengths, axis=None)[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(i, strengths.shape)) for i in flat]


# ---------------------------------------------------------------------------
# Function evaluation: basis functions and learned edge curves
# ---------------------------------------------------------------------------

def _dim_range(kan_layer, in_index: int) -> tuple[float, float]:
    """Interior knot range of one input dimension (where the splines live)."""
    k = kan_layer.k
    grid = kan_layer.grid.detach()
    return float(grid[in_index, k]), float(grid[in_index, -(k + 1)])


def basis_functions(
    kan_layer, in_index: int, n_points: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the ``G + k`` B-spline basis functions of one input dimension.

    Returns:
        ``(x, B)`` where ``x`` is ``(n_points,)`` spanning the dimension's
        interior knot range and ``B`` is ``(n_points, G + k)``. Every learned
        edge function on this dimension is a linear combination of exactly
        these curves, so they show the "vocabulary" of shapes available
        before any training: local bumps whose position/width follow the
        (adapted) knot spacing.
    """
    lo, hi = _dim_range(kan_layer, in_index)
    device = kan_layer.grid.device
    x = torch.linspace(lo, hi, n_points, device=device)
    with torch.no_grad():
        basis = B_batch(x[:, None], kan_layer.grid[in_index : in_index + 1], k=kan_layer.k)
    return x.cpu().numpy(), basis[:, 0, :].cpu().numpy()


@dataclass(frozen=True)
class EdgeCurve:
    """One learned edge function ``phi_ij`` sampled on a 1-D input sweep.

    Attributes:
        in_index / out_index: the edge ``(i -> j)`` within its layer.
        x: sweep over input dimension ``i``'s interior knot range.
        spline: the *learned* part, ``mask * scale_sp * spline_ij(x)``.
        base: the residual part, ``mask * scale_base * silu(x)`` (fixed shape,
            learned magnitude) -- analogous to a skip connection.
        full: ``spline + base``, the complete activation on the edge.
    """

    in_index: int
    out_index: int
    x: np.ndarray
    spline: np.ndarray
    base: np.ndarray
    full: np.ndarray


def edge_curve(kan_layer, in_index: int, out_index: int, n_points: int = 256) -> EdgeCurve:
    """Sample the full learned activation ``phi_ij`` on edge ``(i -> j)``.

    This is *the* core KAN interpretability object: where an MLP has a
    single scalar weight per edge, this curve shows the entire univariate
    function the KAN learned there (e.g. a saturating S-shape, a bump
    around a particular activation value, or a near-identity).
    """
    lo, hi = _dim_range(kan_layer, in_index)
    device = kan_layer.grid.device
    x = torch.linspace(lo, hi, n_points, device=device)
    with torch.no_grad():
        spline = coef2curve(
            x[:, None],
            kan_layer.grid[in_index : in_index + 1],
            kan_layer.coef[in_index : in_index + 1, out_index : out_index + 1],
            kan_layer.k,
        )[:, 0, 0]
        mask = kan_layer.mask[in_index, out_index]
        spline_part = kan_layer.scale_sp[in_index, out_index] * spline * mask
        base_part = kan_layer.scale_base[in_index, out_index] * kan_layer.base_fun(x) * mask
    return EdgeCurve(
        in_index=in_index,
        out_index=out_index,
        x=x.cpu().numpy(),
        spline=spline_part.cpu().numpy(),
        base=base_part.cpu().numpy(),
        full=(spline_part + base_part).cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Data-dependent components: forward activations and importance measures
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_capture(model, x: torch.Tensor) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    """Run the model and capture what each spline layer actually receives.

    Returns:
        ``{"layer_inputs": [x_0, x_1, x_2, x_3], "logits": logits}`` where
        ``x_l`` is the input to layer ``l`` (so ``x_0`` is the embedding
        batch and ``x_3`` feeds the logits layer). These are the
        "spline-transformed outputs during the forward pass": each
        ``x_{l+1}`` is the sum of layer ``l``'s edge functions.
    """
    was_training = model.training
    model.eval()
    x = x[:MAX_FORWARD_SAMPLES]
    inputs: list[torch.Tensor] = []
    h = x
    for layer in model.layers:
        inputs.append(h)
        h = layer(h)
    if was_training:
        model.train()
    return {"layer_inputs": inputs, "logits": h}


def layer_importances(model, x: torch.Tensor) -> list[dict[str, np.ndarray]]:
    """Score every input dimension of every layer by three importance measures.

    For each of the four layers, returns a dict of per-input-dimension
    scores (each an array of length ``in_dim``):

    * ``coef_magnitude`` -- mean `edge_strengths` over outgoing edges: how
      much learned spline mass sits on this dimension. Data-independent.
    * ``activation_variance`` -- variance of the values this dimension
      actually takes on the batch ``x``. A dimension that barely varies
      cannot drive the output regardless of its spline shape.
    * ``gradient`` -- mean ``|d logit_pred / d x_l[dim]|`` (saliency of the
      predicted-class logit w.r.t. this layer's input). Captures how
      sensitive the decision is to the dimension *in context*.

    High-dimensional layers (the 768-dim embedding input in particular)
    should be visualised only on the top-k dimensions under one of these
    measures -- see `select_top_k_dims`.
    """
    was_training = model.training
    model.eval()

    # Single differentiable pass (small batch: KAN forwards are memory-heavy).
    x_grad = x[:MAX_GRADIENT_SAMPLES].clone().requires_grad_(True)
    acts: list[torch.Tensor] = [x_grad]
    h = x_grad
    for layer in model.layers:
        h = layer(h)
        acts.append(h)
    # Saliency target: the logit of whichever class the model predicts.
    score = h.max(dim=1).values.sum()
    grads = torch.autograd.grad(score, acts[:-1])

    results: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        captured = forward_capture(model, x)
        for layer_idx, (_name, _wrapper, kan_layer) in enumerate(iter_kan_layers(model)):
            layer_input = captured["layer_inputs"][layer_idx]
            # Non-B-spline layers (RBF/thin-plate in mixed-spline models) have
            # no pykan coefficients; score them on the data-dependent measures
            # only, with zero spline mass.
            coef_magnitude = (
                edge_strengths(kan_layer).mean(axis=1)
                if kan_layer is not None
                else np.zeros(layer_input.shape[1])
            )
            results.append({
                "coef_magnitude": coef_magnitude,
                "activation_variance": layer_input.var(dim=0).cpu().numpy(),
                "gradient": grads[layer_idx].abs().mean(dim=0).cpu().numpy(),
            })
    if was_training:
        model.train()
    return results


def select_top_k_dims(
    importance: dict[str, np.ndarray], k: int, measure: str = "combined"
) -> list[int]:
    """Pick the ``k`` most important input dimensions of one layer.

    Args:
        importance: one entry of `layer_importances` output.
        k: number of dimensions to keep (clamped to ``in_dim``).
        measure: one of the keys in ``importance``, or ``"combined"`` --
            the mean of all measures after min-max normalisation, so that a
            dimension must rank well on several criteria to be selected.
    """
    if measure == "combined":
        normalised = []
        for values in importance.values():
            span = values.max() - values.min()
            normalised.append(
                (values - values.min()) / span if span > 0 else np.zeros_like(values)
            )
        scores = np.mean(normalised, axis=0)
    else:
        scores = importance[measure]
    k = min(k, len(scores))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]
