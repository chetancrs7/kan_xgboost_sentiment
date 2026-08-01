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
``adapt_grid`` convenience method).

The four *role* classes used by the model are deliberately **not** the same
building block -- each stage uses the spline family that matches its job:

* Layer 1 ``SemanticCompressionLayer`` -- B-spline (``AdaptiveSplineKANLayer``)
* Layer 2 ``ContextInteractionLayer``  -- B-spline (``AdaptiveSplineKANLayer``)
* Layer 3 ``SentimentDistortionLayer`` -- thin-plate spline (``r^2 log r``)
* Layer 4 ``EmotionHotspotLayer``      -- Gaussian RBF spline (``exp(-r^2/2)``)

The thin-plate and RBF layers share :class:`RadialBasisKANLayer`, which
realises the same ``y_j = sum_i phi_ij(x_i)`` Kolmogorov-Arnold form using a
grid of radial centres instead of a B-spline knot vector. All four expose the
identical ``forward`` / ``adapt_grid`` contract, so they remain drop-in
interchangeable and every layer's grid still adapts to the data.
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
        # Retained so the grid can be *grown* later (see ``refine``) without the
        # caller having to remember the construction arguments.
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = list(grid_range)
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

    @torch.no_grad()
    def refine(self, new_grid_size: int, x: torch.Tensor) -> bool:
        """Grow the number of B-spline knots to ``new_grid_size`` intervals.

        Grid refinement is the KAN answer to "spline complexity should increase
        where the data is dense": a finer knot grid is fitted to reproduce the
        current edge functions on ``x`` (via pykan's
        ``initialize_grid_from_parent``), so the layer *keeps what it learned*
        but gains resolution. Combined with ``adapt_grid`` (which places those
        extra knots at the data's quantiles) the new resolution lands exactly in
        the high-density regions. Creates new parameters, so the optimiser must
        be rebuilt afterwards (the training loop handles this).

        Guarded: if the transfer would produce non-finite weights or outputs,
        the layer keeps its current grid and returns ``False`` rather than
        corrupting the model. Returns ``True`` when the refinement was applied.
        """
        if new_grid_size <= self.grid_size:
            return False
        device = self.kan_layer.coef.device
        parent = self.kan_layer
        finer = KANLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            num=int(new_grid_size),
            k=self.spline_order,
            grid_range=self.grid_range,
            sp_trainable=True,
            sb_trainable=True,
        ).to(device)
        # Transfers the parent's learned function onto the finer knot grid.
        try:
            finer.initialize_grid_from_parent(parent, x)
        except Exception:
            return False  # pykan solve failed -> keep the current grid
        self.kan_layer = finer
        # Verify the finer grid reproduced a finite function; else revert.
        if not (torch.isfinite(finer.coef).all() and torch.isfinite(self.forward(x)).all()):
            self.kan_layer = parent
            return False
        self.grid_size = int(new_grid_size)
        return True


class RadialBasisKANLayer(nn.Module):
    """KAN layer whose per-edge functions are built from *radial* bases.

    Like :class:`AdaptiveSplineKANLayer` this realises the Kolmogorov-Arnold
    form ``y_j = sum_i phi_ij(x_i)``, but each univariate edge function is a
    weighted sum of radial kernels ``K(|x_i - c_g|)`` placed at a shared grid
    of centres ``c_g`` (plus a smooth ``SiLU`` base term)::

        phi_ij(x) = base_ij * silu(x) + sum_g coef_ijg * K((x - c_g) / h)

    This is the standard RBF-network view of a learnable univariate function
    and is the natural non-B-spline instantiation of a KAN edge. Subclasses
    only supply the kernel ``K`` via :meth:`_kernel`; everything else
    (centres, bandwidth, adaptive refitting, the base term) is shared.

    The constructor signature intentionally matches
    :class:`AdaptiveSplineKANLayer` so the two are drop-in interchangeable
    inside :class:`~src.models.kan.model.KANSentimentClassifier`:

    * ``grid_size`` sets the number of radial centres (``grid_size + 1``).
    * ``spline_order`` is accepted for signature parity and ignored (radial
      bases have no polynomial order).
    * ``grid_range`` is the initial span the centres are spread over, later
      refit to the real activation quantiles by :meth:`adapt_grid`.
    """

    #: Human-readable kernel name, surfaced in ``__repr__`` and diagnostics.
    kernel_name: str = "radial"
    _EPS: float = 1e-6

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
        self.num_centers = max(2, int(grid_size) + 1)
        self.spline_order = spline_order  # kept for API parity; unused

        centers = torch.linspace(grid_range[0], grid_range[1], self.num_centers)
        init_h = float(
            (grid_range[1] - grid_range[0]) / max(1, self.num_centers - 1)
        )
        # Centres and bandwidth are *buffers*: like the B-spline knot grid they
        # are refit to the data by ``adapt_grid`` rather than trained by SGD.
        self.register_buffer("centers", centers)
        self.register_buffer("h", torch.tensor(max(init_h, self._EPS)))

        # Trainable pieces: one coefficient per (edge, centre) and a base scale
        # per edge. Small init keeps the radial branch quiet early in training
        # so the well-behaved SiLU base term dominates until the coefficients
        # learn useful shapes (important for the unbounded thin-plate kernel).
        self.coef = nn.Parameter(
            torch.randn(in_dim, out_dim, self.num_centers) * (0.1 / self.num_centers)
        )
        self.base_weight = nn.Parameter(
            torch.randn(in_dim, out_dim) * (in_dim ** -0.5)
        )

    def _kernel(self, r: torch.Tensor) -> torch.Tensor:
        """Radial kernel evaluated on the *normalised* distance ``r = (x-c)/h``."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> normalised distances to every centre: [B, in_dim, G]
        r = (x.unsqueeze(-1) - self.centers.view(1, 1, -1)) / self.h
        basis = self._kernel(r)  # [B, in_dim, G]
        # Sum the radial edge functions into each output unit.
        spline = torch.einsum("big,iog->bo", basis, self.coef)
        base = torch.nn.functional.silu(x) @ self.base_weight  # [B, out_dim]
        return base + spline

    @torch.no_grad()
    def adapt_grid(self, x: torch.Tensor) -> None:
        """Refit the radial centres to the quantiles of real activations.

        The RBF/TPS analogue of ``update_grid_from_samples``: the centres move
        to where this layer's input actually lives (denser centres in denser
        regions), and the bandwidth ``h`` tracks the new mean centre spacing.
        """
        flat = x.reshape(-1).float()
        if flat.numel() < self.num_centers:
            return
        qs = torch.linspace(0.0, 1.0, self.num_centers, device=flat.device)
        new_centers = torch.quantile(flat, qs)
        # Guard against duplicate centres when activations are near-constant.
        if torch.allclose(new_centers[0], new_centers[-1]):
            return
        self.centers.copy_(new_centers.to(self.centers.dtype))
        spacing = (self.centers[1:] - self.centers[:-1]).abs().mean()
        self.h.copy_(spacing.clamp_min(self._EPS).to(self.h.dtype))

    @torch.no_grad()
    def refine(self, new_grid_size: int, x: torch.Tensor) -> bool:
        """Grow the number of radial centres and refit, preserving the function.

        The radial analogue of B-spline grid refinement. Because every edge in
        the layer shares one set of centres, the current edge functions are
        sampled on a 1-D sweep and a single **ridge-regularised** solve transfers
        them onto a larger, data-quantile-placed centre set — so the layer keeps
        its learned shape but gains resolution where the activations are dense.

        The ridge term is essential for the thin-plate kernel: its ``r^2 log|r|``
        basis is unbounded, so on real (wide-range) activations a raw
        least-squares fit is ill-conditioned and can return huge/NaN
        coefficients. The solve is regularised and then verified: if the result
        is non-finite the layer keeps its current centres and returns ``False``
        rather than corrupting the model. Returns ``True`` when applied.
        """
        new_num = max(2, int(new_grid_size) + 1)  # mirror __init__: grid_size + 1
        if new_num <= self.num_centers:
            return False
        device = self.coef.device
        flat = x.reshape(-1).float().to(device)
        if flat.numel() < new_num or not torch.isfinite(flat).all():
            return False

        # Sweep points spanning the bulk of the data range.
        lo, hi = torch.quantile(flat, torch.tensor([0.01, 0.99], device=device))
        if torch.allclose(lo, hi):
            return False
        pts = torch.linspace(float(lo), float(hi), 256, device=device)

        # Current edge functions on the sweep: [P, in_dim, out_dim].
        old_basis = self._kernel((pts.unsqueeze(-1) - self.centers.view(1, -1)) / self.h)
        old_spline = torch.einsum("pg,iog->pio", old_basis, self.coef)

        # New, denser centres placed at the data quantiles.
        qs = torch.linspace(0.0, 1.0, new_num, device=device)
        new_centers = torch.quantile(flat, qs)
        new_h = (new_centers[1:] - new_centers[:-1]).abs().mean().clamp_min(self._EPS)
        new_basis = self._kernel((pts.unsqueeze(-1) - new_centers.view(1, -1)) / new_h)

        # Ridge-regularised normal equations solve all edges at once (shared
        # basis): (BᵀB + λI) C = Bᵀ target. λ scales with the basis magnitude so
        # it stays meaningful for the unbounded thin-plate kernel.
        target = old_spline.reshape(pts.shape[0], -1)          # [P, in*out]
        bt_b = new_basis.T @ new_basis                          # [G_new, G_new]
        lam = 1e-3 * torch.diagonal(bt_b).mean().clamp_min(self._EPS)
        eye = torch.eye(new_num, device=device, dtype=bt_b.dtype)
        try:
            sol = torch.linalg.solve(bt_b + lam * eye, new_basis.T @ target)
        except Exception:
            return False
        new_coef = sol.reshape(new_num, self.in_dim, self.out_dim).permute(1, 2, 0)
        if not torch.isfinite(new_coef).all():
            return False  # ill-conditioned fit -> keep current centres

        # Tentatively apply, then verify the forward pass is finite; else revert.
        old_state = (self.num_centers, self.centers, self.h, self.coef)
        self.num_centers = new_num
        self.centers = new_centers.to(self.centers.dtype)  # buffer: reassign to resize
        self.h = new_h.to(self.h.dtype)
        self.coef = nn.Parameter(new_coef.contiguous().to(device))
        if not torch.isfinite(self.forward(x)).all():
            self.num_centers, self.centers, self.h, self.coef = old_state
            return False
        return True

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"num_centers={self.num_centers}, kernel={self.kernel_name}"
        )


class ThinPlateSplineKANLayer(RadialBasisKANLayer):
    """Radial KAN layer using the thin-plate-spline kernel ``U(r) = r^2 log|r|``.

    The thin-plate spline is the classical minimum-curvature interpolant; its
    radial basis ``r^2 log r`` bends smoothly and globally, which is why it is
    a good fit for *warping* an input space rather than carving local bumps.
    Here it is applied per edge on the normalised distance ``r = (x - c) / h``,
    written as ``0.5 * r^2 * log(r^2 + eps)`` for a numerically safe evaluation
    at ``r = 0``.
    """

    kernel_name = "thin_plate_spline"

    def _kernel(self, r: torch.Tensor) -> torch.Tensor:
        r2 = r * r
        return 0.5 * r2 * torch.log(r2 + self._EPS)


class GaussianRBFKANLayer(RadialBasisKANLayer):
    """Radial KAN layer using the Gaussian RBF kernel ``exp(-r^2 / 2)``.

    Gaussian radial bases are *localised*: each centre responds only to inputs
    near it, so the layer learns a set of soft "hotspots" in its input space
    and turns them on/off per output class. The response is bounded in
    ``(0, 1]``, which makes this the stable, well-conditioned counterpart to
    the global thin-plate layer.
    """

    kernel_name = "gaussian_rbf"

    def _kernel(self, r: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * r * r)


class SemanticCompressionLayer(AdaptiveSplineKANLayer):
    """Layer 1: compresses raw DeBERTa embeddings into a compact semantic code.

    Takes the full-width pretrained embedding (e.g. 768-dim DeBERTa-v3
    hidden state) and reduces it to ``out_dim`` learned semantic components,
    each a nonlinear **B-spline** combination of the input dimensions rather
    than a plain linear projection.
    """


class ContextInteractionLayer(AdaptiveSplineKANLayer):
    """Layer 2: models interactions between the compressed semantic features.

    Every output unit is a **B-spline**-weighted mixture of *all* semantic
    components from layer 1, letting the network capture feature
    co-occurrence patterns (e.g. negation interacting with intensifiers)
    that a single compressed dimension cannot express alone.
    """


class SentimentDistortionLayer(ThinPlateSplineKANLayer):
    """Layer 3: warps contextual features into sentiment-salient directions.

    Implemented with **thin-plate splines** so the transformation is exactly
    the smooth, global "distortion" the name implies: the ``r^2 log r`` basis
    stretches, compresses, or folds the context representation so
    sentiment-relevant structure (polarity, intensity) becomes more linearly
    separable for the final classification layer.
    """


class EmotionHotspotLayer(GaussianRBFKANLayer):
    """Layer 4: final classification layer producing per-class logits.

    Implemented with **radial basis function (Gaussian) splines**: each RBF
    centre is an "emotional hotspot" in the distorted sentiment space, and the
    per-class output weights turn the relevant hotspots on for each class. Its
    ``out_dim`` equals the number of sentiment classes and its raw output is
    used directly as logits for cross-entropy / softmax.
    """
