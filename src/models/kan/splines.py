"""Spline registry / factory for the mixed-spline KAN.

Each KAN layer replaces a weight matrix with a matrix of learnable
univariate functions phi_ij. This module provides several *families* of
basis functions for those phi_ij and a factory that builds a layer of the
requested family, so each of the model's four semantic stages can use the
spline type best matched to its role:

======================  =====================================================
Registry name           Basis / intended role
======================  =====================================================
``bspline``             pykan B-splines of order ``spline_order`` (default 3)
                        on an adaptive knot grid. Stable, smooth, global
                        support per interval -- suited to compressing
                        transformer embeddings (layer 1).
``cubic_bspline``       B-splines with the order pinned to cubic (k=3); pair
                        with a finer ``grid_size`` to smoothly model subtle
                        contextual shifts such as negation and modifiers
                        (layer 2).
``thin_plate``          Thin-plate radial basis functions phi(r) = r^2 ln r
                        over data-adapted centres. Very flexible, minimally
                        curved interpolants for complex nonlinear sentiment
                        distortion (layer 3). This is the 1-D per-edge RBF
                        *approximation* of a thin-plate spline; a true
                        multivariate TPS does not fit the KAN edge-function
                        factorisation.
``rbf``                 Gaussian radial basis functions. Strictly localised
                        bumps -- suited to detecting "emotion hotspots",
                        i.e. strong responses in narrow regions of the
                        activation space (layer 4).
``linear``              A plain ``nn.Linear`` head. No spline capacity at
                        all -- the low-variance output classifier.
======================  =====================================================

Every layer built here exposes the same minimal interface the training
pipeline relies on:

* ``forward(x: (B, in_dim)) -> (B, out_dim)``
* ``adapt_grid(x)`` -- refit knots / centres to a batch of real activations
  (a no-op where it does not apply, e.g. ``linear``).
* ``in_dim`` / ``out_dim`` attributes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.kan.layers import AdaptiveSplineKANLayer  # noqa: E402

#: Fraction of uniform spacing blended into the quantile-based centre update,
#: mirroring pykan's ``grid_eps`` so centres never collapse onto ties.
_CENTER_UNIFORM_BLEND = 0.02


class RadialBasisKANLayer(nn.Module):
    """KAN layer whose edge functions are radial-basis expansions.

    Each edge computes

        phi_ij(x_i) = scale_base_ij * silu(x_i)
                      + scale_sp_ij * sum_c coef[i, j, c] * K(|x_i - center_ic| / width_i)

    with kernel ``K`` either a Gaussian bump ``exp(-r^2)`` (``kernel=
    "gaussian"``) or the thin-plate kernel ``r^2 ln r`` (``kernel=
    "thin_plate"``), and outputs ``y_j = sum_i phi_ij(x_i)`` -- the same
    edge-function structure as the pykan B-spline layer, only with a radial
    basis instead of a B-spline basis.

    Centres and widths play the role of the B-spline knot grid: they are
    non-trainable parameters (so they persist in checkpoints) refit to the
    empirical activation distribution by ``adapt_grid``, while ``coef`` and
    the two scales are learned by backprop.

    Args:
        in_dim: Input feature dimensionality.
        out_dim: Output feature dimensionality.
        num_centers: Radial basis functions per input dimension. The factory
            defaults this to ``grid_size + spline_order`` so RBF layers get
            the same per-edge coefficient count as the B-spline layers they
            replace (fair ablation comparisons).
        grid_range: Initial ``(min, max)`` range the centres span, later
            refit by ``adapt_grid``.
        kernel: ``"gaussian"`` or ``"thin_plate"``.
    """

    KERNELS = ("gaussian", "thin_plate")

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_centers: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        kernel: str = "gaussian",
    ) -> None:
        super().__init__()
        if kernel not in self.KERNELS:
            raise ValueError(f"Unknown radial kernel '{kernel}'. Expected one of {self.KERNELS}.")
        if num_centers < 2:
            raise ValueError(f"num_centers must be >= 2, got {num_centers}.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_centers = num_centers
        self.kernel = kernel

        lo, hi = float(grid_range[0]), float(grid_range[1])
        centers = torch.linspace(lo, hi, num_centers).repeat(in_dim, 1)
        width = (hi - lo) / (num_centers - 1)
        # Non-trainable parameters (not buffers) to match how pykan stores its
        # knot grid: saved in the state dict, moved with .to(device), but
        # updated only by adapt_grid, never by the optimizer.
        self.centers = nn.Parameter(centers, requires_grad=False)
        self.widths = nn.Parameter(torch.full((in_dim, 1), width), requires_grad=False)

        # Init mirrors pykan's KANLayer scales: small random spline
        # contribution, 1/sqrt(in_dim) so the sum over in_dim edges keeps
        # unit-order variance.
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_centers) * 0.1 / num_centers**0.5)
        self.scale_base = nn.Parameter(
            (torch.rand(in_dim, out_dim) * 2 - 1) / in_dim**0.5
        )
        self.scale_sp = nn.Parameter(torch.full((in_dim, out_dim), 1.0 / in_dim**0.5))
        self.base_fun = nn.SiLU()

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the radial basis, ``(B, in_dim) -> (B, in_dim, num_centers)``."""
        r = (x.unsqueeze(-1) - self.centers) / self.widths
        if self.kernel == "gaussian":
            return torch.exp(-r.pow(2))
        # Thin-plate kernel r^2 * ln(r): the eps keeps ln finite at r == 0,
        # where the r^2 factor makes the correct limit 0 anyway.
        r = r.abs()
        return r.pow(2) * torch.log(r + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._basis(x)  # (B, in, C)
        spline = torch.einsum("bic,ioc->bio", basis, self.coef)  # (B, in, out)
        y = torch.einsum("bio,io->bo", spline, self.scale_sp)
        y = y + self.base_fun(x) @ self.scale_base
        return y

    @torch.no_grad()
    def adapt_grid(self, x: torch.Tensor) -> None:
        """Refit centres/widths to a batch, analogous to B-spline knot refits.

        Centres move to the empirical quantiles of each input dimension
        (blended with a small uniform component so ties cannot collapse
        them); widths are set to the mean centre spacing so neighbouring
        kernels keep a consistent overlap after the move.
        """
        x = x.detach().to(self.centers.dtype)
        qs = torch.linspace(0.0, 1.0, self.num_centers, device=x.device, dtype=x.dtype)
        quantiles = torch.quantile(x, qs, dim=0).T  # (in_dim, C)
        lo = x.min(dim=0).values.unsqueeze(1)
        hi = x.max(dim=0).values.unsqueeze(1)
        uniform = lo + (hi - lo) * qs.unsqueeze(0)
        centers = _CENTER_UNIFORM_BLEND * uniform + (1 - _CENTER_UNIFORM_BLEND) * quantiles
        spacing = (centers[:, -1] - centers[:, 0]) / (self.num_centers - 1)
        self.centers.data = centers.to(self.centers.device)
        self.widths.data = spacing.clamp_min(1e-3).unsqueeze(1).to(self.widths.device)


class LinearKANHead(nn.Module):
    """Plain linear classification head with the KAN-layer interface.

    Deliberately spline-free: the output stage should separate the already
    sentiment-warped representation without adding overfittable nonlinear
    capacity. ``adapt_grid`` is a no-op so the head slots into
    ``KANSentimentClassifier.update_grids`` unchanged.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.no_grad()
    def adapt_grid(self, x: torch.Tensor) -> None:  # noqa: ARG002 - interface parity
        return None


def _build_bspline(in_dim, out_dim, *, grid_size, spline_order, grid_range, **_):
    return AdaptiveSplineKANLayer(in_dim, out_dim, grid_size, spline_order, grid_range)


def _build_cubic_bspline(in_dim, out_dim, *, grid_size, grid_range, **_):
    # Order pinned to cubic regardless of the global spline_order setting.
    return AdaptiveSplineKANLayer(in_dim, out_dim, grid_size, 3, grid_range)


def _build_radial(kernel):
    def build(in_dim, out_dim, *, grid_size, spline_order, grid_range, num_centers=None, **_):
        if num_centers is None:
            # Match the B-spline per-edge coefficient count (G + k).
            num_centers = grid_size + spline_order
        return RadialBasisKANLayer(in_dim, out_dim, num_centers, grid_range, kernel=kernel)

    return build


def _build_linear(in_dim, out_dim, **_):
    return LinearKANHead(in_dim, out_dim)


#: name -> builder(in_dim, out_dim, **spline_kwargs). Builders accept the
#: full common kwarg set and ignore what they do not use, so per-layer
#: overrides can be passed uniformly.
SPLINE_REGISTRY = {
    "bspline": _build_bspline,
    "cubic_bspline": _build_cubic_bspline,
    "thin_plate": _build_radial("thin_plate"),
    "rbf": _build_radial("gaussian"),
    "linear": _build_linear,
}


def build_spline_layer(
    spline_type: str,
    in_dim: int,
    out_dim: int,
    *,
    grid_size: int = 5,
    spline_order: int = 3,
    grid_range: tuple[float, float] = (-2.0, 2.0),
    num_centers: int | None = None,
) -> nn.Module:
    """Build one KAN layer of the requested spline family.

    Args:
        spline_type: A key of :data:`SPLINE_REGISTRY` (case-insensitive).
        in_dim / out_dim: Layer width.
        grid_size: B-spline grid intervals (B-spline families only).
        spline_order: B-spline order; ignored by ``cubic_bspline`` (always
            3), used as part of the default ``num_centers`` by radial types.
        grid_range: Initial knot/centre range before adaptive refitting.
        num_centers: Radial basis count (radial families only); defaults to
            ``grid_size + spline_order`` for parameter parity with B-splines.

    Raises:
        ValueError: If ``spline_type`` is not registered.
    """
    key = str(spline_type).strip().lower()
    if key not in SPLINE_REGISTRY:
        available = ", ".join(sorted(SPLINE_REGISTRY))
        raise ValueError(f"Unknown spline type '{spline_type}'. Available types: {available}")
    return SPLINE_REGISTRY[key](
        in_dim,
        out_dim,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=tuple(grid_range),
        num_centers=num_centers,
    )
