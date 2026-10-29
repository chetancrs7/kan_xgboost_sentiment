"""Publication-style (grayscale, high-DPI) visualizations of KAN splines.

Every plotting function documents what the figure *means* for interpreting
the KAN sentiment model. All figures share one visual language:

* grayscale only (colors are gray levels ``0.0``–``0.8``), distinguishable
  in black-and-white print via line styles;
* serif fonts, inward ticks, no legend frames, 300 DPI on save;
* one figure per file so individual panels can be dropped into a paper.

The module is completely decoupled from training: it only reads model
state via :mod:`src.analysis.spline_extractor` and writes PNGs. If it is
never imported, training behaves exactly as before.

Typical use::

    from src.analysis.spline_visualizer import generate_all_visualizations
    generate_all_visualizations(model, X=train_embeddings[:256],
                                before_model=snapshot_at_init)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers / during training
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.spline_extractor import (
    LAYER_TITLES,
    basis_functions,
    edge_curve,
    edge_strengths,
    interior_knots,
    iter_kan_layers,
    layer_importances,
    select_top_k_dims,
    top_edges,
)

DEFAULT_OUTPUT_DIR = Path("outputs/kan_spline_visualizations")

# Matplotlib rc settings applied (via rc_context, i.e. without leaking
# global state into the training script's own plots) to every figure.
PUBLICATION_RC = {
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
}

# Curves cycle through gray levels and line styles together so up to ~8
# series stay distinguishable in pure black-and-white print.
_GRAYS = ("0.0", "0.35", "0.55", "0.7", "0.8")
_STYLES = ("-", "--", "-.", ":")


def _series_style(i: int) -> dict[str, str]:
    return {"color": _GRAYS[i % len(_GRAYS)], "linestyle": _STYLES[i % len(_STYLES)]}


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _grid_of_axes(n: int, ncols: int = 2, panel: tuple[float, float] = (3.4, 2.4)):
    """A subplot grid sized per-panel, with unused axes hidden."""
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


# 1. Knot grid

def plot_knot_grid(kan_layer, layer_title: str, dims: list[int], path: Path) -> Path:
    """Knot positions of selected input dimensions (one row per dimension).

    Interpretation: each tick is a B-spline knot after adaptive grid
    refitting (``update_grid_from_samples``). Uniform spacing means the
    dimension's activations are spread evenly; ticks bunching up in a region
    means the KAN concentrated spline resolution where that dimension's
    values are dense -- direct evidence of the "adaptive" part of the
    adaptive-spline layers.
    """
    knots = interior_knots(kan_layer)
    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(5.0, 0.45 * len(dims) + 1.2))
        ax.eventplot(
            [knots[d] for d in dims],
            lineoffsets=np.arange(len(dims)),
            linelengths=0.7,
            colors="black",
            linewidths=1.0,
        )
        ax.set_yticks(np.arange(len(dims)))
        ax.set_yticklabels([f"dim {d}" for d in dims])
        ax.set_xlabel("Activation value")
        ax.set_title(f"{layer_title}\nAdaptive B-spline knot positions (top input dims)")
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
    return _save(fig, path)


# 2. Basis functions

def plot_basis_functions(kan_layer, layer_title: str, in_index: int, path: Path) -> Path:
    """The B-spline basis of one input dimension, with knots marked.

    Interpretation: every learned edge function leaving this dimension is a
    weighted sum of exactly these ``G + k`` local bumps (degree-``k``
    B-splines over the adapted knot grid). Narrow bumps = fine resolution;
    the trainable coefficients only choose *how much* of each bump to use,
    so this figure shows the hypothesis space of the spline before/after
    knot adaptation.
    """
    x, basis = basis_functions(kan_layer, in_index)
    knots = interior_knots(kan_layer)[in_index]
    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        for b in range(basis.shape[1]):
            ax.plot(x, basis[:, b], **_series_style(b))
        for knot in knots:
            ax.axvline(knot, color="0.8", linewidth=0.6, zorder=0)
        ax.set_xlabel(f"Input activation (dim {in_index})")
        ax.set_ylabel("Basis value $B_b(x)$")
        ax.set_title(
            f"{layer_title}\nB-spline basis functions, input dim {in_index} "
            f"(order k={kan_layer.k}, {basis.shape[1]} bases; knots in gray)"
        )
    return _save(fig, path)


# 3. Learned edge functions (spline transformation curves)

def edge_curves_figure(kan_layer, layer_title: str, edges: list[tuple[int, int]]):
    """Build (but do not save) a small-multiples figure of learned edge functions.

    Shared by `plot_spline_curves` and the TensorBoard logger. Each panel
    decomposes one edge activation into its parts:
    solid = full ``phi_ij``, dashed = learned spline component, dotted =
    fixed SiLU residual. Where the dashed curve dominates, the KAN relies on
    its learned nonlinearity rather than the default activation.
    """
    curves = [edge_curve(kan_layer, i, j) for i, j in edges]
    fig, axes = _grid_of_axes(len(curves))
    for ax, curve in zip(axes, curves):
        ax.plot(curve.x, curve.full, color="black", linestyle="-", label="full $\\phi$")
        ax.plot(curve.x, curve.spline, color="0.45", linestyle="--", label="spline part")
        ax.plot(curve.x, curve.base, color="0.65", linestyle=":", label="SiLU residual")
        ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
        ax.set_title(f"$\\phi$: in {curve.in_index} $\\to$ out {curve.out_index}")
        ax.set_xlabel("Input activation")
        ax.set_ylabel("Edge output")
    axes[0].legend(loc="best")
    fig.suptitle(f"{layer_title}\nStrongest learned edge functions", y=1.0)
    fig.tight_layout()
    return fig


def plot_spline_curves(
    kan_layer, layer_title: str, edges: list[tuple[int, int]], path: Path
) -> Path:
    """Save the learned edge-function decomposition (see `edge_curves_figure`).

    Interpretation: these curves are the KAN's replacement for scalar
    weights -- e.g. a saturating shape acts like a soft clamp on a semantic
    feature, a bump detects a specific activation range ("hotspot"), and a
    near-linear curve means the edge degenerated to an ordinary weight.
    """
    with plt.rc_context(PUBLICATION_RC):
        fig = edge_curves_figure(kan_layer, layer_title, edges)
    return _save(fig, path)


# 4. Coefficient heatmap

def plot_coefficient_heatmap(
    kan_layer, layer_title: str, path: Path, max_rows: int = 32, max_cols: int = 32
) -> Path:
    """Grayscale heatmap of per-edge spline strength ``|scale_sp|·||coef||₂``.

    Interpretation: dark cells are edges whose learned spline carries real
    signal; pale rows are input dimensions the layer effectively ignores.
    Sparsity patterns here indicate how much the KAN pruned itself during
    training. For wide layers only the strongest ``max_rows × max_cols``
    block is shown (rows/columns ranked by their mean strength), with the
    original dimension indices kept as tick labels.
    """
    strengths = edge_strengths(kan_layer)
    row_order = np.argsort(strengths.mean(axis=1))[::-1][:max_rows]
    col_order = np.argsort(strengths.mean(axis=0))[::-1][:max_cols]
    block = strengths[np.ix_(row_order, col_order)]

    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        image = ax.imshow(block, cmap="gray_r", aspect="auto", interpolation="nearest")
        fig.colorbar(image, ax=ax, label="Edge strength $|s_{sp}|\\,\\|c\\|_2$", shrink=0.85)
        tick_fontsize = 6 if max(len(row_order), len(col_order)) > 16 else 8
        ax.set_xticks(np.arange(len(col_order)))
        ax.set_xticklabels(col_order, rotation=90, fontsize=tick_fontsize)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=tick_fontsize)
        ax.set_xlabel("Output dimension")
        ax.set_ylabel("Input dimension")
        ax.set_title(
            f"{layer_title}\nSpline coefficient strength per edge "
            f"(top {len(row_order)}×{len(col_order)} block)"
        )
    return _save(fig, path)


# 5. Before vs. after training

def plot_before_after(
    kan_layer_before,
    kan_layer_after,
    layer_title: str,
    edges: list[tuple[int, int]],
    path: Path,
) -> Path:
    """Edge functions at initialisation (dashed) vs. after training (solid).

    Interpretation: the "before" state is the model right after data-seeded
    grid initialisation but zero gradient steps -- splines are still small
    random noise around the SiLU residual. The gap between the two curves is
    literally *what training learned* on that edge; edges whose curves
    barely move were not useful for the sentiment task.
    """
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = _grid_of_axes(len(edges))
        for ax, (i, j) in zip(axes, edges):
            before = edge_curve(kan_layer_before, i, j)
            after = edge_curve(kan_layer_after, i, j)
            ax.plot(before.x, before.full, color="0.55", linestyle="--", label="before training")
            ax.plot(after.x, after.full, color="black", linestyle="-", label="after training")
            ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
            ax.set_title(f"$\\phi$: in {i} $\\to$ out {j}")
            ax.set_xlabel("Input activation")
            ax.set_ylabel("Edge output")
        axes[0].legend(loc="best")
        fig.suptitle(f"{layer_title}\nEdge functions before vs. after training", y=1.0)
        fig.tight_layout()
    return _save(fig, path)


# 6. Input feature response curves (end-to-end)

def plot_feature_response(
    model,
    X: np.ndarray,
    dims: list[int],
    path: Path,
    n_points: int = 41,
) -> Path:
    """End-to-end sentiment response to single embedding dimensions.

    For each selected embedding dimension, sweep its value across the
    1st–99th percentile of the real data while holding every other
    dimension at its mean, and plot the model's positive-class probability.

    Interpretation: this is a 1-D partial-dependence view of the whole
    4-layer spline stack -- it shows the *net* nonlinear semantic
    transformation the KAN applies to one DeBERTa embedding direction
    (e.g. saturating, thresholded, or non-monotone responses that a linear
    probe could not express). Only meaningful with real embeddings, since
    the sweep range and the frozen means come from ``X``.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    mean_vector = X.mean(axis=0)
    n_classes = model.emotion_hotspot.out_dim

    with plt.rc_context(PUBLICATION_RC):
        fig, axes = _grid_of_axes(len(dims), panel=(3.4, 2.5))
        for ax, dim in zip(axes, dims):
            low, high = np.percentile(X[:, dim], [1, 99])
            sweep = np.linspace(low, high, n_points)
            batch = np.tile(mean_vector, (n_points, 1)).astype(np.float32)
            batch[:, dim] = sweep
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(batch).to(device)), dim=1)
            probs = probs.cpu().numpy()
            if n_classes == 2:
                ax.plot(sweep, probs[:, 1], color="black")
                ax.set_ylabel("P(positive)")
            else:
                for c in range(n_classes):
                    ax.plot(sweep, probs[:, c], label=f"class {c}", **_series_style(c))
                ax.set_ylabel("Class probability")
                if dim == dims[0]:
                    ax.legend(loc="best")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(f"Embedding dim {dim}")
            ax.axhline(0.5, color="0.85", linewidth=0.6, zorder=0)
        fig.suptitle(
            "End-to-end response: sentiment probability vs. single embedding "
            "dimension\n(other dimensions fixed at their mean)",
            y=1.0,
        )
        fig.tight_layout()
    if was_training:
        model.train()
    return _save(fig, path)


# 7. Layer-wise comparison

def plot_layer_comparison(model, path: Path, edges_per_layer: int = 3) -> Path:
    """Strongest learned edge functions of all four layers, side by side.

    Interpretation: comparing spline shapes across depth shows the division
    of labour in the stack -- typically smoother, compressive curves in the
    semantic-compression layer, sharper / more localised nonlinearities in
    the sentiment-distortion and emotion-hotspot layers where class
    structure is carved out. Curves are each layer's top edges by spline
    strength, so panels are representative rather than exhaustive.
    """
    layers = list(iter_kan_layers(model))
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 2.8))
        for ax, (name, _wrapper, kan_layer) in zip(np.atleast_1d(axes), layers):
            for rank, (i, j) in enumerate(top_edges(kan_layer, k=edges_per_layer)):
                curve = edge_curve(kan_layer, i, j)
                ax.plot(curve.x, curve.full, label=f"{i}$\\to${j}", **_series_style(rank))
            ax.axhline(0.0, color="0.85", linewidth=0.6, zorder=0)
            # First line of the long title, e.g. "Layer 1 — Semantic Compression".
            ax.set_title(LAYER_TITLES[name].split(" (")[0])
            ax.set_xlabel("Input activation")
            ax.legend(loc="best")
        np.atleast_1d(axes)[0].set_ylabel("Edge output $\\phi(x)$")
        fig.suptitle("Layer-wise comparison of learned spline transformations", y=1.02)
        fig.tight_layout()
    return _save(fig, path)


# Orchestrator

def generate_all_visualizations(
    model,
    X: np.ndarray | None = None,
    before_model=None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    top_k: int = 8,
    n_response_dims: int = 6,
    n_curve_edges: int = 6,
    seed: int = 0,
) -> list[Path]:
    """Generate the full spline interpretability report for a trained KAN.

    Produces, per layer (in ``layer<n>_<name>/`` subdirectories): the knot
    grid, basis functions, learned edge curves, coefficient heatmap, and --
    when ``before_model`` is given -- before-vs-after training curves. At
    the top level it adds the end-to-end input feature response curves, the
    layer-wise comparison, and ``selected_dimensions.json`` recording which
    dimensions/edges were selected and by what importance scores.

    Args:
        model: a trained ``KANSentimentClassifier``.
        X: real embedding batch used for importance scoring and response
            sweeps. When ``None``, a standard-normal surrogate is used and
            data-dependent figures are marked as synthetic.
        before_model: deep copy of the model *before* training (grids
            seeded, no gradient steps) for the before/after figures.
        output_dir: destination root (default
            ``outputs/kan_spline_visualizations``).
        top_k: input dimensions per layer to show in knot/heatmap figures.
        n_response_dims: embedding dimensions for the response-curve figure.
        n_curve_edges: edges per layer in the curve figures.

    Returns:
        Paths of every figure written. Never raises on individual-figure
        problems is *not* guaranteed here -- callers inside training loops
        should wrap this in try/except (see ``src/models/kan/train.py``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    synthetic = X is None
    if synthetic:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((256, model.semantic_compression.in_dim)).astype(np.float32)
        print(
            "[WARN] No embeddings provided: importance scores and response "
            "curves use synthetic N(0,1) inputs and should not be interpreted."
        )
    X = np.asarray(X, dtype=np.float32)
    x_tensor = torch.from_numpy(X).to(device)

    importances = layer_importances(model, x_tensor)
    written: list[Path] = []
    summary: dict[str, object] = {"synthetic_inputs": synthetic, "n_samples": int(len(X))}

    for layer_idx, (name, _wrapper, kan_layer) in enumerate(iter_kan_layers(model)):
        title = LAYER_TITLES[name]
        subdir = output_dir / f"layer{layer_idx + 1}_{name}"
        dims = select_top_k_dims(importances[layer_idx], k=top_k)
        edges = top_edges(kan_layer, k=n_curve_edges)

        written.append(plot_knot_grid(kan_layer, title, dims, subdir / "knot_grid.png"))
        written.append(
            plot_basis_functions(kan_layer, title, dims[0], subdir / "basis_functions.png")
        )
        written.append(plot_spline_curves(kan_layer, title, edges, subdir / "spline_curves.png"))
        written.append(
            plot_coefficient_heatmap(kan_layer, title, subdir / "coefficient_heatmap.png")
        )
        if before_model is not None:
            before_layer = getattr(before_model, name).kan_layer
            written.append(
                plot_before_after(
                    before_layer, kan_layer, title, edges,
                    subdir / "before_after_training.png",
                )
            )

        summary[name] = {
            "selected_input_dims": dims,
            "importance": {
                measure: [float(values[d]) for d in dims]
                for measure, values in importances[layer_idx].items()
            },
            "top_edges": [list(edge) for edge in edges],
        }

    response_dims = select_top_k_dims(importances[0], k=n_response_dims)
    written.append(
        plot_feature_response(model, X, response_dims, output_dir / "input_feature_response.png")
    )
    written.append(plot_layer_comparison(model, output_dir / "layer_comparison.png"))

    summary_path = output_dir / "selected_dimensions.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)

    print(f"[INFO] Spline visualizations written to: {output_dir.resolve()}")
    return written
