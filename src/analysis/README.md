# KAN Spline Interpretability (`src/analysis`)

Tools for inspecting and visualizing what the four `AdaptiveSplineKANLayer`
stages of the KAN sentiment classifier actually learn. The model architecture
is untouched — everything here is read-only analysis on top of the pykan
tensors (`grid`, `coef`, `scale_base`, `scale_sp`).

## Quick start

```bash
# After any KAN training run (visualization is on by default):
python src/train.py train.stages=[kan]                 # figures land in outputs/kan_spline_visualizations/

# Re-analyze a saved checkpoint without retraining:
python -m src.analysis.run_spline_analysis
python -m src.analysis.run_spline_analysis --embeddings-dir data/fusion_inputs/embeddings --top-k 12

# Live spline monitoring during training:
python src/train.py train.stages=[kan] train.kan.enable_tensorboard=true
tensorboard --logdir outputs/tensorboard/kan

# Turn the report off entirely:
python src/train.py train.stages=[kan] train.kan.enable_spline_visualization=false
```

Standalone (non-Hydra) equivalents: `python src/models/kan/train.py
--enable-tensorboard` / `--no-enable-spline-visualization`.

## Files

| File | Purpose |
|---|---|
| `spline_extractor.py` | Read-only access to knots, B-spline bases, coefficients, edge functions, forward activations, and per-dimension importance measures |
| `spline_visualizer.py` | Publication-style figures (grayscale, serif, 300 DPI); `generate_all_visualizations` builds the full report |
| `tensorboard_logger.py` | `SplineTensorBoardLogger` — failure-proof optional TB logging during training |
| `run_spline_analysis.py` | Standalone CLI over saved checkpoints |

## What each figure means

All figures are written under `outputs/kan_spline_visualizations/`, with one
`layer<n>_<name>/` subdirectory per KAN layer.

| Figure | Interpretation |
|---|---|
| `knot_grid.png` | Positions of the adaptive B-spline knots per (top-k) input dimension. Non-uniform tick spacing is direct evidence of grid adaptation: knots concentrate where that dimension's activations are dense, i.e. where spline resolution is spent. |
| `basis_functions.png` | The `G + k` cubic B-spline basis bumps over one dimension's knot grid — the "vocabulary" of shapes every learned edge function on that dimension is mixed from. |
| `spline_curves.png` | The strongest learned edge functions `φ_ij(x)`, decomposed into learned spline part (dashed), fixed SiLU residual (dotted), and their sum (solid). These curves are the KAN's replacement for scalar weights. |
| `coefficient_heatmap.png` | Per-edge spline strength `\|scale_sp\|·‖coef‖₂` as a grayscale matrix. Dark cells = edges carrying real learned nonlinearity; pale rows = ignored inputs (self-pruning). |
| `before_after_training.png` | Edge functions at initialization (grids seeded, zero gradient steps; dashed) vs. after training (solid). The gap is literally what gradient descent learned. Requires the `kan_model_init.pt` snapshot saved during training. |
| `input_feature_response.png` | End-to-end partial dependence: sweep one DeBERTa embedding dimension over its observed range (others fixed at their mean) and plot P(positive). Shows the *net* nonlinear semantic transformation of the full 4-layer stack. |
| `layer_comparison.png` | Top edge functions of all four layers side by side — reveals the division of labour across depth (compressive early curves vs. sharper class-carving curves late). |
| `selected_dimensions.json` | Which dimensions/edges were selected for plotting and their importance scores, for traceability. |

## Top-k dimension selection

Embedding inputs are 768-dimensional, so figures never show every dimension.
`spline_extractor.layer_importances` scores each input dimension of each
layer by three measures — **coefficient magnitude** (mean edge strength),
**activation variance** (on a real data batch), and **gradient saliency**
(|∂ predicted-class logit / ∂ dimension|) — and `select_top_k_dims` ranks by
their min-max-normalized average (`measure="combined"`), or any single
measure by name.

## Guarantees

- **Training can never be broken by this module**: the training script only
  imports it lazily behind config flags, the post-training report is wrapped
  in try/except, and the TensorBoard logger disables itself on any internal
  error instead of raising.
- With `enable_spline_visualization=false` and `enable_tensorboard=false`,
  the KAN training path is byte-identical to its previous behaviour.
