# kan_xgboost_sentiment
Capstone Project 2

To install the required libraries:
`pip install -r library_requirements.txt`

The pipeline is configured with Hydra — see [conf/README.md](conf/README.md) for configuration, CLI overrides, and multirun experiment usage.

## Mixed-spline KAN

Each of the KAN's four layers can use a different spline family, selected via
the registry in `src/models/kan/splines.py` (`bspline`, `cubic_bspline`,
`thin_plate`, `rbf`, plus a `linear` output head). The research-motivated
assignment pairs each stage with the basis suited to its role:

| Stage | Spline | Rationale |
|---|---|---|
| 1. Semantic compression | `bspline` | stable, smooth embedding compression |
| 2. Context interaction | `cubic_bspline` (finer grid) | subtle shifts: negation, contrast, modifiers |
| 3. Sentiment distortion | `thin_plate` (1-D RBF approximation) | complex nonlinear warping: mixed sentiment, sarcasm |
| 4. Emotion hotspot | `rbf` (Gaussian) | localised detection of strong emotional regions |
| Output classifier | `linear` | low capacity to avoid overfitting |

Configure via Hydra (`model.kan.spline_types`, `model.kan.output_type`) or the
standalone CLI:

```bash
python src/models/kan/train.py --hidden-dims 256,128,64,32 --output-type linear \
    --spline-types bspline,cubic_bspline,thin_plate,rbf
```

The all-`bspline` default reproduces the original architecture (and its
checkpoints) exactly.

### Ablation study

`src/models/kan/run_ablation.py` trains four spline configurations —
`all_bspline`, `bspline_cubic`, `bspline_cubic_rbf`, `full_mixed` — with
identical widths, seed, and training settings, saving per-experiment
checkpoints, metrics (accuracy, precision/recall, macro & weighted F1,
confusion matrices, loss curves, prediction probabilities) under
`models/kan_ablation/<experiment>/`, plus a cross-experiment
`ablation_summary.{json,md}`:

```bash
python src/models/kan/run_ablation.py            # 6k/2k/2k fusion subsample
python src/models/kan/run_ablation.py --experiments full_mixed --epochs 100
```
