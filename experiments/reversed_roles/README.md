# Reversed-roles experiment

**Question:** what if the two experts swap data modalities — KAN on the
**structural/linguistic** features and XGBoost on the **semantic DeBERTa
embeddings** — instead of the production assignment (KAN→embeddings,
XGBoost→structural)?

This directory answers it **without touching any existing code**. It reuses the
production feature extractor, the production XGBoost Optuna/fit/eval helpers, and
the *unmodified* KAN trainer. The only trick is the clean seam: the KAN loader
already reads `{split}_embeddings.npy` + `{split}_labels.npy`, so we write the
**structural** matrix under those filenames and point the existing KAN trainer at
it.

| | Production | This experiment (reversed) |
|---|---|---|
| KAN input | DeBERTa embeddings (768-d, dense) | Structural features (50-d, standardized) |
| XGBoost input | Structural features (50-d) | DeBERTa embeddings (768-d) |

Outputs are written to separate directories, so nothing overwrites your existing
`models/kan*` or `models/xgboost*` artifacts:
- `data/reversed_roles/structural_features/` — reversed KAN inputs
- `models/reversed_roles/kan_on_structural/` — reversed KAN
- `models/reversed_roles/xgboost_on_embeddings/` — reversed XGBoost

## Two dataset scales

| Scale | Split CSVs | Embeddings dir | Rows (tr/va/te) |
|---|---|---|---|
| 100k sample | `data/splits/` | `data/embeddings/deberta_v3_base/` | 80k / 10k / 10k |
| **Full 1.28M** | `data/splitsfull/` | `data/embeddingsfull/` | 1,280,000 / 160,000 / 160,000 |

Both the full split CSVs and the matching full DeBERTa embeddings already exist
on disk, so **no regeneration is needed** — just point the flags at the `*full`
directories.

## Run on the FULL 1.28M dataset

Run from the repo root with the project venv active. Output dirs are suffixed
`_full` so they don't collide with the 100k-sample run.

```bash
# 1. Build reversed KAN inputs from the full split CSVs (1.28M rows).
#    Heaviest cost is the spaCy dependency parse — expect ~1-3 h on CPU.
python experiments/reversed_roles/build_structural_features.py \
    --train-path data/splitsfull/train.csv \
    --val-path data/splitsfull/val.csv \
    --test-path data/splitsfull/test.csv \
    --output-dir data/reversed_roles/structural_features_full

# 2. Reversed KAN on the full structural features.
#    Use THIS trainer, not src/models/kan/train.py — the production KAN NaNs on
#    structural input (see "The layer-3 NaN" below). This one uses a bounded
#    layer 3 and skips the destabilising grid-seed refit.
python experiments/reversed_roles/train_kan_on_structural.py \
    --embeddings-dir data/reversed_roles/structural_features_full \
    --output-dir models/reversed_roles/kan_on_structural_full \
    --batch-size 512

# 3. Reversed XGBoost on the full 1.28M DeBERTa embeddings.
#    This is the expensive step: each Optuna trial fits XGBoost on 1.28M x 768.
#    Lower --n-trials to keep it tractable on CPU (10 is a reasonable start).
python experiments/reversed_roles/train_xgboost_on_embeddings.py \
    --embeddings-dir data/embeddingsfull \
    --output-dir models/reversed_roles/xgboost_on_embeddings_full \
    --n-trials 10
```

## Run on the 100k sample (faster smoke of the whole idea)

```bash
python experiments/reversed_roles/build_structural_features.py
python experiments/reversed_roles/train_kan_on_structural.py \
    --embeddings-dir data/reversed_roles/structural_features \
    --output-dir models/reversed_roles/kan_on_structural
python experiments/reversed_roles/train_xgboost_on_embeddings.py \
    --embeddings-dir data/embeddings/deberta_v3_base \
    --output-dir models/reversed_roles/xgboost_on_embeddings --n-trials 50
```

Compare `models/reversed_roles/*/metrics.json` against your production
`models/kan*/metrics.json` and `models/xgboost*/metrics.json` (same splits, same
evaluator, so test accuracy / macro-F1 / AUC line up 1:1).

## The layer-3 NaN (why there's a dedicated KAN trainer)

Feeding structural features to the production `KANSentimentClassifier` via
`src/models/kan/train.py` produces `train_loss=nan` from epoch 1. Two distinct
causes, both fixed by `train_kan_on_structural.py` +
`StructuralKANClassifier` — the production code is left untouched:

1. **Unbounded layer 3.** Layer 3 is a thin-plate spline (`r^2 log|r|`). On
   spiky structural inputs its activation grows without bound (traced: abs-max
   9 → 19 → 26 → … → inf) while the B-spline layers 1-2 and the Gaussian-RBF
   layer 4 stay flat. Fix: `StructuralKANClassifier` swaps **only** layer 3 for a
   bounded B-spline (same block as layers 1-2).

2. **The grid-seed refit destabilises training.** Before training, the
   production trainer calls `update_grids` (pykan `update_grid_from_samples` →
   `curve2coef`, a least-squares fit) to place the B-spline knots at the data
   quantiles. On the discrete, heavily-tied structural values this produces a
   grid geometry on which B-spline *training* diverges on the very first
   gradient step — the seeded model is finite, but one step blows up. Trained on
   the plain **uniform grid** (refit skipped) the same model is stable and
   learns (~0.72 val acc in 40 steps). So the dedicated trainer skips the refit
   by default (`--seed-grids` re-enables it, guarded) and also wraps each step in
   a finite-guard that rolls back any non-finite update.

None of this affects the production KAN-on-embeddings path, where smooth
embeddings keep both the thin-plate layer and the grid refit well-behaved.

## Notes

- **Feature scaling (and why clipping is mandatory).** The structural columns
  are heavy-tailed count features: e.g. `hashtag_neg_count` is nonzero in ~0.06%
  of rows, so a plain `StandardScaler` turns those few rows into ~260-sigma
  outliers. The KAN's thin-plate (`r^2 log|r|`) and RBF layers are *unbounded*,
  so those outliers make the forward activations explode and the loss go **NaN**
  within the first epoch (gradient clipping can't help — the blow-up is in the
  forward pass, not the gradient). The default `--scaler standard --clip 5.0`
  clips standardized features to +/-5 (touches ~0.37% of cells) and fixes it.
  `--scaler quantile` is an alternative that's naturally bounded and arguably a
  better fit for spiky discrete features. XGBoost-on-embeddings gets the raw
  embeddings — trees are scale-invariant, mirroring how the production KAN eats
  raw embeddings.
- The `.pkl`/`.json` provenance (feature extractor, scaler, feature names,
  manifest) is saved alongside the arrays for reproducibility and interpretation.
- Expectation: this reversal should *underperform* the production assignment —
  it fights both models' inductive biases (splines on discrete count columns;
  axis-aligned tree splits on rotation-encoded embedding dims). Running it turns
  that argument into a measured ablation.
