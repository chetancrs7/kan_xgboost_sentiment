# KAN Expert — Adaptive-Spline Kolmogorov-Arnold Network

A sentiment classifier built on a **Kolmogorov-Arnold Network (KAN)**: instead
of linear weight matrices, every layer is a grid of learnable univariate
B-spline functions (via [pykan](https://github.com/KindXiaoming/pykan)) that
adapt their shape during training. The model consumes **precomputed DeBERTa-v3
embeddings** (see `src/embeddings/`) and is designed as an *expert* in the
same hybrid ensemble as `src/models/xgboost`.

## Module layout

```
src/models/kan/
├── __init__.py       # package exports
├── data_loader.py     # load precomputed {split}_embeddings.npy / _labels.npy
├── layers.py          # the four adaptive-spline KAN layers
├── model.py            # KANSentimentClassifier: chains the four layers
├── evaluate.py         # metrics + softmax probs + entropy-based confidence
├── train.py             # end-to-end orchestrator (entry point)
└── README.md
```

## Architecture

Four adaptive-spline KAN layers, chained in a fixed sequence with no extra
linear/normalisation layers in between:

| # | Layer | Role | Default shape |
|---|-------|------|----------------|
| 1 | `SemanticCompressionLayer` | Reduces the raw DeBERTa embedding to a compact semantic code | 768 → 256 |
| 2 | `ContextInteractionLayer` | Mixes semantic components to capture feature interactions | 256 → 128 |
| 3 | `SentimentDistortionLayer` | Nonlinearly warps context into sentiment-salient directions | 128 → 64 |
| 4 | `EmotionHotspotLayer` | Final classification layer (logits) | 64 → n_classes |

Every layer is an instance of `AdaptiveSplineKANLayer` (`layers.py`), which
wraps `kan.KANLayer`. Each edge `i -> j` computes

```
phi_ij(x_i) = scale_base * base_fun(x_i) + scale_sp * spline_ij(x_i)
y_j          = sum_i phi_ij(x_i)
```

`spline_ij` is a B-spline over a trainable coefficient vector — gradients
from the cross-entropy loss update these coefficients directly. The spline
**grid** (knot positions) is additionally refit to the real activation
distribution via `update_grid_from_samples` (exposed as `adapt_grid` /
`KANSentimentClassifier.update_grids`), called once before training and then
periodically (`--grid-update-freq`) during training — this is what makes the
spline basis genuinely *adaptive* rather than fixed at initialisation.

## Quick start

```bash
# from the repo root
python src/models/kan/train.py
python src/models/kan/train.py --embeddings-dir data/embeddings/deberta_v3_base
python src/models/kan/train.py --epochs 300 --patience 20 --hidden-dims 256,128,64
```

Key options (`python src/models/kan/train.py --help`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--embeddings-dir` | `data/embeddings/deberta_v3_base` | Directory with `{split}_embeddings.npy` / `_labels.npy` |
| `--hidden-dims` | `256,128,64` | Output widths of layers 1-3 |
| `--grid-size` | `5` | B-spline grid intervals per layer |
| `--spline-order` | `3` | B-spline polynomial degree |
| `--epochs` | `200` | Maximum training epochs |
| `--patience` | `15` | Early-stopping patience (epochs without val-loss improvement) |
| `--grid-update-freq` | `10` | Refit spline grids every N epochs (0 disables) |

## Pipeline

1. Loads precomputed `data/embeddings/deberta_v3_base/{train,val,test}_{embeddings,labels}.npy`.
2. Builds the 4-layer `KANSentimentClassifier`.
3. Refits spline grids to the training-data distribution, then trains with
   cross-entropy loss and Adam, periodically re-adapting the grids.
4. Applies early stopping on validation loss and restores the best checkpoint
   in memory.
5. Evaluates train/val/test (accuracy, precision, recall, F1, AUC, mean
   confidence).
6. Exports probabilities, entropy-based confidence, training curves, and the
   model checkpoint.

## Outputs (`models/kan/`)

| File | Contents |
|------|----------|
| `kan_model.pt` | `{"model_state_dict", "config"}` — reloadable checkpoint |
| `training_curves.png` | Loss + accuracy curves, train vs. validation |
| `history.json` | Per-epoch train/val loss & accuracy (backs the plot) |
| `{train,val,test}_probs.npy` | `[n_samples, n_classes]` softmax probabilities |
| `{train,val,test}_confidence.npy` | `[n_samples]` entropy-based confidence in `[0, 1]` |
| `metrics.json` | Per-split metrics + best validation loss + stopped epoch + config |

Confidence is derived from the Shannon entropy of each sample's softmax
distribution, normalised by `log(n_classes)` and inverted: `confidence = 1 -
H(p) / log(n_classes)`, so a fully peaked prediction scores 1 and a uniform
(maximally uncertain) prediction scores 0.

## Reloading the model

```python
import torch
from src.models.kan import KANSentimentClassifier

ckpt = torch.load("models/kan/kan_model.pt", map_location="cpu")
model = KANSentimentClassifier(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```
