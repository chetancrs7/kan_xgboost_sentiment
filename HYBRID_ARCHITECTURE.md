# Parallel Adaptive Hybrid Ensemble — architecture completion

This document covers three additions that make the codebase match the full
architecture spec, where each was added, and exactly what to run.

| # | Capability | Lives in |
|---|---|---|
| 1 | KAN layers 3 & 4 use **thin-plate** and **RBF** splines (not B-splines) | [`src/models/kan/layers.py`](src/models/kan/layers.py) |
| 2 | XGBoost expert gains **dependency-parse** and **hashtag-sentiment** features | [`src/models/xgboost/features.py`](src/models/xgboost/features.py) |
| 3 | **Uncertainty-aware fallback** mechanism after fusion | [`src/fusion/fallback.py`](src/fusion/fallback.py) |
| 4 | **Grid refinement** — spline complexity grows during training | [`src/models/kan/`](src/models/kan/) |
| 5 | **Quotation-mark** sarcasm cue + **emoji preservation** in cleaning | [`features.py`](src/models/xgboost/features.py), [`text_cleaning.py`](src/data/text_cleaning.py) |
| 6 | **SHAP explainability** for the XGBoost expert and the fusion gate | [`src/explainability/`](src/explainability/) |

> **Items 1–5 change model inputs/architecture, so the existing trained
> artifacts are stale.** You must retrain in the order below. Nothing is
> deleted for you. Item 6 (SHAP) runs on the trained models afterward.

---

## 1. Thin-plate (layer 3) and RBF (layer 4) KAN splines

Previously all four KAN layers were the same B-spline `AdaptiveSplineKANLayer`.
Now each stage uses the spline family its role implies:

```
embedding → SemanticCompressionLayer  (B-spline)
          → ContextInteractionLayer   (B-spline)
          → SentimentDistortionLayer  (thin-plate spline,  U(r) = r² log|r|)
          → EmotionHotspotLayer       (Gaussian RBF spline, exp(-r²/2))  → logits
```

- The thin-plate and RBF layers share a new `RadialBasisKANLayer`, which
  realises the same Kolmogorov–Arnold form `yⱼ = Σᵢ φᵢⱼ(xᵢ)` using a grid of
  radial centres plus a smooth SiLU base term, instead of a B-spline knot
  vector.
- They keep the **identical constructor and `forward`/`adapt_grid` contract**,
  so `KANSentimentClassifier` is unchanged and adaptive knot refitting
  (`update_grids`) still moves the radial centres to the data quantiles.
- `grid_size` = number of radial centres for layers 3–4; `spline_order` is
  accepted but ignored there (radial bases have no polynomial order).

**Stability note:** the thin-plate kernel is unbounded, so training the KAN
expert with gradient clipping is recommended:

```bash
python src/models/kan/train.py --gradient-clip-norm 1.0
```

## 2. Dependency-parse + hashtag-sentiment features (XGBoost expert)

`FeatureExtractor` now appends **9 columns** (feature width 41 → 50), always at
the end of the schema so existing name-based lookups (including the fusion
meta-features) are unaffected:

- **Hashtag sentiment** (4): `hashtag_count`, `hashtag_sentiment`,
  `hashtag_pos_count`, `hashtag_neg_count` — VADER over camel-split hashtag
  tokens (`#FollowFriday → "follow friday"`).
- **Dependency parse** (5): `dep_neg_count`, `dep_amod_count`,
  `dep_advmod_count`, `dep_max_depth`, `dep_mean_depth` — a batched spaCy pass.

Both groups **degrade gracefully**: the columns are always present and
zero-filled when spaCy / its model / VADER data is missing, so feature width is
stable across machines. The dependency features need spaCy's English model:

```bash
pip install spacy && python -m spacy download en_core_web_sm
```

Flags on `FeatureExtractor`: `use_spacy_dep` (default on) and `use_hashtag`
(default on). The fusion meta-feature builder constructs it with
`use_spacy_dep=False` on purpose — the gate only reads the front lexical
columns, so it skips the heavy parse.

## 3. Uncertainty-aware fallback (fusion)

A post-fusion decision layer that detects hard/OOD inputs and overrides the
fused label instead of emitting confident errors. Signals: fused entropy, fused
confidence, expert disagreement (total-variation), and an embedding-space OOD
score (diagonal-Gaussian Mahalanobis to the train embeddings). Actions:
`keep`, `route_kan`/`route_xgb` (to the lower-expected-residual expert),
`conservative` (temperature-soften), `abstain_review` (neutral + human-review
flag). Thresholds are calibrated from the **gate-holdout** split; the detector
is fit on **train** embeddings — no test leakage. Everything serialises into
`fusion_model.pt`. Enabled by default (`--enable-fallback`), fully documented in
[`src/fusion/README.md`](src/fusion/README.md).

## 4. Adaptive grid refinement (growing spline complexity)

The spec says "spline complexity automatically increases in regions of high
semantic density." Before, grids only *moved* to dense regions (`adapt_grid`);
now they also *grow*. New `refine()` on both layer families + `model.refine_grids`:

- **B-spline layers** use pykan's `initialize_grid_from_parent` to fit a finer
  knot grid that reproduces the current edge functions (keeps what was learned).
- **Radial layers** sample their current edge functions and do a single
  least-squares refit onto more centres (shared centres → one solve for all
  edges).
- After growing, `update_grids` re-places the new knots/centres at the data
  quantiles — so the added resolution lands in the dense regions.

Driven by a schedule flag on the KAN trainer, e.g. `--grid-refine "40:8,80:12"`
(grow to 8 knots at epoch 40, 12 at epoch 80). Because refinement replaces
parameters, the training loop rebuilds the optimizer/scheduler and resets
early-stopping state at each step. Off by default (empty schedule).

## 5. Sarcasm quotation cue + emoji preservation

- **Quotation-mark cue** (`FeatureExtractor._sarcasm_score`): scare-quotes
  (`she "loved" it`) now add to the sarcasm score, weighted higher next to
  positive sentiment — the spec's "quotation mark usage" signal.
- **Emoji preservation** (`clean_text`): the tokenizer now keeps emoji as their
  own tokens instead of stripping them, so the cleaned text fed to the embedding
  model retains emoji polarity. (Sentiment140 is a 2009 latin-1 corpus with
  almost no unicode emoji, so this mainly benefits modern data.)

## 6. SHAP explainability

New [`src/explainability/`](src/explainability/) package. `TreeExplainer` gives
**exact** Shapley values for the XGBoost expert; `KernelExplainer` attributes the
fusion gate's trust `α` (why the ensemble leaned on the KAN vs. XGBoost expert).
SHAP is chosen over LIME because it is exact for trees, consistent, and gives one
attribution scheme spanning both models. Full detail + the LIME rationale in
[`src/explainability/README.md`](src/explainability/README.md).

---

## What to run

### 0. One-time setup

```bash
pip install -r library_requirements.txt   # now includes shap
pip install spacy && python -m spacy download en_core_web_sm   # for dependency features
```

### Full pipeline (retrain everything)

```bash
# 1. XGBoost structural expert — now with dependency + hashtag features (50 cols).
#    Writes models/xgboost/ incl. {train,val,test}_probs.npy used downstream.
python src/models/xgboost/train.py

# 2. Row-aligned fusion inputs (stratified subsample + DeBERTa embeddings +
#    sliced XGBoost probabilities), CPU-friendly scale.
python scripts/prepare_fusion_inputs.py --train-size 6000 --val-size 2000 --test-size 2000

# 3. KAN semantic expert on the subsample embeddings — thin-plate (layer 3) and
#    RBF (layer 4) splines. Clip gradients for the TPS layer; optionally grow
#    spline resolution mid-training with --grid-refine.
python src/models/kan/train.py \
    --embeddings-dir data/fusion_inputs/embeddings \
    --output-dir models/kan_subset \
    --gradient-clip-norm 1.0 \
    --grid-refine "40:8,80:12"

# 4. Train the fusion gate AND fit the uncertainty-aware fallback, then export
#    all artifacts (fallback is on by default; add --no-enable-fallback to skip).
python -m src.fusion.trainer

# 5. (Optional) re-run inference later from the checkpoint alone — the fallback
#    is reconstructed from fusion_model.pt automatically.
python -m src.fusion.predictor --checkpoint models/fusion/fusion_model.pt

# 6. SHAP explainability for the XGBoost expert and the fusion gate.
pip install shap
python -m src.explainability.run_explainability
```

### Fast regression check (no data or trained models needed)

Runs the whole fusion stack — feature build, gate training, **fallback fit +
apply**, checkpoint round-trip, artifact export — on synthetic data:

```bash
python scripts/smoke_test_fusion.py
```

### New / changed outputs (`models/fusion/`)

- `fusion_metrics.json` → gains a `fallback` block (action mix, review/abstain
  rates, coverage, selective accuracy, base-vs-final accuracy).
- `fusion_predictions.csv` → gains `fallback_action`, `fallback_review`, and
  `uncertainty_*` columns; the fused columns are **post-fallback**.
- `fusion_probabilities.npy` → post-fallback probabilities.
- `plots/fallback_actions.png` → fallback action distribution.

### Tuning knobs

| Flag | Where | Effect |
|---|---|---|
| `--gradient-clip-norm 1.0` | `kan/train.py` | stabilises the thin-plate layer |
| `--grid-size` | `kan/train.py` | initial B-spline intervals / radial-centre count |
| `--grid-refine "40:8,80:12"` | `kan/train.py` | grow spline resolution at the given epochs |
| `--no-enable-fallback` | `src.fusion.trainer` | train the gate without the fallback |
| `--fallback-temperature` | `src.fusion.trainer` | softening strength in the conservative branch |
| `--only xgboost` / `--only fusion` | `src.explainability.run_explainability` | run one SHAP report |
| `--n-explain` | `src.explainability.shap_fusion` | rows KernelSHAP attributes (speed vs. coverage) |
