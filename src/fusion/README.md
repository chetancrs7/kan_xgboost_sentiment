# Residual-Aware Adaptive Fusion Network

Per-sample fusion of the two frozen sentiment experts:

* **KAN semantic expert** — adaptive-spline Kolmogorov–Arnold Network over
  DeBERTa-v3 embeddings (`src/models/kan/`).
* **XGBoost structural expert** — gradient-boosted trees over engineered
  linguistic features (`src/models/xgboost/`).

Instead of averaging, a lightweight **gating network** looks at *how* the two
experts disagree on each individual sample and outputs a trust value
`α ∈ [0, 1]`:

```
p_fused = α · p_KAN + (1 − α) · p_XGB
```

## Fusion feature vector

| Group | Features |
|---|---|
| KAN beliefs | per-class probabilities, predicted class, softmax confidence, Shannon entropy |
| XGBoost beliefs | per-class probabilities, predicted class, softmax confidence, Shannon entropy |
| Cross-model | signed per-class probability differences, total L1 difference, prediction agreement, KL(KAN‖XGB), KL(XGB‖KAN), confidence gap |
| Linguistic cues | sarcasm score, emoji density, negation indicator, intensifier score, punctuation density, capitalisation ratio (recomputed with the XGBoost expert's own `FeatureExtractor`) |
| Embedding statistics | L2 norm, per-dimension variance of the DeBERTa embedding |
| Residual estimates | expected residual per expert + residual gap, from confidence-binned **historical** residuals |

### Residual features, leak-free

During expert training we observe each expert's residual
`r = 1 − p(y_true)` on every *training-split* sample. Those historical
residuals are grouped into quantile bins of the expert's confidence; at
inference a sample receives the mean historical residual of its confidence
bin. Evaluation labels are never touched, so the features are valid on
val/test and in deployment.

## Gating network

```
Input → Linear → LayerNorm → ReLU → Dropout → Linear → Sigmoid → α
```

Trained with cross-entropy on the **fused** distribution, using the
validation split's expert predictions (never the experts' own training split,
where they are overconfident). A stratified 20 % holdout of the validation
split drives early stopping, `ReduceLROnPlateau` scheduling, and checkpoint
selection; the test split is only ever used for the final report.

## Uncertainty-aware fallback (`fallback.py`)

The gate always emits a convex combination — the right default, but confident
labels on genuinely hard inputs are where catastrophic errors come from. After
fusion, an opt-in decision layer (`--enable-fallback`, on by default) watches
four per-sample signals and overrides the output when they hit calibrated
thresholds:

| Signal | Meaning |
|---|---|
| Fused entropy | the ensemble itself is unsure |
| Fused confidence | max fused class probability |
| Expert disagreement | total-variation distance KAN↔XGBoost |
| Embedding OOD score | diagonal-Gaussian Mahalanobis distance to the train embeddings |

Actions, in priority order:

1. **abstain_review** — extreme entropy or strong OOD → return the training
   class prior with reduced confidence and raise a human-review flag.
2. **route_kan / route_xgb** — strong disagreement → defer to the expert with
   the lower *expected residual* (the same leak-free residual the gate uses).
3. **conservative** — moderately uncertain → keep the label but
   temperature-soften the probabilities so confidence is honest.
4. **keep** — confident and in-distribution → pass through untouched.

Thresholds are **percentiles of the gate-holdout signal distributions** and
the OOD detector is fit on the experts' **train** embeddings, so nothing about
the test split leaks in. All of it (thresholds, OOD statistics, class prior)
serialises into `fusion_model.pt`, so standalone inference reproduces the exact
decisions.

**Annotation-only by default.** The fallback is a *reliability* layer, not an
accuracy booster, so by default it only **flags** risky samples (review flag +
recommended action) and leaves the fused prediction untouched — `final_accuracy`
equals the true fused `base_accuracy`. Report `base_accuracy` as the fusion
accuracy; `accuracy_on_covered` vs `accuracy_on_reviewed` shows the flags are
meaningful (confident samples beat flagged ones). Pass
`--fallback-apply-overrides` to actually apply abstain/route/soften for a
conservative deployment (this trades accuracy for safety).

## Expert calibration (`calibration.py`)

The gate keys on the experts' confidence/entropy, so those must be trustworthy.
Before the gate sees anything, each expert's probabilities are **temperature
scaled** (one scalar `T` per expert, fit on the validation split by minimising
NLL): `p' = softmax(log(p) / T)`. This is monotonic — an expert's own hard
predictions never change — but it re-scales confidence, giving the gate honest
signals and reshaping the confidence-weighted fusion. On by default
(`--no-calibrate-experts` to disable); the temperatures serialise into the
checkpoint and are re-applied at inference.

## Pipeline

```bash
# 1. Row-aligned inputs: stratified subsample + DeBERTa embeddings +
#    sliced XGBoost probabilities (CPU-friendly scale).
python scripts/prepare_fusion_inputs.py --train-size 6000 --val-size 2000 --test-size 2000

# 2. Retrain the KAN expert on the subsample embeddings (its own pipeline).
python src/models/kan/train.py \
    --embeddings-dir data/fusion_inputs/embeddings \
    --output-dir models/kan_subset

# 3. Train the fusion gate + export all artifacts.
python -m src.fusion.trainer

# 4. (Optional) re-run inference later from the checkpoint alone.
python -m src.fusion.predictor --checkpoint models/fusion/fusion_model.pt

# Monitor training
tensorboard --logdir models/fusion/tensorboard
```

`prepare_fusion_inputs.py` slices the XGBoost expert's **full-split**
probabilities (`{split}_probs.npy`, one row per raw-CSV row). By default it
reads them from `models/xgboost/` and subsamples `data/splits/`.

### Variant: full-dataset TF-IDF XGBoost expert

To fuse against the stronger TF-IDF XGBoost expert
(`src/models/xgboost_tfidf/`) trained on the full `data/splitsfull` splits,
run step 0 first and point steps 1–2 at the full splits / new probs dir:

```bash
# 0. Train the TF-IDF XGBoost expert on the full dataset and export the
#    per-split probability arrays the fusion prep consumes (--save-split-probs
#    writes {train,val,test}_probs.npy, row-aligned with the raw CSVs).
python src/models/xgboost_tfidf/train.py \
    --train-path data/splitsfull/train.csv \
    --val-path data/splitsfull/val.csv \
    --test-path data/splitsfull/test.csv \
    --output-dir models/xgboost_tfidf_full \
    --save-split-probs

# 1. Subsample the full splits and slice the full-dataset XGBoost probs.
python scripts/prepare_fusion_inputs.py \
    --splits-dir data/splitsfull \
    --xgb-probs-dir models/xgboost_tfidf_full \
    --train-size 6000 --val-size 2000 --test-size 2000

# 2–4. As above (retrain KAN on the new embeddings, then train the gate).
```

Steps 1–2 overwrite `data/fusion_inputs/` and `models/kan_subset/`; pass
alternate `--output-dir` / `--inputs-dir` / `--kan-dir` values to keep the
`data/splits`-based artifacts alongside these.

## Outputs (`models/fusion/`)

| Artifact | Contents |
|---|---|
| `fusion_model.pt` | self-contained checkpoint: gate weights + config, feature scaler, residual lookup tables, feature schema |
| `fusion_predictions.csv` | per-sample text, label, expert predictions, fused prediction, α |
| `fusion_probabilities.npy` | fused test probabilities `[n, 2]` |
| `fusion_metrics.json` | accuracy, precision, recall, macro/weighted F1, ROC-AUC, confusion matrix, ECE — for the fusion *and* both experts, on gate-holdout and test |
| `training_history.json` | per-epoch losses/accuracies, best epoch |
| `plots/confusion_matrix.png` | fused test confusion matrix |
| `plots/calibration_curve.png` | KAN vs XGBoost vs Fusion calibration |
| `plots/reliability_diagram.png` | confidence-binned reliability + ECE |
| `plots/alpha_distribution.png` | histogram of learned trust values |
| `plots/fallback_actions.png` | bar chart of fallback action counts (when `--enable-fallback`) |
| `tensorboard/` | training curves, mean α, learning rate |

When the fallback is enabled, `fusion_metrics.json` gains a `fallback` block
(action counts/fractions, review & abstain rates, coverage, selective accuracy,
base-vs-final accuracy), `fusion_probabilities.npy` holds the **post-fallback**
probabilities, and `fusion_predictions.csv` gains `fallback_action`,
`fallback_review`, and `uncertainty_*` columns.

## Module layout

```
src/fusion/
    fusion_dataset.py   meta-feature assembly, residual estimator, alignment validation
    fusion_model.py     FusionGate (Linear→LayerNorm→ReLU→Dropout→Linear→Sigmoid)
    fallback.py         uncertainty-aware fallback: OOD detector, calibrated
                        thresholds, routing/abstention policy, reporting
    trainer.py          training loop: CE loss, early stopping, checkpointing,
                        TensorBoard, LR scheduler, fallback fitting
                        (entry point: python -m src.fusion.trainer)
    predictor.py        checkpoint loading, inference, fallback, artifact export
    metrics.py          metric suite + calibration/reliability/α figures
    utils.py            seeding, logging, JSON helpers
```

## Reproducibility

Every stage is seeded (default 42): subsample indices, embedding generation,
KAN training, gate-holdout split, dataloader shuffling, and weight
initialisation. `data/fusion_inputs/manifest.json` records the subsample
provenance; the checkpoint embeds everything inference needs.
