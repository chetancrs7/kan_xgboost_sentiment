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
| `tensorboard/` | training curves, mean α, learning rate |

## Module layout

```
src/fusion/
    fusion_dataset.py   meta-feature assembly, residual estimator, alignment validation
    fusion_model.py     FusionGate (Linear→LayerNorm→ReLU→Dropout→Linear→Sigmoid)
    trainer.py          training loop: CE loss, early stopping, checkpointing,
                        TensorBoard, LR scheduler  (entry point: python -m src.fusion.trainer)
    predictor.py        checkpoint loading, inference, artifact export
    metrics.py          metric suite + calibration/reliability/α figures
    utils.py            seeding, logging, JSON helpers
```

## Reproducibility

Every stage is seeded (default 42): subsample indices, embedding generation,
KAN training, gate-holdout split, dataloader shuffling, and weight
initialisation. `data/fusion_inputs/manifest.json` records the subsample
provenance; the checkpoint embeds everything inference needs.
