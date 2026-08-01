# Residual-Aware Complementarity (KAN specialises where XGBoost fails)

This adds the **residual-aware specialisation** methodology on top of the
existing hybrid, *without changing any existing training path*. The two experts
keep their strict separation of duties — XGBoost sees only interpretable
structural/linguistic features, the KAN sees only DeBERTa semantic embeddings —
and complementarity is introduced purely through the **KAN training objective**,
never through feature sharing.

Everything here is additive. The stock `src/models/kan/train.py`,
`src/models/xgboost/train.py`, and the fusion pipeline are untouched and behave
exactly as before.

## What was added

| File | Role |
| --- | --- |
| `src/models/xgboost/oof.py` | Leakage-safe **out-of-fold** XGBoost predictions on the structural features → per-sample true-class probability → bounded, capped KAN training weights (percentile-rank based). |
| `src/models/kan/model.py` | Optional **auxiliary competence head** on `KANSentimentClassifier` that predicts *P(XGBoost fails)* from the embedding. **Off by default** — a default model has zero extra parameters, so every existing checkpoint still loads and `forward` still returns class logits only. |
| `src/models/kan/residual_loss.py` | `ComplementarityLoss` = unweighted **base** CE (protects overall accuracy) + residual-**weighted** CE (specialisation pressure) + **semantic-rescue** margin hinge + optional **aux** BCE. Plus a weighted dataloader helper. |
| `src/fusion/complementarity.py` | Metric suite: **KAN rescue rate**, **XGBoost preservation rate**, **disagreement accuracy**, **double-fault rate**, **oracle ensemble accuracy**. |
| `scripts/train_residual_kan.py` | Drop-in KAN trainer that wires it all together and emits the **same artifact layout** as `src/models/kan/train.py` (so the result fuses unchanged). |
| `scripts/smoke_test_residual_kan.py` | Fast synthetic self-check of all of the above (exit 0 = pass). |

## How the objective works

1. **Leakage-safe residuals.** XGBoost is fit with stratified K-fold CV; each
   training tweet gets a structural probability from a fold that never saw it.
   The residual is `1 − p_xgb(true class)` — high when the structural expert
   struggled.
2. **Bounded weights.** Residuals become percentile ranks, mapped to a training
   weight in `[floor, w_max]` (easy tweets ≈ `1.0`, hard tweets *moderately*
   higher, hard-capped at `w_max`). Optional `--double-fault-cap` gives only
   limited emphasis to samples XGBoost already fails, so noisy/mislabeled
   outliers can't dominate.
3. **The KAN still optimises the true sentiment label** (base CE), so it stays an
   independent semantic classifier, not a direct XGBoost-correction model. The
   weighted term adds the complementarity pressure on top.
4. **Semantic rescue margin.** On tweets XGBoost found hard, a hinge encourages
   the KAN's true-class probability to exceed XGBoost's by a small margin.
5. **Auxiliary competence signal** (optional): predicts whether XGBoost will
   fail, from the embedding only — an interpretable input for the fusion gate,
   with no effect on the sentiment prediction.

## Run order

The XGBoost expert must exist first (for its params + probabilities). Then:

```bash
python scripts/train_residual_kan.py \
    --embeddings-dir data/fusion_inputs/embeddings \
    --splits-dir data/fusion_inputs/splits \
    --xgb-probs-dir data/fusion_inputs/xgboost \
    --xgb-params models/xgboost/best_params.json \
    --output-dir models/kan_residual \
    --aux-failure-head --gradient-clip-norm 1.0
```

Then fuse as usual, pointing the gate at the new KAN directory:

```bash
python -m src.fusion.trainer --kan-dir models/kan_residual --output-dir models/fusion_residual
```

`--embeddings-dir`, `--splits-dir`, and `--xgb-probs-dir` must be **row-aligned**
(the `data/fusion_inputs*` layout already is). The script asserts alignment and
fails loudly otherwise.

## Key knobs

| Flag | Meaning |
| --- | --- |
| `--weight-method {percentile,linear}` | Residual→weight mapping; `percentile` is robust to XGBoost miscalibration. |
| `--weight-max` / `--weight-floor` | Hard cap / baseline weight. |
| `--double-fault-cap` | Extra cap on XGBoost-failed samples (limits emphasis on ambiguous cases). |
| `--base-weight` / `--comp-weight` / `--rescue-weight` / `--aux-weight` | Turn each loss term up/down; set to `0` to disable it. |
| `--rescue-margin` / `--hard-threshold` | Rescue hinge margin and the true-class-prob cutoff defining a "hard for XGBoost" tweet. |
| `--aux-failure-head` | Attach the competence head (adds `aux` term). |
| `--oof-splits` | Stratified CV folds for the OOF difficulty model. |

Standard KAN knobs (`--hidden-dims`, `--grid-size`, `--spline-order`, `--epochs`,
`--patience`, `--grid-update-freq`, `--gradient-clip-norm`, `--seed`, `--device`)
behave as in `src/models/kan/train.py`.

## Outputs

`models/kan_residual/` contains the usual `kan_model.pt`, `{split}_probs.npy`,
`{split}_confidence.npy`, `metrics.json`, `history.json`, `training_curves.png`,
plus:

- `train_residual_weights.npy`, `train_xgb_oof_probs.npy` — the derived signals.
- `complementarity.json` and a `residual_weighting` block in `metrics.json` —
  rescue/preservation/disagreement/double-fault/oracle metrics (train uses the
  leakage-safe OOF probs; val/test use the real XGBoost probs when
  `--xgb-probs-dir` is given).
