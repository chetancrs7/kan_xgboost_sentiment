# SHAP explainability

Feature-attribution reports for the two trainable models in the ensemble, both
under one Shapley framework so their importances are comparable.

| Target | Explainer | Why |
|---|---|---|
| XGBoost expert (log-odds) | `shap.TreeExplainer` | **exact** Shapley values for tree ensembles, fast, no surrogate |
| Fusion gate trust `α` | `shap.KernelExplainer` | model-agnostic Shapley sampling around the gate's forward pass |

### Why SHAP and not LIME

- The XGBoost expert is a gradient-boosted tree ensemble, where `TreeExplainer`
  computes **exact** Shapley values in polynomial time. LIME only fits a local
  linear surrogate that is approximate and unstable run-to-run.
- Shapley values are additive and consistent, giving one attribution scheme
  that covers **both** the tree expert and the neural gate — so the two
  models' feature importances live on the same footing.
- SHAP gives global (beeswarm / mean-|value| ranking) **and** local
  (waterfall) views from the same values; LIME is local-only.

### What is explained

- **XGBoost:** SHAP values over the 50 engineered features, in log-odds space
  (positive → pushes toward the positive class).
- **Fusion gate:** SHAP values of **`α`** (trust in the KAN expert). `α` is a
  clean function of the meta-features, so "why did the ensemble trust the KAN
  vs. the XGBoost expert here?" is exactly what the attribution answers. The
  *fused probability* is intentionally not attributed to the meta-features —
  it also depends on the experts' per-sample probabilities, which KernelSHAP
  would not perturb, making that attribution ill-posed.

## Run

```bash
pip install shap    # not in the base install

# both reports
python -m src.explainability.run_explainability

# individually
python -m src.explainability.shap_xgboost --model-dir models/xgboost --split-path data/splits/test.csv
python -m src.explainability.shap_fusion  --checkpoint models/fusion/fusion_model.pt \
    --inputs-dir data/fusion_inputs --kan-dir models/kan_subset
```

Requires the models to be trained first (see [../../HYBRID_ARCHITECTURE.md](../../HYBRID_ARCHITECTURE.md)).
KernelSHAP is slow, so the fusion explainer attributes a subsample
(`--n-explain`, default 300) against a k-means background (`--n-background`).

## Outputs

`models/xgboost/shap/`
- `shap_summary_beeswarm.png`, `shap_global_importance.png`,
  `shap_vs_gain_importance.png`, `shap_waterfall_*.png`
- `shap_global_importance.json`, `shap_feature_importance.csv`

`models/fusion/shap/`
- `fusion_shap_summary_beeswarm.png`, `fusion_shap_global_importance.png`,
  `fusion_shap_waterfall_trust_kan.png`, `fusion_shap_waterfall_trust_xgb.png`
- `fusion_shap_global_importance.json`

## Module layout

```
src/explainability/
    __init__.py           require_shap() import guard
    shap_xgboost.py        TreeExplainer report for the XGBoost expert
    shap_fusion.py         KernelExplainer report for the gate's trust α
    run_explainability.py  orchestrator (python -m src.explainability.run_explainability)
```
