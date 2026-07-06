# Hydra Configuration — KAN + XGBoost Sentiment Pipeline

This directory holds all configuration for every model in the project. The `model` config group selects what `src/train.py` trains:

| `model=` | What it is |
|---|---|
| `kan_xgboost` *(default)* | The ensemble: KAN expert (adaptive-spline network over DeBERTa embeddings) → XGBoost expert (engineered linguistic features) → adaptive fusion gate |
| `svm` | TF-IDF + Linear SVM baseline |
| `knn` | TF-IDF + k-Nearest-Neighbours baseline |
| `rf` | TF-IDF + Random Forest baseline |
| `bilstm_cnn` | BiLSTM-CNN deep-learning baseline (TensorFlow, reads the raw Sentiment140 CSV) |

Every dataset path, hyperparameter, training setting, seed, and output location lives in the YAML files here, and any value can be overridden (or swept) from the command line via [Hydra](https://hydra.cc). The entry point is `src/train.py`.

## Why Hydra instead of traditional CLI commands?

With plain argparse, every tunable needs its own `--flag`, defaults are buried in Python constants, and an experiment is only reproducible if you remember the exact command you typed. Hydra fixes each of these:

1. **Single source of truth.** All defaults live in versioned YAML files instead of being scattered across `argparse` definitions and hard-coded constants in four different scripts. Changing a default is a one-line YAML edit, not a code change.
2. **Override anything without new code.** Any config key — even deeply nested ones like `model.xgboost.max_depth` — is overridable from the CLI with `key=value`. With argparse, each new knob means editing `parse_args()` and threading the value through the call stack.
3. **Built-in experiment sweeps.** `-m key=a,b,c other=x,y` runs the full Cartesian product of settings as separate jobs. Reproducing that with traditional CLI commands means hand-written shell loops.
4. **Automatic reproducibility.** Every run gets its own directory (`outputs/<date>/<time>/`, or `multirun/.../<job-id>/`) containing logs plus a `.hydra/` snapshot of the exact composed config and the overrides used — you can always answer "what settings produced this model?".
5. **Composable config groups.** `data/`, `model/`, and `train/` are swappable groups: add `conf/data/subsample.yaml` and select it with `data=subsample`, instead of maintaining near-duplicate command lines or config files.
6. **Type-safe, structured access.** Code reads `cfg.train.kan.lr` from a validated hierarchy instead of a flat namespace of stringly-typed flags; interpolation like `${model.xgboost.max_depth}` lets one value reference another.
7. **The full config is printed at startup** (`OmegaConf.to_yaml(cfg)`), so each log begins with exactly what was run.

## Config files

| File | Keys |
|---|---|
| `config.yaml` | defaults list, `seed`, `device` (null = auto), `outputs.{kan_dir,xgboost_dir,fusion_dir,rf_dir,bilstm_cnn_dir}` |
| `data/default.yaml` | `splits.{train,val,test}`, `embeddings_dir`, `raw.{path,encoding,header,text_column,label_column,sample_size}`, `fusion.{inputs_dir,kan_dir}` |
| `model/kan_xgboost.yaml` | `kan.{hidden_dims,grid_size,spline_order,grid_range}` · `xgboost.{use_optuna,n_trials,n_estimators,max_depth,learning_rate,subsample,colsample_bytree,min_child_weight,gamma,reg_alpha,reg_lambda}` · `fusion.{hidden_dim,dropout,residual_bins,kan_weight}` |
| `model/svm.yaml` | `max_features`, `c_values`, `class_weights`, `scoring` |
| `model/knn.yaml` | `max_features`, `k_values`, `weights`, `scoring` |
| `model/rf.yaml` | `max_features` |
| `model/bilstm_cnn.yaml` | `num_classes`, `preprocessing.{vocab_size,max_length,padding,truncating,lowercase}`, `params.{embedding_dim,lstm_units,filters,kernel_size,dense_units,dropout_rate}` |
| `train/default.yaml` | `stages` · `kan.{epochs,batch_size,lr,weight_decay,patience,grid_update_freq}` · `fusion.{epochs,batch_size,lr,weight_decay,patience,scheduler_patience,scheduler_factor,min_lr,gate_holdout}` · `bilstm_cnn.{epochs,batch_size,learning_rate,test_size,validation_split,early_stopping_patience,verbose}` |

Every `model/*.yaml` carries a `name:` discriminator that `src/train.py` dispatches on. New config-group variants can be added as extra files, e.g. `conf/data/subsample.yaml`, then selected with `python src/train.py data=subsample`.

## Running the pipeline

```bash
# Full ensemble pipeline with defaults: KAN -> XGBoost -> fusion
python src/train.py

# A single stage (bare value, no brackets needed)
python src/train.py train.stages=xgboost

# Multiple stages: list syntax, or a comma-separated string in quotes
python src/train.py train.stages=[kan,fusion]
python src/train.py "train.stages='kan,fusion'"

# Train a baseline instead of the ensemble
python src/train.py model=svm
python src/train.py model=rf model.max_features=30000
python src/train.py model=bilstm_cnn train.bilstm_cnn.epochs=5 data.raw.sample_size=100000

# Show the composed config without training
python src/train.py --cfg job
```

The resolved config is printed at the start of every run. Hydra also snapshots it (plus logs) under `outputs/<date>/<time>/.hydra/`, while model artifacts always go to the project-root paths in `outputs.*` (default `models/kan`, `models/xgboost`, `models/fusion`).

### Stages

| Stage | What it does | Inputs | Artifacts |
|---|---|---|---|
| `kan` | Trains the KAN expert on precomputed embeddings | `data.embeddings_dir` | `outputs.kan_dir` (weights, probs, curves, metrics) |
| `xgboost` | Trains the XGBoost expert on the split CSVs | `data.splits.*` | `outputs.xgboost_dir` (model, extractor, probs, importance) |
| `fusion` | Trains the adaptive gate on the experts' validation predictions | `data.fusion.*` | `outputs.fusion_dir` (checkpoint, metrics, TensorBoard) |
| `static_fusion` | Evaluates a **fixed** blend `w·P_kan + (1−w)·P_xgb` (no training) | `data.fusion.*` + `model.fusion.kan_weight` | `static_fusion_w<w>.json` |

Note: the fusion stage consumes the prepared subsample in `data/fusion_inputs` and KAN probabilities from `models/kan_subset` (a KAN retrained on the subsample embeddings). Run `scripts/prepare_fusion_inputs.py` first if those are missing.

### Baselines

Selecting a baseline with `model=<name>` replaces the ensemble stages entirely (`train.stages` is ignored). Baseline hyperparameters sit directly under `model.*`, so overrides are short:

```bash
python src/train.py model=svm model.c_values=[0.5,1.0] model.class_weights=[balanced]
python src/train.py model=knn model.k_values=[5,9] model.weights=[distance]
python src/train.py model=rf model.max_features=30000
python src/train.py model=bilstm_cnn model.params.lstm_units=64 train.bilstm_cnn.epochs=5
```

| Baseline | Data | Saves |
|---|---|---|
| `svm`, `knn` | `data.splits.*` (`clean_text` column) | nothing — prints evaluation |
| `rf` | `data.splits.*` (`clean_text` column) | pipeline to `outputs.rf_dir` |
| `bilstm_cnn` | `data.raw` (raw Sentiment140; does its own split; set `data.raw.sample_size` for quick runs) | model + tokenizer to `outputs.bilstm_cnn_dir` |

`svm`, `knn`, and `rf` share the `max_features` key, so one override can span a model sweep: `python src/train.py -m model=svm,knn,rf model.max_features=30000`.

## CLI overrides

Any config value can be overridden with `key=value`:

```bash
python src/train.py model.xgboost.max_depth=3 model.fusion.kan_weight=0.7
python src/train.py train.kan.epochs=50 train.kan.lr=5e-4 seed=123
python src/train.py data.embeddings_dir=data/fusion_inputs/embeddings outputs.kan_dir=models/kan_subset train.stages=[kan]
```

Relative paths are resolved against the project root (via `hydra.utils.to_absolute_path`), so they work no matter where Hydra puts its run directory.

**Why the quotes for `train.stages='kan,fusion'`?** An unquoted comma (`train.stages=kan,fusion`) is reserved by Hydra's override grammar for multirun sweeps — in single-run mode Hydra rejects it as ambiguous before the application even starts. Quoting turns it into an ordinary string, which `src/train.py` then splits on commas. The quotes must reach Hydra itself, so wrap the whole override in double quotes with single quotes inside (works in PowerShell, bash, and cmd).

### XGBoost: fixed hyperparameters vs. Optuna

By default `model.xgboost.use_optuna=false`, so the fixed values in `model/kan_xgboost.yaml` (the best Optuna trial found so far) are used directly — this is what makes overrides like `model.xgboost.max_depth=3` take effect. To re-run the original Optuna search instead:

```bash
python src/train.py train.stages=[xgboost] model.xgboost.use_optuna=true model.xgboost.n_trials=50
```

When `use_optuna=true`, the fixed hyperparameter values are ignored.

### Fusion: adaptive gate vs. fixed weight

The learned gate needs no weight setting. Setting `model.fusion.kan_weight` additionally reports a *static* blend evaluation (`fused = w·P_kan + (1−w)·P_xgb`) — useful as a baseline against the adaptive gate. Combine it with the `static_fusion` stage to skip gate training entirely:

```bash
python src/train.py train.stages=[static_fusion] model.fusion.kan_weight=0.7
```

## Multirun experiments (sweeps)

Add `-m` (or `--multirun`) and comma-separated values to run the Cartesian product of settings:

```bash
# 3 x 2 = 6 XGBoost runs
python src/train.py -m train.stages=[xgboost] \
    model.xgboost.max_depth=3,5,7 model.xgboost.learning_rate=0.01,0.05

# Compare all classic baselines in one command
python src/train.py -m model=svm,knn,rf

# Sweep the static fusion weight (fast, no training)
python src/train.py -m train.stages=[static_fusion] model.fusion.kan_weight=0.3,0.5,0.7

# Range sweep syntax
python src/train.py -m train.stages=[kan] "train.kan.lr=range(0.0005,0.002,0.0005)"
```

Each job's logs and config snapshot land in `multirun/<date>/<time>/<job-id>/`. Because all jobs share the same `outputs.*` directories by default, give each sweep point its own artifact directory with interpolation when you want to keep every model:

```bash
python src/train.py -m train.stages=[xgboost] \
    model.xgboost.max_depth=3,5,7 \
    'outputs.xgboost_dir=models/sweeps/xgb_depth${model.xgboost.max_depth}'
```

## Standalone entry points

The original per-model scripts still work independently of Hydra:

```bash
python src/models/kan/train.py --epochs 300 --hidden-dims 256,128,64
python src/models/xgboost/train.py --n-trials 100          # always runs Optuna standalone
python -m src.fusion.trainer --epochs 300 --hidden-dim 64
python src/baselines/svm/train.py --c-values 0.5,1.0
python src/baselines/knn/train.py --k-values 3,5,7
python src/baselines/rf/train.py --max-features 30000
python train.py --config configs/bilstm_cnn.yaml           # legacy BiLSTM-CNN entry (non-Hydra)
```
