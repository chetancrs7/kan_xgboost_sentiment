# KAN–XGBoost Hybrid Sentiment Analysis

A parallel, adaptive **hybrid ensemble** for tweet sentiment classification on the
[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset
(1.6M tweets, binary positive/negative). The system pairs two complementary
experts and blends them with a learned gate:

- **KAN expert** — a Kolmogorov–Arnold Network with adaptive spline layers
  (B-spline → B-spline → thin-plate → RBF) operating on
  `microsoft/deberta-v3-base` sentence embeddings. It captures the *semantic*
  signal.
- **XGBoost expert** — gradient boosting over 50 engineered *linguistic*
  features (lexical, VADER, negation, hashtag-sentiment, and spaCy
  dependency-parse cues). It captures the *structural* signal.
- **Adaptive fusion gate** — a small neural network that learns a per-sample
  weight `α` to combine the two experts, with an uncertainty-aware fallback for
  low-confidence predictions.

The whole pipeline is configured with [Hydra](https://hydra.cc), so every
dataset path, hyperparameter, and output location lives in versioned YAML under
[`conf/`](conf/) and any value can be overridden or swept from the command line.

---

## Results

Test-set metrics on the full 160,000-tweet held-out split:

| Model | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| KNN (TF-IDF) | 0.759 | 0.759 | — |
| Random Forest (TF-IDF) | 0.762 | 0.762 | — |
| XGBoost (50 engineered features) | 0.743 | 0.743 | 0.823 |
| XGBoost (TF-IDF, 30k n-grams) | 0.813 | 0.813 | 0.894 |
| SVM (TF-IDF, linear) | 0.825 | 0.825 | — |
| BiLSTM-CNN | 0.831 | 0.831 | — |
| **KAN (semantic expert)** | 0.875 | 0.875 | 0.947 |
| **KAN–XGBoost fusion (proposed)** | **0.875** | **0.875** | **0.947** |

The KAN expert does the heavy lifting; the learned gate leans on it heavily
(mean α ≈ 0.98) and adds a small but consistent margin. The hybrid leads the
strongest classical/deep baseline (BiLSTM-CNN) by ~4.4 points.

---

## Repository structure

```
.
├── conf/                     # Hydra configuration (see conf/README.md)
├── scripts/                  # Data download + experiment/utility scripts
├── src/
│   ├── data/                 # Text cleaning, feature extraction, train/val/test split
│   ├── embeddings/           # DeBERTa-v3 embedding generation (+ encoder fine-tuning)
│   ├── models/
│   │   ├── kan/              # KAN expert: adaptive spline layers, training, residual loss
│   │   ├── xgboost/          # XGBoost expert: engineered features, OOF stacking
│   │   └── xgboost_tfidf/    # TF-IDF XGBoost variant
│   ├── baselines/            # SVM, KNN, Random Forest, BiLSTM-CNN
│   ├── fusion/               # Adaptive gate, calibration, complementarity, fallback
│   ├── analysis/             # KAN spline extraction + visualization, TensorBoard
│   ├── explainability/       # SHAP for the XGBoost expert and the fusion gate
│   └── train.py              # Hydra entry point for every model
├── eda/notebooks/            # Exploratory data analysis notebooks
├── experiments/              # Standalone research experiments (e.g. reversed roles)
├── HYBRID_ARCHITECTURE.md    # Architecture deep-dive
├── RESIDUAL_COMPLEMENTARITY.md
└── library_requirements.txt
```

> **Note:** trained weights (`models/`) and generated embeddings/data (`data/`)
> are git-ignored — they are large, reproducible artifacts. Regenerate them with
> the steps below.

---

## Installation

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r library_requirements.txt

# 3. (Optional) spaCy model for dependency-parse features
python -m spacy download en_core_web_sm
```

Key dependencies: `torch`, `transformers`, `xgboost`, `tensorflow` (BiLSTM-CNN
baseline), `hydra-core`, `shap`, and `spacy`.

A CUDA GPU is recommended for embedding generation and KAN/encoder training but
is not required — set `device=cpu` to force CPU.

---

## Data preparation

Run these once to produce the inputs the models consume:

```bash
# 1. Download Sentiment140 into data/raw/ (needs a Kaggle account for kagglehub)
python scripts/download_dataset.py

# 2. Clean the raw tweets -> data/processed/cleaned_tweets.csv
python src/data/text_cleaning.py

# 3. Split into train/val/test -> data/splits/
python src/data/split_dataset.py

# 4. Generate DeBERTa-v3 embeddings for each split -> data/embeddings/
python src/embeddings/generate_embeddings.py
```

The `.npy` embedding files produced in step 4 are the shared input for the KAN
and fusion stages.

---

## Running the models

The single entry point is `src/train.py`. The `model` config group selects what
to train — the `kan_xgboost` ensemble (default) or a baseline.

```bash
# Full ensemble pipeline: KAN -> XGBoost -> fusion
python src/train.py

# Run a single stage
python src/train.py train.stages=xgboost

# Run a subset of stages
python src/train.py train.stages=[kan,fusion]

# Show the composed config without training
python src/train.py --cfg job
```

### Stages

| Stage | What it does |
|---|---|
| `kan` | Trains the KAN expert on the precomputed embeddings |
| `xgboost` | Trains the XGBoost expert on the engineered features |
| `fusion` | Trains the adaptive gate on the experts' validation predictions |
| `static_fusion` | Evaluates a fixed blend `w·P_kan + (1−w)·P_xgb` (no training) |

### Baselines

Selecting a baseline replaces the ensemble stages entirely:

```bash
python src/train.py model=svm
python src/train.py model=knn
python src/train.py model=rf model.max_features=30000
python src/train.py model=bilstm_cnn train.bilstm_cnn.epochs=5
```

### Overriding and sweeping hyperparameters

Any nested config key can be overridden with `key=value`, and `-m` runs a
Cartesian-product sweep:

```bash
# Override individual values
python src/train.py model.xgboost.max_depth=3 train.kan.epochs=50 seed=123

# Multirun sweep (3 x 2 = 6 runs)
python src/train.py -m train.stages=[xgboost] \
    model.xgboost.max_depth=3,5,7 model.xgboost.learning_rate=0.01,0.05

# Compare all classic baselines in one command
python src/train.py -m model=svm,knn,rf
```

See [`conf/README.md`](conf/README.md) for the full configuration reference,
every overridable key, and multirun examples.

---

## Analysis & explainability

```bash
# SHAP explanations for the XGBoost expert and the fusion gate
python src/explainability/run_explainability.py

# Extract and visualize the learned KAN spline activations
python src/analysis/run_spline_analysis.py
```

TensorBoard logs (KAN training curves, spline evolution) are written under
`outputs/tensorboard/` — view with `tensorboard --logdir outputs/tensorboard`.

---

## Documentation

- [`HYBRID_ARCHITECTURE.md`](HYBRID_ARCHITECTURE.md) — architecture deep-dive:
  the thin-plate/RBF KAN layers, the dependency-parse + hashtag features, and
  the uncertainty-aware fallback.
- [`RESIDUAL_COMPLEMENTARITY.md`](RESIDUAL_COMPLEMENTARITY.md) — residual-aware
  fusion and the expert-complementarity / performance-ceiling analysis.
- [`conf/README.md`](conf/README.md) — Hydra configuration and CLI usage.

---

## License

This project was developed as a capstone research project. See the repository
owner for usage terms.
