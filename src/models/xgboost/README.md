# XGBoost Expert — Engineered Linguistic Features

A production-ready XGBoost sentiment classifier that learns from **interpretable
linguistic features** (negation, intensifiers, emoji sentiment, sarcasm cues,
punctuation, capitalisation, POS distribution, subjectivity) rather than raw
text. It is designed as an *expert* in a hybrid ensemble: it exports calibrated
prediction probabilities that can be stacked with the DeBERTa-v3 and other
baseline models in this repo.

## Module layout

```
src/models/xgboost/
├── __init__.py        # package exports
├── data_loader.py     # load pre-split train/val/test CSVs (CSV/pickle/parquet)
├── features.py        # FeatureExtractor: raw text -> numeric feature matrix
├── model.py           # XGBoost build + Optuna hyperparameter search
├── evaluate.py        # per-split metrics + feature-importance reporting
├── train.py           # end-to-end orchestrator (entry point)
└── README.md
```

## Quick start

```bash
# from the repo root
python src/models/xgboost/train.py                 # default: 50 Optuna trials
python src/models/xgboost/train.py --n-trials 100  # deeper search
python src/models/xgboost/train.py \
    --train-path data/splits/train.csv \
    --val-path   data/splits/val.csv \
    --test-path  data/splits/test.csv \
    --output-dir models/xgboost
```

The pipeline:

1. Loads the existing `data/splits/{train,val,test}.csv` (same rows the other
   models see — no re-splitting).
2. Fits a `FeatureExtractor` on training text and transforms all three splits.
3. Runs Optuna to maximise validation F1.
4. Refits the final model with early stopping.
5. Evaluates train/val/test (accuracy, precision, recall, F1, AUC).
6. Exports probabilities and all artifacts.

## Features (30 columns)

| Group           | Features |
|-----------------|----------|
| Negation        | `negation_count`, `negation_ratio` |
| Intensifiers    | `intensifier_count`, `intensifier_score` (weighted) |
| Emoji           | `emoji_sentiment`, `emoji_count`, `emoji_density` |
| Sarcasm         | `sarcasm_score` (mixed polarity, cue phrases, ellipsis) |
| Punctuation     | `exclamation_count`, `question_count`, `ellipsis_count`, `punctuation_density` |
| Capitalisation  | `caps_word_count`, `caps_ratio` |
| Lengths         | `char_count`, `token_count`, `avg_word_length` |
| Subjectivity    | `subjectivity_score` |
| POS distribution| `pos_{noun,verb,adj,adv,pron,adp,det,conj,num,prt,x,punct}_ratio` |

The `FeatureExtractor` follows the scikit-learn transformer protocol
(`fit` / `transform` / `fit_transform`), so one instance is fitted on train and
reused on val/test with identical column ordering. It handles empty text,
`None`/NaN, and missing NLTK data gracefully (regex fallbacks).

```python
from features import FeatureExtractor
fx = FeatureExtractor()
X_train = fx.fit_transform(train_texts)   # [n, 30]
X_val   = fx.transform(val_texts)         # identical columns
```

## Artifacts (written to `models/xgboost/`)

| File | Description |
|------|-------------|
| `xgboost_model.json` / `.pkl` | Trained model (native + joblib) |
| `feature_extractor.pkl` | Fitted extractor for inference/ensembling |
| `best_params.json` | Winning Optuna hyperparameters |
| `metrics.json` | Accuracy/precision/recall/F1/AUC per split |
| `feature_importance.json` / `.csv` | Ranked feature importances |
| `optuna_trials.csv` | Full trial log |
| `train_probs.npy` | `[n_train, n_classes]` probabilities |
| `val_probs.npy` | `[n_val, n_classes]` probabilities |
| `test_probs.npy` | `[n_test, n_classes]` probabilities |

## Ensemble integration

The `*_probs.npy` files align row-for-row with the split CSVs (rows are never
dropped), so they stack directly with the DeBERTa-v3 embeddings/probabilities
under `data/embeddings/`. To score new text:

```python
import joblib
fx    = joblib.load("models/xgboost/feature_extractor.pkl")
model = joblib.load("models/xgboost/xgboost_model.pkl")
proba = model.predict_proba(fx.transform(["I absolutely love this!! 😍"]))
```

## Dependencies

`xgboost`, `optuna`, `scikit-learn`, `nltk`, `emot`, `pandas`, `numpy`,
`joblib`. The first run auto-downloads the required NLTK resources
(`punkt`, `punkt_tab`, `averaged_perceptron_tagger_eng`, `universal_tagset`).

> **Note on the sample splits:** the committed `data/splits/*.csv` are a tiny
> 160/20/20 sample, so headline metrics will look noisy. Regenerate the full
> splits with `python src/data/split_dataset.py` for representative numbers.
