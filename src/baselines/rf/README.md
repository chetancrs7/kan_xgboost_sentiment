# Random Forest Sentiment Baseline

A TF-IDF vectorizer followed by a Random Forest of decision trees for binary
tweet sentiment classification. Tree votes are averaged into the prediction.

## Files

| File | Responsibility |
| --- | --- |
| `model.py` | Hyperparameter constants, the `RandomForestParams` dataclass, TF-IDF + Random Forest pipeline construction, validation-split tuning, and final training. |
| `train.py` | CLI entry point: loads the pre-split CSVs, tunes, trains the final model, evaluates, and saves the pipeline. |
| `evaluate.py` | Computes and prints accuracy, macro precision/recall/F1, and the confusion matrix. |

## Prerequisites

1. Install dependencies (from the repo root):

   ```bash
   pip install -r library_requirements.txt
   ```

2. Make sure the train/validation/test splits exist. `train.py` reads
   pre-split CSVs and does **not** split the raw dataset itself. It expects:

   - `data/splits/train.csv`
   - `data/splits/val.csv`
   - `data/splits/test.csv`

   Each must contain a `clean_text` column and an integer `label` column
   (`0` = negative, `1` = positive). If these files are missing, generate them
   with the dataset split script (`python src/data/split_dataset.py`).

## Running

From the repo root, with the default split paths:

```bash
python src/baselines/rf/train.py
```

Override paths or key hyperparameters via flags:

```bash
python src/baselines/rf/train.py \
    --train-path data/splits/train.csv \
    --val-path data/splits/val.csv \
    --test-path data/splits/test.csv \
    --max-features 30000 \
    --save-dir outputs/rf
```

| Flag | Default | Description |
| --- | --- | --- |
| `--train-path` | `data/splits/train.csv` | Training split CSV. |
| `--val-path` | `data/splits/val.csv` | Validation split CSV (used for tuning). |
| `--test-path` | `data/splits/test.csv` | Held-out test split CSV. |
| `--max-features` | `50000` | Maximum TF-IDF vocabulary size. |
| `--save-dir` | `outputs/rf` | Output directory for the trained pipeline. |

## How it works

1. **Load** the three pre-split CSVs and validate that each has at least two
   classes with two or more samples.
2. **Vectorize**: a `TfidfVectorizer` (unigrams + bigrams, sublinear TF) is kept
   inside the pipeline so its vocabulary is fitted on the training split only,
   which prevents validation/test leakage.
3. **Tune**: grid-search forest parameters (`n_estimators`, `max_depth`,
   `min_samples_leaf`) on the validation split, selecting the best macro-F1.
4. **Train** the selected configuration on the training split. Class weights are
   balanced to offset label imbalance.
5. **Evaluate** on the test split and **save** the trained pipeline
   (`rf.joblib`) to the output directory.

## Tuning the search grid

The forest search space lives in `model.py`:

```python
MAX_FEATURES_SPLIT       = "sqrt"      # fixed (features per split)
N_ESTIMATORS_VALUES      = (200, 400)  # tuned
MAX_DEPTH_VALUES         = (None, 50)  # tuned
MIN_SAMPLES_LEAF_VALUES  = (1, 2)      # tuned
```

Edit these tuples to widen or narrow the grid. A grid with a single combination
skips tuning entirely.

> **Note:** the evaluation report assumes binary sentiment (target names
> `Negative (0)` / `Positive (1)`). Adjust `evaluate.py` before using this on a
> multi-class label set.
