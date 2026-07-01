# TF-IDF + SVM Baseline

A linear SVM (`LinearSVC`) over TF-IDF features for binary tweet sentiment
classification (Negative `0` / Positive `1`). This is a baseline for comparison
against the main models.

## Files

| File          | Purpose                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------- |
| `model.py`    | Hyperparameters, the TF-IDF + LinearSVC pipeline, validation-set tuning, final refit.    |
| `evaluate.py` | Computes and prints accuracy, macro precision/recall/F1, and the confusion matrix.       |
| `train.py`    | Entry point — loads the splits, tunes, refits on train+val, and evaluates on the test set. |

## Prerequisites

1. Install dependencies (from the repo root):

   ```bash
   pip install -r requirements.txt
   ```

2. Generate the data splits. `train.py` reads pre-split CSVs from
   `data/splits/` — it does **not** split the data itself:

   ```bash
   python src/data/split_dataset.py
   ```

   This produces `data/splits/train.csv`, `data/splits/val.csv`, and
   `data/splits/test.csv`, each with a `clean_text` column and a binary
   `label` column.

## Usage

Run from the **repo root**:

```bash
# Train + evaluate with default settings
python src/baselines/svm/train.py

# Point at custom split files
python src/baselines/svm/train.py \
    --train-path data/splits/train.csv \
    --val-path   data/splits/val.csv \
    --test-path  data/splits/test.csv

# Tune over a custom grid of C values
python src/baselines/svm/train.py --c-values 0.25,0.5,1.0,2.0
```

### CLI options

| Flag             | Default                   | Description                                  |
| ---------------- | ------------------------- | -------------------------------------------- |
| `--train-path`   | `data/splits/train.csv`   | Path to the training split CSV.              |
| `--val-path`     | `data/splits/val.csv`     | Path to the validation split CSV.            |
| `--test-path`    | `data/splits/test.csv`    | Path to the test split CSV.                  |
| `--max-features` | `50000`                   | Maximum TF-IDF vocabulary size.              |
| `--c-values`     | `0.1,0.25,0.5,1.0,2.0,4.0`| Comma-separated `C` values to tune.          |
| `--random-state` | `42`                      | Random seed for reproducibility.             |

## How it works

1. **Load** the train / validation / test splits and drop empty or null rows.
2. **Tune** the regularisation strength `C` and `class_weight`
   (`None` vs `balanced`) on the validation split, selecting the combination
   with the best macro-F1.
3. **Refit** the best pipeline on the combined train + validation data.
4. **Evaluate** on the held-out test set and print the metrics report.
