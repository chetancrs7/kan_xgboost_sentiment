# BiLSTM-CNN Sentiment Baseline

A bidirectional LSTM (contextual sequence encoding) followed by a 1D
convolution (local n-gram features) and a softmax classifier for binary
tweet sentiment classification.

## Files

| File | Responsibility |
| --- | --- |
| `model.py` | Hyperparameter constants, the `BiLSTMCNNParams` dataclass, tokenizer fitting/vectorization, model construction, validation-split tuning, and final training. |
| `train.py` | CLI entry point: loads the pre-split CSVs, fits the tokenizer, tunes, trains the final model, evaluates, and saves artifacts. |
| `evaluate.py` | Computes and prints accuracy, macro precision/recall/F1, and the confusion matrix. |
| `bilstm_cnn.py` | Original monolithic implementation, kept temporarily and slated for removal. |

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
python src/baselines/bilstm_cnn/train.py
```

Override paths or key hyperparameters via flags:

```bash
python src/baselines/bilstm_cnn/train.py \
    --train-path data/splits/train.csv \
    --val-path data/splits/val.csv \
    --test-path data/splits/test.csv \
    --vocab-size 30000 \
    --max-length 150 \
    --epochs 10 \
    --save-dir outputs/bilstm_cnn
```

| Flag | Default | Description |
| --- | --- | --- |
| `--train-path` | `data/splits/train.csv` | Training split CSV. |
| `--val-path` | `data/splits/val.csv` | Validation split CSV (used for tuning and early stopping). |
| `--test-path` | `data/splits/test.csv` | Held-out test split CSV. |
| `--vocab-size` | `20000` | Maximum tokenizer vocabulary size. |
| `--max-length` | `200` | Padded/truncated sequence length. |
| `--epochs` | `15` | Epochs for final model training. |
| `--save-dir` | `outputs/bilstm_cnn` | Output directory for the model and tokenizer. |

## How it works

1. **Load** the three pre-split CSVs and validate that each has at least two
   classes with two or more samples.
2. **Tokenize**: fit a Keras `Tokenizer` on the training texts only, then
   convert all splits to padded integer sequences.
3. **Tune**: grid-search architecture parameters (`lstm_units`,
   `dropout_rate`) on the validation split, selecting the best macro-F1 over a
   short number of tuning epochs.
4. **Train** the selected configuration on the training split, using the
   validation split for early stopping and learning-rate reduction. Class
   weights are balanced to offset label imbalance.
5. **Evaluate** on the test split and **save** the trained model
   (`bilstm_cnn.keras`) and fitted tokenizer (`tokenizer.joblib`) to the
   output directory.

## Tuning the search grid

The architecture search space lives in `model.py`:

```python
EMBEDDING_DIM       = 128          # fixed
FILTERS             = 128          # fixed
KERNEL_SIZE         = 3            # fixed
DENSE_UNITS         = 64           # fixed
LSTM_UNITS_VALUES   = (64, 128)    # tuned
DROPOUT_RATES       = (0.3, 0.5)   # tuned
```

Edit `LSTM_UNITS_VALUES` / `DROPOUT_RATES` to widen or narrow the grid. A grid
with a single combination skips tuning entirely.

> **Note:** the evaluation report assumes binary sentiment (target names
> `Negative (0)` / `Positive (1)`). Adjust `evaluate.py` before using this on a
> multi-class label set.
