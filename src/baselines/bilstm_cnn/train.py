"""Train and evaluate the BiLSTM-CNN sentiment baseline.

Usage:
    python src/baselines/bilstm_cnn/train.py
    python src/baselines/bilstm_cnn/train.py --train-path data/splits/train.csv --val-path data/splits/val.csv --test-path data/splits/test.csv
    python src/baselines/bilstm_cnn/train.py --vocab-size 30000 --max-length 150 --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from evaluate import evaluate
from visualize import generate_plots
from model import (
    EPOCHS,
    MAX_LENGTH,
    RANDOM_STATE,
    VOCAB_SIZE,
    fit_tokenizer,
    set_reproducibility,
    train_final_model,
    tune_model,
    vectorize_texts,
)

DEFAULT_TRAIN_PATH = "data/splits/train.csv"
DEFAULT_VAL_PATH = "data/splits/val.csv"
DEFAULT_TEST_PATH = "data/splits/test.csv"
DEFAULT_SAVE_DIR = "outputs/bilstm_cnn"
TEXT_COLUMN: str = "clean_text"
LABEL_COLUMN: str = "label"


def load_split(data_path: str, split_name: str) -> pd.DataFrame:
    """Load one pre-split CSV produced by split_dataset.py."""
    path = Path(data_path)
    if not path.exists():
        sys.exit(
            f"[ERROR] Data file not found: {path.resolve()}\n"
            "Run 'python src/data/split_dataset.py' first to generate the split CSV files."
        )

    df = pd.read_csv(path)

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")

    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""].reset_index(drop=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows from {split_name} with null/empty text or label.")

    validate_dataframe(df, split_name)
    print(f"[INFO] Loaded {len(df):,} {split_name} samples from '{path}'.")
    return df


def validate_dataframe(df: pd.DataFrame, split_name: str) -> None:
    """Fail early with actionable messages for common data issues."""
    if df.empty:
        sys.exit(f"[ERROR] No usable rows found in the {split_name} split.")

    class_counts = df[LABEL_COLUMN].value_counts()
    if len(class_counts) < 2:
        sys.exit(f"[ERROR] The {split_name} split needs at least two sentiment classes.")

    rarest_class_count = int(class_counts.min())
    if rarest_class_count < 2:
        sys.exit(
            f"[ERROR] Each class in the {split_name} split needs at least two samples. "
            f"Class counts: {class_counts.to_dict()}"
        )


def load_splits(
    train_path: str,
    val_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, validation, and test CSV files without re-splitting."""
    train_df = load_split(train_path, "train")
    val_df = load_split(val_path, "validation")
    test_df = load_split(test_path, "test")
    print(
        f"[INFO] Using existing splits | Train: {len(train_df):,} | "
        f"Validation: {len(val_df):,} | Test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


def save_artifacts(model: object, tokenizer: object, save_dir: str) -> None:
    """Persist the trained model and fitted tokenizer."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, out_dir / "tokenizer.joblib")
    model.save(out_dir / "bilstm_cnn.keras")  # type: ignore[attr-defined]
    print(f"[INFO] Saved model and tokenizer to '{out_dir}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiLSTM-CNN sentiment baseline.")
    parser.add_argument(
        "--train-path",
        default=DEFAULT_TRAIN_PATH,
        help="Path to the training split CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--val-path",
        default=DEFAULT_VAL_PATH,
        help="Path to the validation split CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help="Path to the test split CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help="Maximum tokenizer vocabulary size (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Padded/truncated sequence length (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Epochs for final model training (default: %(default)s).",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory for the trained model and tokenizer (default: %(default)s).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    # The Hydra entry point passes cfg.seed; the standalone CLI keeps the
    # module default.
    set_reproducibility(getattr(args, "seed", RANDOM_STATE))

    train_df, val_df, test_df = load_splits(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    num_classes = int(
        pd.concat([train_df[LABEL_COLUMN], val_df[LABEL_COLUMN], test_df[LABEL_COLUMN]]).nunique()
    )

    tokenizer = fit_tokenizer(train_df[TEXT_COLUMN], vocab_size=args.vocab_size)
    print(f"[INFO] Vocabulary size: {len(tokenizer.word_index):,} terms.")

    x_train = vectorize_texts(tokenizer, train_df[TEXT_COLUMN], max_length=args.max_length)
    x_val = vectorize_texts(tokenizer, val_df[TEXT_COLUMN], max_length=args.max_length)
    x_test = vectorize_texts(tokenizer, test_df[TEXT_COLUMN], max_length=args.max_length)
    y_train = train_df[LABEL_COLUMN].to_numpy()
    y_val = val_df[LABEL_COLUMN].to_numpy()
    y_test = test_df[LABEL_COLUMN].to_numpy()

    best_params = tune_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_classes=num_classes,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )

    model = train_final_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        best_params=best_params,
        num_classes=num_classes,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        epochs=args.epochs,
    )

    evaluate(model, x_test, y_test)
    save_artifacts(model, tokenizer, args.save_dir)

    # Keras attaches the final fit's History callback to the model.
    history = getattr(model, "history", None)
    generate_plots(
        model,
        x_test,
        y_test,
        Path(args.save_dir) / "plots",
        history=history.history if history is not None else None,
    )


if __name__ == "__main__":
    main()
