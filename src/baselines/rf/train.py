"""Train and evaluate the TF-IDF + Random Forest sentiment baseline.

Usage:
    python src/baselines/rf/train.py
    python src/baselines/rf/train.py --train-path data/splits/train.csv --val-path data/splits/val.csv --test-path data/splits/test.csv
    python src/baselines/rf/train.py --max-features 30000 --save-dir outputs/rf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

# Make the project root importable so this file works both as a script
# (`python src/baselines/rf/train.py`) and imported from the Hydra entry
# point (`src/train.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.baselines.rf.evaluate import evaluate  # noqa: E402
from src.baselines.rf.model import (  # noqa: E402
    MAX_FEATURES,
    RANDOM_STATE,
    set_reproducibility,
    train_final_model,
    tune_model,
)
from src.baselines.rf.visualize import generate_plots  # noqa: E402

DEFAULT_TRAIN_PATH = "data/splits/train.csv"
DEFAULT_VAL_PATH = "data/splits/val.csv"
DEFAULT_TEST_PATH = "data/splits/test.csv"
DEFAULT_SAVE_DIR = "outputs/rf"
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


def save_artifacts(model: object, save_dir: str) -> None:
    """Persist the trained TF-IDF + Random Forest pipeline."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "rf.joblib")
    print(f"[INFO] Saved model pipeline to '{out_dir}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + Random Forest sentiment baseline.")
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
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help="Maximum TF-IDF vocabulary size (default: %(default)s).",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory for the trained model pipeline (default: %(default)s).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    set_reproducibility(getattr(args, "seed", RANDOM_STATE))

    train_df, val_df, test_df = load_splits(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    best_params, tuning_history = tune_model(
        train_texts=train_df[TEXT_COLUMN],
        y_train=train_df[LABEL_COLUMN],
        val_texts=val_df[TEXT_COLUMN],
        y_val=val_df[LABEL_COLUMN],
        max_features=args.max_features,
    )

    model = train_final_model(
        train_texts=train_df[TEXT_COLUMN],
        y_train=train_df[LABEL_COLUMN],
        best_params=best_params,
        max_features=args.max_features,
    )

    vocab_size = len(model.named_steps["tfidf"].vocabulary_)
    print(f"[INFO] Vocabulary size: {vocab_size:,} terms.")

    evaluate(model, test_df[TEXT_COLUMN], test_df[LABEL_COLUMN])
    save_artifacts(model, args.save_dir)
    generate_plots(
        model,
        test_df[TEXT_COLUMN],
        test_df[LABEL_COLUMN],
        Path(args.save_dir) / "plots",
        tuning_history=tuning_history,
    )


if __name__ == "__main__":
    main()
