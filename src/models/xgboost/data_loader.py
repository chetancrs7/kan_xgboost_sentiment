"""Data loading for the XGBoost expert sentiment model.

Reads the pre-split CSV files produced by ``src/data/split_dataset.py`` without
re-splitting, so the XGBoost expert sees exactly the same train/val/test rows as
the other models in the ensemble. The loader is intentionally tolerant of the
standard serialisation formats used across this repo (CSV today; ``.pkl`` /
``.npy`` companions are accepted transparently for forward compatibility).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# The raw ``text`` column is preferred for feature engineering because casing,
# punctuation and emoji carry sentiment signal that ``clean_text`` strips away.
# We fall back to ``clean_text`` when raw text is unavailable.
RAW_TEXT_COLUMN: str = "text"
FALLBACK_TEXT_COLUMN: str = "clean_text"
LABEL_COLUMN: str = "label"

DEFAULT_TRAIN_PATH = "data/splits/train.csv"
DEFAULT_VAL_PATH = "data/splits/val.csv"
DEFAULT_TEST_PATH = "data/splits/test.csv"


def _read_any(path: Path) -> pd.DataFrame:
    """Read a dataframe from CSV or pickle based on the file suffix."""
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".csv", ".txt", ""}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    # Last resort: try CSV so a mislabelled extension still loads.
    return pd.read_csv(path)


def _resolve_text_column(df: pd.DataFrame, split_name: str) -> str:
    """Pick the best available text column, preferring raw text."""
    if RAW_TEXT_COLUMN in df.columns:
        return RAW_TEXT_COLUMN
    if FALLBACK_TEXT_COLUMN in df.columns:
        print(
            f"[WARN] '{RAW_TEXT_COLUMN}' missing in {split_name}; "
            f"falling back to '{FALLBACK_TEXT_COLUMN}'."
        )
        return FALLBACK_TEXT_COLUMN
    sys.exit(
        f"[ERROR] {split_name} split has neither '{RAW_TEXT_COLUMN}' nor "
        f"'{FALLBACK_TEXT_COLUMN}' column. Found: {list(df.columns)}"
    )


def load_split(data_path: str, split_name: str) -> tuple[pd.Series, pd.Series]:
    """Load one split and return ``(text, label)`` aligned series.

    Args:
        data_path: Path to the split file (CSV / pickle / parquet).
        split_name: Human-readable name used in log/error messages.

    Returns:
        A ``(X_text, y_label)`` tuple with a clean RangeIndex. ``X_text`` holds
        raw strings; ``y_label`` holds integer class labels.
    """
    path = Path(data_path)
    if not path.exists():
        sys.exit(
            f"[ERROR] Data file not found: {path.resolve()}\n"
            "Run 'python src/data/split_dataset.py' first to generate the splits."
        )

    df = _read_any(path)

    text_column = _resolve_text_column(df, split_name)
    if LABEL_COLUMN not in df.columns:
        sys.exit(
            f"[ERROR] {split_name} split is missing the '{LABEL_COLUMN}' column. "
            f"Found: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    # Empty/NaN text is allowed through as an empty string: the FeatureExtractor
    # handles it gracefully and dropping rows would desync ensemble alignment.
    df[text_column] = df[text_column].fillna("").astype(str)

    try:
        labels = df[LABEL_COLUMN].astype(int)
    except (ValueError, TypeError):
        sys.exit(
            f"[ERROR] Non-integer labels found in the {split_name} split. "
            "Expected binary 0/1 sentiment labels."
        )

    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows from {split_name} with null labels.")

    _validate(labels, split_name)
    print(
        f"[INFO] Loaded {len(df):,} {split_name} samples from '{path}' "
        f"(text column: '{text_column}')."
    )
    return df[text_column].reset_index(drop=True), labels.reset_index(drop=True)


def _validate(labels: pd.Series, split_name: str) -> None:
    """Fail early on degenerate label distributions."""
    if labels.empty:
        sys.exit(f"[ERROR] No usable rows found in the {split_name} split.")
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        print(
            f"[WARN] The {split_name} split contains a single class "
            f"({class_counts.to_dict()}). Metrics may be degenerate."
        )


def load_splits(
    train_path: str = DEFAULT_TRAIN_PATH,
    val_path: str = DEFAULT_VAL_PATH,
    test_path: str = DEFAULT_TEST_PATH,
) -> tuple[
    pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
]:
    """Load all three splits.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.
    """
    X_train, y_train = load_split(train_path, "train")
    X_val, y_val = load_split(val_path, "validation")
    X_test, y_test = load_split(test_path, "test")
    print(
        f"[INFO] Using existing splits | Train: {len(X_train):,} | "
        f"Validation: {len(X_val):,} | Test: {len(X_test):,}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
