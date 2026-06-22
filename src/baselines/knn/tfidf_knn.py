# TODO: Remove this file once testing is done for the other files
"""
TF-IDF + KNN Baseline for Tweet Sentiment Classification
========================================================
Loads cleaned tweet data, applies TF-IDF vectorization, tunes a
KNeighborsClassifier on the training split, and reports evaluation metrics on
the held-out test set.

Usage:
    python src/baselines/knn/tfidf_knn.py
    python src/baselines/knn/tfidf_knn.py --data-path data/processed/cleaned_tweets.csv
    python src/baselines/knn/tfidf_knn.py --k-values 3,5,7,9,11 --cv-folds 5 --test-size 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Hyperparameters - adjust these without touching the rest of the script
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "data/processed/cleaned_tweets.csv"
MAX_FEATURES: int | None = 50_000      # TF-IDF vocabulary cap (None = unlimited)
NGRAM_RANGE: tuple[int, int] = (1, 2)  # unigrams + bigrams
MIN_DF: int = 2                        # ignore terms that appear in fewer docs
MAX_DF: float = 0.95                   # ignore terms that appear in >95% of docs
K_VALUES: tuple[int, ...] = (3, 5, 7, 9, 11, 15, 21)
WEIGHTS: tuple[str, ...] = ("uniform", "distance")
METRIC: str = "cosine"                 # cosine distance works well for TF-IDF
CV_FOLDS: int = 5
TEST_SIZE: float = 0.20                # fraction of data held out for evaluation
RANDOM_STATE: int = 42                 # reproducibility seed
TEXT_COLUMN: str = "clean_text"        # column that holds the pre-cleaned text
LABEL_COLUMN: str = "label"            # column that holds the binary sentiment label
SCORING: str = "f1_macro"              # robust default for sentiment class balance
N_JOBS: int = -1
# ---------------------------------------------------------------------------


def load_data(data_path: str) -> pd.DataFrame:
    """Load the processed CSV produced by text_cleaning.py."""
    path = Path(data_path)
    if not path.exists():
        sys.exit(
            f"[ERROR] Data file not found: {path.resolve()}\n"
            "Run 'python src/data/text_cleaning.py' first to generate it."
        )

    df = pd.read_csv(path)

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")

    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows with null/empty text or label.")

    validate_dataframe(df)
    print(f"[INFO] Loaded {len(df):,} samples from '{path}'.")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Fail early with actionable messages for common data issues."""
    if df.empty:
        sys.exit("[ERROR] No usable rows found after cleaning null/empty values.")

    class_counts = df[LABEL_COLUMN].value_counts()
    if len(class_counts) < 2:
        sys.exit("[ERROR] At least two sentiment classes are required.")

    rarest_class_count = int(class_counts.min())
    if rarest_class_count < 2:
        sys.exit(
            "[ERROR] Each class needs at least two samples for a stratified "
            f"train/test split. Class counts: {class_counts.to_dict()}"
        )


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split to preserve class balance."""
    if not 0 < test_size < 1:
        sys.exit("[ERROR] --test-size must be between 0 and 1.")

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] Could not create stratified train/test split: {exc}")

    print(
        f"[INFO] Train: {len(X_train):,} samples | "
        f"Test: {len(X_test):,} samples  "
        f"(test_size={test_size})"
    )
    return X_train, X_test, y_train, y_test


def parse_k_values(raw_k_values: str) -> tuple[int, ...]:
    """Parse comma-separated k values from the CLI."""
    try:
        values = tuple(sorted({int(value.strip()) for value in raw_k_values.split(",")}))
    except ValueError:
        sys.exit("[ERROR] --k-values must be a comma-separated list of integers.")

    if not values or any(value <= 0 for value in values):
        sys.exit("[ERROR] --k-values must contain positive integers.")

    return values


def build_pipeline(
    max_features: int | None,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
) -> Pipeline:
    """Construct the TF-IDF + KNN pipeline.

    Keeping vectorization inside the Pipeline lets GridSearchCV fit TF-IDF only
    on each training fold, which prevents validation-fold leakage.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\S+",  # whitespace-tokenised (text is already cleaned)
    )

    knn = KNeighborsClassifier(
        algorithm="brute",
        metric=METRIC,
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("knn", knn),
        ]
    )


def available_cv_folds(y_train: pd.Series, requested_cv_folds: int) -> int:
    """Choose the largest safe number of stratified folds for the data."""
    if requested_cv_folds < 2:
        sys.exit("[ERROR] --cv-folds must be at least 2.")

    rarest_class_count = int(y_train.value_counts().min())
    return min(requested_cv_folds, rarest_class_count)


def filter_k_values(k_values: tuple[int, ...], n_train_samples: int, cv_folds: int) -> tuple[int, ...]:
    """Keep k values that are valid for every training fold."""
    min_fold_train_size = n_train_samples - int(np.ceil(n_train_samples / cv_folds))
    safe_k_values = tuple(value for value in k_values if value <= min_fold_train_size)

    if not safe_k_values:
        fallback_k = max(1, min_fold_train_size)
        print(
            "[WARN] No requested k values fit the cross-validation folds; "
            f"using k={fallback_k}."
        )
        return (fallback_k,)

    skipped = sorted(set(k_values) - set(safe_k_values))
    if skipped:
        print(f"[INFO] Skipped k values too large for CV folds: {skipped}")

    return safe_k_values


def tune_knn(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    k_values: tuple[int, ...],
    weights: tuple[str, ...],
    cv_folds: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
) -> GridSearchCV:
    """Tune k and vote weighting with stratified cross-validation."""
    safe_cv_folds = available_cv_folds(y_train, cv_folds)
    if safe_cv_folds != cv_folds:
        print(
            f"[INFO] Reduced CV folds from {cv_folds} to {safe_cv_folds} "
            "because of class counts."
        )

    safe_k_values = filter_k_values(k_values, len(X_train), safe_cv_folds)
    cv = StratifiedKFold(
        n_splits=safe_cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    param_grid: dict[str, tuple[Any, ...]] = {
        "knn__n_neighbors": safe_k_values,
        "knn__weights": weights,
    }

    print(
        "[INFO] Tuning KNN with GridSearchCV "
        f"(k_values={safe_k_values}, weights={weights}, metric={METRIC}, "
        f"cv={safe_cv_folds}, scoring={scoring}) ..."
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    print(f"[INFO] Best CV {scoring}: {search.best_score_:.4f}")
    print(f"[INFO] Best parameters: {search.best_params_}")
    print("[INFO] Training complete.")
    return search


def evaluate(model: GridSearchCV | Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Print and return accuracy, precision, recall, F1, and confusion matrix."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "model": "TF-IDF + KNN",
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("Model    : TF-IDF + KNN")
    print(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    print(f"Precision: {precision:.4f}  (macro avg)")
    print(f"Recall   : {recall:.4f}  (macro avg)")
    print(f"F1-score : {f1:.4f}  (macro avg)")
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("  Labels : [Negative, Positive]")
    print(f"  {cm}")
    print("=" * 60)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + KNN sentiment baseline."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned tweets CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help="Maximum TF-IDF vocabulary size (default: %(default)s).",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(value) for value in K_VALUES),
        help="Comma-separated k values to tune (default: %(default)s).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help="Number of stratified CV folds for tuning (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data used for testing (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="Parallel jobs for GridSearchCV (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    pipeline = build_pipeline(
        max_features=args.max_features,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    try:
        model = tune_knn(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            k_values=parse_k_values(args.k_values),
            weights=WEIGHTS,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            scoring=SCORING,
            n_jobs=args.n_jobs,
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] KNN training failed: {exc}")

    best_pipeline = model.best_estimator_
    X_train_tfidf = best_pipeline.named_steps["tfidf"].transform(X_train)
    if sparse.issparse(X_train_tfidf):
        vocab_size = len(best_pipeline.named_steps["tfidf"].vocabulary_)
        print(f"[INFO] Vocabulary size: {vocab_size:,} terms.")

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
