# TODO: Delete later as files have already been brokken up into smaller modules. This file is kept for reference only.

"""
TF-IDF + SVM Baseline for Tweet Sentiment Classification
=========================================================
Loads cleaned tweet data, applies TF-IDF vectorization, trains a LinearSVC
classifier, and reports evaluation metrics on the held-out test set.

Usage:
    python src/baselines/svm/tfidf_svm.py
    python src/baselines/svm/tfidf_svm.py --data-path data/processed/cleaned_tweets.csv
    python src/baselines/svm/tfidf_svm.py --max-features 30000 --C 0.5 --test-size 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Hyperparameters — adjust these without touching the rest of the script
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "data/processed/cleaned_tweets.csv"
MAX_FEATURES: int | None = 50_000   # TF-IDF vocabulary cap (None = unlimited)
NGRAM_RANGE: tuple[int, int] = (1, 2)  # unigrams + bigrams
MIN_DF: int = 2                     # ignore terms that appear in fewer docs
MAX_DF: float = 0.95                # ignore terms that appear in >95 % of docs
SVM_C: float = 1.0                  # regularisation strength (smaller = stronger)
SVM_MAX_ITER: int = 2_000           # max iterations for LinearSVC solver
TEST_SIZE: float = 0.20             # fraction of data held out for evaluation
RANDOM_STATE: int = 42              # reproducibility seed
TEXT_COLUMN: str = "clean_text"     # column that holds the pre-cleaned text
LABEL_COLUMN: str = "label"         # column that holds the binary sentiment label
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

    # Validate required columns
    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")

    # Drop rows where the text or label is null
    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows with null text/label.")

    print(f"[INFO] Loaded {len(df):,} samples from '{path}'.")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split to preserve class balance."""
    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(
        f"[INFO] Train: {len(X_train):,} samples | "
        f"Test: {len(X_test):,} samples  "
        f"(test_size={test_size})"
    )
    return X_train, X_test, y_train, y_test


def build_vectorizer(
    max_features: int | None,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer with the given settings."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,   # apply log(1 + tf) to dampen high-frequency terms
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\S+",  # whitespace-tokenised (text is already cleaned)
    )


def vectorize(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series,
    X_test: pd.Series,
) -> tuple:
    """Fit the vectorizer on training data, then transform both splits.

    Fitting only on training data prevents data leakage from the test set.
    """
    print("[INFO] Fitting TF-IDF vectorizer on training data …")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"[INFO] Vocabulary size: {vocab_size:,} terms.")
    return X_train_tfidf, X_test_tfidf


def train_svm(
    X_train_tfidf,
    y_train: pd.Series,
    C: float,
    max_iter: int,
    random_state: int,
) -> LinearSVC:
    """Train a LinearSVC on the TF-IDF features.

    LinearSVC is preferred over SVC(kernel='linear') for large sparse matrices
    because it uses a more efficient optimisation algorithm (liblinear).
    """
    print(f"[INFO] Training LinearSVC  (C={C}, max_iter={max_iter}) …")
    clf = LinearSVC(C=C, max_iter=max_iter, random_state=random_state)
    clf.fit(X_train_tfidf, y_train)
    print("[INFO] Training complete.")
    return clf


def evaluate(clf: LinearSVC, X_test_tfidf, y_test: pd.Series) -> None:
    """Print accuracy, per-class precision/recall/F1, and the confusion matrix."""
    y_pred = clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f} %)")
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels : [Negative, Positive]")
    print(f"  {cm}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + LinearSVC sentiment baseline."
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
        "--C",
        type=float,
        default=SVM_C,
        dest="svm_c",
        help="LinearSVC regularisation parameter C (default: %(default)s).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load cleaned data
    df = load_data(args.data_path)

    # 2. Train / test split
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 3. Build and fit TF-IDF vectorizer; transform both splits
    vectorizer = build_vectorizer(
        max_features=args.max_features,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    X_train_tfidf, X_test_tfidf = vectorize(vectorizer, X_train, X_test)

    # 4. Train LinearSVC on TF-IDF features
    clf = train_svm(
        X_train_tfidf,
        y_train,
        C=args.svm_c,
        max_iter=SVM_MAX_ITER,
        random_state=args.random_state,
    )

    # 5. Evaluate on the held-out test set
    evaluate(clf, X_test_tfidf, y_test)


if __name__ == "__main__":
    main()
