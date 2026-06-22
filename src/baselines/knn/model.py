"""Model setup and training helpers for the TF-IDF + KNN baseline."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

MAX_FEATURES: int | None = 50_000
NGRAM_RANGE: tuple[int, int] = (1, 2)
MIN_DF: int = 2
MAX_DF: float = 0.95
K_VALUES: tuple[int, ...] = (3, 5, 7, 9, 11, 15, 21)
WEIGHTS: tuple[str, ...] = ("uniform", "distance")
METRIC: str = "cosine"
SCORING: str = "f1_macro"


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
    """Construct the TF-IDF + KNN pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\S+",
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


def filter_k_values(k_values: tuple[int, ...], n_train_samples: int) -> tuple[int, ...]:
    """Keep k values that are valid for the training split."""
    safe_k_values = tuple(value for value in k_values if value <= n_train_samples)

    if not safe_k_values:
        fallback_k = max(1, n_train_samples)
        print(
            "[WARN] No requested k values fit the training split; "
            f"using k={fallback_k}."
        )
        return (fallback_k,)

    skipped = sorted(set(k_values) - set(safe_k_values))
    if skipped:
        print(f"[INFO] Skipped k values too large for training split: {skipped}")

    return safe_k_values


def tune_knn(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    max_features: int | None,
    k_values: tuple[int, ...],
    weights: tuple[str, ...],
    scoring: str,
) -> dict[str, Any]:
    """Tune k and vote weighting on the existing validation split."""
    safe_k_values = filter_k_values(k_values, len(X_train))
    print(
        "[INFO] Tuning KNN on validation split "
        f"(k_values={safe_k_values}, weights={weights}, metric={METRIC}, "
        f"scoring={scoring}) ..."
    )

    best_score = -np.inf
    best_params: dict[str, Any] = {}

    for k in safe_k_values:
        for weight in weights:
            pipeline = build_pipeline(
                max_features=max_features,
                ngram_range=NGRAM_RANGE,
                min_df=MIN_DF,
                max_df=MAX_DF,
            )
            pipeline.set_params(
                knn__n_neighbors=k,
                knn__weights=weight,
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)

            if score > best_score:
                best_score = score
                best_params = {
                    "knn__n_neighbors": k,
                    "knn__weights": weight,
                }

    print(f"[INFO] Best validation {scoring}: {best_score:.4f}")
    print(f"[INFO] Best parameters: {best_params}")
    return best_params


def train_final_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    max_features: int | None,
    best_params: dict[str, Any],
) -> Pipeline:
    """Refit the selected model on train + validation data."""
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    pipeline = build_pipeline(
        max_features=max_features,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    pipeline.set_params(**best_params)
    print(
        "[INFO] Fitting final model on train + validation splits "
        f"({len(X_train_val):,} samples) ..."
    )
    pipeline.fit(X_train_val, y_train_val)
    print("[INFO] Training complete.")
    return pipeline
