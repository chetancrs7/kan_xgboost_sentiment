"""Model setup and training helpers for the TF-IDF + SVM baseline."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MAX_FEATURES: int | None = 50_000
NGRAM_RANGE: tuple[int, int] = (1, 2)
MIN_DF: int = 2
MAX_DF: float = 0.95
C_VALUES: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")
SVM_MAX_ITER: int = 2_000
SCORING: str = "f1_macro"


def parse_c_values(raw_c_values: str) -> tuple[float, ...]:
    """Parse comma-separated C values from the CLI."""
    try:
        values = tuple(sorted({float(value.strip()) for value in raw_c_values.split(",")}))
    except ValueError:
        sys.exit("[ERROR] --c-values must be a comma-separated list of numbers.")

    if not values or any(value <= 0 for value in values):
        sys.exit("[ERROR] --c-values must contain positive numbers.")

    return values


def build_pipeline(
    max_features: int | None,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    random_state: int,
) -> Pipeline:
    """Construct the TF-IDF + LinearSVC pipeline."""
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

    svm = LinearSVC(
        max_iter=SVM_MAX_ITER,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("svm", svm),
        ]
    )


def tune_svm(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    max_features: int | None,
    c_values: tuple[float, ...],
    class_weights: tuple[str | None, ...],
    scoring: str,
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune C and class weighting on the existing validation split.

    Returns the best parameters plus the full tuning history (one record
    per candidate) for visualization.
    """
    print(
        "[INFO] Tuning SVM on validation split "
        f"(c_values={c_values}, class_weights={class_weights}, "
        f"scoring={scoring}) ..."
    )

    best_score = -np.inf
    best_params: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for c in c_values:
        for class_weight in class_weights:
            pipeline = build_pipeline(
                max_features=max_features,
                ngram_range=NGRAM_RANGE,
                min_df=MIN_DF,
                max_df=MAX_DF,
                random_state=random_state,
            )
            pipeline.set_params(
                svm__C=c,
                svm__class_weight=class_weight,
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            history.append(
                {"C": c, "class_weight": class_weight, "score": float(score)}
            )

            if score > best_score:
                best_score = score
                best_params = {
                    "svm__C": c,
                    "svm__class_weight": class_weight,
                }

    print(f"[INFO] Best validation {scoring}: {best_score:.4f}")
    print(f"[INFO] Best parameters: {best_params}")
    return best_params, history


def train_final_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    max_features: int | None,
    best_params: dict[str, Any],
    random_state: int,
) -> Pipeline:
    """Refit the selected model on train + validation data."""
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    pipeline = build_pipeline(
        max_features=max_features,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        random_state=random_state,
    )
    pipeline.set_params(**best_params)
    print(
        "[INFO] Fitting final model on train + validation splits "
        f"({len(X_train_val):,} samples) ..."
    )
    pipeline.fit(X_train_val, y_train_val)
    print("[INFO] Training complete.")
    return pipeline
