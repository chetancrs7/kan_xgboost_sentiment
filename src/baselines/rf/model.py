"""Model setup, tuning, and training helpers for the TF-IDF + Random Forest baseline.

The model vectorizes pre-cleaned tweets with TF-IDF and classifies sentiment with
a Random Forest of decision trees, whose votes are averaged for the prediction.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# TF-IDF defaults.
MAX_FEATURES: int = 50_000             # TF-IDF vocabulary cap (None = unlimited)
NGRAM_RANGE: tuple[int, int] = (1, 2)  # unigrams + bigrams
MIN_DF: int = 2                        # ignore terms that appear in fewer docs
MAX_DF: float = 0.95                   # ignore terms that appear in >95% of docs

# Training defaults.
CLASS_WEIGHT: str = "balanced"         # counter class imbalance during training
RANDOM_STATE: int = 42                 # reproducibility seed
SCORING: str = "f1_macro"              # robust default for sentiment class balance
N_JOBS: int = -1                       # cores used to grow trees (-1 = all)

# Fixed Random Forest hyperparameters.
MAX_FEATURES_SPLIT: str = "sqrt"       # features considered per split

# Tuned Random Forest hyperparameters (grid searched on the validation split).
N_ESTIMATORS_VALUES: tuple[int, ...] = (200, 400)
MAX_DEPTH_VALUES: tuple[int | None, ...] = (None, 50)
MIN_SAMPLES_LEAF_VALUES: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class RandomForestParams:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    """Seed Python and NumPy for repeatable runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def param_candidates(
    n_estimators_values: tuple[int, ...] = N_ESTIMATORS_VALUES,
    max_depth_values: tuple[int | None, ...] = MAX_DEPTH_VALUES,
    min_samples_leaf_values: tuple[int, ...] = MIN_SAMPLES_LEAF_VALUES,
) -> list[RandomForestParams]:
    """Enumerate the forest configurations to evaluate during tuning."""
    return [
        RandomForestParams(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            max_features=MAX_FEATURES_SPLIT,
        )
        for n_estimators in n_estimators_values
        for max_depth in max_depth_values
        for min_samples_leaf in min_samples_leaf_values
    ]


def build_vectorizer(max_features: int | None = MAX_FEATURES) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer for the pre-cleaned tweet text."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\S+",  # whitespace-tokenised (text is already cleaned)
    )


def build_model(
    params: RandomForestParams,
    max_features: int | None = MAX_FEATURES,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
) -> Pipeline:
    """Construct the TF-IDF + Random Forest pipeline for the given params.

    Keeping vectorization inside the Pipeline means the TF-IDF vocabulary is
    fitted on the training split only, which prevents validation/test leakage.
    """
    forest = RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        class_weight=CLASS_WEIGHT,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer(max_features)),
            ("rf", forest),
        ]
    )


def tune_model(
    train_texts: pd.Series,
    y_train: pd.Series,
    val_texts: pd.Series,
    y_val: pd.Series,
    max_features: int | None = MAX_FEATURES,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    candidates: list[RandomForestParams] | None = None,
) -> tuple[RandomForestParams, list[dict[str, object]]]:
    """Grid-search forest params on the validation split (macro-F1).

    Returns the best parameters plus the full tuning history (one record
    per candidate) for visualization.
    """
    candidates = candidates if candidates is not None else param_candidates()
    if len(candidates) == 1:
        print(f"[INFO] Single candidate; skipping tuning: {candidates[0]}")
        return candidates[0], []

    print(
        f"[INFO] Tuning Random Forest on validation split "
        f"({len(candidates)} candidates, scoring={SCORING}) ..."
    )

    best_score = -np.inf
    best_params = candidates[0]
    history: list[dict[str, object]] = []

    for params in candidates:
        model = build_model(
            params,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(train_texts, y_train)
        y_pred = model.predict(val_texts)
        score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"[INFO]   {params} -> val macro-F1={score:.4f}")
        history.append(
            {
                "n_estimators": params.n_estimators,
                "max_depth": params.max_depth,
                "min_samples_leaf": params.min_samples_leaf,
                "score": float(score),
            }
        )

        if score > best_score:
            best_score = score
            best_params = params

    print(f"[INFO] Best validation {SCORING}: {best_score:.4f}")
    print(f"[INFO] Best parameters: {best_params}")
    return best_params, history


def train_final_model(
    train_texts: pd.Series,
    y_train: pd.Series,
    best_params: RandomForestParams,
    max_features: int | None = MAX_FEATURES,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
) -> Pipeline:
    """Train the selected configuration on the training split."""
    print(
        f"[INFO] Fitting final model on {len(train_texts):,} samples "
        f"(n_estimators={best_params.n_estimators}, max_depth={best_params.max_depth}, "
        f"min_samples_leaf={best_params.min_samples_leaf}) ..."
    )
    model = build_model(
        best_params,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(train_texts, y_train)
    print("[INFO] Training complete.")
    return model
