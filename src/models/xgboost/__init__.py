"""XGBoost expert sentiment classifier with engineered linguistic features.

This package builds a production-ready XGBoost sentiment model on top of an
interpretable ``FeatureExtractor`` (negation, intensifiers, emoji sentiment,
sarcasm cues, punctuation, capitalisation, POS distribution, subjectivity).

Public entry points:
    - ``FeatureExtractor``     : raw text -> numeric feature matrix
    - ``load_splits``          : read pre-split train/val/test CSVs
    - ``optimize_xgboost``     : Optuna hyperparameter search
    - ``train_final_model``    : fit XGBoost on the best parameters
    - ``evaluate_split``       : metrics for a single split
"""

from __future__ import annotations

from .data_loader import load_split, load_splits
from .evaluate import evaluate_split, summarize_feature_importance
from .features import FeatureExtractor
from .model import optimize_xgboost, train_final_model

__all__ = [
    "FeatureExtractor",
    "load_split",
    "load_splits",
    "optimize_xgboost",
    "train_final_model",
    "evaluate_split",
    "summarize_feature_importance",
]
