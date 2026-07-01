"""XGBoost model construction and Optuna hyperparameter optimisation.

The search optimises macro F1 on the validation split. The final model is then
refit on the best parameters with early stopping against the validation set.
All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)

RANDOM_STATE: int = 42
DEFAULT_N_TRIALS: int = 50
EARLY_STOPPING_ROUNDS: int = 30


def _scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negative to positive samples for class-imbalance correction."""
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def build_classifier(params: dict[str, Any], n_classes: int) -> XGBClassifier:
    """Instantiate an ``XGBClassifier`` with sensible fixed defaults merged in.

    Args:
        params: Tunable hyperparameters (depth, learning rate, ...).
        n_classes: Number of target classes (selects the objective).

    Returns:
        An unfitted :class:`xgboost.XGBClassifier`.
    """
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    fixed: dict[str, Any] = {
        "objective": objective,
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if n_classes > 2:
        fixed["num_class"] = n_classes
    return XGBClassifier(**fixed, **params)


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter configuration from the search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], optuna.Study]:
    """Run Optuna to maximise validation macro F1.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels (search objective).
        n_trials: Number of Optuna trials.
        random_state: Seed for the sampler and models.

    Returns:
        ``(best_params, study)`` where ``best_params`` are the winning
        hyperparameters and ``study`` is the completed Optuna study.
    """
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))
    scale_pos_weight = _scale_pos_weight(y_train)
    average = "binary" if n_classes == 2 else "macro"

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        if n_classes == 2:
            params["scale_pos_weight"] = scale_pos_weight
        model = build_classifier(params, n_classes)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average=average, zero_division=0)
        return float(score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"[INFO] Starting Optuna search: {n_trials} trials, objective=val F1 ({average}).")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    if n_classes == 2:
        best_params["scale_pos_weight"] = scale_pos_weight

    print(f"[INFO] Best validation F1: {study.best_value:.4f}")
    print("[INFO] Best parameters:")
    for key, value in best_params.items():
        print(f"         {key}: {value}")
    return best_params, study


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Fit the final XGBoost model with early stopping on the validation set.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data used for early stopping.
        best_params: Hyperparameters from :func:`optimize_xgboost`.
        random_state: Reproducibility seed.

    Returns:
        The fitted :class:`xgboost.XGBClassifier`.
    """
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val]))))
    params = dict(best_params)
    model = build_classifier(params, n_classes)
    # Re-instantiate with early stopping so the best boosting round is kept.
    model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        print(f"[INFO] Final model trained. Best iteration: {best_iteration}.")
    else:
        print("[INFO] Final model trained.")
    return model
