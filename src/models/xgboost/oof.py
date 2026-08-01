"""Leakage-safe out-of-fold XGBoost residuals and KAN training weights.

This module implements the *residual-aware specialisation* signal described in
the hybrid methodology: it teaches the KAN semantic expert to concentrate on the
tweets the XGBoost structural expert finds hard, **without** ever sharing
features between the two experts and **without** leaking labels.

Pipeline
--------
1. :func:`compute_oof_probs` fits XGBoost on the *interpretable structural
   feature matrix only* using stratified K-fold cross-validation. For each fold
   the model is trained on the remaining folds and used to predict the held-out
   samples, so every training tweet receives a structural probability from a
   model that has never seen it (leakage-safe out-of-fold prediction).
2. :func:`true_class_probs` converts those probabilities into the per-sample
   probability XGBoost assigns to the *correct* class. A low value means the
   structural expert struggled; a high value means the tweet was structurally
   easy.
3. :func:`residual_weights` maps the residual ``1 - p_true`` into a bounded,
   capped KAN training weight (percentile-rank based by default, to be robust to
   imperfect probability calibration). Structurally easy tweets keep a normal
   weight of ``1.0``; structurally hard tweets get a *moderately* higher weight,
   never exceeding ``w_max``.

:func:`build_residual_targets` runs the whole pipeline and returns a
:class:`ResidualTargets` bundle (weights + XGBoost true-class probs + a
hard-failure indicator) that the residual-aware KAN trainer consumes.

Nothing here mutates the trained XGBoost expert on disk; the folds are fitted on
in-memory copies purely to derive the complementarity signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.xgboost.model import RANDOM_STATE, _scale_pos_weight, build_classifier


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean rank), 1-indexed.

    A tiny dependency-free stand-in for ``scipy.stats.rankdata`` so this module
    stays importable without scipy. Used to turn residuals into percentile
    ranks, which are insensitive to the (imperfect) absolute calibration of the
    XGBoost probabilities.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average tied ranks so equal residuals get equal weight.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    avg = (starts + cumulative + 1) / 2.0  # mean of 1-indexed ranks in each tie group
    return avg[inverse]


def compute_oof_probs(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> np.ndarray:
    """Leakage-safe out-of-fold XGBoost probabilities on structural features.

    Args:
        X: Structural / linguistic feature matrix ``[n, d]`` (the XGBoost
            expert's own features — never the DeBERTa embeddings).
        y: Integer labels ``[n]``.
        params: Optional fixed XGBoost hyperparameters (e.g. loaded from the
            trained expert's ``best_params.json``). ``scale_pos_weight`` is
            filled per-fold for binary tasks when absent. ``None`` uses library
            defaults, which is fine for deriving a difficulty signal.
        n_splits: Number of stratified CV folds.
        seed: RNG seed for the fold split and the models.
        verbose: Print per-fold progress.

    Returns:
        ``oof_probs`` of shape ``[n, n_classes]`` where row ``i`` is the
        prediction for sample ``i`` from a model that did **not** train on it.
    """
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) are misaligned.")
    classes = np.unique(y)
    n_classes = int(len(classes))
    if n_classes < 2:
        raise ValueError("Need at least two classes for out-of-fold prediction.")

    n_splits = max(2, min(n_splits, int(np.min(np.bincount(y)))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), n_classes), dtype=np.float64)

    base_params = dict(params or {})
    base_params.pop("scale_pos_weight", None)  # refilled per fold from the fold's y

    for fold, (train_idx, held_idx) in enumerate(skf.split(X, y), start=1):
        fold_params = dict(base_params)
        if n_classes == 2:
            fold_params["scale_pos_weight"] = _scale_pos_weight(y[train_idx])
        model = build_classifier(fold_params, n_classes)
        model.fit(X[train_idx], y[train_idx], verbose=False)
        proba = model.predict_proba(X[held_idx])
        # Align the estimator's class order to 0..n_classes-1.
        oof[held_idx[:, None], model.classes_[None, :]] = proba
        if verbose:
            fold_acc = float((proba.argmax(axis=1) == y[held_idx]).mean())
            print(
                f"[OOF] fold {fold}/{n_splits}: "
                f"{len(train_idx):,} train / {len(held_idx):,} held-out "
                f"| held-out acc={fold_acc:.4f}"
            )
    return oof


def true_class_probs(oof_probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability the (out-of-fold) XGBoost assigns to each sample's true class."""
    oof_probs = np.asarray(oof_probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    return oof_probs[np.arange(len(y)), y]


def residual_weights(
    oof_probs: np.ndarray,
    y: np.ndarray,
    method: str = "percentile",
    w_max: float = 3.0,
    floor: float = 1.0,
    scale: float = 2.0,
    double_fault_cap: float | None = None,
    xgb_fail: np.ndarray | None = None,
) -> np.ndarray:
    """Convert XGBoost residuals into bounded, capped KAN training weights.

    The residual is ``1 - p_true`` (high when the structural expert struggled).

    Args:
        oof_probs: Out-of-fold XGBoost probabilities ``[n, C]``.
        y: Integer labels ``[n]``.
        method: ``"percentile"`` (default, robust to calibration — the weight is
            a monotone map of the residual's rank) or ``"linear"``
            (``clip(floor + scale * residual, floor, w_max)``).
        w_max: Hard cap on any sample's weight; structurally easy tweets get
            ``floor``. Keeps noisy/mislabeled outliers from dominating.
        floor: Baseline weight for structurally easy tweets (normally ``1.0``).
        scale: Slope for ``method="linear"``.
        double_fault_cap: Optional lower cap applied to samples flagged in
            ``xgb_fail`` that the KAN is *also* expected to miss. When both
            ``double_fault_cap`` and ``xgb_fail`` are given, weights of those
            "hard for XGBoost" samples are additionally capped at this value so
            genuinely ambiguous / mislabeled cases receive only limited emphasis
            (see the methodology's double-fault caveat). Left ``None`` the plain
            ``w_max`` cap already bounds them.
        xgb_fail: Optional boolean mask used with ``double_fault_cap``.

    Returns:
        A ``[n]`` float64 weight vector in ``[floor, w_max]``.
    """
    p_true = true_class_probs(oof_probs, y)
    residual = 1.0 - p_true

    if method == "percentile":
        ranks = _rankdata(residual)
        # Normalise to [0, 1]; the smallest residual maps to 0 (weight == floor).
        rank01 = (ranks - 1.0) / max(1.0, len(residual) - 1.0)
        weights = floor + (w_max - floor) * rank01
    elif method == "linear":
        weights = floor + scale * residual
    else:
        raise ValueError(f"Unknown weighting method: {method!r}.")

    weights = np.clip(weights, floor, w_max)

    if double_fault_cap is not None and xgb_fail is not None:
        xgb_fail = np.asarray(xgb_fail, dtype=bool)
        capped = np.minimum(weights, float(double_fault_cap))
        weights = np.where(xgb_fail, capped, weights)

    return weights.astype(np.float64)


@dataclass
class ResidualTargets:
    """Per-sample complementarity signals derived from the structural expert.

    Attributes:
        weights: Bounded KAN training weight (structurally hard → higher).
        xgb_true_prob: Out-of-fold XGBoost probability of the true class.
        xgb_fail: 1.0 where the out-of-fold XGBoost misclassified the sample
            (the target for the optional auxiliary competence head).
        oof_probs: The raw out-of-fold probability matrix ``[n, C]``.
    """

    weights: np.ndarray
    xgb_true_prob: np.ndarray
    xgb_fail: np.ndarray
    oof_probs: np.ndarray

    def summary(self) -> dict[str, float]:
        """Compact stats for logging / the training manifest."""
        return {
            "n": int(len(self.weights)),
            "weight_mean": float(self.weights.mean()),
            "weight_max": float(self.weights.max()),
            "weight_min": float(self.weights.min()),
            "xgb_oof_accuracy": float(1.0 - self.xgb_fail.mean()),
            "xgb_true_prob_mean": float(self.xgb_true_prob.mean()),
            "frac_hard_xgb": float((self.xgb_true_prob < 0.5).mean()),
        }


def build_residual_targets(
    X_struct: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
    method: str = "percentile",
    w_max: float = 3.0,
    floor: float = 1.0,
    scale: float = 2.0,
    double_fault_cap: float | None = None,
    verbose: bool = True,
) -> ResidualTargets:
    """Run the full leakage-safe residual → weight pipeline on structural feats.

    Args:
        X_struct: Structural feature matrix (XGBoost's own features).
        y: Integer labels.
        params, n_splits, seed: Passed to :func:`compute_oof_probs`.
        method, w_max, floor, scale, double_fault_cap: Passed to
            :func:`residual_weights`.
        verbose: Print progress.

    Returns:
        A :class:`ResidualTargets` bundle aligned row-for-row with ``X_struct``.
    """
    oof = compute_oof_probs(
        X_struct, y, params=params, n_splits=n_splits, seed=seed, verbose=verbose
    )
    xgb_fail = (oof.argmax(axis=1) != np.asarray(y, dtype=np.int64)).astype(np.float64)
    weights = residual_weights(
        oof,
        y,
        method=method,
        w_max=w_max,
        floor=floor,
        scale=scale,
        double_fault_cap=double_fault_cap,
        xgb_fail=xgb_fail,
    )
    targets = ResidualTargets(
        weights=weights,
        xgb_true_prob=true_class_probs(oof, y),
        xgb_fail=xgb_fail,
        oof_probs=oof,
    )
    if verbose:
        print(f"[OOF] residual targets: {targets.summary()}")
    return targets
