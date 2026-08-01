"""Shared helper for persisting baseline evaluation metrics to JSON.

Every baseline's ``evaluate()`` returns the same results dict (model name,
macro accuracy/precision/recall/F1, and the confusion matrix) but only prints
it. This module serializes that dict to ``<out_dir>/metrics.json`` so a
baseline run leaves a machine-readable record next to its plots, matching the
``metrics.json`` files the KAN and XGBoost experts already write.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Fallback serializer for numpy scalars/arrays that json can't handle."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_metrics(
    results: dict[str, Any],
    out_dir: str | Path,
    *,
    n_test_samples: int | None = None,
    best_params: dict[str, Any] | None = None,
    filename: str = "metrics.json",
) -> Path:
    """Write an ``evaluate()`` results dict to ``<out_dir>/<filename>``.

    ``results`` is the dict every baseline's ``evaluate()`` returns: ``model``,
    ``accuracy``, ``macro_precision``, ``macro_recall``, ``macro_f1``, and
    ``confusion_matrix``. Numpy scalars and the confusion-matrix array are
    converted to native Python types so the JSON is portable. ``best_params``
    (the tuned hyperparameters) and ``n_test_samples`` are recorded when given.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": results["model"],
        "split": "test",
        "accuracy": float(results["accuracy"]),
        "macro_precision": float(results["macro_precision"]),
        "macro_recall": float(results["macro_recall"]),
        "macro_f1": float(results["macro_f1"]),
        "confusion_matrix": np.asarray(results["confusion_matrix"]).tolist(),
    }
    if n_test_samples is not None:
        payload["n_samples"] = int(n_test_samples)
    if best_params is not None:
        # best_params may be a plain dict (KNN/SVM) or a frozen dataclass
        # (RF's RandomForestParams); normalize both to a JSON-friendly dict.
        if dataclasses.is_dataclass(best_params) and not isinstance(best_params, type):
            payload["best_params"] = dataclasses.asdict(best_params)
        else:
            payload["best_params"] = dict(best_params)

    out_path = out_dir / filename
    out_path.write_text(
        json.dumps(payload, indent=2, default=_to_jsonable), encoding="utf-8"
    )
    print(f"[INFO] Saved evaluation metrics to '{out_path}'.")
    return out_path
