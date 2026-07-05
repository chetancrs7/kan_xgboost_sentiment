"""Shared utilities for the adaptive fusion pipeline.

Small, dependency-light helpers used across the fusion modules: RNG seeding,
logging configuration, device resolution, and JSON persistence. Kept separate
so every entry point (training, prediction, ad-hoc analysis) initialises the
environment identically.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG that affects fusion training for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | None = None) -> torch.device:
    """Return the requested device, or CUDA when available, else CPU."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "fusion", level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return a namespaced logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def save_json(payload: dict[str, Any], path: Path | str) -> None:
    """Write ``payload`` as pretty-printed JSON, coercing numpy scalars/arrays."""

    def _default(obj: object) -> object:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def load_json(path: Path | str) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
