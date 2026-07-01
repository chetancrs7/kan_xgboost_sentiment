"""Data loading for the KAN sentiment classifier.

Reads the *precomputed* DeBERTa-v3 embeddings produced by
``src/embeddings/generate_embeddings.py`` (``{split}_embeddings.npy`` +
``{split}_labels.npy``). This module never re-runs DeBERTa or touches raw
text -- it only loads the ``.npy`` arrays already on disk and wraps them as
PyTorch tensors / ``DataLoader``s for the KAN training loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_EMBEDDINGS_DIR = "data/embeddings/deberta_v3_base"
SPLITS: tuple[str, ...] = ("train", "val", "test")


def _load_array(path: Path, split_name: str, kind: str) -> np.ndarray:
    if not path.exists():
        sys.exit(
            f"[ERROR] Missing {kind} file for the {split_name} split: {path.resolve()}\n"
            "Generate embeddings first with 'python src/embeddings/generate_embeddings.py'."
        )
    return np.load(path)


def load_embeddings(embeddings_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split's ``(embeddings, labels)`` arrays from ``embeddings_dir``.

    Args:
        embeddings_dir: Directory containing ``{split}_embeddings.npy`` and
            ``{split}_labels.npy``.
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        ``(X, y)`` where ``X`` is ``[n_samples, embedding_dim]`` float32 and
        ``y`` is ``[n_samples]`` int64.
    """
    directory = Path(embeddings_dir)
    X = _load_array(directory / f"{split}_embeddings.npy", split, "embeddings").astype(
        np.float32
    )
    y = _load_array(directory / f"{split}_labels.npy", split, "labels").astype(np.int64)

    if len(X) != len(y):
        sys.exit(
            f"[ERROR] {split} embeddings ({len(X)}) and labels ({len(y)}) "
            "have mismatched lengths."
        )
    print(f"[INFO] Loaded {len(X):,} {split} embeddings (dim={X.shape[1]}) from '{directory}'.")
    return X, y


def load_splits(
    embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load train/val/test embeddings + labels.

    Returns:
        Mapping ``split -> (X, y)`` for ``"train"``, ``"val"``, ``"test"``.
    """
    splits = {split: load_embeddings(embeddings_dir, split) for split in SPLITS}
    dims = {X.shape[1] for X, _ in splits.values()}
    if len(dims) > 1:
        sys.exit(f"[ERROR] Embedding dimensionality mismatch across splits: {dims}")
    print(
        f"[INFO] Using existing embeddings | "
        + " | ".join(f"{s}: {len(X):,}" for s, (X, _) in splits.items())
    )
    return splits


def build_dataloaders(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Wrap loaded splits into ``TensorDataset`` / ``DataLoader`` pairs.

    Only the training loader is shuffled; validation and test loaders keep
    row order so predictions can be aligned back to the source split later.
    """
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split, (X, y) in splits.items():
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator if split == "train" else None,
        )
    return loaders
