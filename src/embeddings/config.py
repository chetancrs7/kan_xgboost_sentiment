"""Configuration for the DeBERTa-v3 embedding generation pipeline.

A single immutable ``EmbeddingConfig`` holds every knob the pipeline needs so
that the downstream KAN, XGBoost, and hybrid ensemble models all consume
embeddings produced under identical, reproducible settings.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Defaults -------------------------------------------------------------- #

# HuggingFace model identifier. DeBERTa-v3-base ships a 768-dim hidden state.
DEFAULT_MODEL_NAME: str = "microsoft/deberta-v3-base"

# The split CSVs produced by ``src/data/split_dataset.py`` live here.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "deberta_v3_base"

DEFAULT_TEXT_COLUMN: str = "clean_text"
DEFAULT_LABEL_COLUMN: str = "label"

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_MAX_LENGTH: int = 128
DEFAULT_SEED: int = 42

# Logical split name -> CSV filename. The same keys drive the output filenames.
SPLIT_FILENAMES: dict[str, str] = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Immutable settings for one embedding-generation run."""

    model_name: str = DEFAULT_MODEL_NAME
    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR))
    output_dir: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_DIR))
    text_column: str = DEFAULT_TEXT_COLUMN
    label_column: str = DEFAULT_LABEL_COLUMN
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = DEFAULT_SEED
    # ``None`` selects CUDA when available, otherwise CPU. Set to "cpu"/"cuda"
    # to force a device.
    device: str | None = None

    def __post_init__(self) -> None:
        # Coerce string paths (e.g. from argparse) into Path objects without
        # breaking the frozen dataclass contract.
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def split_csv_path(self, split: str) -> Path:
        """Return the input CSV path for a logical split (train/val/test)."""
        return self.data_dir / SPLIT_FILENAMES[split]

    def embeddings_path(self, split: str) -> Path:
        """Return the output ``*_embeddings.npy`` path for a split."""
        return self.output_dir / f"{split}_embeddings.npy"

    def labels_path(self, split: str) -> Path:
        """Return the output ``*_labels.npy`` path for a split."""
        return self.output_dir / f"{split}_labels.npy"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG that affects embedding generation for reproducibility.

    Importing torch lazily keeps ``config`` usable (e.g. for path helpers)
    even in environments where torch is not installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Trade a little kernel-selection speed for deterministic results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the pipeline logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("embeddings")
