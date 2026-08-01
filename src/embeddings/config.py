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

# Sentence-pooling strategy for turning token vectors into one vector:
#   "mean"   - masked mean over tokens (default; works with any encoder).
#   "cls"    - the first-token ([CLS]) last-hidden-state vector. Aligns with how
#              DeBERTa's sequence-classification head reads the sequence, so it
#              matches a fine-tuned encoder better than mean pooling.
#   "pooler" - the fine-tuned ContextPooler output (the exact vector the trained
#              classifier consumes). MOST aligned, but requires a fine-tuned
#              sequence-classification checkpoint that actually has a pooler.
DEFAULT_POOLING: str = "mean"
POOLING_STRATEGIES: tuple[str, ...] = ("mean", "cls", "pooler")

# --- Named embedding-method presets --------------------------------------- #
# A "method" is just a convenient name that fills in model_name / pooling /
# text_column / tokenizer options so the CLI can say ``--method bertweet``
# instead of spelling everything out. These built-in defaults are the source of
# truth; ``conf/embeddings.yaml`` (optional) can add or override methods without
# touching code. A missing or malformed YAML is ignored so generation never
# breaks. Explicit CLI flags always win over a preset.
_BUILTIN_EMBEDDING_METHODS: dict[str, dict] = {
    # The existing DeBERTa-v3 path — identical to the prior default behaviour.
    "deberta": {
        "model_name": DEFAULT_MODEL_NAME,
        "pooling": "mean",
        "text_column": "clean_text",
        "tokenizer_kwargs": {},
    },
    # BERTweet: RoBERTa pretrained on ~850M English tweets. Domain-matched to
    # this task; reads the RAW ``text`` column (it has its own tweet-aware BPE).
    "bertweet": {
        "model_name": "vinai/bertweet-base",
        # Always load the tokenizer from the hub base, never a fine-tuned dir:
        # BERTweet's slow tokenizer does not round-trip through save_pretrained
        # (it loses its BPE merges and tokenizes char-by-char), which silently
        # wrecks accuracy. The base tokenizer is identical and correct.
        "tokenizer_name": "vinai/bertweet-base",
        "pooling": "mean",
        "text_column": "text",
        "tokenizer_kwargs": {"use_fast": False},
    },
}

_METHODS_YAML = PROJECT_ROOT / "conf" / "embeddings.yaml"


def load_embedding_methods() -> dict[str, dict]:
    """Return the method presets, merging ``conf/embeddings.yaml`` over defaults.

    The YAML is optional; any error reading it falls back to the built-in
    presets so embedding generation can never be broken by config problems.
    """
    methods = {name: dict(cfg) for name, cfg in _BUILTIN_EMBEDDING_METHODS.items()}
    if _METHODS_YAML.exists():
        try:
            import yaml

            data = yaml.safe_load(_METHODS_YAML.read_text(encoding="utf-8")) or {}
            for name, cfg in (data.get("methods") or {}).items():
                merged = dict(methods.get(name, {}))
                merged.update(cfg or {})
                methods[name] = merged
        except Exception as exc:  # never let a bad YAML break generation
            import logging

            logging.getLogger("embeddings").warning(
                "Ignoring %s (%s); using built-in method presets.", _METHODS_YAML, exc
            )
    return methods

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
    pooling: str = DEFAULT_POOLING
    # Load the tokenizer from a different source than the model. ``None`` uses
    # ``model_name``. Needed for BERTweet: its saved tokenizer is broken on
    # reload, so a fine-tuned BERTweet model must pair with the hub base tokenizer.
    tokenizer_name: str | None = None
    # Extra kwargs forwarded to ``AutoTokenizer.from_pretrained`` (e.g. BERTweet
    # needs ``{"use_fast": False}``). Empty for the default DeBERTa path.
    tokenizer_kwargs: dict = field(default_factory=dict)
    # ``None`` selects CUDA when available, otherwise CPU. Set to "cpu"/"cuda"
    # to force a device.
    device: str | None = None

    def __post_init__(self) -> None:
        # Coerce string paths (e.g. from argparse) into Path objects without
        # breaking the frozen dataclass contract.
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.pooling not in POOLING_STRATEGIES:
            raise ValueError(
                f"pooling must be one of {POOLING_STRATEGIES}, got '{self.pooling}'."
            )

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
