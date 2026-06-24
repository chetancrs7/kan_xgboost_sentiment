"""Generate and persist DeBERTa-v3 sentence embeddings for every data split.

Usage:
    python src/embeddings/generate_embeddings.py
    python src/embeddings/generate_embeddings.py --data-dir data/splits --output-dir embeddings
    python src/embeddings/generate_embeddings.py --batch-size 64 --max-length 160 --device cuda

Outputs, one pair per split (train/val/test), under ``--output-dir``:
    {split}_embeddings.npy   float32 array of shape (n_samples, hidden_size)
    {split}_labels.npy       label array aligned row-for-row with embeddings

These ``.npy`` files are the shared input for the downstream KAN, XGBoost,
and hybrid ensemble models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    SPLIT_FILENAMES,
    EmbeddingConfig,
    configure_logging,
    set_seed,
)
from embedding_generator import EmbeddingGenerator

logger = configure_logging()


def load_split(config: EmbeddingConfig, split: str) -> pd.DataFrame:
    """Load and sanity-check one split CSV, dropping null/empty text rows."""
    path = config.split_csv_path(split)
    if not path.exists():
        sys.exit(
            f"[ERROR] Split file not found: {path.resolve()}\n"
            "Run 'python src/data/split_dataset.py' first to generate the split CSVs."
        )

    df = pd.read_csv(path)
    missing = {config.text_column, config.label_column} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns {missing} in {path}.")

    before = len(df)
    df = df.dropna(subset=[config.text_column, config.label_column]).reset_index(drop=True)
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df = df[df[config.text_column] != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows from '%s' with null/empty text or label.", dropped, split)

    if df.empty:
        sys.exit(f"[ERROR] No usable rows in the '{split}' split.")

    logger.info("Loaded %d %s samples from '%s'.", len(df), split, path)
    return df


def process_split(generator: EmbeddingGenerator, config: EmbeddingConfig, split: str) -> None:
    """Embed one split and write its embeddings + labels to ``.npy`` files."""
    df = load_split(config, split)

    logger.info("Generating embeddings for '%s' split ...", split)
    embeddings = generator.generate(df[config.text_column].tolist())
    labels = df[config.label_column].to_numpy()

    embeddings_path = config.embeddings_path(split)
    labels_path = config.labels_path(split)
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)

    logger.info(
        "Saved '%s': embeddings %s -> %s | labels %s -> %s",
        split,
        embeddings.shape,
        embeddings_path,
        labels.shape,
        labels_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeBERTa-v3 sentence embeddings for train/val/test splits."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HuggingFace model id (default: %(default)s).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory holding the split CSVs (default: %(default)s).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the output .npy files (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size (default: %(default)s).")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max token length per text (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility (default: %(default)s).")
    parser.add_argument("--device", default=None, help="Force a device, e.g. 'cuda' or 'cpu' (default: auto-detect).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbeddingConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        device=args.device,
    )

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", config.output_dir.resolve())

    generator = EmbeddingGenerator(config)

    for split in SPLIT_FILENAMES:
        process_split(generator, config, split)

    logger.info("Done. Embeddings for %d splits written to '%s'.", len(SPLIT_FILENAMES), config.output_dir)


if __name__ == "__main__":
    main()
