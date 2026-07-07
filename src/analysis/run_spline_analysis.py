"""Standalone CLI: spline interpretability report for a saved KAN checkpoint.

Analyses ``models/kan/kan_model.pt`` (or any checkpoint saved by
``src/models/kan/train.py``) without retraining anything. If the training
run also saved the pre-training snapshot ``kan_model_init.pt`` (written
when spline visualization is enabled), before-vs-after figures are
included automatically.

Usage::

    python -m src.analysis.run_spline_analysis
    python src/analysis/run_spline_analysis.py --checkpoint models/kan/kan_model.pt
    python -m src.analysis.run_spline_analysis --embeddings-dir data/fusion_inputs/embeddings
    python -m src.analysis.run_spline_analysis --top-k 12 --output-dir outputs/kan_spline_visualizations

Real embeddings are strongly recommended (they drive the importance
scores and the feature-response sweeps); when the embeddings directory is
missing, the report falls back to synthetic N(0,1) inputs and says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Project-root import shim, same pattern as the other entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.spline_visualizer import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    generate_all_visualizations,
)
from src.models.kan.data_loader import DEFAULT_EMBEDDINGS_DIR  # noqa: E402
from src.models.kan.model import KANSentimentClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KAN spline interpretability figures from a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/kan/kan_model.pt",
        help="Trained checkpoint saved by src/models/kan/train.py (default: %(default)s).",
    )
    parser.add_argument(
        "--before-checkpoint",
        default=None,
        help="Pre-training snapshot for before/after figures. Default: "
        "kan_model_init.pt next to --checkpoint, if it exists.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory with {split}_embeddings.npy for data-dependent figures "
        "(default: %(default)s; falls back to synthetic inputs when missing).",
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Embedding rows used for importance scoring / response sweeps.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=8, help="Input dims per layer to plot.")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> KANSentimentClassifier:
    """Rebuild a ``KANSentimentClassifier`` from a training checkpoint.

    Checkpoints store ``{"model_state_dict": ..., "config": ...}`` where
    ``config`` are the exact constructor kwargs (with tuples serialised as
    lists).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["config"])
    config["hidden_dims"] = tuple(config["hidden_dims"])
    config["grid_range"] = tuple(config["grid_range"])
    model = KANSentimentClassifier(**config)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def load_embedding_sample(
    embeddings_dir: str, split: str, max_samples: int
) -> np.ndarray | None:
    """Load up to ``max_samples`` embedding rows, or ``None`` when unavailable."""
    path = Path(embeddings_dir) / f"{split}_embeddings.npy"
    if not path.exists():
        print(
            f"[WARN] Embeddings not found at {path.resolve()} — data-dependent "
            "figures will use synthetic inputs. Pass --embeddings-dir to fix."
        )
        return None
    X = np.load(path).astype(np.float32)
    print(f"[INFO] Loaded {len(X):,} {split} embeddings (dim={X.shape[1]}) from '{path}'.")
    return X[:max_samples]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        sys.exit(
            f"[ERROR] Checkpoint not found: {checkpoint.resolve()}\n"
            "Train the KAN first (python src/train.py train.stages=[kan] or "
            "python src/models/kan/train.py)."
        )
    model = load_checkpoint(checkpoint, device)
    print(f"[INFO] Loaded trained KAN from '{checkpoint}'.")

    # Optional pre-training snapshot for before/after-training figures.
    before_path = (
        Path(args.before_checkpoint)
        if args.before_checkpoint
        else checkpoint.with_name("kan_model_init.pt")
    )
    before_model = None
    if before_path.exists():
        before_model = load_checkpoint(before_path, device)
        print(f"[INFO] Loaded pre-training snapshot from '{before_path}'.")
    else:
        print(
            f"[INFO] No pre-training snapshot at '{before_path}' — skipping "
            "before/after figures."
        )

    X = load_embedding_sample(args.embeddings_dir, args.split, args.max_samples)

    written = generate_all_visualizations(
        model,
        X=X,
        before_model=before_model,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
    )
    print(f"[DONE] Wrote {len(written)} files to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
