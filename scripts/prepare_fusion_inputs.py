"""Prepare row-aligned expert inputs for the adaptive fusion network.

Why this script exists
----------------------
The fusion gate consumes *per-sample aligned* outputs from both experts:

* the KAN semantic expert reads DeBERTa embeddings (``.npy``), and
* the XGBoost structural expert reads the split CSVs directly.

The repository's full splits hold 80k/10k/10k tweets, which is more text than
DeBERTa-v3 can embed in a reasonable time on CPU-only hardware. This script
therefore draws a *stratified, seeded* subsample from each split and emits
every artifact the fusion pipeline needs, all sharing one row order per split:

``data/fusion_inputs/``
    ``indices/{split}_indices.npy``      row indices into ``data/splits/{split}.csv``
    ``splits/{split}.csv``               subsampled rows (all original columns
                                         plus ``source_row`` for traceability)
    ``embeddings/{split}_embeddings.npy``DeBERTa-v3 sentence embeddings
    ``embeddings/{split}_labels.npy``    labels aligned with the embeddings
    ``xgboost/{split}_probs.npy``        existing full-split XGBoost probabilities
                                         sliced by the same indices
    ``manifest.json``                    sizes, seed, and provenance record

The KAN expert is then retrained on the subsample with its own pipeline::

    python src/models/kan/train.py \
        --embeddings-dir data/fusion_inputs/embeddings \
        --output-dir models/kan_subset

Usage::

    python scripts/prepare_fusion_inputs.py
    python scripts/prepare_fusion_inputs.py --train-size 6000 --val-size 2000 --test-size 2000
    python scripts/prepare_fusion_inputs.py --skip-embeddings   # re-slice only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "embeddings"))

SPLITS = ("train", "val", "test")
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
XGB_PROBS_DIR = PROJECT_ROOT / "models" / "xgboost"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "fusion_inputs"

TEXT_COLUMN = "clean_text"  # column DeBERTa embeddings are generated from
LABEL_COLUMN = "label"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build row-aligned subsampled inputs for the fusion network."
    )
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory holding {train,val,test}.csv to subsample from.",
    )
    parser.add_argument(
        "--xgb-probs-dir",
        type=Path,
        default=XGB_PROBS_DIR,
        help="Directory holding full-split {train,val,test}_probs.npy to slice.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=128, help="Embedding max tokens.")
    parser.add_argument("--device", default=None, help="Force 'cpu'/'cuda' for embedding.")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only re-draw indices and re-slice XGBoost probabilities.",
    )
    return parser.parse_args()


def stratified_indices(labels: pd.Series, eligible: pd.Series, size: int, seed: int) -> np.ndarray:
    """Draw a label-stratified sample of row positions.

    Args:
        labels: Full label column of the split, indexed by row position.
        eligible: Boolean mask of rows that may be sampled (non-empty text).
        size: Total sample size; allocated across classes proportionally.
        seed: RNG seed shared by all splits for reproducibility.

    Returns:
        Sorted array of sampled row positions (sorted so downstream artifacts
        keep the original CSV ordering, which eases manual inspection).
    """
    pool = labels[eligible]
    if size > len(pool):
        sys.exit(f"[ERROR] Requested {size} samples but only {len(pool)} eligible rows.")
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    class_counts = (pool.value_counts(normalize=True) * size).round().astype(int)
    # Rounding can leave the total off by a row or two; fix up on the largest class.
    drift = size - int(class_counts.sum())
    class_counts.iloc[0] += drift
    for cls, n_cls in class_counts.items():
        cls_rows = pool.index[pool == cls].to_numpy()
        chosen.append(rng.choice(cls_rows, size=min(n_cls, len(cls_rows)), replace=False))
    return np.sort(np.concatenate(chosen))


def main() -> None:
    args = parse_args()
    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}

    splits_dir = args.splits_dir
    xgb_probs_dir = args.xgb_probs_dir
    out = args.output_dir
    for sub in ("indices", "splits", "embeddings", "xgboost"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    generator = None
    if not args.skip_embeddings:
        from config import EmbeddingConfig, set_seed  # noqa: E402 (src/embeddings)
        from embedding_generator import EmbeddingGenerator  # noqa: E402

        set_seed(args.seed)
        config = EmbeddingConfig(
            batch_size=args.batch_size, max_length=args.max_length, device=args.device
        )
        generator = EmbeddingGenerator(config)

    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "text_column": TEXT_COLUMN,
        "sizes": {},
        "source_splits_dir": str(splits_dir),
        "source_xgb_probs_dir": str(xgb_probs_dir),
    }

    for split in SPLITS:
        csv_path = splits_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
        probs = np.load(xgb_probs_dir / f"{split}_probs.npy")
        if len(probs) != len(df):
            sys.exit(
                f"[ERROR] {split}: XGBoost probs ({len(probs)}) and split CSV "
                f"({len(df)}) are misaligned. Regenerate the XGBoost expert outputs."
            )

        text = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
        eligible = (text != "") & df[LABEL_COLUMN].notna()
        n_dropped = int((~eligible).sum())
        if n_dropped:
            print(f"[INFO] {split}: {n_dropped} rows ineligible (empty '{TEXT_COLUMN}').")

        idx = stratified_indices(df[LABEL_COLUMN], eligible, sizes[split], args.seed)
        np.save(out / "indices" / f"{split}_indices.npy", idx)

        sub_df = df.iloc[idx].copy()
        sub_df.insert(0, "source_row", idx)
        sub_df.to_csv(out / "splits" / f"{split}.csv", index=False)

        np.save(out / "xgboost" / f"{split}_probs.npy", probs[idx])
        labels = sub_df[LABEL_COLUMN].to_numpy(dtype=np.int64)
        np.save(out / "embeddings" / f"{split}_labels.npy", labels)

        counts = {int(k): int(v) for k, v in sub_df[LABEL_COLUMN].value_counts().items()}
        manifest["sizes"][split] = {"n": len(idx), "class_counts": counts}
        print(f"[INFO] {split}: sampled {len(idx)} rows | class counts {counts}")

        if generator is not None:
            print(f"[INFO] {split}: generating DeBERTa embeddings ...")
            embeddings = generator.generate(sub_df[TEXT_COLUMN].astype(str).tolist())
            np.save(out / "embeddings" / f"{split}_embeddings.npy", embeddings)
            print(f"[INFO] {split}: embeddings {embeddings.shape} saved.")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] Fusion inputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
