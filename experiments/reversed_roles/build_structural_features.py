"""Build standardized structural-feature arrays for the *reversed* experiment.

Role reversal
-------------
In the main architecture the KAN reads dense DeBERTa embeddings (semantic) and
XGBoost reads engineered linguistic features (structural). This experiment
swaps the two. Here we produce the input the reversed **KAN** consumes: the
50-column structural feature matrix from
:class:`src.models.xgboost.features.FeatureExtractor`, standardized and written
under the *same filenames the KAN data loader already expects*
(``{split}_embeddings.npy`` + ``{split}_labels.npy``).

Because the existing KAN trainer
(``src/models/kan/train.py``) auto-detects its input dimension and reads exactly
those two arrays per split, pointing it at this output directory trains a KAN on
structural features with **no change to any existing file**::

    python src/models/kan/train.py \
        --embeddings-dir data/reversed_roles/structural_features \
        --output-dir models/reversed_roles/kan_on_structural \
        --gradient-clip-norm 1.0

Why standardize
---------------
Trees are scale-invariant, so the original XGBoost expert never needed scaling.
The KAN's spline edges are *not*: the structural columns mix raw counts
(``char_count`` in the hundreds) with ratios in ``[0, 1]``, and unscaled inputs
would blow past the spline grid range and waste knot resolution. We fit a
``StandardScaler`` on train and apply it to every split; the scaler is saved so
the transform is reproducible and invertible for interpretation.

Usage::

    python experiments/reversed_roles/build_structural_features.py
    python experiments/reversed_roles/build_structural_features.py \
        --train-path data/splits/train.csv \
        --val-path data/splits/val.csv \
        --test-path data/splits/test.csv \
        --output-dir data/reversed_roles/structural_features
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Make the project root importable so this file runs as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.xgboost.data_loader import (  # noqa: E402
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    load_splits,
)
from src.models.xgboost.features import (  # noqa: E402
    FeatureExtractor,
    ensure_nltk_resources,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "reversed_roles" / "structural_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standardized structural features for the reversed KAN."
    )
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scaler",
        choices=["standard", "quantile", "none"],
        default="standard",
        help="Feature transform for the KAN. 'standard' = StandardScaler then clip "
        "to +/-CLIP (default, recommended). The structural columns are heavy-tailed "
        "count features whose rare spikes become extreme z-scores (seen: 261 sigma) "
        "that drive the KAN's UNBOUNDED thin-plate/RBF layers to overflow -> NaN "
        "loss; clipping bounds them. 'quantile' = QuantileTransformer(normal), "
        "naturally bounded and better suited to spiky discrete features. 'none' = "
        "raw features (will likely NaN the KAN — for diagnostics only).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="Clip standardized features to [-clip, clip] (0 disables). Only used "
        "with --scaler standard. This is the guard that keeps the thin-plate/RBF "
        "spline layers finite on the rare extreme-outlier rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_nltk_resources()

    # 1. Load the same train/val/test rows every other model in the repo sees.
    #    load_splits prefers the raw ``text`` column, which is exactly what the
    #    structural FeatureExtractor wants (casing/punctuation/emoji preserved).
    X_train_text, y_train, X_val_text, y_val, X_test_text, y_test = load_splits(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    # 2. Engineer the 50-column structural matrix (fit schema on train only).
    print("[INFO] Extracting structural/linguistic features...")
    extractor = FeatureExtractor()
    Xtr = extractor.fit_transform(X_train_text)
    Xva = extractor.transform(X_val_text)
    Xte = extractor.transform(X_test_text)
    print(
        f"[INFO] Structural matrices | train: {Xtr.shape} | "
        f"val: {Xva.shape} | test: {Xte.shape}"
    )

    # 3. Transform (fit on train) so the KAN's spline layers see BOUNDED inputs.
    #    Critical: the KAN's thin-plate (r^2 log|r|) and RBF layers are unbounded,
    #    so a single 261-sigma outlier row makes the forward activations explode
    #    and the loss go NaN within the first epoch. Clipping (standard) or a
    #    quantile map (quantile) bounds the inputs and fixes it.
    scaler = None
    if args.scaler == "standard":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)
        if args.clip and args.clip > 0:
            Xtr, Xva, Xte = (np.clip(a, -args.clip, args.clip) for a in (Xtr, Xva, Xte))
            print(f"[INFO] Standardized (train-fit) + clipped to +/-{args.clip}.")
        else:
            print("[WARN] Standardized without clipping; the KAN may overflow to NaN.")
    elif args.scaler == "quantile":
        from sklearn.preprocessing import QuantileTransformer

        scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(Xtr)),
            subsample=1_000_000,
            random_state=42,
        )
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)
        print("[INFO] Quantile-transformed features (normal output, bounded).")
    else:
        print("[WARN] No scaling (--scaler none); the KAN's spline layers may NaN.")
    Xtr, Xva, Xte = (a.astype(np.float32) for a in (Xtr, Xva, Xte))

    # 4. Write under the *KAN loader's* expected filenames so the existing
    #    trainer consumes this directory unchanged.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": (Xtr, y_train.to_numpy(dtype=np.int64)),
        "val": (Xva, y_val.to_numpy(dtype=np.int64)),
        "test": (Xte, y_test.to_numpy(dtype=np.int64)),
    }
    for split, (X, y) in splits.items():
        np.save(out / f"{split}_embeddings.npy", X)
        np.save(out / f"{split}_labels.npy", y)
        print(f"[INFO] {split}: wrote features {X.shape} + labels {y.shape}")

    # 5. Provenance + the transforms needed to reproduce/interpret the inputs.
    joblib.dump(extractor, out / "feature_extractor.pkl")
    if scaler is not None:
        joblib.dump(scaler, out / "scaler.pkl")
    (out / "feature_names.json").write_text(
        json.dumps(extractor.feature_names_, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "reversed-roles experiment: KAN input = structural features",
        "scaler": args.scaler,
        "clip": args.clip if args.scaler == "standard" else None,
        "n_features": int(extractor.n_features_),
        "feature_names": extractor.feature_names_,
        "sizes": {s: int(len(y)) for s, (_, y) in splits.items()},
        "source_splits": {
            "train": str(args.train_path),
            "val": str(args.val_path),
            "test": str(args.test_path),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] Reversed KAN inputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
