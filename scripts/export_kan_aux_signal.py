"""Export the KAN's auxiliary XGBoost-competence signal for the fusion gate.

The residual-aware KAN can carry an auxiliary head that predicts, from the
DeBERTa embedding alone, the probability that the XGBoost structural expert will
*fail* on a sample (see ``KANSentimentClassifier.aux_failure_logit``). This
script runs that head over each split's embeddings and writes
``{split}_xgb_fail_prob.npy`` next to the KAN's probabilities, so the fusion gate
can consume it as an extra per-sample competence feature.

It is read-only w.r.t. the KAN and does nothing if the checkpoint has no aux
head (prints a notice and exits cleanly).

Usage::

    python scripts/export_kan_aux_signal.py \
        --kan-dir models/kan_residual_full \
        --embeddings-dir data/embeddings_finetuned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kan.data_loader import load_embeddings  # noqa: E402
from src.models.kan.model import KANSentimentClassifier  # noqa: E402
from src.models.kan.train import resolve_device  # noqa: E402

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kan-dir", default="models/kan_residual_full",
                   help="Dir with kan_model.pt; outputs land here.")
    p.add_argument("--embeddings-dir", default="data/embeddings_finetuned")
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def aux_probs(model: KANSentimentClassifier, X: np.ndarray, device, batch: int) -> np.ndarray:
    """P(XGBoost fails) per row from the aux head (sigmoid of its logit)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).to(device)
        logit = model.aux_failure_logit(xb)
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    kan_dir = Path(args.kan_dir)
    ckpt = torch.load(kan_dir / "kan_model.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    config = {k: (tuple(v) if isinstance(v, list) else v) for k, v in config.items()}
    model = KANSentimentClassifier(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if not getattr(model, "aux_failure_head", False) or model.aux_head is None:
        print("[INFO] This KAN checkpoint has no auxiliary head; nothing to export "
              "(train it with --aux-failure-head). Exiting.")
        return

    for split in args.splits:
        X, _ = load_embeddings(args.embeddings_dir, split)
        probs = aux_probs(model, X.astype(np.float32), device, args.batch_size)
        out_path = kan_dir / f"{split}_xgb_fail_prob.npy"
        np.save(out_path, probs)
        print(f"[INFO] {split}: aux P(xgb fails) mean={probs.mean():.3f} "
              f"-> {out_path}")

    print("[DONE] Aux competence signal exported.")


if __name__ == "__main__":
    main()
