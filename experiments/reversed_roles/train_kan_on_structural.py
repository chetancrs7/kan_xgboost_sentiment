"""Train the reversed KAN on structural features — numerically robust.

Why this has its own training loop
----------------------------------
Two things make the production KAN trainer (``src/models/kan/train.py``) NaN on
structural-feature input, both traced empirically:

1. **Unbounded layer 3.** The production layer 3 is a thin-plate spline
   (``r^2 log|r|``); on spiky structural counts its activation grows without
   bound and overflows. We fix this by using
   :class:`~experiments.reversed_roles.structural_kan.StructuralKANClassifier`,
   whose layer 3 is a *bounded* B-spline (same block as layers 1-2).

2. **Init-sensitive early divergence.** With the production trainer's exact
   seed/build order, most initializations send an early batch to inf within the
   first epoch (a bad init + the ``curve2coef`` grid refit compound). A run whose
   init happens to be tame trains fine to ~0.73 acc. To make this *reliable*
   rather than luck, this loop (a) seeds the RNG immediately before building the
   model, (b) seeds the spline grids once and does **no** periodic mid-training
   grid refits, and (c) wraps every step in a **finite-guard**: if a step
   produces a non-finite loss or parameter, the model is rolled back to the last
   good state and the batch is skipped. Occasional blow-ups become skipped steps
   instead of a dead, all-NaN model.

Everything else (data loader, evaluator, artifact writer, layer classes) is the
production code, imported unchanged — nothing in ``src/`` is modified.

Usage (full 1.28M dataset)::

    python experiments/reversed_roles/train_kan_on_structural.py \
        --embeddings-dir data/reversed_roles/structural_features_full \
        --output-dir models/reversed_roles/kan_on_structural_full \
        --batch-size 512
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kan.data_loader import build_dataloaders, load_splits  # noqa: E402
from src.models.kan.evaluate import (  # noqa: E402
    confidence_from_entropy,
    evaluate_split,
    logits_to_probs,
)
from src.models.kan.model import DEFAULT_HIDDEN_DIMS  # noqa: E402
from src.models.kan.train import (  # noqa: E402
    predict_logits,
    resolve_device,
    save_artifacts,
)

from experiments.reversed_roles.structural_kan import (  # noqa: E402
    DISTORTION_KERNELS,
    StructuralKANClassifier,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reversed KAN on structural features (bounded layer 3, NaN-guarded)."
    )
    p.add_argument("--embeddings-dir", default="data/reversed_roles/structural_features_full")
    p.add_argument("--output-dir", default="models/reversed_roles/kan_on_structural_full")
    p.add_argument("--distortion-kernel", choices=DISTORTION_KERNELS, default="bspline")
    p.add_argument("--hidden-dims", default=",".join(str(d) for d in DEFAULT_HIDDEN_DIMS))
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--grid-range", default="-2,2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0)
    p.add_argument(
        "--seed-grids",
        action="store_true",
        help="Re-enable the pre-training B-spline grid refit (update_grids). OFF "
        "by default because on structural features it makes training diverge; "
        "the uniform grid is stable. Guarded when enabled.",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=100,
        help="Steps between 'last-good' parameter snapshots used to roll back a "
        "non-finite step (smaller = safer rollback, slightly slower).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _all_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(p).all() for p in model.parameters())


def train_guarded(model, loaders, X_train, X_val, y_val, device, args):
    """Finite-guarded training loop with early stopping. Returns a history dict."""
    criterion = nn.CrossEntropyLoss()

    # NOTE: we deliberately do NOT call ``model.update_grids`` here.
    #
    # The production trainer seeds the B-spline grids from a data batch before
    # training (``update_grids`` -> pykan ``update_grid_from_samples`` ->
    # ``curve2coef``). On smooth embeddings that helps. On the discrete, tied
    # structural-feature values it is the direct cause of the NaN: the refit
    # produces a grid geometry on which B-spline *training* diverges every step
    # (the seeded model is finite, but the first gradient step blows up). Trained
    # on the plain uniform grid instead, the same model is stable and learns
    # (verified: ~0.72 val acc). ``--seed-grids`` re-enables the refit, guarded.
    if args.seed_grids:
        seed_batch = torch.from_numpy(X_train[: min(len(X_train), 512)]).to(device)
        pre = copy.deepcopy(model.state_dict())
        model.update_grids(seed_batch)
        with torch.no_grad():
            ok = _all_finite(model) and bool(torch.isfinite(model(seed_batch)).all())
        if not ok:
            model.load_state_dict(pre)
            print("[WARN] Grid seeding produced a non-finite model; reverted to init.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "skipped": 0}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    last_good = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    stopped_epoch = args.epochs
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = tot_correct = tot = 0
        for X_batch, y_batch in loaders["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            global_step += 1
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Finite-guard: a bad batch/step rolls back instead of poisoning
            # the whole model with NaNs.
            if not torch.isfinite(loss):
                model.load_state_dict(last_good)
                history["skipped"] += 1
                continue
            loss.backward()
            if args.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            if not _all_finite(model):
                model.load_state_dict(last_good)
                history["skipped"] += 1
                continue

            if global_step % args.snapshot_every == 0:
                last_good = copy.deepcopy(model.state_dict())

            bs = y_batch.size(0)
            tot_loss += loss.item() * bs
            tot_correct += (logits.argmax(1) == y_batch).sum().item()
            tot += bs

        train_loss = tot_loss / max(tot, 1)
        train_acc = tot_correct / max(tot, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = predict_logits(model, X_val, device)
            val_loss = criterion(v_logits, torch.from_numpy(y_val.astype(np.int64))).item()
            val_acc = (v_logits.argmax(1).numpy() == y_val).mean()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | skipped={history['skipped']}"
            + ("  * new best" if val_loss < best_val else "")
        )

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            last_good = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                stopped_epoch = epoch
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["stopped_epoch"] = stopped_epoch
    return history


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    splits = load_splits(args.embeddings_dir)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    input_dim = X_train.shape[1]
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    print(f"[INFO] Feature dim: {input_dim} | Detected {n_classes} target classes.")

    loaders = build_dataloaders(splits, batch_size=args.batch_size, seed=args.seed)

    hidden_dims = tuple(int(d) for d in args.hidden_dims.split(","))
    grid_range = tuple(float(v) for v in args.grid_range.split(","))
    model_config = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "n_classes": n_classes,
        "grid_size": args.grid_size,
        "spline_order": args.spline_order,
        "grid_range": grid_range,
        "distortion_kernel": args.distortion_kernel,
    }
    # Seed the RNG immediately before constructing the model so the (init-
    # sensitive) spline weights are drawn from a fixed, tame state.
    torch.manual_seed(args.seed)
    model = StructuralKANClassifier(**model_config).to(device)
    print(
        f"[INFO] Built StructuralKANClassifier (layer3={args.distortion_kernel}): "
        f"{input_dim} -> {hidden_dims[0]} -> {hidden_dims[1]} -> {hidden_dims[2]} -> {n_classes}"
    )

    history = train_guarded(model, loaders, X_train, X_val, y_val, device, args)
    if history["skipped"]:
        print(f"[INFO] Finite-guard rolled back {history['skipped']} non-finite step(s).")

    print("\n" + "=" * 64 + "\nEVALUATION\n" + "=" * 64)
    probs, confidence, metrics = {}, {}, {}
    for split_name, (X, y) in (
        ("train", (X_train, y_train)),
        ("val", (X_val, y_val)),
        ("test", (X_test, y_test)),
    ):
        logits = predict_logits(model, X, device)
        split_probs = logits_to_probs(logits)
        probs[split_name] = split_probs
        confidence[split_name] = confidence_from_entropy(split_probs)
        metrics[split_name] = evaluate_split(split_probs, y, split_name, n_classes)

    serializable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_config.items()}
    metrics["model"] = f"KAN reversed (structural features, layer3={args.distortion_kernel})"
    metrics["n_classes"] = n_classes
    metrics["best_val_loss"] = history["best_val_loss"]
    metrics["stopped_epoch"] = history["stopped_epoch"]
    metrics["skipped_steps"] = history["skipped"]
    metrics["config"] = serializable

    save_artifacts(
        output_dir=Path(args.output_dir),
        model=model,
        model_config=serializable,
        history=history,
        metrics=metrics,
        probs=probs,
        confidence=confidence,
    )
    print("\n[DONE] Reversed KAN-on-structural pipeline complete.")


if __name__ == "__main__":
    main()
