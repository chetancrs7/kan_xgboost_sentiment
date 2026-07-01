"""End-to-end training pipeline for the KAN expert sentiment model.

Execution flow
--------------
1. Load precomputed DeBERTa embeddings + labels for train/val/test (no text
   processing, no embedding regeneration).
2. Build a 4-layer adaptive-spline KAN
   (``SemanticCompressionLayer -> ContextInteractionLayer ->
   SentimentDistortionLayer -> EmotionHotspotLayer``).
3. Train with cross-entropy loss, periodic spline-grid adaptation, and early
   stopping on validation loss.
4. Evaluate train/val/test, export probabilities + entropy-based confidence.
5. Persist model weights, training-curve plot, and metrics.

Usage:
    python src/models/kan/train.py
    python src/models/kan/train.py --embeddings-dir data/embeddings/deberta_v3_base
    python src/models/kan/train.py --epochs 300 --patience 20 --hidden-dims 256,128,64
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Allow running both as a script (`python src/models/kan/train.py`) and as a
# module (`python -m src.models.kan.train`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import DEFAULT_EMBEDDINGS_DIR, build_dataloaders, load_splits  # noqa: E402
from evaluate import (  # noqa: E402
    confidence_from_entropy,
    evaluate_split,
    logits_to_probs,
)
from model import DEFAULT_HIDDEN_DIMS, KANSentimentClassifier  # noqa: E402

DEFAULT_OUTPUT_DIR = "models/kan"
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 15
DEFAULT_GRID_UPDATE_FREQ = 10
RANDOM_STATE = 42


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KAN expert sentiment classifier over pretrained DeBERTa embeddings."
    )
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hidden-dims",
        default=",".join(str(d) for d in DEFAULT_HIDDEN_DIMS),
        help="Comma-separated widths for layers 1-3, e.g. '256,128,64' (default: %(default)s).",
    )
    parser.add_argument("--grid-size", type=int, default=5, help="B-spline grid intervals.")
    parser.add_argument("--spline-order", type=int, default=3, help="B-spline polynomial degree.")
    parser.add_argument(
        "--grid-range",
        default="-2,2",
        help="Initial spline grid range as 'min,max' (default: %(default)s).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience in epochs without validation-loss improvement.",
    )
    parser.add_argument(
        "--grid-update-freq",
        type=int,
        default=DEFAULT_GRID_UPDATE_FREQ,
        help="Refit adaptive spline grids every N epochs (0 disables periodic updates).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--device", default=None, help="Force 'cpu' or 'cuda' (default: auto-detect)."
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns:
        ``(mean_loss, accuracy)`` for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def predict_logits(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    """Batched forward pass over a full split, preserving row order."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            outputs.append(model(X_batch).cpu())
    return torch.cat(outputs, dim=0)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    train_X: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grid_update_freq: int,
) -> dict[str, list[float]]:
    """Train with cross-entropy loss, periodic grid adaptation, and early stopping.

    Returns:
        A ``history`` dict with per-epoch ``train_loss``, ``train_acc``,
        ``val_loss``, ``val_acc`` lists.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Seed the spline grids from the real training-data distribution before
    # the first gradient step, rather than leaving them at their fixed
    # initial `grid_range`.
    seed_batch = torch.from_numpy(train_X[: min(len(train_X), 512)]).to(device)
    model.update_grids(seed_batch)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if grid_update_freq and epoch % grid_update_freq == 0 and epoch < epochs:
            model.update_grids(seed_batch)

        improved = val_loss < best_val_loss
        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + ("  * new best" if improved else "")
        )

        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[INFO] Early stopping at epoch {epoch} "
                    f"(no val-loss improvement for {patience} epochs)."
                )
                stopped_epoch = epoch
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    history["stopped_epoch"] = stopped_epoch
    return history


def plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save a two-panel (loss, accuracy) train-vs-validation curve figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs_range, history["train_loss"], label="train")
    ax_loss.plot(epochs_range, history["val_loss"], label="validation")
    ax_loss.set_title("Cross-Entropy Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs_range, history["train_acc"], label="train")
    ax_acc.plot(epochs_range, history["val_acc"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.suptitle("KAN Sentiment Classifier — Training Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved training curves to: {output_path.resolve()}")


def save_artifacts(
    output_dir: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    history: dict[str, Any],
    metrics: dict[str, Any],
    probs: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
) -> None:
    """Persist all model artifacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model weights: state dict + the config needed to reconstruct the module.
    torch.save(
        {"model_state_dict": model.state_dict(), "config": model_config},
        output_dir / "kan_model.pt",
    )

    # Prediction probabilities + entropy-based confidence, per split.
    for split, array in probs.items():
        np.save(output_dir / f"{split}_probs.npy", array)
    for split, array in confidence.items():
        np.save(output_dir / f"{split}_confidence.npy", array)

    # Training curves.
    plot_training_curves(history, output_dir / "training_curves.png")

    # Per-epoch history (backing data for the plot).
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Metrics for every split.
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[INFO] All artifacts saved to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    # 1. Load precomputed embeddings ---------------------------------------
    splits = load_splits(args.embeddings_dir)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    input_dim = X_train.shape[1]
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    print(f"[INFO] Embedding dim: {input_dim} | Detected {n_classes} target classes.")

    loaders = build_dataloaders(splits, batch_size=args.batch_size, seed=args.seed)

    # 2. Build the 4-layer adaptive-spline KAN -----------------------------
    hidden_dims = tuple(int(d) for d in args.hidden_dims.split(","))
    grid_range = tuple(float(v) for v in args.grid_range.split(","))
    model_config = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "n_classes": n_classes,
        "grid_size": args.grid_size,
        "spline_order": args.spline_order,
        "grid_range": grid_range,
    }
    model = KANSentimentClassifier(**model_config).to(device)
    print(
        f"[INFO] Built KANSentimentClassifier: {input_dim} -> {hidden_dims[0]} -> "
        f"{hidden_dims[1]} -> {hidden_dims[2]} -> {n_classes}"
    )

    # 3. Train with early stopping ------------------------------------------
    history = train_model(
        model,
        loaders["train"],
        loaders["val"],
        X_train,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grid_update_freq=args.grid_update_freq,
    )

    # 4. Evaluate + export probabilities / confidence -----------------------
    print("\n" + "=" * 64)
    print("EVALUATION")
    print("=" * 64)
    probs, confidence, metrics = {}, {}, {}
    for split_name, (X, y) in (("train", (X_train, y_train)), ("val", (X_val, y_val)), ("test", (X_test, y_test))):
        logits = predict_logits(model, X, device)
        split_probs = logits_to_probs(logits)
        probs[split_name] = split_probs
        confidence[split_name] = confidence_from_entropy(split_probs)
        metrics[split_name] = evaluate_split(split_probs, y, split_name, n_classes)

    metrics["model"] = "KAN (adaptive-spline, DeBERTa embeddings)"
    metrics["n_classes"] = n_classes
    metrics["best_val_loss"] = history["best_val_loss"]
    metrics["stopped_epoch"] = history["stopped_epoch"]
    metrics["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_config.items()}

    # 5. Persist --------------------------------------------------------------
    save_artifacts(
        output_dir=Path(args.output_dir),
        model=model,
        model_config={k: (list(v) if isinstance(v, tuple) else v) for k, v in model_config.items()},
        history=history,
        metrics=metrics,
        probs=probs,
        confidence=confidence,
    )

    print("\n[DONE] KAN expert pipeline complete.")


if __name__ == "__main__":
    main()
