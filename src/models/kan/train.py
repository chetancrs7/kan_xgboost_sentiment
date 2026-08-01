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
6. Optionally (on by default, see --enable-spline-visualization) generate the
   spline interpretability report via ``src/analysis``; optionally
   (--enable-tensorboard) log spline diagnostics to TensorBoard during
   training. Both are best-effort and can never fail the training run.

Usage:
    python src/models/kan/train.py
    python src/models/kan/train.py --embeddings-dir data/embeddings/deberta_v3_base
    python src/models/kan/train.py --epochs 300 --patience 20 --hidden-dims 256,128,64
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Make the project root importable so this file works as a script
# (`python src/models/kan/train.py`), as a module (`python -m
# src.models.kan.train`), and imported from the Hydra entry point
# (`src/train.py`). Package-qualified imports avoid clashing with the
# XGBoost expert's identically named data_loader/evaluate/model modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.kan.data_loader import (  # noqa: E402
    DEFAULT_EMBEDDINGS_DIR,
    build_dataloaders,
    load_splits,
)
from src.models.kan.evaluate import (  # noqa: E402
    confidence_from_entropy,
    evaluate_split,
    logits_to_probs,
)
from src.models.kan.model import DEFAULT_HIDDEN_DIMS, KANSentimentClassifier  # noqa: E402

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
    parser.add_argument(
        "--grid-refine",
        default="",
        help="Grid-refinement schedule as 'epoch:grid_size' pairs, e.g. "
        "'40:8,80:12' grows spline resolution to 8 knots at epoch 40 and 12 at "
        "epoch 80 (keeping the learned function). Empty disables refinement. "
        "The optimiser is rebuilt and early-stopping state reset at each step.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
        help="Clip gradients to this global L2 norm (0 disables). ~1.0 stabilizes "
        "the spline updates and smooths the learning curve.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "plateau"],
        default="none",
        help="LR schedule. 'cosine' warms up then decays every optimizer step "
        "(smoothest gradual curve); 'plateau' halves LR when val-loss stalls.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear LR warmup over the first N optimizer steps (cosine schedule).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Floor learning rate for the cosine/plateau schedules.",
    )
    parser.add_argument(
        "--step-eval-every",
        type=int,
        default=0,
        help="Evaluate on a validation subsample every N optimizer steps and "
        "record a step-level learning curve. Useful when the dataset is large "
        "enough that most learning happens inside the first epoch (0 disables).",
    )
    parser.add_argument(
        "--step-eval-samples",
        type=int,
        default=20000,
        help="Validation subsample size for --step-eval-every evaluations.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--device", default=None, help="Force 'cpu' or 'cuda' (default: auto-detect)."
    )
    # --- Spline interpretability (src/analysis) -----------------------------
    # These are strictly additive: with visualization and TensorBoard both
    # disabled, training behaves exactly as before.
    viz = parser.add_argument_group("spline interpretability (src/analysis)")
    viz.add_argument(
        "--enable-spline-visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the spline interpretability figures after training "
        "(disable with --no-enable-spline-visualization).",
    )
    viz.add_argument(
        "--spline-viz-dir",
        default="outputs/kan_spline_visualizations",
        help="Where the interpretability figures are written.",
    )
    viz.add_argument(
        "--spline-viz-top-k",
        type=int,
        default=8,
        help="Input dimensions per layer to visualize (importance-ranked).",
    )
    viz.add_argument(
        "--enable-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log spline diagnostics (coefficient histograms, knot ranges, "
        "edge curves) to TensorBoard during training.",
    )
    viz.add_argument(
        "--tensorboard-dir",
        default="outputs/tensorboard/kan",
        help="TensorBoard log directory (used with --enable-tensorboard).",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_grid_refine(spec: str) -> dict[int, int]:
    """Parse a ``'epoch:grid_size,...'`` refinement spec into ``{epoch: size}``.

    Empty / whitespace strings yield an empty schedule (refinement disabled).
    """
    schedule: dict[int, int] = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        epoch_str, _, size_str = pair.partition(":")
        schedule[int(epoch_str)] = int(size_str)
    return schedule


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    step_every: int = 0,
    step_callback=None,
    global_step_start: int = 0,
    clip_norm: float = 0.0,
    scheduler=None,
    step_scheduler: bool = False,
) -> tuple[float, float]:
    """Run one pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Args:
        step_every: when training, invoke ``step_callback`` every N optimizer
            steps (0 disables).
        step_callback: called with the absolute optimizer-step index; used for
            mid-epoch validation snapshots.
        global_step_start: optimizer steps completed in previous epochs.
        clip_norm: max global gradient norm; 0 disables clipping.
        scheduler: LR scheduler stepped once per optimizer step when
            ``step_scheduler`` is True (used for warmup/cosine schedules).
        step_scheduler: whether ``scheduler`` should advance inside this loop.

    Returns:
        ``(mean_loss, accuracy)`` for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (X_batch, y_batch) in enumerate(loader, start=1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                loss.backward()
                if clip_norm and clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                if step_scheduler and scheduler is not None:
                    scheduler.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += batch_size

            if is_train and step_every and step_callback is not None:
                step = global_step_start + batch_idx
                if step % step_every == 0:
                    step_callback(step)

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
    tb_logger=None,
    step_eval_every: int = 0,
    step_eval_samples: int = 20000,
    val_eval_data: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = RANDOM_STATE,
    clip_norm: float = 0.0,
    lr_scheduler: str = "none",
    warmup_steps: int = 0,
    min_lr: float = 1e-6,
    grid_refine_schedule: dict[int, int] | None = None,
) -> dict[str, list[float]]:
    """Train with cross-entropy loss, periodic grid adaptation, and early stopping.

    Args:
        tb_logger: optional ``src.analysis.tensorboard_logger.
            SplineTensorBoardLogger``. Logging is failure-proof by design
            (any internal error disables it); ``None`` keeps training
            byte-identical to the pre-analysis behaviour.
        grid_refine_schedule: optional ``{epoch: new_grid_size}`` map. At the
            start of a listed epoch the spline resolution is grown (see
            ``model.refine_grids``); because that replaces parameters, the
            optimiser/scheduler are rebuilt and early-stopping state is reset.

    Returns:
        A ``history`` dict with per-epoch ``train_loss``, ``train_acc``,
        ``val_loss``, ``val_acc`` lists.
    """
    criterion = nn.CrossEntropyLoss()
    grid_refine_schedule = grid_refine_schedule or {}

    # LR schedule. 'cosine' is stepped every optimizer step (linear warmup then
    # cosine decay to min_lr) so the val curve keeps improving smoothly instead
    # of converging in epoch 1 and plateauing; 'plateau' is stepped per epoch on
    # val loss. 'none' keeps the flat-LR behaviour.
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * epochs)

    def build_optimizer(step_offset: int):
        """(Re)build the optimizer + scheduler; ``step_offset`` keeps the cosine
        schedule continuous across grid-refinement rebuilds."""
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = None
        step_scheduler = False
        if lr_scheduler == "cosine":
            min_factor = min_lr / lr

            def _cosine(local_step: int) -> float:
                step = step_offset + local_step
                if step < warmup_steps:
                    return (step + 1) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return min_factor + (1.0 - min_factor) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine)
            step_scheduler = True
        elif lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5,
                patience=max(1, patience // 3), min_lr=min_lr,
            )
        return optimizer, scheduler, step_scheduler

    optimizer, scheduler, step_scheduler = build_optimizer(0)

    # Seed the spline grids from the real training-data distribution before
    # the first gradient step, rather than leaving them at their fixed
    # initial `grid_range`.
    seed_batch = torch.from_numpy(train_X[: min(len(train_X), 512)]).to(device)
    model.update_grids(seed_batch)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }

    # Optional mid-epoch validation snapshots. On large datasets the model
    # converges within the first epoch, so per-epoch curves look flat; the
    # step-level curve shows the actual learning trajectory.
    step_eval_cb = None
    if step_eval_every and val_eval_data is not None:
        X_val_full, y_val_full = val_eval_data
        if len(X_val_full) > step_eval_samples:
            rng = np.random.default_rng(seed)
            eval_idx = rng.choice(len(X_val_full), size=step_eval_samples, replace=False)
        else:
            eval_idx = np.arange(len(X_val_full))
        X_step_eval = X_val_full[eval_idx]
        y_step_eval = torch.from_numpy(y_val_full[eval_idx].astype(np.int64))
        history["step_eval"] = {"step": [], "val_loss": [], "val_acc": []}

        def step_eval_cb(step: int) -> None:
            logits = predict_logits(model, X_step_eval, device, batch_size=1024)
            loss = criterion(logits, y_step_eval).item()
            acc = (logits.argmax(dim=1) == y_step_eval).float().mean().item()
            history["step_eval"]["step"].append(step)
            history["step_eval"]["val_loss"].append(loss)
            history["step_eval"]["val_acc"].append(acc)
            if tb_logger is not None:
                tb_logger.log_step_metrics(step, {"val_loss": loss, "val_acc": acc})
            model.train()  # predict_logits leaves the model in eval mode
            print(f"    [Step {step:06d}] val_loss={loss:.4f} val_acc={acc:.4f}")

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        # Grid refinement: grow spline resolution before this epoch trains.
        # Each layer's refine is self-guarding (a layer that would go non-finite
        # keeps its current grid), so the model is always finite afterwards. Only
        # rebuild the optimizer/scheduler and reset early-stopping when at least
        # one layer actually grew (its parameters were replaced).
        if epoch in grid_refine_schedule:
            new_size = grid_refine_schedule[epoch]
            refined = model.refine_grids(new_size, seed_batch)
            if refined:
                optimizer, scheduler, step_scheduler = build_optimizer(
                    (epoch - 1) * steps_per_epoch
                )
                best_val_loss = float("inf")
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(
                    f"[INFO] Grid refined to {new_size} at epoch {epoch}: "
                    f"{refined}/{len(model.layers)} layers grown "
                    f"({sum(p.numel() for p in model.parameters()):,} params); "
                    "optimizer rebuilt, early-stopping reset."
                )
            else:
                print(
                    f"[WARN] Grid refinement to {new_size} at epoch {epoch} made "
                    "no change (all layers kept their grid for numerical safety); "
                    "continuing without it."
                )

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer,
            step_every=step_eval_every,
            step_callback=step_eval_cb,
            global_step_start=(epoch - 1) * len(train_loader),
            clip_norm=clip_norm,
            scheduler=scheduler,
            step_scheduler=step_scheduler,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        if lr_scheduler == "plateau":
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history.setdefault("lr", []).append(current_lr)

        if tb_logger is not None:
            tb_logger.log_epoch_metrics(epoch, {
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
            })
            tb_logger.log_spline_state(model, epoch)

        if grid_update_freq and epoch % grid_update_freq == 0 and epoch < epochs:
            model.update_grids(seed_batch)

        improved = val_loss < best_val_loss
        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.2e}"
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

    # Step-level learning curve (recorded with --step-eval-every). This is the
    # curve that shows gradual learning when convergence happens inside epoch 1.
    step_eval = history.get("step_eval")
    if step_eval and step_eval["step"]:
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        ax_loss.plot(step_eval["step"], step_eval["val_loss"], color="tab:orange")
        ax_loss.set_title("Validation Loss (subsample)")
        ax_loss.set_xlabel("Optimizer step")
        ax_loss.set_ylabel("Loss")

        ax_acc.plot(step_eval["step"], step_eval["val_acc"], color="tab:blue")
        ax_acc.set_title("Validation Accuracy (subsample)")
        ax_acc.set_xlabel("Optimizer step")
        ax_acc.set_ylabel("Accuracy")

        fig.suptitle("KAN Sentiment Classifier — Step-Level Learning Curve")
        fig.tight_layout()
        step_path = output_path.with_name("training_curves_steps.png")
        fig.savefig(step_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved step-level learning curve to: {step_path.resolve()}")


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


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
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

    # 2b. Optional spline interpretability hooks (src/analysis) -------------
    # getattr defaults keep this backward compatible with callers that pass a
    # bare Namespace without the new flags (e.g. older scripts).
    viz_enabled = bool(getattr(args, "enable_spline_visualization", False))
    tb_enabled = bool(getattr(args, "enable_tensorboard", False))
    serializable_config = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in model_config.items()
    }

    tb_logger = None
    if tb_enabled:
        try:
            from src.analysis.tensorboard_logger import SplineTensorBoardLogger

            tb_logger = SplineTensorBoardLogger(
                getattr(args, "tensorboard_dir", "outputs/tensorboard/kan")
            )
        except Exception as exc:  # analysis extras must never block training
            print(f"[WARN] Could not set up TensorBoard spline logging: {exc}")

    before_model = None
    if viz_enabled:
        # Snapshot the model *before* any gradient step but *after* seeding
        # the knot grids from real data, so before/after figures isolate what
        # gradient descent learned (train_model repeats the same seeding;
        # refitting the grid twice on the same batch is a harmless no-op).
        seed_batch = torch.from_numpy(X_train[: min(len(X_train), 512)]).to(device)
        model.update_grids(seed_batch)
        before_model = copy.deepcopy(model)
        init_path = Path(args.output_dir) / "kan_model_init.pt"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": before_model.state_dict(), "config": serializable_config},
            init_path,
        )
        print(f"[INFO] Saved pre-training spline snapshot to: {init_path}")

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
        tb_logger=tb_logger,
        step_eval_every=getattr(args, "step_eval_every", 0),
        step_eval_samples=getattr(args, "step_eval_samples", 20000),
        val_eval_data=(X_val, y_val),
        seed=args.seed,
        clip_norm=getattr(args, "gradient_clip_norm", 0.0),
        lr_scheduler=getattr(args, "lr_scheduler", "none"),
        warmup_steps=getattr(args, "warmup_steps", 0),
        min_lr=getattr(args, "min_lr", 1e-6),
        grid_refine_schedule=parse_grid_refine(getattr(args, "grid_refine", "")),
    )
    if tb_logger is not None:
        tb_logger.close()

    # Grid refinement may have grown the spline resolution during training;
    # record the FINAL grid size so the saved checkpoint reconstructs to
    # matching tensor shapes when reloaded (all layers are refined uniformly).
    model_config["grid_size"] = model.semantic_compression.grid_size

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

    # 6. Optional spline interpretability report ------------------------------
    # Best-effort by design: a visualization failure must never invalidate a
    # finished training run, so everything is wrapped and only warns.
    if viz_enabled:
        try:
            from src.analysis.spline_visualizer import generate_all_visualizations

            written = generate_all_visualizations(
                model,
                X=X_train[: min(len(X_train), 256)],
                before_model=before_model,
                output_dir=Path(getattr(args, "spline_viz_dir", "outputs/kan_spline_visualizations")),
                top_k=int(getattr(args, "spline_viz_top_k", 8)),
            )
            print(f"[INFO] Spline interpretability report: {len(written)} files.")
        except Exception as exc:
            print(
                f"[WARN] Spline visualization failed ({exc}). "
                "Training artifacts above are unaffected."
            )

    print("\n[DONE] KAN expert pipeline complete.")


if __name__ == "__main__":
    main()
