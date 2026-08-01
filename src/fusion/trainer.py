"""Training pipeline for the Residual-Aware Adaptive Fusion gate.

The gate is trained on the *validation* split's expert predictions -- never on
the experts' own training split, where both experts are overconfident and the
meta-features would not reflect deployment conditions. The validation split is
further divided (stratified) into a gate-train and a gate-holdout portion; the
holdout drives early stopping, LR scheduling, and checkpoint selection, while
the untouched test split is reserved for the final report.

Features of the loop:

* cross-entropy loss on the fused distribution (``fused_cross_entropy``)
* early stopping on gate-holdout loss
* best-model checkpointing (single self-contained ``fusion_model.pt``)
* TensorBoard logging (losses, accuracies, mean gate value, learning rate)
* ``ReduceLROnPlateau`` learning-rate scheduling

Usage::

    python -m src.fusion.trainer
    python -m src.fusion.trainer --epochs 300 --hidden-dim 64 --dropout 0.2
    tensorboard --logdir models/fusion/tensorboard
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.fusion.fusion_dataset import (
    FusionFeatureBuilder,
    FusionSplit,
    StandardScaler,
    load_fusion_inputs,
    make_dataloader,
)
from src.fusion.calibration import TemperatureScaler
from src.fusion.fallback import FallbackConfig, UncertaintyFallback
from src.fusion.fusion_model import FusionGate, fused_cross_entropy
from src.fusion.predictor import FusionPredictor, evaluate_and_export
from src.fusion.utils import get_logger, resolve_device, save_json, set_seed

logger = get_logger("fusion.trainer")

DEFAULT_OUTPUT_DIR = "models/fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the residual-aware adaptive fusion gate."
    )
    # Data locations
    parser.add_argument("--inputs-dir", default="data/fusion_inputs")
    parser.add_argument("--kan-dir", default="models/kan_subset")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    # Feature construction
    parser.add_argument("--residual-bins", type=int, default=10)
    parser.add_argument(
        "--gate-holdout",
        type=float,
        default=0.2,
        help="Fraction of the validation split held out for early stopping.",
    )
    # Gate architecture
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    # Optimisation
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    # Uncertainty-aware fallback
    parser.add_argument(
        "--enable-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit the uncertainty-aware fallback layer and apply it at "
        "evaluation (disable with --no-enable-fallback).",
    )
    parser.add_argument(
        "--fallback-temperature",
        type=float,
        default=2.0,
        help="Softmax temperature (>1) used by the conservative fallback branch.",
    )
    parser.add_argument(
        "--fallback-apply-overrides",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the fallback override predictions (abstain/route/soften). Off by "
        "default: the fallback only flags risky samples and never lowers accuracy.",
    )
    parser.add_argument(
        "--calibrate-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temperature-scale each expert's probabilities (fit on val) before "
        "the gate, so the confidence signals it relies on are well-calibrated.",
    )
    # Expert-balance regularisation (give the structural expert more weight).
    parser.add_argument(
        "--alpha-reg-weight",
        type=float,
        default=0.0,
        help="Load-balancing penalty pushing the gate's mean trust toward "
        "--alpha-target, so it leans less on the dominant expert (default 0.0 = "
        "off, accuracy-optimal). Larger = the structural (XGBoost) expert gets "
        "more weight, trading a little accuracy. Try 0.1-1.0.",
    )
    parser.add_argument(
        "--alpha-target",
        type=float,
        default=0.5,
        help="Target mean gate value alpha for --alpha-reg-weight (0.5 = balanced "
        "KAN/XGBoost; lower favours XGBoost, higher favours the KAN).",
    )
    # Environment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def stratified_holdout(
    labels: np.ndarray, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions into (train, holdout) preserving class balance."""
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    holdout_idx: list[np.ndarray] = []
    for cls in np.unique(labels):
        rows = np.where(labels == cls)[0]
        rng.shuffle(rows)
        n_holdout = max(1, int(round(len(rows) * holdout_fraction)))
        holdout_idx.append(rows[:n_holdout])
        train_idx.append(rows[n_holdout:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(holdout_idx))


def subset_split(split: FusionSplit, idx: np.ndarray, name: str) -> FusionSplit:
    """Index a :class:`FusionSplit` down to the given rows."""
    return FusionSplit(
        name=name,
        features=split.features[idx],
        kan_probs=split.kan_probs[idx],
        xgb_probs=split.xgb_probs[idx],
        labels=split.labels[idx],
        feature_names=split.feature_names,
        source_rows=None if split.source_rows is None else split.source_rows[idx],
        texts=None if split.texts is None else [split.texts[i] for i in idx],
        embeddings=None if split.embeddings is None else split.embeddings[idx],
    )


@torch.no_grad()
def evaluate_loader(
    model: FusionGate, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    """Return (mean CE loss, accuracy, mean alpha) over a dataloader."""
    model.eval()
    total_loss, total_correct, total_alpha, total_n = 0.0, 0, 0.0, 0
    for features, kan_probs, xgb_probs, labels in loader:
        features, labels = features.to(device), labels.to(device)
        kan_probs, xgb_probs = kan_probs.to(device), xgb_probs.to(device)
        alpha = model(features)
        loss = fused_cross_entropy(alpha, kan_probs, xgb_probs, labels)
        fused = FusionGate.fuse(alpha, kan_probs, xgb_probs)
        n = len(labels)
        total_loss += loss.item() * n
        total_correct += (fused.argmax(dim=1) == labels).sum().item()
        total_alpha += alpha.sum().item()
        total_n += n
    return total_loss / total_n, total_correct / total_n, total_alpha / total_n


def train_gate(
    model: FusionGate,
    train_loader: DataLoader,
    holdout_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    writer: SummaryWriter,
) -> dict[str, object]:
    """Run the full training loop; returns history and restores best weights."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    # Optional expert-balance regularisation (default off -> behaviour unchanged).
    alpha_reg_weight = float(getattr(args, "alpha_reg_weight", 0.0))
    alpha_target = float(getattr(args, "alpha_target", 0.5))
    if alpha_reg_weight > 0:
        logger.info(
            "Alpha-balance regularisation ON: weight=%.3f target=%.2f "
            "(pushes the gate to give XGBoost more weight).",
            alpha_reg_weight, alpha_target,
        )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "holdout_loss": [], "holdout_acc": [],
    }
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        for features, kan_probs, xgb_probs, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            kan_probs, xgb_probs = kan_probs.to(device), xgb_probs.to(device)

            optimizer.zero_grad()
            alpha = model(features)
            loss = fused_cross_entropy(alpha, kan_probs, xgb_probs, labels)
            if alpha_reg_weight > 0:
                # Penalise the batch mean trust for straying from the target, so
                # the gate spreads weight onto the structural expert instead of
                # collapsing onto the dominant one. Per-sample discrimination is
                # preserved (only the mean is constrained).
                loss = loss + alpha_reg_weight * (alpha.mean() - alpha_target) ** 2
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                fused = FusionGate.fuse(alpha, kan_probs, xgb_probs)
                epoch_correct += (fused.argmax(dim=1) == labels).sum().item()
            epoch_loss += loss.item() * len(labels)
            epoch_n += len(labels)

        train_loss = epoch_loss / epoch_n
        train_acc = epoch_correct / epoch_n
        holdout_loss, holdout_acc, holdout_alpha = evaluate_loader(
            model, holdout_loader, device
        )
        scheduler.step(holdout_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["holdout_loss"].append(holdout_loss)
        history["holdout_acc"].append(holdout_acc)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/holdout", holdout_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/holdout", holdout_acc, epoch)
        writer.add_scalar("gate/mean_alpha_holdout", holdout_alpha, epoch)
        writer.add_scalar("optim/lr", current_lr, epoch)

        improved = holdout_loss < best_loss
        if improved:
            best_loss = holdout_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            logger.info(
                "[Epoch %03d/%d] train_loss=%.4f train_acc=%.4f | "
                "holdout_loss=%.4f holdout_acc=%.4f | lr=%.2e%s",
                epoch, args.epochs, train_loss, train_acc,
                holdout_loss, holdout_acc, current_lr,
                "  * new best" if improved else "",
            )

        if patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (no holdout-loss improvement "
                "for %d epochs).", epoch, args.patience,
            )
            break

    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_holdout_loss": best_loss,
        "best_epoch": best_epoch,
        "trained_epochs": len(history["train_loss"]),
    }


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using device: %s", device)

    # 1. Load row-aligned expert outputs -----------------------------------
    inputs = load_fusion_inputs(args.inputs_dir, args.kan_dir)

    # 1b. Calibrate each expert's probabilities (temperature scaling, fit on
    # val) so the confidence/entropy signals the gate depends on are honest.
    # Applied to every split before any feature is built, so the whole pipeline
    # (residuals, meta-features, fusion, fallback) sees calibrated probabilities.
    kan_temp, xgb_temp = TemperatureScaler(), TemperatureScaler()
    if args.calibrate_experts:
        val_labels = inputs["val"]["df"]["label"].to_numpy(dtype=np.int64)
        kan_temp.fit(inputs["val"]["kan_probs"], val_labels)
        xgb_temp.fit(inputs["val"]["xgb_probs"], val_labels)
        for split_inputs in inputs.values():
            split_inputs["kan_probs"] = kan_temp.transform(split_inputs["kan_probs"])
            split_inputs["xgb_probs"] = xgb_temp.transform(split_inputs["xgb_probs"])
        logger.info(
            "Calibrated experts (temperature scaling): T_kan=%.3f  T_xgb=%.3f "
            "(>1 = softened an over-confident expert).",
            kan_temp.temperature, xgb_temp.temperature,
        )

    # 2. Build fusion features ----------------------------------------------
    # Residual history comes from the experts' TRAIN split; the gate itself
    # never sees that split, so residual features stay leak-free.
    builder = FusionFeatureBuilder(n_residual_bins=args.residual_bins)
    train_labels = inputs["train"]["df"]["label"].to_numpy(dtype=np.int64)
    builder.fit_residuals(
        inputs["train"]["kan_probs"], inputs["train"]["xgb_probs"], train_labels
    )

    splits = {
        name: builder.build(
            name,
            inputs[name]["df"],
            inputs[name]["embeddings"],
            inputs[name]["kan_probs"],
            inputs[name]["xgb_probs"],
            aux_fail=inputs[name].get("aux_fail"),
        )
        for name in ("val", "test")
    }
    if "aux_fail" in inputs["val"]:
        logger.info("Using KAN aux competence signal as an extra gate feature.")
    logger.info(
        "Fusion features: %d columns | val=%d test=%d",
        splits["val"].features.shape[1], len(splits["val"]), len(splits["test"]),
    )

    # 3. Gate-train / gate-holdout split of the validation predictions ------
    train_idx, holdout_idx = stratified_holdout(
        splits["val"].labels, args.gate_holdout, args.seed
    )
    gate_train = subset_split(splits["val"], train_idx, "gate_train")
    gate_holdout = subset_split(splits["val"], holdout_idx, "gate_holdout")
    logger.info(
        "Gate training on %d samples, early-stopping holdout %d samples.",
        len(gate_train), len(gate_holdout),
    )

    scaler = StandardScaler().fit(gate_train.features)
    train_loader = make_dataloader(
        scaler.transform(gate_train.features),
        gate_train.kan_probs, gate_train.xgb_probs, gate_train.labels,
        batch_size=args.batch_size, shuffle=True, seed=args.seed,
    )
    holdout_loader = make_dataloader(
        scaler.transform(gate_holdout.features),
        gate_holdout.kan_probs, gate_holdout.xgb_probs, gate_holdout.labels,
        batch_size=args.batch_size,
    )

    # 4. Train ---------------------------------------------------------------
    model = FusionGate(
        input_dim=gate_train.features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    logger.info(
        "FusionGate: %d -> %d -> 1 (dropout %.2f), %d parameters.",
        model.input_dim, model.hidden_dim, args.dropout,
        sum(p.numel() for p in model.parameters()),
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    t0 = time.time()
    result = train_gate(model, train_loader, holdout_loader, device, args, writer)
    writer.close()
    logger.info(
        "Training finished in %.1fs | best epoch %d | best holdout loss %.4f",
        time.time() - t0, result["best_epoch"], result["best_holdout_loss"],
    )

    # 5. Checkpoint (self-contained: model + scaler + residual tables) ------
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config(),
        "scaler": scaler.state_dict(),
        "residuals": {
            "kan": builder.kan_residuals.state_dict(),
            "xgb": builder.xgb_residuals.state_dict(),
        },
        "calibration": {
            "enabled": bool(args.calibrate_experts),
            "kan": kan_temp.state_dict(),
            "xgb": xgb_temp.state_dict(),
        },
        "feature_names": gate_train.feature_names,
        "text_column": builder.text_column,
        "train_args": vars(args),
        "training_summary": {
            "best_epoch": result["best_epoch"],
            "best_holdout_loss": result["best_holdout_loss"],
            "trained_epochs": result["trained_epochs"],
        },
    }
    checkpoint_path = output_dir / "fusion_model.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path.resolve())

    save_json(
        {"args": vars(args), **{k: v for k, v in result.items() if k != "history"},
         "history": result["history"]},
        output_dir / "training_history.json",
    )

    # 5b. Fit the uncertainty-aware fallback and fold it into the checkpoint -
    # Thresholds are calibrated on the gate-holdout split (never on test), the
    # OOD detector is fit on the experts' TRAIN embeddings (in-distribution
    # reference), so nothing about the test split leaks into the fallback.
    predictor = FusionPredictor.from_checkpoint(checkpoint_path, device=device)
    if args.enable_fallback:
        holdout_outputs = predictor.predict(gate_holdout)
        fallback = UncertaintyFallback(
            FallbackConfig(
                conservative_temperature=args.fallback_temperature,
                apply_overrides=args.fallback_apply_overrides,
            )
        )
        fallback.fit(
            train_embeddings=inputs["train"]["embeddings"],
            holdout_fused_probs=holdout_outputs.fused_probs,
            holdout_kan_probs=gate_holdout.kan_probs,
            holdout_xgb_probs=gate_holdout.xgb_probs,
            holdout_embeddings=gate_holdout.embeddings,
            holdout_labels=gate_holdout.labels,
        )
        checkpoint["fallback"] = fallback.state_dict()
        torch.save(checkpoint, checkpoint_path)  # re-save with fallback embedded
        predictor.fallback = fallback
        logger.info(
            "Uncertainty-aware fallback fitted and saved "
            "(entropy_high=%.3f entropy_abstain=%.3f ood_abstain=%.3f).",
            fallback.config.entropy_high, fallback.config.entropy_abstain,
            fallback.config.ood_abstain,
        )

    # 6. Final evaluation + artifact export on the untouched test split -----
    evaluate_and_export(
        predictor,
        test_split=splits["test"],
        gate_holdout_split=gate_holdout,
        output_dir=output_dir,
    )
    logger.info("[DONE] Fusion pipeline complete. Artifacts in %s", output_dir.resolve())


if __name__ == "__main__":
    main()
