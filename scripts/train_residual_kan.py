"""Residual-aware specialisation training for the KAN semantic expert.

This is a *drop-in alternative* to ``src/models/kan/train.py``: it produces the
exact same artifact layout (``kan_model.pt``, ``{split}_probs.npy``,
``{split}_confidence.npy``, ``metrics.json``, ``history.json``,
``training_curves.png``) so the resulting model plugs straight into the fusion
pipeline — but it trains the KAN with the residual-aware complementarity
objective instead of plain cross-entropy.

What it adds over the standard KAN trainer
------------------------------------------
1. **Leakage-safe residual weighting.** It fits XGBoost on the *structural*
   feature matrix with stratified out-of-fold CV (:mod:`src.models.xgboost.oof`),
   turns each training tweet's true-class probability into a bounded, capped
   training weight, and up-weights the tweets XGBoost found hard.
2. **Complementarity loss** (:mod:`src.models.kan.residual_loss`): unweighted
   base CE + residual-weighted CE + semantic-rescue margin + optional auxiliary
   "will XGBoost fail?" head.
3. **Complementarity report** (:mod:`src.fusion.complementarity`): KAN rescue
   rate, XGBoost preservation rate, disagreement accuracy, double-fault rate,
   oracle ensemble accuracy — on train (via the OOF probs) and, when XGBoost
   probabilities are supplied, on val/test too.

The two experts never share features: XGBoost sees only structural indicators,
the KAN sees only DeBERTa embeddings. The complementarity is introduced purely
through the training objective.

Examples
--------
Fusion-subsample layout (embeddings + splits + xgboost probs all row-aligned)::

    python scripts/train_residual_kan.py \
        --embeddings-dir data/fusion_inputs/embeddings \
        --splits-dir data/fusion_inputs/splits \
        --xgb-probs-dir data/fusion_inputs/xgboost \
        --xgb-params models/xgboost/best_params.json \
        --output-dir models/kan_residual \
        --aux-failure-head --gradient-clip-norm 1.0

Then fuse as usual, pointing the gate at the new KAN dir::

    python -m src.fusion.trainer --kan-dir models/kan_residual --output-dir models/fusion_residual
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.complementarity import (  # noqa: E402
    complementarity_metrics,
    format_complementarity,
)
from src.models.kan.data_loader import load_splits as load_embedding_splits  # noqa: E402
from src.models.kan.evaluate import (  # noqa: E402
    confidence_from_entropy,
    evaluate_split,
    logits_to_probs,
)
from src.models.kan.model import DEFAULT_HIDDEN_DIMS, KANSentimentClassifier  # noqa: E402
from src.models.kan.residual_loss import (  # noqa: E402
    ComplementarityConfig,
    ComplementarityLoss,
    make_residual_dataloader,
)
from src.models.kan.train import (  # noqa: E402
    plot_training_curves,
    predict_logits,
    resolve_device,
    set_global_seed,
)
from src.models.xgboost.data_loader import (  # noqa: E402
    FALLBACK_TEXT_COLUMN,
    LABEL_COLUMN,
    RAW_TEXT_COLUMN,
)
from src.models.xgboost.features import FeatureExtractor, ensure_nltk_resources  # noqa: E402
from src.models.xgboost.oof import ResidualTargets, build_residual_targets  # noqa: E402

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # --- Inputs -------------------------------------------------------------
    p.add_argument(
        "--embeddings-dir",
        default="data/fusion_inputs/embeddings",
        help="Dir with {split}_embeddings.npy / {split}_labels.npy (KAN input).",
    )
    p.add_argument(
        "--splits-dir",
        default="data/fusion_inputs/splits",
        help="Dir with {split}.csv (raw text) for XGBoost structural features. "
        "Must be row-aligned with the embeddings.",
    )
    p.add_argument(
        "--xgb-probs-dir",
        default=None,
        help="Optional dir with row-aligned {split}_probs.npy from the trained "
        "XGBoost expert; enables val/test complementarity metrics.",
    )
    p.add_argument(
        "--xgb-params",
        default=None,
        help="Optional path to the XGBoost expert's best_params.json so the "
        "out-of-fold difficulty model matches the deployed expert.",
    )
    p.add_argument("--output-dir", default="models/kan_residual")

    # --- Residual weighting -------------------------------------------------
    p.add_argument("--oof-splits", type=int, default=5, help="Stratified CV folds for OOF XGBoost.")
    p.add_argument(
        "--weight-method",
        choices=["percentile", "linear"],
        default="percentile",
        help="Residual->weight mapping (percentile is calibration-robust).",
    )
    p.add_argument("--weight-max", type=float, default=3.0, help="Cap on any training weight.")
    p.add_argument("--weight-floor", type=float, default=1.0, help="Weight for easy tweets.")
    p.add_argument("--weight-scale", type=float, default=2.0, help="Slope for --weight-method linear.")
    p.add_argument(
        "--double-fault-cap",
        type=float,
        default=None,
        help="Extra lower cap on weights of XGBoost-failed samples so genuinely "
        "ambiguous cases get limited emphasis (default: rely on --weight-max).",
    )

    # --- Complementarity loss ----------------------------------------------
    p.add_argument("--base-weight", type=float, default=1.0)
    p.add_argument("--comp-weight", type=float, default=1.0)
    p.add_argument("--rescue-weight", type=float, default=0.5)
    p.add_argument("--rescue-margin", type=float, default=0.1)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument(
        "--aux-failure-head",
        action="store_true",
        help="Attach the auxiliary XGBoost-competence head to the KAN.",
    )
    p.add_argument("--aux-hidden", type=int, default=64)

    # --- KAN architecture ---------------------------------------------------
    p.add_argument("--hidden-dims", default=",".join(str(d) for d in DEFAULT_HIDDEN_DIMS))
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--spline-order", type=int, default=3)
    p.add_argument("--grid-range", default="-2,2")

    # --- Optimisation -------------------------------------------------------
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--grid-update-freq", type=int, default=10)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    # --- Crash resilience ---------------------------------------------------
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume KAN training from <output-dir>/train_state.pt if present "
        "(also reuses the cached OOF residual targets). Safe to pass on the "
        "first run too — it simply starts fresh when no checkpoint exists.",
    )
    p.add_argument(
        "--reuse-oof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse <output-dir>/residual_targets.npz if it matches the train "
        "split, skipping the expensive feature-extraction + out-of-fold XGBoost "
        "stage (--no-reuse-oof forces recomputation).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Structural feature loading (row-aligned with the embeddings)
# --------------------------------------------------------------------------- #
def _resolve_text_column(df: pd.DataFrame) -> str:
    if RAW_TEXT_COLUMN in df.columns:
        return RAW_TEXT_COLUMN
    if FALLBACK_TEXT_COLUMN in df.columns:
        print(f"[WARN] '{RAW_TEXT_COLUMN}' missing; using '{FALLBACK_TEXT_COLUMN}'.")
        return FALLBACK_TEXT_COLUMN
    sys.exit(
        f"[ERROR] split CSV has neither '{RAW_TEXT_COLUMN}' nor "
        f"'{FALLBACK_TEXT_COLUMN}'. Columns: {list(df.columns)}"
    )


def load_structural_features(
    splits_dir: Path, split: str, expected_len: int, extractor: FeatureExtractor
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the XGBoost structural feature matrix for one split's CSV.

    Rows are kept in file order (never dropped) so the matrix stays aligned with
    the precomputed embeddings; empty text becomes an empty string.
    """
    csv_path = splits_dir / f"{split}.csv"
    if not csv_path.exists():
        sys.exit(f"[ERROR] Missing split CSV for structural features: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    if len(df) != expected_len:
        sys.exit(
            f"[ERROR] {split}: split CSV ({len(df)}) is not row-aligned with the "
            f"embeddings ({expected_len}). Regenerate them from the same subsample."
        )
    text_col = _resolve_text_column(df)
    texts = df[text_col].fillna("").astype(str).tolist()
    features = extractor.transform(texts)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    return features, labels


# --------------------------------------------------------------------------- #
# Training loop (residual-aware)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _val_ce(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, float]:
    """Plain cross-entropy + accuracy on a split (used for early stopping)."""
    logits = predict_logits(model, X, device)
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    loss = nn.functional.cross_entropy(logits, y_t).item()
    acc = float((logits.argmax(dim=1) == y_t).float().mean())
    return loss, acc


def _compact_state(state: dict) -> dict:
    """Clone tensors so saved checkpoints don't drag oversized backing storage.

    pykan's grid adaptation leaves ``coef`` as a small *view* into a large
    intermediate B-spline basis storage; ``torch.save`` serialises the whole
    storage (hundreds of MB) unless the view is cloned into a compact tensor
    first. Cloning shrinks each checkpoint from ~470 MB to ~10 MB.
    """
    return {
        k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
        for k, v in state.items()
    }


def _save_train_state(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    best_state: dict,
    wait: int,
    history: dict,
    config: dict,
) -> None:
    """Persist a resumable training checkpoint as safely as Windows allows.

    Writes to a temp file then atomically renames it over the target. On Windows
    the rename can transiently fail with ``PermissionError`` (WinError 32) when
    antivirus/indexing briefly locks the freshly-written file, so we retry with a
    short backoff and, as a last resort, write the payload directly over the
    target (non-atomic but still leaves a valid checkpoint).
    """
    payload = {
        "epoch": epoch,
        "model_state": _compact_state(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "best_val": best_val,
        "best_state": _compact_state(best_state),
        "wait": wait,
        "history": history,
        "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    for attempt in range(5):
        try:
            os.replace(tmp, path)  # atomic on the same filesystem
            return
        except PermissionError:
            time.sleep(0.4 * (attempt + 1))
    # Rename kept losing the race with a file lock: write straight to the target.
    print(f"[WARN] atomic rename to {path.name} kept failing; writing in place.")
    torch.save(payload, path)
    try:
        tmp.unlink()
    except OSError:
        pass


def train_residual_kan(
    model: KANSentimentClassifier,
    train_loader,
    X_train: np.ndarray,
    val_data: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    loss_fn: ComplementarityLoss,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grid_update_freq: int,
    clip_norm: float,
    out_dir: Path | None = None,
    config: dict | None = None,
    resume_state: dict | None = None,
) -> dict[str, Any]:
    """Train the KAN with the complementarity objective and early stop on val CE.

    When ``out_dir`` is given, a resumable ``train_state.pt`` is written after
    every epoch and the best model so far is mirrored to ``kan_model.pt``, so an
    interruption never loses more than the in-progress epoch. Passing
    ``resume_state`` (loaded from a prior ``train_state.pt``) continues training
    from where it stopped.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_val, y_val = val_data
    seed_batch = torch.from_numpy(X_train[: min(len(X_train), 512)]).to(device)

    if resume_state is not None:
        # Grids/weights come straight from the checkpoint — do NOT re-seed.
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_val = float(resume_state["best_val"])
        best_state = resume_state["best_state"]
        wait = int(resume_state["wait"])
        history = resume_state["history"]
        print(f"[INFO] Resumed from epoch {resume_state['epoch']} "
              f"(best_val_loss={best_val:.4f}); continuing at epoch {start_epoch}.")
    else:
        # Seed spline grids from the real training distribution before step 1.
        model.update_grids(seed_batch)
        start_epoch = 1
        best_val = float("inf")
        best_state = _compact_state(model.state_dict())
        wait = 0
        history = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "components": [],
        }

    stopped_epoch = epochs
    state_path = (out_dir / "train_state.pt") if out_dir is not None else None
    best_path = (out_dir / "kan_model.pt") if out_dir is not None else None

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        epoch_components: dict[str, float] = {}
        for X_b, y_b, w_b, ptrue_b, fail_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            w_b = w_b.to(device)
            ptrue_b = ptrue_b.to(device)
            fail_b = fail_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            aux_logit = model.aux_failure_logit(X_b)
            loss, components = loss_fn(
                logits, y_b,
                sample_weight=w_b,
                xgb_true_prob=ptrue_b,
                xgb_fail=fail_b,
                aux_logit=aux_logit,
            )
            loss.backward()
            if clip_norm and clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            bs = y_b.size(0)
            tot_loss += float(loss.detach()) * bs
            correct += int((logits.argmax(dim=1) == y_b).sum())
            n += bs
            for key, value in components.items():
                epoch_components[key] = epoch_components.get(key, 0.0) + value * bs

        train_loss = tot_loss / n
        train_acc = correct / n
        val_loss, val_acc = _val_ce(model, X_val, y_val, device)
        epoch_components = {k: v / n for k, v in epoch_components.items()}

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["components"].append(epoch_components)

        if grid_update_freq and epoch % grid_update_freq == 0 and epoch < epochs:
            model.update_grids(seed_batch)

        improved = val_loss < best_val - 1e-6
        breakdown = " ".join(f"{k}={v:.3f}" for k, v in epoch_components.items() if k != "total")
        print(
            f"[Epoch {epoch:03d}/{epochs}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| {breakdown}" + ("  * new best" if improved else "")
        )

        early_stop = False
        if improved:
            best_val = val_loss
            best_state = _compact_state(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch
                early_stop = True

        # Persist after every epoch so an interruption loses at most this epoch.
        # kan_model.pt always holds the best model so far (deployable); it is
        # mirrored only on improvement to avoid regressing the saved best.
        if state_path is not None:
            _save_train_state(
                state_path, epoch=epoch, model=model, optimizer=optimizer,
                best_val=best_val, best_state=best_state, wait=wait,
                history=history, config=config or {},
            )
        if improved and best_path is not None:
            torch.save({"model_state_dict": best_state, "config": config or {}}, best_path)

        if early_stop:
            print(f"[INFO] Early stopping at epoch {epoch} (no val improvement for {patience}).")
            break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["stopped_epoch"] = stopped_epoch
    return history


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")
    ensure_nltk_resources()

    # 1. Embeddings (KAN input) -------------------------------------------
    emb_splits = load_embedding_splits(args.embeddings_dir)
    X_train, y_train = emb_splits["train"]
    X_val, y_val = emb_splits["val"]
    X_test, y_test = emb_splits["test"]
    input_dim = X_train.shape[1]
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    print(f"[INFO] Embedding dim: {input_dim} | classes: {n_classes}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 2-3. Leakage-safe residual weighting — cached, since feature extraction +
    # out-of-fold XGBoost on the full split is the most expensive stage and must
    # survive an interruption. On a cache hit we skip both entirely.
    cache_path = out / "residual_targets.npz"
    targets = None
    if args.reuse_oof and cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["weights"]) == len(X_train):
            targets = ResidualTargets(
                weights=cached["weights"],
                xgb_true_prob=cached["xgb_true_prob"],
                xgb_fail=cached["xgb_fail"],
                oof_probs=cached["oof_probs"],
            )
            print(f"[INFO] Reusing cached OOF residual targets from {cache_path} "
                  f"(skipping feature extraction + OOF). {targets.summary()}")
        else:
            print(f"[WARN] Cached targets ({len(cached['weights'])}) do not match "
                  f"train ({len(X_train)}); recomputing.")

    if targets is None:
        print("[INFO] Extracting structural features for the OOF XGBoost model ...")
        extractor = FeatureExtractor()
        extractor.fit([])  # stateless schema; transform does the work
        splits_dir = Path(args.splits_dir)
        X_struct_train, y_struct_train = load_structural_features(
            splits_dir, "train", len(X_train), extractor
        )
        if not np.array_equal(y_struct_train, y_train):
            sys.exit(
                "[ERROR] train labels from the split CSV disagree with the embedding "
                "labels — the structural features and embeddings are not aligned."
            )
        print(f"[INFO] Structural train matrix: {X_struct_train.shape}")

        xgb_params = None
        if args.xgb_params:
            xgb_params = json.loads(Path(args.xgb_params).read_text(encoding="utf-8"))
            xgb_params.pop("scale_pos_weight", None)  # refilled per fold
            print(f"[INFO] Loaded XGBoost params from {args.xgb_params}.")
        targets = build_residual_targets(
            X_struct_train, y_train,
            params=xgb_params,
            n_splits=args.oof_splits,
            seed=args.seed,
            method=args.weight_method,
            w_max=args.weight_max,
            floor=args.weight_floor,
            scale=args.weight_scale,
            double_fault_cap=args.double_fault_cap,
        )
        np.savez(
            cache_path,
            weights=targets.weights,
            xgb_true_prob=targets.xgb_true_prob,
            xgb_fail=targets.xgb_fail,
            oof_probs=targets.oof_probs,
        )
        print(f"[INFO] Cached OOF residual targets to {cache_path}.")

    # 4. Build the KAN (optionally with the aux competence head) -----------
    hidden_dims = tuple(int(d) for d in args.hidden_dims.split(","))
    grid_range = tuple(float(v) for v in args.grid_range.split(","))
    model_config = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "n_classes": n_classes,
        "grid_size": args.grid_size,
        "spline_order": args.spline_order,
        "grid_range": grid_range,
        "aux_failure_head": bool(args.aux_failure_head),
        "aux_hidden": args.aux_hidden,
    }
    model = KANSentimentClassifier(**model_config).to(device)
    serializable_config = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in model_config.items()
    }
    print(
        f"[INFO] KAN: {input_dim} -> {hidden_dims} -> {n_classes} "
        f"| aux_failure_head={args.aux_failure_head}"
    )

    # Resume: reload a prior training checkpoint if requested and present.
    resume_state = None
    state_path = out / "train_state.pt"
    if args.resume and state_path.exists():
        resume_state = torch.load(state_path, map_location=device, weights_only=False)
        print(f"[INFO] Found resumable checkpoint at {state_path} (epoch {resume_state['epoch']}).")
    elif args.resume:
        print(f"[INFO] --resume given but no checkpoint at {state_path}; starting fresh.")

    # 5. Train with the complementarity objective --------------------------
    loss_cfg = ComplementarityConfig(
        base_weight=args.base_weight,
        comp_weight=args.comp_weight,
        rescue_weight=args.rescue_weight,
        aux_weight=args.aux_weight if args.aux_failure_head else 0.0,
        rescue_margin=args.rescue_margin,
        hard_threshold=args.hard_threshold,
    )
    loss_fn = ComplementarityLoss(loss_cfg)
    train_loader = make_residual_dataloader(
        X_train, y_train,
        sample_weight=targets.weights,
        xgb_true_prob=targets.xgb_true_prob,
        xgb_fail=targets.xgb_fail,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    history = train_residual_kan(
        model, train_loader, X_train, (X_val, y_val), device, loss_fn,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, grid_update_freq=args.grid_update_freq,
        clip_norm=args.gradient_clip_norm,
        out_dir=out, config=serializable_config, resume_state=resume_state,
    )

    # 6. Evaluate + export (same layout as src/models/kan/train.py) --------
    print("\n" + "=" * 64 + "\nEVALUATION\n" + "=" * 64)
    probs, confidence, metrics = {}, {}, {}
    for split, (X, y) in (("train", (X_train, y_train)), ("val", (X_val, y_val)), ("test", (X_test, y_test))):
        logits = predict_logits(model, X, device)
        split_probs = logits_to_probs(logits)
        probs[split] = split_probs
        confidence[split] = confidence_from_entropy(split_probs)
        metrics[split] = evaluate_split(split_probs, y, split, n_classes)

    metrics["model"] = "KAN (residual-aware complementarity, DeBERTa embeddings)"
    metrics["n_classes"] = n_classes
    metrics["best_val_loss"] = history["best_val_loss"]
    metrics["stopped_epoch"] = history["stopped_epoch"]
    metrics["config"] = serializable_config
    metrics["residual_weighting"] = {
        "summary": targets.summary(),
        "method": args.weight_method,
        "w_max": args.weight_max,
        "floor": args.weight_floor,
        "oof_splits": args.oof_splits,
        "loss": vars(loss_cfg),
    }

    # 7. Complementarity report -------------------------------------------
    print("\n" + "=" * 64 + "\nCOMPLEMENTARITY\n" + "=" * 64)
    complementarity: dict[str, dict[str, float]] = {}
    # Train: leakage-safe OOF XGBoost probs.
    comp_train = complementarity_metrics(probs["train"], targets.oof_probs, y_train)
    complementarity["train"] = comp_train
    print("[train] (XGBoost = out-of-fold)")
    print(format_complementarity(comp_train))
    # Val/test: real XGBoost probs when supplied.
    if args.xgb_probs_dir:
        xgb_dir = Path(args.xgb_probs_dir)
        for split, y in (("val", y_val), ("test", y_test)):
            xp = xgb_dir / f"{split}_probs.npy"
            if not xp.exists():
                print(f"[WARN] {xp} missing; skipping {split} complementarity.")
                continue
            xgb_probs = np.load(xp)
            if len(xgb_probs) != len(y):
                print(f"[WARN] {split}: XGBoost probs ({len(xgb_probs)}) misaligned with labels ({len(y)}); skipping.")
                continue
            comp = complementarity_metrics(probs[split], xgb_probs, y)
            complementarity[split] = comp
            print(f"\n[{split}]")
            print(format_complementarity(comp))
    metrics["complementarity"] = complementarity

    # 8. Persist -----------------------------------------------------------
    # `out` and the best kan_model.pt already exist (checkpointed during
    # training); this final save records the loaded best model + full metrics.
    torch.save({"model_state_dict": _compact_state(model.state_dict()), "config": serializable_config}, out / "kan_model.pt")
    for split, array in probs.items():
        np.save(out / f"{split}_probs.npy", array)
    for split, array in confidence.items():
        np.save(out / f"{split}_confidence.npy", array)
    np.save(out / "train_residual_weights.npy", targets.weights)
    np.save(out / "train_xgb_oof_probs.npy", targets.oof_probs)
    plot_training_curves(history, out / "training_curves.png")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "complementarity.json").write_text(json.dumps(complementarity, indent=2), encoding="utf-8")
    print(f"\n[DONE] Residual-aware KAN artifacts saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
