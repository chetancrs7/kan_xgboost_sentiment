"""Hydra-driven entry point for the KAN + XGBoost + fusion sentiment pipeline.

All dataset paths, hyperparameters, training settings, seeds, and output
locations live in ``conf/`` (see ``conf/config.yaml`` and the ``data``,
``model``, ``train`` config groups). This script only wires resolved config
values into the existing expert pipelines -- the model code itself is
unchanged.

Usage::

    # Full pipeline (KAN -> XGBoost -> fusion) with defaults
    python src/train.py

    # Run a subset of stages
    python src/train.py train.stages=[xgboost]

    # CLI overrides
    python src/train.py model.xgboost.max_depth=3 model.fusion.kan_weight=0.7

    # Multirun sweeps
    python src/train.py -m model.xgboost.max_depth=3,5,7 model.xgboost.learning_rate=0.01,0.05

    # Show the composed config without training
    python src/train.py --cfg job
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable regardless of how this script is launched
# (``python src/train.py`` puts ``src/`` -- not the repo root -- on sys.path).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

VALID_STAGES = ("kan", "xgboost", "fusion", "static_fusion")


def _normalize_stages(raw: object) -> list[str]:
    """Accept every reasonable way of writing the stages override.

    Supported forms::

        train.stages=[kan,xgboost]        # Hydra list syntax
        train.stages=xgboost              # bare single stage
        "train.stages='kan,fusion'"       # comma-separated string (quoted so
                                          # Hydra doesn't parse it as a sweep)
    """
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    return [str(s).strip().lower() for s in items if str(s).strip()]


def run_kan(cfg: DictConfig) -> None:
    """Train the KAN expert with hyperparameters taken from the config."""
    from src.models.kan.train import main as kan_main

    model_cfg, train_cfg = cfg.model.kan, cfg.train.kan
    kan_main(argparse.Namespace(
        embeddings_dir=to_absolute_path(cfg.data.embeddings_dir),
        output_dir=to_absolute_path(cfg.outputs.kan_dir),
        # The KAN pipeline parses these two as comma-separated strings.
        hidden_dims=",".join(str(int(d)) for d in model_cfg.hidden_dims),
        grid_range=",".join(str(float(v)) for v in model_cfg.grid_range),
        grid_size=int(model_cfg.grid_size),
        spline_order=int(model_cfg.spline_order),
        epochs=int(train_cfg.epochs),
        batch_size=int(train_cfg.batch_size),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
        patience=int(train_cfg.patience),
        grid_update_freq=int(train_cfg.grid_update_freq),
        seed=int(cfg.seed),
        device=cfg.device,
    ))


def run_xgboost(cfg: DictConfig) -> None:
    """Train the XGBoost expert; fixed params or Optuna search per config."""
    from src.models.xgboost.train import main as xgb_main

    model_cfg = OmegaConf.to_container(cfg.model.xgboost, resolve=True)
    use_optuna = bool(model_cfg.pop("use_optuna"))
    n_trials = int(model_cfg.pop("n_trials"))
    if use_optuna:
        print(
            "[INFO] model.xgboost.use_optuna=true: Optuna will search "
            "hyperparameters; fixed values in the config are ignored."
        )
    xgb_main(argparse.Namespace(
        train_path=to_absolute_path(cfg.data.splits.train),
        val_path=to_absolute_path(cfg.data.splits.val),
        test_path=to_absolute_path(cfg.data.splits.test),
        output_dir=to_absolute_path(cfg.outputs.xgboost_dir),
        n_trials=n_trials,
        seed=int(cfg.seed),
        use_optuna=use_optuna,
        fixed_params=model_cfg,
    ))


def run_fusion(cfg: DictConfig) -> None:
    """Train the residual-aware adaptive fusion gate."""
    from src.fusion.trainer import main as fusion_main

    model_cfg, train_cfg = cfg.model.fusion, cfg.train.fusion
    fusion_main(argparse.Namespace(
        inputs_dir=to_absolute_path(cfg.data.fusion.inputs_dir),
        kan_dir=to_absolute_path(cfg.data.fusion.kan_dir),
        output_dir=to_absolute_path(cfg.outputs.fusion_dir),
        residual_bins=int(model_cfg.residual_bins),
        hidden_dim=int(model_cfg.hidden_dim),
        dropout=float(model_cfg.dropout),
        gate_holdout=float(train_cfg.gate_holdout),
        epochs=int(train_cfg.epochs),
        batch_size=int(train_cfg.batch_size),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
        patience=int(train_cfg.patience),
        scheduler_patience=int(train_cfg.scheduler_patience),
        scheduler_factor=float(train_cfg.scheduler_factor),
        min_lr=float(train_cfg.min_lr),
        seed=int(cfg.seed),
        device=cfg.device,
    ))


def run_static_fusion(cfg: DictConfig) -> None:
    """Evaluate a fixed-weight blend of the experts' saved probabilities.

    ``fused = kan_weight * P_kan + (1 - kan_weight) * P_xgb`` on the fusion
    val/test splits. Evaluation only -- the learned gate is untouched.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    weight = float(cfg.model.fusion.kan_weight)
    inputs_dir = Path(to_absolute_path(cfg.data.fusion.inputs_dir))
    kan_dir = Path(to_absolute_path(cfg.data.fusion.kan_dir))

    print("\n" + "=" * 64)
    print(f"STATIC FUSION EVALUATION (kan_weight={weight:g})")
    print("=" * 64)
    results: dict[str, object] = {"kan_weight": weight}
    for split in ("val", "test"):
        kan_probs = np.load(kan_dir / f"{split}_probs.npy")
        xgb_probs = np.load(inputs_dir / "xgboost" / f"{split}_probs.npy")
        labels = pd.read_csv(inputs_dir / "splits" / f"{split}.csv")["label"].to_numpy()
        preds = (weight * kan_probs + (1.0 - weight) * xgb_probs).argmax(axis=1)
        results[split] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }
        print(
            f"  {split:<5} accuracy={results[split]['accuracy']:.4f} "
            f"f1_macro={results[split]['f1_macro']:.4f}"
        )

    output_dir = Path(to_absolute_path(cfg.outputs.fusion_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"static_fusion_w{weight:g}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[INFO] Static fusion metrics saved to: {out_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 64)
    print("RESOLVED CONFIGURATION")
    print("=" * 64)
    print(OmegaConf.to_yaml(cfg))

    stages = _normalize_stages(cfg.train.stages)
    unknown = sorted(set(stages) - set(VALID_STAGES))
    if unknown:
        raise ValueError(
            f"Unknown stage(s) {unknown}. Valid stages: {list(VALID_STAGES)}"
        )
    if "static_fusion" in stages and cfg.model.fusion.kan_weight is None:
        raise ValueError(
            "The static_fusion stage requires model.fusion.kan_weight to be "
            "set, e.g. model.fusion.kan_weight=0.7"
        )

    if "kan" in stages:
        print("\n>>> STAGE: KAN expert")
        run_kan(cfg)
    if "xgboost" in stages:
        print("\n>>> STAGE: XGBoost expert")
        run_xgboost(cfg)
    if "fusion" in stages:
        print("\n>>> STAGE: adaptive fusion gate")
        run_fusion(cfg)
    if cfg.model.fusion.kan_weight is not None and (
        "fusion" in stages or "static_fusion" in stages
    ):
        run_static_fusion(cfg)

    print("\n[DONE] Pipeline finished for stages:", ", ".join(stages) or "(none)")


if __name__ == "__main__":
    main()
