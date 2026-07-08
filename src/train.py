"""Hydra-driven entry point for every model in this project.

The ``model`` config group selects what to train: the default
``kan_xgboost`` ensemble (KAN -> XGBoost -> fusion stages) or one of the
baselines (``svm``, ``knn``, ``rf``, ``bilstm_cnn``). All dataset paths,
hyperparameters, training settings, seeds, and output locations live in
``conf/``. This script only wires resolved config values into the existing
pipelines -- the model code itself is unchanged.

Usage::

    # Full ensemble pipeline (KAN -> XGBoost -> fusion) with defaults
    python src/train.py

    # Run a subset of ensemble stages
    python src/train.py train.stages=xgboost

    # Train a baseline instead
    python src/train.py model=svm
    python src/train.py model=rf model.max_features=30000

    # CLI overrides
    python src/train.py model.xgboost.max_depth=3 model.fusion.kan_weight=0.7

    # Multirun sweeps (over values, or over whole models)
    python src/train.py -m model.xgboost.max_depth=3,5,7 model.xgboost.learning_rate=0.01,0.05
    python src/train.py -m model=svm,knn,rf

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
from omegaconf import DictConfig, OmegaConf

VALID_STAGES = ("kan", "xgboost", "fusion", "static_fusion")


def to_project_path(path: object) -> str:
    """Resolve a (possibly relative) config path against the project root.

    Unlike Hydra's ``to_absolute_path``, which anchors relative paths to the
    directory the command was launched from, this anchors them to the
    repository root derived from ``__file__``. That makes every path resolve to
    the same location no matter where the job is launched -- essential on cloud
    / HPC runners that start the process from an arbitrary working directory.
    Absolute paths are returned unchanged.
    """
    return str((PROJECT_ROOT / str(path)).resolve())


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
        embeddings_dir=to_project_path(cfg.data.embeddings_dir),
        output_dir=to_project_path(cfg.outputs.kan_dir),
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
        # Spline interpretability (src/analysis). .get() defaults keep older
        # custom train/*.yaml files without these keys working unchanged.
        enable_spline_visualization=bool(train_cfg.get("enable_spline_visualization", True)),
        spline_viz_top_k=int(train_cfg.get("spline_viz_top_k", 8)),
        enable_tensorboard=bool(train_cfg.get("enable_tensorboard", False)),
        spline_viz_dir=to_project_path(
            str(cfg.outputs.get("spline_viz_dir", "outputs/kan_spline_visualizations"))
        ),
        tensorboard_dir=to_project_path(
            str(cfg.outputs.get("kan_tensorboard_dir", "outputs/tensorboard/kan"))
        ),
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
        train_path=to_project_path(cfg.data.splits.train),
        val_path=to_project_path(cfg.data.splits.val),
        test_path=to_project_path(cfg.data.splits.test),
        output_dir=to_project_path(cfg.outputs.xgboost_dir),
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
        inputs_dir=to_project_path(cfg.data.fusion.inputs_dir),
        kan_dir=to_project_path(cfg.data.fusion.kan_dir),
        output_dir=to_project_path(cfg.outputs.fusion_dir),
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
    inputs_dir = Path(to_project_path(cfg.data.fusion.inputs_dir))
    kan_dir = Path(to_project_path(cfg.data.fusion.kan_dir))

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

    output_dir = Path(to_project_path(cfg.outputs.fusion_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"static_fusion_w{weight:g}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[INFO] Static fusion metrics saved to: {out_path}")


def run_svm(cfg: DictConfig) -> None:
    """Train and evaluate the TF-IDF + SVM baseline."""
    from src.baselines.svm.train import main as svm_main

    model_cfg = cfg.model
    svm_main(argparse.Namespace(
        train_path=to_project_path(cfg.data.splits.train),
        val_path=to_project_path(cfg.data.splits.val),
        test_path=to_project_path(cfg.data.splits.test),
        max_features=None if model_cfg.max_features is None else int(model_cfg.max_features),
        # The SVM pipeline parses this as a comma-separated string.
        c_values=",".join(str(float(c)) for c in model_cfg.c_values),
        class_weights=tuple(OmegaConf.to_container(model_cfg.class_weights, resolve=True)),
        scoring=str(model_cfg.scoring),
        random_state=int(cfg.seed),
        plots_dir=str(Path(to_project_path(cfg.outputs.svm_dir)) / "plots"),
    ))


def run_knn(cfg: DictConfig) -> None:
    """Train and evaluate the TF-IDF + KNN baseline."""
    from src.baselines.knn.train import main as knn_main

    model_cfg = cfg.model
    knn_main(argparse.Namespace(
        train_path=to_project_path(cfg.data.splits.train),
        val_path=to_project_path(cfg.data.splits.val),
        test_path=to_project_path(cfg.data.splits.test),
        max_features=None if model_cfg.max_features is None else int(model_cfg.max_features),
        # The KNN pipeline parses this as a comma-separated string.
        k_values=",".join(str(int(k)) for k in model_cfg.k_values),
        weights=tuple(str(w) for w in model_cfg.weights),
        scoring=str(model_cfg.scoring),
        plots_dir=str(Path(to_project_path(cfg.outputs.knn_dir)) / "plots"),
    ))


def run_rf(cfg: DictConfig) -> None:
    """Train, evaluate, and save the TF-IDF + Random Forest baseline."""
    from src.baselines.rf.train import main as rf_main

    model_cfg = cfg.model
    rf_main(argparse.Namespace(
        train_path=to_project_path(cfg.data.splits.train),
        val_path=to_project_path(cfg.data.splits.val),
        test_path=to_project_path(cfg.data.splits.test),
        max_features=None if model_cfg.max_features is None else int(model_cfg.max_features),
        save_dir=to_project_path(cfg.outputs.rf_dir),
        seed=int(cfg.seed),
    ))


def run_bilstm_cnn(cfg: DictConfig) -> None:
    """Train, evaluate, and save the BiLSTM-CNN baseline."""
    from src.baselines.bilstm_cnn.train import main as bilstm_cnn_main

    model_cfg = cfg.model
    bilstm_cnn_main(argparse.Namespace(
        train_path=to_project_path(cfg.data.splits.train),
        val_path=to_project_path(cfg.data.splits.val),
        test_path=to_project_path(cfg.data.splits.test),
        vocab_size=int(model_cfg.vocab_size),
        max_length=int(model_cfg.max_length),
        epochs=int(cfg.train.bilstm_cnn.epochs),
        save_dir=to_project_path(cfg.outputs.bilstm_cnn_dir),
        seed=int(cfg.seed),
    ))


BASELINE_RUNNERS = {
    "svm": run_svm,
    "knn": run_knn,
    "rf": run_rf,
    "bilstm_cnn": run_bilstm_cnn,
}


def run_ensemble(cfg: DictConfig) -> None:
    """Run the KAN -> XGBoost -> fusion stages selected in train.stages."""
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 64)
    print("RESOLVED CONFIGURATION")
    print("=" * 64)
    print(OmegaConf.to_yaml(cfg))

    model_name = str(cfg.model.get("name", "kan_xgboost"))
    if model_name == "kan_xgboost":
        run_ensemble(cfg)
    elif model_name in BASELINE_RUNNERS:
        print(f"\n>>> BASELINE: {model_name}")
        BASELINE_RUNNERS[model_name](cfg)
        print(f"\n[DONE] Baseline '{model_name}' finished.")
    else:
        raise ValueError(
            f"Unknown model.name '{model_name}'. Expected 'kan_xgboost' or "
            f"one of: {sorted(BASELINE_RUNNERS)}"
        )


if __name__ == "__main__":
    main()
