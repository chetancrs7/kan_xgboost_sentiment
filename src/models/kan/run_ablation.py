"""Mixed-spline KAN ablation study.

Trains the same 4-stage KAN architecture (identical widths, optimiser,
schedule, seed, and linear output head) four times, varying ONLY the spline
family assigned to each stage, so metric differences are attributable to the
spline design rather than capacity or tuning:

    all_bspline        bspline | bspline       | bspline    | bspline
    bspline_cubic      bspline | cubic_bspline | bspline    | bspline
    bspline_cubic_rbf  bspline | cubic_bspline | rbf        | rbf
    full_mixed         bspline | cubic_bspline | thin_plate | rbf

(stages: semantic_compression | context_interaction | sentiment_distortion |
emotion_hotspot; the cubic layer uses a finer grid -- 8 intervals vs 5 -- to
give the "subtle contextual shift" stage extra resolution.)

Each experiment reuses the existing training pipeline end to end
(``src/models/kan/train.py``: dataset loading, early stopping, checkpoints,
probability export) and writes its own artifact directory:

    models/kan_ablation/<experiment>/
        kan_model.pt, metrics.json, history.json, training_curves.png,
        confusion_matrices.png, {split}_probs.npy, {split}_confidence.npy

followed by a cross-experiment comparison:

    models/kan_ablation/ablation_summary.json
    models/kan_ablation/ablation_summary.md

Usage:
    python src/models/kan/run_ablation.py
    python src/models/kan/run_ablation.py --experiments full_mixed
    python src/models/kan/run_ablation.py --epochs 100 --patience 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.kan import train as kan_train  # noqa: E402
from src.models.kan.model import DEFAULT_SPLINE_TYPES  # noqa: E402

# The seeded 6k/2k/2k fusion subsample -- the canonical dataset for KAN
# experiments (data/embeddings/deberta_v3_base holds only a stale 160-sample
# smoke-test set).
DEFAULT_EMBEDDINGS_DIR = "data/fusion_inputs/embeddings"
DEFAULT_OUTPUT_ROOT = "models/kan_ablation"

# Constant across all experiments: 4 KAN stages + linear head, so the only
# varying factor is the spline family per stage.
ABLATION_HIDDEN_DIMS = "256,128,64,32"
ABLATION_OUTPUT_TYPE = "linear"

# Finer knot grid for the cubic context-interaction stage (see module docstring).
_CUBIC_PARAMS = [{}, {"grid_size": 8}, {}, {}]

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "all_bspline": {
        "description": "Baseline: B-splines at every stage.",
        "spline_types": ",".join(DEFAULT_SPLINE_TYPES),
        "spline_params": None,
    },
    "bspline_cubic": {
        "description": "Cubic B-spline (finer grid) for context interaction.",
        "spline_types": "bspline,cubic_bspline,bspline,bspline",
        "spline_params": _CUBIC_PARAMS,
    },
    "bspline_cubic_rbf": {
        "description": "Adds Gaussian RBFs for distortion + emotion hotspots.",
        "spline_types": "bspline,cubic_bspline,rbf,rbf",
        "spline_params": _CUBIC_PARAMS,
    },
    "full_mixed": {
        "description": "Full design: thin-plate distortion, RBF hotspots.",
        "spline_types": "bspline,cubic_bspline,thin_plate,rbf",
        "spline_params": _CUBIC_PARAMS,
    },
}

# Test-split metrics surfaced in the comparison table.
SUMMARY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "weighted_f1",
    "auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the mixed-spline KAN ablation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Each experiment writes to <output-root>/<experiment>/.",
    )
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS),
        help="Comma-separated subset of experiments to run.",
    )
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--grid-range", default="-2,2")
    parser.add_argument("--epochs", type=int, default=kan_train.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=kan_train.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=kan_train.DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=kan_train.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=kan_train.DEFAULT_PATIENCE)
    parser.add_argument("--grid-update-freq", type=int, default=kan_train.DEFAULT_GRID_UPDATE_FREQ)
    parser.add_argument("--seed", type=int, default=kan_train.RANDOM_STATE)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--enable-spline-visualization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Per-experiment interpretability figures (B-spline stages only; "
        "off by default to keep CPU runs fast).",
    )
    return parser.parse_args()


def build_train_args(name: str, spec: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Assemble the full Namespace ``kan_train.main`` expects for one experiment."""
    output_dir = str(Path(args.output_root) / name)
    return argparse.Namespace(
        embeddings_dir=args.embeddings_dir,
        output_dir=output_dir,
        hidden_dims=ABLATION_HIDDEN_DIMS,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        grid_range=args.grid_range,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grid_update_freq=args.grid_update_freq,
        seed=args.seed,
        device=args.device,
        spline_types=spec["spline_types"],
        spline_params=spec["spline_params"],
        output_type=ABLATION_OUTPUT_TYPE,
        enable_spline_visualization=args.enable_spline_visualization,
        spline_viz_dir=str(Path(output_dir) / "spline_visualizations"),
        spline_viz_top_k=8,
        enable_tensorboard=False,
        tensorboard_dir=str(Path(output_dir) / "tensorboard"),
    )


def summarise(results: dict[str, dict[str, Any]], output_root: Path) -> None:
    """Write the cross-experiment JSON + Markdown comparison tables."""
    summary: dict[str, Any] = {}
    for name, metrics in results.items():
        test = metrics.get("test", {})
        summary[name] = {
            "description": EXPERIMENTS[name]["description"],
            "spline_types": EXPERIMENTS[name]["spline_types"],
            "test": {k: test.get(k) for k in SUMMARY_METRICS},
            "test_confusion_matrix": test.get("confusion_matrix"),
            "best_val_loss": metrics.get("best_val_loss"),
            "stopped_epoch": metrics.get("stopped_epoch"),
            "artifacts_dir": str(output_root / name),
        }
    (output_root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    header = ["experiment", "spline types (L1..L4)", *SUMMARY_METRICS, "best_val_loss", "epochs"]
    lines = [
        "# Mixed-spline KAN ablation (test split)",
        "",
        "All experiments share widths 768->256->128->64->32->classes, a linear "
        "output head, and identical training settings; only the per-layer "
        "spline families differ.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for name, entry in summary.items():
        cells = [name, entry["spline_types"].replace(",", " / ")]
        for metric in SUMMARY_METRICS:
            value = entry["test"].get(metric)
            cells.append(f"{value:.4f}" if isinstance(value, float) else "n/a")
        best_val = entry["best_val_loss"]
        cells.append(f"{best_val:.4f}" if isinstance(best_val, float) else "n/a")
        cells.append(str(entry["stopped_epoch"]))
        lines.append("| " + " | ".join(cells) + " |")
    (output_root / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[INFO] Ablation summary written to: {(output_root / 'ablation_summary.md').resolve()}")
    print("\n".join(lines[4:]))


def main() -> None:
    args = parse_args()
    requested = [s.strip() for s in args.experiments.split(",") if s.strip()]
    unknown = [name for name in requested if name not in EXPERIMENTS]
    if unknown:
        sys.exit(f"[ERROR] Unknown experiment(s) {unknown}. Available: {', '.join(EXPERIMENTS)}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(requested, start=1):
        spec = EXPERIMENTS[name]
        print("\n" + "=" * 72)
        print(f"ABLATION {idx}/{len(requested)}: {name} — {spec['description']}")
        print(f"  spline types: {spec['spline_types']}")
        print("=" * 72)
        kan_train.main(build_train_args(name, spec, args))
        metrics_path = output_root / name / "metrics.json"
        results[name] = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Include previously completed experiments in the summary when only a
    # subset was re-run, so the comparison table stays complete.
    for name in EXPERIMENTS:
        metrics_path = output_root / name / "metrics.json"
        if name not in results and metrics_path.exists():
            results[name] = json.loads(metrics_path.read_text(encoding="utf-8"))

    summarise(results, output_root)
    print("\n[DONE] Mixed-spline ablation complete.")


if __name__ == "__main__":
    main()
