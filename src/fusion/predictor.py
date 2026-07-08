"""Inference and artifact export for the trained fusion gate.

:class:`FusionPredictor` reconstructs the full inference stack (gating network,
feature scaler, and historical-residual lookup tables) from the single
self-contained ``fusion_model.pt`` checkpoint, so downstream consumers never
need the training-time objects.

``evaluate_and_export`` produces the deliverables:

* ``fusion_predictions.csv``    per-sample expert/fused predictions + alpha
* ``fusion_probabilities.npy``  fused test probabilities ``[n, C]``
* ``fusion_metrics.json``       fusion and per-expert metrics (holdout + test)
* ``plots/``                    confusion matrix, calibration curve,
                                reliability diagram, gate distribution

Standalone usage (re-run inference from a checkpoint)::

    python -m src.fusion.predictor --checkpoint models/fusion/fusion_model.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fusion.fusion_dataset import (
    FusionFeatureBuilder,
    FusionSplit,
    ResidualEstimator,
    StandardScaler,
    load_fusion_inputs,
)
from src.fusion.fusion_model import FusionGate
from src.fusion.metrics import (
    compute_classification_metrics,
    plot_alpha_distribution,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_reliability_diagram,
)
from src.fusion.utils import get_logger, resolve_device, save_json

logger = get_logger("fusion.predictor")

# Repo root (src/fusion/ -> parents[2]) so CLI defaults resolve relative to the
# project regardless of the launch directory (cloud/HPC-safe).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FusionOutputs:
    """Per-sample outputs of the fusion gate on one split."""

    alpha: np.ndarray  # [n] trust in the KAN expert
    fused_probs: np.ndarray  # [n, C]

    @property
    def preds(self) -> np.ndarray:
        return self.fused_probs.argmax(axis=1)


class FusionPredictor:
    """Load a fusion checkpoint and run gated inference on fusion splits."""

    def __init__(
        self,
        model: FusionGate,
        scaler: StandardScaler,
        feature_names: list[str],
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.scaler = scaler
        self.feature_names = feature_names
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path | str, device: torch.device | str | None = None
    ) -> "FusionPredictor":
        """Reconstruct the predictor from a self-contained checkpoint file."""
        device = resolve_device(str(device) if device is not None else None)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = FusionGate(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = StandardScaler.from_state_dict(checkpoint["scaler"])
        predictor = cls(model, scaler, checkpoint["feature_names"], device)
        predictor._checkpoint = checkpoint  # kept for builder reconstruction
        return predictor

    def builder_from_checkpoint(self) -> FusionFeatureBuilder:
        """Rebuild the feature builder with the checkpointed residual tables.

        Lets a standalone inference run recreate the exact training-time
        feature pipeline without access to the experts' training split.
        """
        checkpoint = self._checkpoint
        builder = FusionFeatureBuilder(
            n_residual_bins=int(checkpoint["residuals"]["kan"]["n_bins"]),
            text_column=checkpoint["text_column"],
        )
        builder.kan_residuals = ResidualEstimator.from_state_dict(
            checkpoint["residuals"]["kan"]
        )
        builder.xgb_residuals = ResidualEstimator.from_state_dict(
            checkpoint["residuals"]["xgb"]
        )
        builder._residuals_fitted = True
        return builder

    @torch.no_grad()
    def predict(self, split: FusionSplit) -> FusionOutputs:
        """Run the gate over a fusion split and return alpha + fused probs."""
        if split.feature_names != self.feature_names:
            raise ValueError(
                f"{split.name}: feature schema does not match the checkpoint. "
                f"Expected {self.feature_names}, got {split.feature_names}."
            )
        features = torch.as_tensor(
            self.scaler.transform(split.features), dtype=torch.float32
        ).to(self.device)
        kan = torch.as_tensor(split.kan_probs, dtype=torch.float32).to(self.device)
        xgb = torch.as_tensor(split.xgb_probs, dtype=torch.float32).to(self.device)

        alpha = self.model(features)
        fused = FusionGate.fuse(alpha, kan, xgb)
        return FusionOutputs(
            alpha=alpha.squeeze(1).cpu().numpy(),
            fused_probs=fused.cpu().numpy(),
        )


# --------------------------------------------------------------------------- #
# Evaluation + export
# --------------------------------------------------------------------------- #
def split_report(split: FusionSplit, outputs: FusionOutputs) -> dict[str, object]:
    """Fusion metrics next to each frozen expert's metrics on the same rows."""
    report: dict[str, object] = {}
    report.update(compute_classification_metrics(split.labels, outputs.fused_probs, "fusion_"))
    report.update(compute_classification_metrics(split.labels, split.kan_probs, "kan_"))
    report.update(compute_classification_metrics(split.labels, split.xgb_probs, "xgb_"))
    report["gate"] = {
        "alpha_mean": float(outputs.alpha.mean()),
        "alpha_std": float(outputs.alpha.std()),
        "alpha_min": float(outputs.alpha.min()),
        "alpha_max": float(outputs.alpha.max()),
        "fraction_trusting_kan": float((outputs.alpha > 0.5).mean()),
    }
    report["n_samples"] = len(split)
    return report


def export_predictions(
    split: FusionSplit, outputs: FusionOutputs, csv_path: Path
) -> None:
    """Write the per-sample prediction table for error analysis."""
    n_classes = outputs.fused_probs.shape[1]
    table: dict[str, object] = {}
    if split.source_rows is not None:
        table["source_row"] = split.source_rows
    if split.texts is not None:
        table["text"] = split.texts
    table.update(
        {
            "label": split.labels,
            "alpha_kan_trust": outputs.alpha,
            "kan_pred": split.kan_probs.argmax(axis=1),
            "xgb_pred": split.xgb_probs.argmax(axis=1),
            "fusion_pred": outputs.preds,
            "fusion_correct": (outputs.preds == split.labels).astype(int),
        }
    )
    for c in range(n_classes):
        table[f"kan_prob_{c}"] = split.kan_probs[:, c]
        table[f"xgb_prob_{c}"] = split.xgb_probs[:, c]
        table[f"fusion_prob_{c}"] = outputs.fused_probs[:, c]
    pd.DataFrame(table).to_csv(csv_path, index=False)


def evaluate_and_export(
    predictor: FusionPredictor,
    test_split: FusionSplit,
    output_dir: Path | str,
    gate_holdout_split: FusionSplit | None = None,
) -> dict[str, object]:
    """Produce every required evaluation artifact for the fusion model."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_outputs = predictor.predict(test_split)
    metrics: dict[str, object] = {"test": split_report(test_split, test_outputs)}
    if gate_holdout_split is not None:
        holdout_outputs = predictor.predict(gate_holdout_split)
        metrics["gate_holdout"] = split_report(gate_holdout_split, holdout_outputs)

    # Required artifacts -----------------------------------------------------
    np.save(output_dir / "fusion_probabilities.npy", test_outputs.fused_probs)
    export_predictions(test_split, test_outputs, output_dir / "fusion_predictions.csv")
    save_json(metrics, output_dir / "fusion_metrics.json")

    # Figures ------------------------------------------------------------------
    plot_confusion_matrix(
        test_split.labels, test_outputs.preds, plots_dir / "confusion_matrix.png"
    )
    plot_calibration_curve(
        test_split.labels,
        {
            "KAN": test_split.kan_probs[:, 1],
            "XGBoost": test_split.xgb_probs[:, 1],
            "Fusion": test_outputs.fused_probs[:, 1],
        },
        plots_dir / "calibration_curve.png",
    )
    plot_reliability_diagram(
        test_split.labels, test_outputs.fused_probs, plots_dir / "reliability_diagram.png"
    )
    plot_alpha_distribution(test_outputs.alpha, plots_dir / "alpha_distribution.png")

    test_report = metrics["test"]
    logger.info(
        "TEST | fusion acc=%.4f macro_f1=%.4f auc=%.4f | "
        "kan acc=%.4f | xgb acc=%.4f | mean alpha=%.3f",
        test_report["fusion_accuracy"], test_report["fusion_macro_f1"],
        test_report["fusion_roc_auc"], test_report["kan_accuracy"],
        test_report["xgb_accuracy"], test_report["gate"]["alpha_mean"],
    )
    logger.info("Artifacts exported to %s", output_dir.resolve())
    return metrics


# --------------------------------------------------------------------------- #
# Standalone CLI (inference from a saved checkpoint)
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fusion inference from a saved checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "models/fusion/fusion_model.pt"))
    parser.add_argument("--inputs-dir", default=str(PROJECT_ROOT / "data/fusion_inputs"))
    parser.add_argument("--kan-dir", default=str(PROJECT_ROOT / "models/kan_subset"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models/fusion"))
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    predictor = FusionPredictor.from_checkpoint(args.checkpoint, device=args.device)
    builder = predictor.builder_from_checkpoint()

    inputs = load_fusion_inputs(args.inputs_dir, args.kan_dir, splits=(args.split,))
    data = inputs[args.split]
    split = builder.build(
        args.split, data["df"], data["embeddings"], data["kan_probs"], data["xgb_probs"]
    )
    evaluate_and_export(predictor, split, args.output_dir)


if __name__ == "__main__":
    main()
