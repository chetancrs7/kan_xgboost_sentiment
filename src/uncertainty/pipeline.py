"""End-to-end uncertainty-aware inference on top of the trained fusion gate.

Sits strictly *after* the adaptive fusion gate: the gate produces the fused
prediction, this layer decides how much to trust it. For every test sample it
computes the four measure families (prediction uncertainty, expert
disagreement, embedding uncertainty, MC-dropout model uncertainty), combines
them into a single weighted-normalised score (normalisation ranges fitted on
the fusion validation split -- never on test), buckets the score into
low / medium / high tiers, and resolves the configured fallback mode.

Deliverables written to ``--output-dir``:

* ``confidence_scores.csv``          per-sample measures, score, tier, decision
* ``uncertainty_report.json``        config echo, tier/fallback summaries,
                                     per-measure diagnostics
* ``reliability_metrics.json``       ECE / MCE / Brier / NLL for fusion + experts
* ``calibration_plots.png``          calibration curve, reliability diagram,
                                     confidence histogram, risk-coverage curve
* ``uncertainty_distribution.png``   score histogram + correctness split

Usage::

    python -m src.uncertainty.pipeline
    python -m src.uncertainty.pipeline --fallback-mode 3 --high-threshold 0.7
    python -m src.uncertainty.pipeline --config configs/uncertainty.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.fusion.fusion_dataset import FusionSplit, load_fusion_inputs
from src.fusion.predictor import FusionOutputs, FusionPredictor
from src.fusion.utils import get_logger, save_json, set_seed
from src.uncertainty.config import FallbackMode, UncertaintyConfig
from src.uncertainty.fallback import NEUTRAL_LABEL, apply_fallback, fallback_summary
from src.uncertainty.measures import EmbeddingUncertaintyEstimator, compute_measures
from src.uncertainty.reliability import (
    compute_reliability_metrics,
    plot_calibration_plots,
    plot_uncertainty_distribution,
)
from src.uncertainty.scorer import TIER_HIGH, UncertaintyScorer

logger = get_logger("uncertainty.pipeline")

DEFAULT_OUTPUT_DIR = "models/fusion/uncertainty"


# --------------------------------------------------------------------------- #
# Core analysis (in-memory objects only; the CLI wires up the file layout)
# --------------------------------------------------------------------------- #
def run_uncertainty_analysis(
    predictor: FusionPredictor,
    val_split: FusionSplit,
    val_embeddings: np.ndarray,
    test_split: FusionSplit,
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    config: UncertaintyConfig,
    output_dir: Path | str,
) -> dict[str, object]:
    """Run the full uncertainty layer and export every deliverable.

    Args:
        predictor: Trained fusion gate wrapped in its inference stack.
        val_split: Fusion validation split -- the *reference* split on which
            the measure-normalisation ranges are fitted.
        val_embeddings: DeBERTa embeddings aligned with ``val_split`` rows.
        test_split: Fusion test split to be scored.
        test_embeddings: DeBERTa embeddings aligned with ``test_split`` rows.
        train_embeddings: Experts' training embeddings; the reference
            distribution for Mahalanobis / LOF / norm statistics.
        config: Uncertainty-layer configuration.
        output_dir: Destination directory for all artifacts.

    Returns:
        The ``uncertainty_report.json`` payload.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fused predictions (the layer operates downstream of the gate) -----
    val_outputs = predictor.predict(val_split)
    test_outputs = predictor.predict(test_split)

    # ---- Uncertainty measures ----------------------------------------------
    logger.info("Fitting embedding-uncertainty estimator on %d training embeddings",
                len(train_embeddings))
    estimator = EmbeddingUncertaintyEstimator(config.lof_neighbors).fit(train_embeddings)

    def measures_for(split: FusionSplit, outputs: FusionOutputs, embeddings: np.ndarray):
        return compute_measures(
            fused_probs=outputs.fused_probs,
            kan_probs=split.kan_probs,
            xgb_probs=split.xgb_probs,
            embeddings=embeddings,
            embedding_estimator=estimator,
            model=predictor.model,
            scaled_features=predictor.scaler.transform(split.features),
            mc_passes=config.mc_passes,
            mc_seed=config.mc_seed,
        )

    logger.info("Computing uncertainty measures (%d MC-dropout passes per split)",
                config.mc_passes)
    val_measures, _ = measures_for(val_split, val_outputs, val_embeddings)
    test_measures, test_aux = measures_for(test_split, test_outputs, test_embeddings)

    # ---- Combined score + tiers + fallback ---------------------------------
    scorer = UncertaintyScorer(config).fit(val_measures)
    score, normalized = scorer.score(test_measures)
    tiers = scorer.assign_tiers(score)
    decisions = apply_fallback(
        tiers, test_outputs.fused_probs, test_split.kan_probs,
        test_split.xgb_probs, config.fallback_mode,
    )

    labels = test_split.labels
    fused_correct = test_outputs.preds == labels

    # ---- Deliverable 1: confidence_scores.csv ------------------------------
    export_confidence_scores(
        output_dir / "confidence_scores.csv",
        test_split, test_outputs, test_measures, test_aux,
        score, tiers, decisions,
    )

    # ---- Deliverable 2: reliability_metrics.json ---------------------------
    reliability = reliability_report(test_split, test_outputs, tiers, decisions)
    save_json(reliability, output_dir / "reliability_metrics.json")

    # ---- Deliverable 3: uncertainty_report.json ----------------------------
    report = build_report(
        config, labels, fused_correct, test_measures, normalized,
        score, tiers, decisions,
    )
    save_json(report, output_dir / "uncertainty_report.json")

    # ---- Deliverables 4+5: figures -----------------------------------------
    plot_calibration_plots(
        labels,
        {
            "KAN": test_split.kan_probs[:, 1],
            "XGBoost": test_split.xgb_probs[:, 1],
            "Fusion": test_outputs.fused_probs[:, 1],
        },
        test_outputs.fused_probs,
        score,
        output_dir / "calibration_plots.png",
    )
    plot_uncertainty_distribution(
        score, fused_correct,
        config.low_threshold, config.high_threshold, tiers,
        output_dir / "uncertainty_distribution.png",
    )

    tier_counts = report["tiers"]["counts"]
    logger.info(
        "TEST | tiers low/medium/high = %d/%d/%d | fused acc=%.4f | "
        "answered acc=%.4f at %.1f%% coverage | error-detection AUROC=%.4f",
        tier_counts["low"], tier_counts["medium"], tier_counts["high"],
        fused_correct.mean(),
        report["fallback"]["accuracy_answered"] or float("nan"),
        100 * report["fallback"]["automation_coverage"],
        report["score"]["error_detection_auroc"] or float("nan"),
    )
    logger.info("Uncertainty artifacts exported to %s", output_dir.resolve())
    return report


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def export_confidence_scores(
    csv_path: Path,
    split: FusionSplit,
    outputs: FusionOutputs,
    measures: dict[str, np.ndarray],
    aux: dict[str, np.ndarray],
    score: np.ndarray,
    tiers: np.ndarray,
    decisions: dict[str, np.ndarray],
) -> None:
    """Write the per-sample uncertainty table (``confidence_scores.csv``)."""
    table: dict[str, object] = {}
    if split.source_rows is not None:
        table["source_row"] = split.source_rows
    if split.texts is not None:
        table["text"] = split.texts
    final = decisions["final_pred"]
    answered = final != NEUTRAL_LABEL
    table.update(
        {
            "label": split.labels,
            "kan_pred": split.kan_probs.argmax(axis=1),
            "xgb_pred": split.xgb_probs.argmax(axis=1),
            "fused_pred": outputs.preds,
            "alpha_kan_trust": outputs.alpha,
            "fused_confidence": aux["msp"],
            "confidence_margin": aux["margin"],
            **measures,
            "embedding_norm": aux["embedding_norm"],
            "mc_alpha_mean": aux["mc_alpha_mean"],
            "uncertainty_score": score,
            "uncertainty_tier": tiers,
            **decisions,
            "fused_correct": (outputs.preds == split.labels).astype(int),
            # Neutral abstentions have no correctness; left empty in the CSV.
            "final_correct": np.where(answered, (final == split.labels).astype(float), np.nan),
        }
    )
    pd.DataFrame(table).to_csv(csv_path, index=False)


def reliability_report(
    split: FusionSplit,
    outputs: FusionOutputs,
    tiers: np.ndarray,
    decisions: dict[str, np.ndarray],
) -> dict[str, object]:
    """ECE / MCE / Brier / NLL for fusion + both experts (+ accepted subset)."""
    labels = split.labels
    report: dict[str, object] = {
        "fusion": compute_reliability_metrics(labels, outputs.fused_probs),
        "kan": compute_reliability_metrics(labels, split.kan_probs),
        "xgb": compute_reliability_metrics(labels, split.xgb_probs),
    }
    accepted = tiers != TIER_HIGH
    if accepted.any():
        report["fusion_low_medium_only"] = {
            **compute_reliability_metrics(labels[accepted], outputs.fused_probs[accepted]),
            "coverage": float(accepted.mean()),
        }
    answered = (decisions["final_pred"] != NEUTRAL_LABEL) & ~decisions["needs_human_review"]
    report["automation_coverage"] = float(answered.mean())
    return report


def build_report(
    config: UncertaintyConfig,
    labels: np.ndarray,
    fused_correct: np.ndarray,
    measures: dict[str, np.ndarray],
    normalized: dict[str, np.ndarray],
    score: np.ndarray,
    tiers: np.ndarray,
    decisions: dict[str, np.ndarray],
) -> dict[str, object]:
    """Assemble the ``uncertainty_report.json`` payload."""
    errors = (~fused_correct).astype(np.float64)

    tier_counts, tier_accuracy = {}, {}
    for tier in ("low", "medium", "high"):
        mask = tiers == tier
        tier_counts[tier] = int(mask.sum())
        tier_accuracy[tier] = float(fused_correct[mask].mean()) if mask.any() else None

    measure_stats: dict[str, object] = {}
    for name in measures:
        raw = np.asarray(measures[name], dtype=np.float64)
        corr = None
        if raw.std() > 1e-12 and errors.std() > 1e-12:
            corr = float(np.corrcoef(raw, errors)[0, 1])
        measure_stats[name] = {
            "mean": float(raw.mean()),
            "std": float(raw.std()),
            "normalized_mean": float(np.asarray(normalized[name]).mean()),
            "weight": config.normalized_weights[name],
            "correlation_with_fused_error": corr,
        }

    auroc = None
    if 0 < errors.sum() < len(errors):
        auroc = float(roc_auc_score(errors, score))

    return {
        "config": config.to_dict(),
        "n_samples": int(len(labels)),
        "fused_accuracy": float(fused_correct.mean()),
        "tiers": {"counts": tier_counts, "fused_accuracy": tier_accuracy},
        "fallback": fallback_summary(decisions, labels),
        "score": {
            "mean": float(score.mean()),
            "std": float(score.std()),
            "quantiles": {
                q: float(np.quantile(score, float(q)))
                for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
            },
            "error_detection_auroc": auroc,
        },
        "measures": measure_stats,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uncertainty-aware inference on the trained fusion gate."
    )
    # Data / model locations
    parser.add_argument("--checkpoint", default="models/fusion/fusion_model.pt")
    parser.add_argument("--inputs-dir", default="data/fusion_inputs")
    parser.add_argument("--kan-dir", default="models/kan_subset")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    # Config: either a JSON file, or individual overrides of the defaults
    parser.add_argument("--config", default=None, help="JSON file with an UncertaintyConfig.")
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--high-threshold", type=float, default=None)
    parser.add_argument(
        "--fallback-mode", type=int, default=None, choices=(1, 2, 3, 4),
        help="1=fused low-confidence, 2=most confident expert, 3=Neutral, 4=human review.",
    )
    parser.add_argument("--mc-passes", type=int, default=None)
    parser.add_argument("--lof-neighbors", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> UncertaintyConfig:
    """Build the config from a JSON file and/or CLI overrides."""
    payload = UncertaintyConfig.from_json(args.config).to_dict() if args.config else {}
    overrides = {
        "low_threshold": args.low_threshold,
        "high_threshold": args.high_threshold,
        "fallback_mode": args.fallback_mode,
        "mc_passes": args.mc_passes,
        "lof_neighbors": args.lof_neighbors,
    }
    payload.update({k: v for k, v in overrides.items() if v is not None})
    if args.fallback_mode is not None:
        payload["fallback_mode"] = FallbackMode(args.fallback_mode)
    return UncertaintyConfig.from_dict(payload)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = config_from_args(args)
    logger.info("Uncertainty config: %s", config.to_dict())

    predictor = FusionPredictor.from_checkpoint(args.checkpoint, device=args.device)
    builder = predictor.builder_from_checkpoint()

    inputs = load_fusion_inputs(args.inputs_dir, args.kan_dir)
    splits: dict[str, FusionSplit] = {}
    for name in ("val", "test"):
        data = inputs[name]
        splits[name] = builder.build(
            name, data["df"], data["embeddings"], data["kan_probs"], data["xgb_probs"]
        )

    run_uncertainty_analysis(
        predictor,
        val_split=splits["val"],
        val_embeddings=inputs["val"]["embeddings"],
        test_split=splits["test"],
        test_embeddings=inputs["test"]["embeddings"],
        train_embeddings=inputs["train"]["embeddings"],
        config=config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
