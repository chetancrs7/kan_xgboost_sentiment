"""Synthetic end-to-end smoke test for the uncertainty package.

Exercises the whole uncertainty layer on fake data (no expert artifacts
needed): every measure family, normalisation + weighted scoring, tier
assignment, all four fallback modes, and the full artifact export of
``run_uncertainty_analysis``. Useful as a fast regression check::

    python scripts/smoke_test_uncertainty.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_test_fusion import make_fake_split
from src.fusion.fusion_dataset import FusionFeatureBuilder, StandardScaler
from src.fusion.fusion_model import FusionGate
from src.fusion.predictor import FusionPredictor
from src.fusion.utils import set_seed
from src.uncertainty.config import FallbackMode, UncertaintyConfig
from src.uncertainty.fallback import NEUTRAL_LABEL, apply_fallback
from src.uncertainty.measures import EmbeddingUncertaintyEstimator, compute_measures
from src.uncertainty.pipeline import run_uncertainty_analysis
from src.uncertainty.scorer import UncertaintyScorer

EXPECTED_ARTIFACTS = {
    "confidence_scores.csv",
    "uncertainty_report.json",
    "reliability_metrics.json",
    "calibration_plots.png",
    "uncertainty_distribution.png",
}


def make_predictor(builder: FusionFeatureBuilder, val_split) -> FusionPredictor:
    """Wrap an untrained gate in a checkpoint round-trip (weights don't matter here)."""
    scaler = StandardScaler().fit(val_split.features)
    model = FusionGate(input_dim=val_split.features.shape[1], hidden_dim=16, dropout=0.3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config(),
        "scaler": scaler.state_dict(),
        "residuals": {
            "kan": builder.kan_residuals.state_dict(),
            "xgb": builder.xgb_residuals.state_dict(),
        },
        "feature_names": val_split.feature_names,
        "text_column": builder.text_column,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "fusion_model.pt"
        torch.save(checkpoint, ckpt_path)
        return FusionPredictor.from_checkpoint(ckpt_path, device="cpu")


def main() -> None:
    set_seed(7)
    rng = np.random.default_rng(7)

    builder = FusionFeatureBuilder(n_residual_bins=5)
    train_df, train_emb, kan_tr, xgb_tr = make_fake_split(400, rng)
    builder.fit_residuals(kan_tr, xgb_tr, train_df["label"].to_numpy())

    val_df, val_emb, kan_val, xgb_val = make_fake_split(300, rng)
    test_df, test_emb, kan_te, xgb_te = make_fake_split(200, rng)
    val_split = builder.build("val", val_df, val_emb, kan_val, xgb_val)
    test_split = builder.build("test", test_df, test_emb, kan_te, xgb_te)
    predictor = make_predictor(builder, val_split)

    config = UncertaintyConfig(mc_passes=10, lof_neighbors=10)

    # ---- Measures ----------------------------------------------------------
    estimator = EmbeddingUncertaintyEstimator(config.lof_neighbors).fit(train_emb)
    outputs = predictor.predict(test_split)
    oriented, aux = compute_measures(
        outputs.fused_probs, test_split.kan_probs, test_split.xgb_probs,
        test_emb, estimator, predictor.model,
        predictor.scaler.transform(test_split.features),
        mc_passes=config.mc_passes, mc_seed=7,
    )
    assert set(config.weights) <= set(oriented), "measure/weight key mismatch"
    for name, values in oriented.items():
        assert len(values) == len(test_split), name
        assert np.isfinite(values).all(), f"non-finite values in {name}"
    assert (oriented["mc_alpha_std"] > 0).any(), "MC dropout produced no spread"
    print(f"[OK] {len(oriented)} measures computed: {', '.join(sorted(oriented))}")

    # ---- Scoring + tiers ---------------------------------------------------
    val_out = predictor.predict(val_split)
    val_measures, _ = compute_measures(
        val_out.fused_probs, val_split.kan_probs, val_split.xgb_probs,
        val_emb, estimator, predictor.model,
        predictor.scaler.transform(val_split.features),
        mc_passes=config.mc_passes, mc_seed=7,
    )
    scorer = UncertaintyScorer(config).fit(val_measures)
    score, _ = scorer.score(oriented)
    tiers = scorer.assign_tiers(score)
    assert score.min() >= 0.0 and score.max() <= 1.0
    assert set(np.unique(tiers)) <= {"low", "medium", "high"}
    print(f"[OK] scores in [{score.min():.3f}, {score.max():.3f}], "
          f"tiers: {dict(zip(*np.unique(tiers, return_counts=True)))}")

    # ---- All four fallback modes ------------------------------------------
    forced = tiers.copy()
    forced[:10] = "high"  # guarantee the high branch is exercised
    for mode in FallbackMode:
        decisions = apply_fallback(
            forced, outputs.fused_probs, test_split.kan_probs, test_split.xgb_probs, mode
        )
        high = forced == "high"
        if mode is FallbackMode.NEUTRAL:
            assert (decisions["final_pred"][high] == NEUTRAL_LABEL).all()
        elif mode is FallbackMode.HUMAN_REVIEW:
            assert decisions["needs_human_review"][high].all()
        elif mode is FallbackMode.MOST_CONFIDENT_EXPERT:
            assert set(decisions["prediction_source"][high]) <= {"kan", "xgb"}
        else:
            assert (decisions["final_pred"][high] == outputs.preds[high]).all()
        assert decisions["uncertainty_flag"][high].all()
        print(f"[OK] fallback mode {int(mode)} ({mode.name}) behaves as specified")

    # ---- Full pipeline + artifact export -----------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        report = run_uncertainty_analysis(
            predictor,
            val_split=val_split, val_embeddings=val_emb,
            test_split=test_split, test_embeddings=test_emb,
            train_embeddings=train_emb,
            config=config, output_dir=tmp,
        )
        produced = {p.name for p in Path(tmp).iterdir() if p.is_file()}
        missing = EXPECTED_ARTIFACTS - produced
        assert not missing, f"missing artifacts: {missing}"
        assert report["n_samples"] == len(test_split)
        print(f"[OK] artifacts: {sorted(produced)}")

    print("\n[PASS] uncertainty package smoke test complete.")


if __name__ == "__main__":
    main()
