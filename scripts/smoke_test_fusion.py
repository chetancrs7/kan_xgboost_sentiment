"""Synthetic end-to-end smoke test for the fusion package.

Exercises the whole stack on fake data (no expert artifacts needed): feature
building, gate training for a few epochs, checkpoint save/load round-trip via
``FusionPredictor``, and artifact export. Useful as a fast regression check::

    python scripts/smoke_test_fusion.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.fallback import FallbackConfig, UncertaintyFallback
from src.fusion.fusion_dataset import FusionFeatureBuilder, StandardScaler, make_dataloader
from src.fusion.fusion_model import FusionGate, fused_cross_entropy
from src.fusion.predictor import FusionPredictor, evaluate_and_export
from src.fusion.trainer import stratified_holdout, subset_split, train_gate
from src.fusion.utils import set_seed


def fake_expert(labels: np.ndarray, accuracy: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesise a binary expert with roughly the requested accuracy."""
    n = len(labels)
    correct = rng.random(n) < accuracy
    pred = np.where(correct, labels, 1 - labels)
    p_pred = rng.uniform(0.55, 0.99, n)
    probs = np.zeros((n, 2))
    probs[np.arange(n), pred] = p_pred
    probs[np.arange(n), 1 - pred] = 1 - p_pred
    return probs


def make_fake_split(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    labels = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "source_row": np.arange(n),
            "text": ["I LOVE this... not really!! :(" if l == 0 else "so happy today :)" for l in labels],
            "label": labels,
        }
    )
    emb = rng.normal(size=(n, 16))
    return df, emb, fake_expert(labels, 0.80, rng), fake_expert(labels, 0.70, rng)


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
    print(f"[OK] features built: {val_split.features.shape[1]} columns")
    print("     " + ", ".join(val_split.feature_names))

    tr_idx, ho_idx = stratified_holdout(val_split.labels, 0.2, seed=7)
    gate_train = subset_split(val_split, tr_idx, "gate_train")
    gate_holdout = subset_split(val_split, ho_idx, "gate_holdout")
    scaler = StandardScaler().fit(gate_train.features)

    train_loader = make_dataloader(
        scaler.transform(gate_train.features), gate_train.kan_probs,
        gate_train.xgb_probs, gate_train.labels, batch_size=64, shuffle=True,
    )
    holdout_loader = make_dataloader(
        scaler.transform(gate_holdout.features), gate_holdout.kan_probs,
        gate_holdout.xgb_probs, gate_holdout.labels, batch_size=64,
    )

    model = FusionGate(input_dim=gate_train.features.shape[1], hidden_dim=16, dropout=0.1)
    args = argparse.Namespace(
        epochs=20, lr=1e-2, weight_decay=1e-4, patience=10,
        scheduler_patience=3, scheduler_factor=0.5, min_lr=1e-6,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        writer = SummaryWriter(log_dir=str(tmp / "tb"))
        result = train_gate(model, train_loader, holdout_loader, torch.device("cpu"), args, writer)
        writer.close()
        print(f"[OK] trained {result['trained_epochs']} epochs, "
              f"best holdout loss {result['best_holdout_loss']:.4f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "scaler": scaler.state_dict(),
            "residuals": {
                "kan": builder.kan_residuals.state_dict(),
                "xgb": builder.xgb_residuals.state_dict(),
            },
            "feature_names": gate_train.feature_names,
            "text_column": builder.text_column,
        }
        ckpt_path = tmp / "fusion_model.pt"
        torch.save(checkpoint, ckpt_path)

        predictor = FusionPredictor.from_checkpoint(ckpt_path, device="cpu")

        # Fit + attach the uncertainty-aware fallback, mirroring the trainer.
        holdout_outputs = predictor.predict(gate_holdout)
        fallback = UncertaintyFallback(FallbackConfig())
        fallback.fit(
            train_embeddings=train_emb,
            holdout_fused_probs=holdout_outputs.fused_probs,
            holdout_kan_probs=gate_holdout.kan_probs,
            holdout_xgb_probs=gate_holdout.xgb_probs,
            holdout_embeddings=gate_holdout.embeddings,
            holdout_labels=gate_holdout.labels,
        )
        predictor.fallback = fallback
        print("[OK] fallback fitted "
              f"(entropy_abstain={fallback.config.entropy_abstain:.3f}, "
              f"ood_abstain={fallback.config.ood_abstain:.3f})")

        rebuilt = predictor.builder_from_checkpoint()
        test_split_rebuilt = rebuilt.build("test", test_df, test_emb, kan_te, xgb_te)
        metrics = evaluate_and_export(
            predictor, test_split_rebuilt, tmp / "out", gate_holdout_split=gate_holdout
        )
        produced = sorted(p.relative_to(tmp / "out").as_posix() for p in (tmp / "out").rglob("*") if p.is_file())
        print("[OK] artifacts:", produced)
        expected = {
            "fusion_metrics.json", "fusion_predictions.csv", "fusion_probabilities.npy",
            "plots/alpha_distribution.png", "plots/calibration_curve.png",
            "plots/confusion_matrix.png", "plots/reliability_diagram.png",
            "plots/fallback_actions.png",
        }
        missing = expected - set(produced)
        assert not missing, f"missing artifacts: {missing}"
        assert "fallback" in metrics, "fallback metrics missing from report"
        pred_cols = pd.read_csv(tmp / "out" / "fusion_predictions.csv").columns
        assert "fallback_action" in pred_cols, "fallback_action column missing"
        acc = metrics["test"]["fusion_accuracy"]
        fb = metrics["fallback"]
        print(f"[OK] fusion test accuracy on synthetic data: {acc:.3f}")
        print(f"[OK] fallback: coverage={fb['coverage']:.3f} "
              f"review_rate={fb['review_rate']:.3f} actions={fb['action_counts']}")

    print("\n[PASS] fusion package smoke test complete.")


if __name__ == "__main__":
    main()
