"""Fast synthetic self-check for the residual-aware complementarity pipeline.

Exercises every new module without needing embeddings/CSVs on disk, and asserts
the backward-compatibility guarantee (a default KAN has no auxiliary-head
parameters, so existing checkpoints still load). Run:

    python scripts/smoke_test_residual_kan.py

Exit code 0 means all checks passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.complementarity import complementarity_metrics  # noqa: E402
from src.models.kan.model import KANSentimentClassifier  # noqa: E402
from src.models.kan.residual_loss import (  # noqa: E402
    ComplementarityConfig,
    ComplementarityLoss,
    make_residual_dataloader,
)
from src.models.xgboost.oof import build_residual_targets, residual_weights  # noqa: E402


def _synthetic(n: int = 400, d_struct: int = 12, d_emb: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Structural features carry weak signal (so XGBoost fails on some).
    X_struct = rng.normal(size=(n, d_struct)).astype(np.float32)
    X_struct[:, 0] += y * 0.8
    # Embeddings carry stronger signal (so the KAN can rescue).
    X_emb = rng.normal(size=(n, d_emb)).astype(np.float32)
    X_emb[:, 0] += y * 1.5
    return X_struct, X_emb, y.astype(np.int64)


def check_oof_and_weights() -> None:
    X_struct, _, y = _synthetic()
    targets = build_residual_targets(X_struct, y, n_splits=4, seed=1, w_max=3.0, verbose=False)
    assert targets.weights.shape == y.shape
    assert targets.weights.min() >= 1.0 - 1e-9 and targets.weights.max() <= 3.0 + 1e-9
    assert targets.xgb_true_prob.min() >= 0.0 and targets.xgb_true_prob.max() <= 1.0
    assert set(np.unique(targets.xgb_fail)).issubset({0.0, 1.0})
    # Harder (lower true-prob) samples must not receive lower weight than easier ones.
    order = np.argsort(targets.xgb_true_prob)
    ranks = residual_weights(targets.oof_probs, y, method="percentile", w_max=3.0)
    corr = np.corrcoef(1.0 - targets.xgb_true_prob, ranks)[0, 1]
    assert corr > 0.9, f"weights should track residuals (corr={corr:.3f})"
    # double-fault cap lowers flagged weights only.
    capped = residual_weights(
        targets.oof_probs, y, method="percentile", w_max=3.0,
        double_fault_cap=1.5, xgb_fail=targets.xgb_fail,
    )
    fail = targets.xgb_fail.astype(bool)
    assert capped[fail].max() <= 1.5 + 1e-9
    print(f"  [oof] ok — xgb_oof_acc={1 - targets.xgb_fail.mean():.3f}, "
          f"weight mean={targets.weights.mean():.3f}")


def check_model_backward_compat() -> None:
    # Default model: no aux head, no aux parameters (checkpoint-compatible).
    plain = KANSentimentClassifier(input_dim=16, hidden_dims=(8, 6, 4), n_classes=2, grid_size=3)
    assert plain.aux_head is None
    assert not any("aux_head" in k for k in plain.state_dict()), "default model must have no aux params"
    x = torch.randn(5, 16)
    assert plain(x).shape == (5, 2)
    assert plain.aux_failure_logit(x) is None

    # Aux-enabled model: forward unchanged, aux logit is [batch]; config round-trips.
    cfg = {"input_dim": 16, "hidden_dims": (8, 6, 4), "n_classes": 2, "grid_size": 3,
           "aux_failure_head": True, "aux_hidden": 8}
    aux_model = KANSentimentClassifier(**cfg)
    assert aux_model(x).shape == (5, 2)
    assert aux_model.aux_failure_logit(x).shape == (5,)
    ser = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    reloaded = KANSentimentClassifier(**ser)
    reloaded.load_state_dict(aux_model.state_dict())  # must not raise
    print("  [model] ok — default has no aux params; aux config round-trips")


def check_loss() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 2, requires_grad=True)
    labels = torch.randint(0, 2, (8,))
    w = torch.rand(8) * 2 + 1
    ptrue = torch.rand(8)
    fail = (torch.rand(8) > 0.5).float()
    aux_logit = torch.randn(8, requires_grad=True)

    loss_fn = ComplementarityLoss(ComplementarityConfig())
    total, comps = loss_fn(logits, labels, sample_weight=w, xgb_true_prob=ptrue,
                           xgb_fail=fail, aux_logit=aux_logit)
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and aux_logit.grad is not None
    assert "base" in comps and "weighted" in comps and "aux" in comps
    # With no extra signals it reduces to plain cross-entropy * base_weight.
    plain, _ = ComplementarityLoss(ComplementarityConfig(comp_weight=0, rescue_weight=0, aux_weight=0))(
        logits, labels
    )
    ce = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(plain, ce, atol=1e-6)
    print(f"  [loss] ok — components={ {k: round(v, 3) for k, v in comps.items()} }")


def check_dataloader() -> None:
    X_struct, X_emb, y = _synthetic(n=64)
    loader = make_residual_dataloader(
        X_emb, y, sample_weight=np.ones(len(y)), xgb_true_prob=np.full(len(y), 0.5),
        xgb_fail=np.zeros(len(y)), batch_size=16, shuffle=True, seed=3,
    )
    batch = next(iter(loader))
    assert len(batch) == 5
    assert batch[0].shape[1] == X_emb.shape[1]
    print("  [dataloader] ok — yields (X, y, w, xgb_ptrue, xgb_fail)")


def check_complementarity() -> None:
    n = 200
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, n)
    # Construct experts with a known relationship.
    def probs_for(correct: np.ndarray) -> np.ndarray:
        p = np.zeros((n, 2))
        p[np.arange(n), np.where(correct, y, 1 - y)] = 0.8
        p[np.arange(n), np.where(correct, 1 - y, y)] = 0.2
        return p
    xgb_correct = rng.random(n) > 0.4
    kan_correct = rng.random(n) > 0.3
    m = complementarity_metrics(probs_for(kan_correct), probs_for(xgb_correct), y)
    for key in ("kan_rescue_rate", "xgb_preservation_rate", "disagreement_accuracy",
                "double_fault_rate", "oracle_ensemble_accuracy"):
        assert key in m, f"missing {key}"
    # Identities.
    assert abs(m["oracle_ensemble_accuracy"] - (1 - m["double_fault_rate"])) < 1e-9
    assert abs(m["kan_accuracy"] - kan_correct.mean()) < 1e-9
    assert abs(m["xgb_accuracy"] - xgb_correct.mean()) < 1e-9
    print(f"  [complementarity] ok — rescue={m['kan_rescue_rate']:.3f}, "
          f"preservation={m['xgb_preservation_rate']:.3f}, oracle={m['oracle_ensemble_accuracy']:.3f}")


def check_end_to_end() -> None:
    """One tiny residual-aware training step through the real modules."""
    X_struct, X_emb, y = _synthetic(n=200, d_emb=16)
    targets = build_residual_targets(X_struct, y, n_splits=3, seed=2, verbose=False)
    model = KANSentimentClassifier(input_dim=16, hidden_dims=(8, 6, 4), n_classes=2,
                                   grid_size=3, aux_failure_head=True, aux_hidden=8)
    loss_fn = ComplementarityLoss(ComplementarityConfig())
    loader = make_residual_dataloader(
        X_emb, y, sample_weight=targets.weights, xgb_true_prob=targets.xgb_true_prob,
        xgb_fail=targets.xgb_fail, batch_size=32, shuffle=True, seed=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.update_grids(torch.from_numpy(X_emb[:128]))
    for Xb, yb, wb, pb, fb in loader:
        opt.zero_grad()
        logits = model(Xb)
        total, _ = loss_fn(logits, yb, sample_weight=wb, xgb_true_prob=pb,
                           xgb_fail=fb, aux_logit=model.aux_failure_logit(Xb))
        assert torch.isfinite(total)
        total.backward()
        opt.step()
        break
    print("  [end-to-end] ok — residual-aware step ran and stayed finite")


def main() -> int:
    print("Residual-aware complementarity smoke test")
    print("=" * 44)
    check_oof_and_weights()
    check_model_backward_compat()
    check_loss()
    check_dataloader()
    check_complementarity()
    check_end_to_end()
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
