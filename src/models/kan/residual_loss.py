"""Residual-aware complementarity objective for the KAN semantic expert.

The KAN keeps optimising the *original* sentiment label (so it stays an
independent semantic classifier, not a direct XGBoost-correction model), but its
training signal is reshaped so it specialises in the context-dependent cases the
structural XGBoost expert gets wrong. Four additive terms (all optional, each
individually switch-off-able by setting its weight to 0):

``base``
    Plain unweighted cross-entropy. Protects the KAN's overall predictive
    performance and prevents excessive specialisation.

``weighted``
    Cross-entropy re-weighted by the bounded, capped residual weights from
    :mod:`src.models.xgboost.oof` (structurally hard tweets weigh more). This is
    the complementarity pressure — it increases the importance of samples the
    structural expert found difficult.

``rescue``
    A semantic-rescue *margin* hinge applied only to samples where XGBoost gave
    the true class a low probability. It encourages the KAN to assign the true
    class a higher probability than XGBoost by a small margin, strengthening its
    recovery of cases that need contextual understanding.

``aux``
    An optional binary cross-entropy that trains the KAN's auxiliary head to
    predict whether XGBoost will fail on the sample (an interpretable competence
    signal for the fusion gate). Does not affect the sentiment prediction.

The forward signature degrades gracefully: with only ``logits``/``labels`` given
it is exactly cross-entropy, so the module is safe to reuse for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ComplementarityConfig:
    """Weights and thresholds for :class:`ComplementarityLoss`."""

    base_weight: float = 1.0
    comp_weight: float = 1.0
    rescue_weight: float = 0.5
    aux_weight: float = 0.5
    rescue_margin: float = 0.1
    hard_threshold: float = 0.5  # XGBoost true-class prob below this == "hard"


class ComplementarityLoss(nn.Module):
    """Residual-aware loss combining base CE, weighted CE, rescue, and aux BCE."""

    def __init__(self, config: ComplementarityConfig | None = None) -> None:
        super().__init__()
        self.config = config or ComplementarityConfig()
        self._ce = nn.CrossEntropyLoss(reduction="none")
        self._bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        xgb_true_prob: torch.Tensor | None = None,
        xgb_fail: torch.Tensor | None = None,
        aux_logit: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss and a per-term breakdown for logging.

        Args:
            logits: KAN class logits ``[batch, C]``.
            labels: Integer labels ``[batch]``.
            sample_weight: Bounded residual weights ``[batch]`` (defaults to
                ones, i.e. plain CE for the weighted term).
            xgb_true_prob: XGBoost's out-of-fold true-class probability
                ``[batch]``; needed for the rescue margin.
            xgb_fail: 1/0 XGBoost-failure target ``[batch]``; needed for aux.
            aux_logit: Auxiliary-head logit ``[batch]`` (from
                :meth:`KANSentimentClassifier.aux_failure_logit`).

        Returns:
            ``(total_loss, components)`` where ``components`` holds the scalar
            value of every active term.
        """
        cfg = self.config
        per_sample_ce = self._ce(logits, labels)  # [batch]
        base = per_sample_ce.mean()

        components: dict[str, float] = {"base": float(base.detach())}
        total = cfg.base_weight * base

        # --- Weighted (complementarity) cross-entropy -----------------------
        if cfg.comp_weight > 0 and sample_weight is not None:
            w = sample_weight.to(per_sample_ce.dtype)
            weighted = (w * per_sample_ce).sum() / w.sum().clamp_min(1e-8)
            total = total + cfg.comp_weight * weighted
            components["weighted"] = float(weighted.detach())

        # --- Semantic-rescue margin hinge -----------------------------------
        # Only on samples XGBoost found hard: push KAN's true-class prob above
        # XGBoost's by `rescue_margin`.
        if cfg.rescue_weight > 0 and xgb_true_prob is not None:
            probs = torch.softmax(logits, dim=1)
            kan_true_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            hard = (xgb_true_prob < cfg.hard_threshold).to(per_sample_ce.dtype)
            if float(hard.sum()) > 0:
                deficit = torch.relu(
                    cfg.rescue_margin + xgb_true_prob - kan_true_prob
                )
                rescue = (hard * deficit).sum() / hard.sum().clamp_min(1e-8)
                total = total + cfg.rescue_weight * rescue
                components["rescue"] = float(rescue.detach())

        # --- Auxiliary XGBoost-failure prediction ---------------------------
        if cfg.aux_weight > 0 and aux_logit is not None and xgb_fail is not None:
            aux = self._bce(aux_logit, xgb_fail.to(aux_logit.dtype))
            total = total + cfg.aux_weight * aux
            components["aux"] = float(aux.detach())

        components["total"] = float(total.detach())
        return total, components


def make_residual_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    xgb_true_prob: np.ndarray | None = None,
    xgb_fail: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Wrap embeddings + labels + residual signals into a ``DataLoader``.

    Every batch yields ``(X, y, sample_weight, xgb_true_prob, xgb_fail)`` so the
    training loop can feed :class:`ComplementarityLoss` directly. Missing signals
    default to neutral values (weight 1, true-prob 1, fail 0) which reduces the
    objective to plain cross-entropy — handy for the validation loader.
    """
    n = len(X)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float32)
    if xgb_true_prob is None:
        xgb_true_prob = np.ones(n, dtype=np.float32)
    if xgb_fail is None:
        xgb_fail = np.zeros(n, dtype=np.float32)

    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(sample_weight, dtype=torch.float32),
        torch.as_tensor(xgb_true_prob, dtype=torch.float32),
        torch.as_tensor(xgb_fail, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )
