"""The gating network of the Residual-Aware Adaptive Fusion Network.

Architecture (exactly as specified)::

    Input
      |
    Fully Connected
      |
    LayerNorm
      |
    ReLU
      |
    Dropout
      |
    Fully Connected
      |
    Sigmoid
      |
    alpha in [0, 1]

``alpha`` is the per-sample trust assigned to the KAN semantic expert;
``1 - alpha`` goes to the XGBoost structural expert. The fused prediction is
the convex combination::

    p_fused = alpha * p_kan + (1 - alpha) * p_xgb

Because the mixture of two valid probability distributions is itself a valid
distribution, no re-normalisation is needed and the gate can be trained with
plain cross-entropy on ``log(p_fused)``.
"""

from __future__ import annotations

import torch
from torch import nn

_EPS = 1e-8


class FusionGate(nn.Module):
    """Lightweight per-sample gating network producing ``alpha`` in [0, 1]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Args:
        input_dim: Width of the fusion meta-feature vector.
        hidden_dim: Width of the single hidden layer.
        dropout: Dropout probability applied after the hidden activation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map meta-features ``[n, d]`` to gate values ``alpha`` of shape ``[n, 1]``."""
        return self.net(features)

    @staticmethod
    def fuse(
        alpha: torch.Tensor, kan_probs: torch.Tensor, xgb_probs: torch.Tensor
    ) -> torch.Tensor:
        """Convex combination of the two expert distributions.

        Args:
            alpha: Gate values ``[n, 1]`` (or ``[n]``).
            kan_probs: KAN probabilities ``[n, C]``.
            xgb_probs: XGBoost probabilities ``[n, C]``.

        Returns:
            Fused probabilities ``[n, C]``.
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)
        return alpha * kan_probs + (1.0 - alpha) * xgb_probs

    def config(self) -> dict[str, float | int]:
        """Constructor arguments, embedded in checkpoints for reconstruction."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


def fused_cross_entropy(
    alpha: torch.Tensor,
    kan_probs: torch.Tensor,
    xgb_probs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy of the fused distribution against the true labels.

    Standard ``nn.CrossEntropyLoss`` expects logits; here the mixture is
    already a probability distribution, so the loss is the negative log of the
    fused probability assigned to the true class.
    """
    fused = FusionGate.fuse(alpha, kan_probs, xgb_probs)
    log_probs = torch.log(fused.clamp_min(_EPS))
    return nn.functional.nll_loss(log_probs, labels)
