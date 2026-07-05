"""Residual-Aware Adaptive Fusion Network.

Learns, per sample, how much to trust the KAN semantic expert versus the
XGBoost structural expert. A lightweight gating network maps a meta-feature
vector (expert probabilities, confidences, entropies, cross-model statistics,
linguistic cues, embedding statistics, and historical residual estimates) to a
scalar gate ``alpha`` in ``[0, 1]``; the fused prediction is::

    p_fused = alpha * p_kan + (1 - alpha) * p_xgb

Modules
-------
``fusion_dataset``  meta-feature assembly + alignment validation
``fusion_model``    the gating network
``trainer``         gate training (CE loss, early stopping, checkpoints,
                    TensorBoard, LR scheduling)
``predictor``       checkpoint loading + inference + artifact export
``metrics``         evaluation metrics, calibration curve, reliability diagram
``utils``           seeding, logging, small I/O helpers
"""

from src.fusion.fusion_model import FusionGate
from src.fusion.fusion_dataset import FusionFeatureBuilder, FusionSplit

__all__ = ["FusionGate", "FusionFeatureBuilder", "FusionSplit"]
