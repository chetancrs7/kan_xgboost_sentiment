"""KAN expert sentiment classifier over pretrained DeBERTa embeddings.

A 4-layer Kolmogorov-Arnold Network (adaptive B-spline edge functions) that
consumes precomputed DeBERTa-v3 embeddings and classifies sentiment:

    SemanticCompressionLayer -> ContextInteractionLayer
        -> SentimentDistortionLayer -> EmotionHotspotLayer

Each layer's spline family is configurable (B-spline, cubic B-spline,
thin-plate, Gaussian RBF, linear head) via the registry in ``splines.py``.

Public entry points:
    - ``load_splits``              : read precomputed embeddings + labels
    - ``KANSentimentClassifier``   : the 4-layer KAN model
    - ``build_spline_layer``       : spline factory (see ``SPLINE_REGISTRY``)
    - ``evaluate_split``           : metrics for a single split
    - ``logits_to_probs``          : softmax logits -> probabilities
    - ``confidence_from_entropy``  : Shannon-entropy-based confidence scores
"""

from __future__ import annotations

from .data_loader import build_dataloaders, load_embeddings, load_splits
from .evaluate import confidence_from_entropy, evaluate_split, logits_to_probs, shannon_entropy
from .layers import (
    ContextInteractionLayer,
    EmotionHotspotLayer,
    SemanticCompressionLayer,
    SentimentDistortionLayer,
)
from .model import KANSentimentClassifier
from .splines import SPLINE_REGISTRY, build_spline_layer

__all__ = [
    "load_splits",
    "load_embeddings",
    "build_dataloaders",
    "KANSentimentClassifier",
    "SPLINE_REGISTRY",
    "build_spline_layer",
    "SemanticCompressionLayer",
    "ContextInteractionLayer",
    "SentimentDistortionLayer",
    "EmotionHotspotLayer",
    "evaluate_split",
    "logits_to_probs",
    "shannon_entropy",
    "confidence_from_entropy",
]
