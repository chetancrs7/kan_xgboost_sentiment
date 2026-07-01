"""KAN expert sentiment classifier over pretrained DeBERTa embeddings.

A 4-layer Kolmogorov-Arnold Network (adaptive B-spline edge functions) that
consumes precomputed DeBERTa-v3 embeddings and classifies sentiment:

    SemanticCompressionLayer -> ContextInteractionLayer
        -> SentimentDistortionLayer -> EmotionHotspotLayer

Public entry points:
    - ``load_splits``              : read precomputed embeddings + labels
    - ``KANSentimentClassifier``   : the 4-layer KAN model
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

__all__ = [
    "load_splits",
    "load_embeddings",
    "build_dataloaders",
    "KANSentimentClassifier",
    "SemanticCompressionLayer",
    "ContextInteractionLayer",
    "SentimentDistortionLayer",
    "EmotionHotspotLayer",
    "evaluate_split",
    "logits_to_probs",
    "shannon_entropy",
    "confidence_from_entropy",
]
