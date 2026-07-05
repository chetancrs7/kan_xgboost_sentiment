"""The individual uncertainty measures of the inference layer.

Four families of signals are computed for every sample, all downstream of the
adaptive fusion gate:

* **Prediction uncertainty** (fused distribution): Shannon entropy, maximum
  softmax probability, and the confidence margin (top-1 minus top-2).
* **Expert disagreement** (KAN vs XGBoost): hard-prediction disagreement,
  Jensen-Shannon divergence, and total-variation probability distance.
* **Embedding uncertainty** (DeBERTa embedding vs the training distribution):
  embedding-norm deviation, Mahalanobis distance under a Ledoit-Wolf
  covariance estimate, and the Local Outlier Factor novelty score.
* **Model uncertainty** (fusion gate): Monte Carlo Dropout over the gate,
  yielding the gate's alpha spread and the variance of the fused prediction.

``compute_measures`` bundles everything into two dicts: ``oriented`` holds the
score-ready versions (every entry monotone in uncertainty, matching the keys
of :data:`src.uncertainty.config.DEFAULT_WEIGHTS`), and ``aux`` holds raw
diagnostics for the per-sample CSV.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor

from src.fusion.fusion_dataset import shannon_entropy
from src.fusion.fusion_model import FusionGate

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Prediction uncertainty
# --------------------------------------------------------------------------- #
def prediction_uncertainty(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Entropy, maximum softmax probability, and confidence margin.

    Args:
        probs: Predictive distribution ``[n, C]`` (the fused probabilities).

    Returns:
        ``entropy`` (normalised to [0, 1] by log C), ``msp``, and ``margin``
        (top-1 minus top-2 probability).
    """
    probs = np.asarray(probs, dtype=np.float64)
    n_classes = probs.shape[1]
    sorted_probs = np.sort(probs, axis=1)
    return {
        "entropy": shannon_entropy(probs) / np.log(n_classes),
        "msp": probs.max(axis=1),
        "margin": sorted_probs[:, -1] - sorted_probs[:, -2],
    }


# --------------------------------------------------------------------------- #
# Expert disagreement
# --------------------------------------------------------------------------- #
def expert_disagreement(
    kan_probs: np.ndarray, xgb_probs: np.ndarray
) -> dict[str, np.ndarray]:
    """Disagreement signals between the two frozen experts.

    Returns:
        ``disagreement`` (hard predictions differ, 0/1), ``js_divergence``
        (base-2 Jensen-Shannon divergence, in [0, 1]), and ``prob_distance``
        (total-variation distance ``0.5 * ||p - q||_1``, in [0, 1]).
    """
    kan = np.asarray(kan_probs, dtype=np.float64)
    xgb = np.asarray(xgb_probs, dtype=np.float64)
    mixture = 0.5 * (kan + xgb)
    jsd = 0.5 * _kl_base2(kan, mixture) + 0.5 * _kl_base2(xgb, mixture)
    return {
        "disagreement": (kan.argmax(axis=1) != xgb.argmax(axis=1)).astype(np.float64),
        "js_divergence": np.clip(jsd, 0.0, 1.0),  # clip fp noise at the bounds
        "prob_distance": 0.5 * np.abs(kan - xgb).sum(axis=1),
    }


def _kl_base2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise KL(p || q) in bits, with numerical clipping."""
    p_c = np.clip(p, _EPS, 1.0)
    q_c = np.clip(q, _EPS, 1.0)
    return (p_c * (np.log2(p_c) - np.log2(q_c))).sum(axis=1)


# --------------------------------------------------------------------------- #
# Embedding uncertainty
# --------------------------------------------------------------------------- #
class EmbeddingUncertaintyEstimator:
    """Distance of a sample's embedding from the training distribution.

    Fit once on the experts' *training* embeddings, then applied to any split:

    * ``embedding_norm_dev`` - |z-score| of the L2 norm relative to the
      training norms (unusually short/long embeddings are both atypical).
    * ``mahalanobis`` - Mahalanobis distance under a Ledoit-Wolf shrinkage
      covariance (well-conditioned even for 768-dim embeddings).
    * ``lof`` - negated Local Outlier Factor novelty score (higher = more
      isolated from the training manifold).
    """

    def __init__(self, lof_neighbors: int = 20) -> None:
        self.lof_neighbors = lof_neighbors
        self._norm_mean: float | None = None
        self._norm_std: float | None = None
        self._covariance: LedoitWolf | None = None
        self._lof: LocalOutlierFactor | None = None

    def fit(self, train_embeddings: np.ndarray) -> "EmbeddingUncertaintyEstimator":
        """Learn the reference statistics from the training embeddings."""
        emb = np.asarray(train_embeddings, dtype=np.float64)
        if emb.ndim != 2:
            raise ValueError(f"Expected [n, d] embeddings, got shape {emb.shape}.")
        norms = np.linalg.norm(emb, axis=1)
        self._norm_mean = float(norms.mean())
        self._norm_std = float(max(norms.std(), _EPS))
        self._covariance = LedoitWolf().fit(emb)
        n_neighbors = min(self.lof_neighbors, len(emb) - 1)
        self._lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(emb)
        return self

    def transform(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        """Return the three embedding-uncertainty measures for each row."""
        if self._covariance is None:
            raise RuntimeError("EmbeddingUncertaintyEstimator.transform before fit().")
        emb = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(emb, axis=1)
        return {
            "embedding_norm": norms,
            "embedding_norm_dev": np.abs(norms - self._norm_mean) / self._norm_std,
            "mahalanobis": np.sqrt(np.maximum(self._covariance.mahalanobis(emb), 0.0)),
            # score_samples: lower = more abnormal, so negate for "higher = odd".
            "lof": -self._lof.score_samples(emb),
        }


# --------------------------------------------------------------------------- #
# Model uncertainty (Monte Carlo Dropout on the fusion gate)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mc_dropout_predict(
    model: FusionGate,
    scaled_features: np.ndarray,
    kan_probs: np.ndarray,
    xgb_probs: np.ndarray,
    n_passes: int = 30,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stochastic forward passes through the gate with dropout enabled.

    Each pass samples a different dropout mask, giving a distribution over the
    gate value ``alpha`` and therefore over the fused prediction. The spread
    of that distribution is the gate's epistemic uncertainty on the sample.

    Args:
        model: Trained fusion gate (its train/eval state is restored on exit).
        scaled_features: Meta-features ``[n, d]``, already scaler-transformed.
        kan_probs / xgb_probs: Frozen expert probabilities ``[n, C]``.
        n_passes: Number of stochastic passes.
        seed: Seed for the dropout masks.

    Returns:
        ``alpha_mean`` / ``alpha_std`` (gate value across passes),
        ``mc_pred_variance`` (variance of the fused probabilities across
        passes, averaged over classes), and ``mc_mean_probs`` ``[n, C]``.
    """
    device = next(model.parameters()).device
    features = torch.as_tensor(scaled_features, dtype=torch.float32, device=device)
    kan = torch.as_tensor(kan_probs, dtype=torch.float32, device=device)
    xgb = torch.as_tensor(xgb_probs, dtype=torch.float32, device=device)

    was_training = model.training
    model.train()  # activate dropout; the gate has no batch-norm to disturb
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        alphas = []
        fused = []
        for _ in range(n_passes):
            alpha = model(features)
            alphas.append(alpha.squeeze(1))
            fused.append(FusionGate.fuse(alpha, kan, xgb))
    finally:
        model.train(was_training)
        torch.random.set_rng_state(generator_state)

    alpha_stack = torch.stack(alphas)  # [T, n]
    fused_stack = torch.stack(fused)  # [T, n, C]
    return {
        "alpha_mean": alpha_stack.mean(dim=0).cpu().numpy(),
        "alpha_std": alpha_stack.std(dim=0).cpu().numpy(),
        "mc_pred_variance": fused_stack.var(dim=0).mean(dim=1).cpu().numpy(),
        "mc_mean_probs": fused_stack.mean(dim=0).cpu().numpy(),
    }


# --------------------------------------------------------------------------- #
# Bundling
# --------------------------------------------------------------------------- #
def compute_measures(
    fused_probs: np.ndarray,
    kan_probs: np.ndarray,
    xgb_probs: np.ndarray,
    embeddings: np.ndarray,
    embedding_estimator: EmbeddingUncertaintyEstimator,
    model: FusionGate,
    scaled_features: np.ndarray,
    mc_passes: int = 30,
    mc_seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute every uncertainty measure for one split.

    Returns:
        ``(oriented, aux)``: ``oriented`` maps each scoring measure name
        (higher = more uncertain) to a ``[n]`` array; ``aux`` holds the raw
        diagnostics (msp, margin, embedding norm, MC alpha statistics) for
        the per-sample export.
    """
    pred = prediction_uncertainty(fused_probs)
    disagree = expert_disagreement(kan_probs, xgb_probs)
    embed = embedding_estimator.transform(embeddings)
    mc = mc_dropout_predict(
        model, scaled_features, kan_probs, xgb_probs, n_passes=mc_passes, seed=mc_seed
    )

    oriented = {
        "entropy": pred["entropy"],
        "one_minus_msp": 1.0 - pred["msp"],
        "one_minus_margin": 1.0 - pred["margin"],
        "disagreement": disagree["disagreement"],
        "js_divergence": disagree["js_divergence"],
        "prob_distance": disagree["prob_distance"],
        "embedding_norm_dev": embed["embedding_norm_dev"],
        "mahalanobis": embed["mahalanobis"],
        "lof": embed["lof"],
        "mc_alpha_std": mc["alpha_std"],
        "mc_pred_variance": mc["mc_pred_variance"],
    }
    aux = {
        "msp": pred["msp"],
        "margin": pred["margin"],
        "embedding_norm": embed["embedding_norm"],
        "mc_alpha_mean": mc["alpha_mean"],
    }
    return oriented, aux
