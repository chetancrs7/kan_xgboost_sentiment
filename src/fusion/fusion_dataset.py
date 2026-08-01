"""Meta-feature assembly for the Residual-Aware Adaptive Fusion Network.

This module turns the raw outputs of the two frozen experts into the fusion
feature vector consumed by the gating network. For every sample it gathers:

* **Expert beliefs** - per-class probabilities, hard prediction, softmax
  confidence (max probability), and Shannon entropy for each expert.
* **Cross-model signals** - signed per-class probability differences, total L1
  probability difference, hard-prediction agreement, symmetric pair of KL
  divergences, and the confidence gap.
* **Linguistic cues** - the interpretable structural features the XGBoost
  expert is built on (sarcasm, emoji density, negation, intensifiers,
  punctuation, capitalisation), recomputed from raw text with the *same*
  ``FeatureExtractor`` so both stages share one definition.
* **Embedding statistics** - L2 norm and variance of the DeBERTa embedding,
  cheap proxies for how "typical" a sample is for the semantic expert.
* **Residual estimates** - each expert's *expected* error on this sample,
  looked up from confidence-binned historical residuals (see
  :class:`ResidualEstimator`). Historical residuals are computed on the
  experts' training split only, so no label information from the fusion
  train/eval splits ever leaks into the features.

All arrays for one split must be row-aligned; :class:`FusionFeatureBuilder`
validates this aggressively because silent misalignment is the classic failure
mode of stacked ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.xgboost.features import FeatureExtractor

_EPS = 1e-12

#: Linguistic features (by ``FeatureExtractor`` column name) fed to the gate.
#: ``negation_indicator`` is derived separately from ``negation_count``.
LINGUISTIC_FEATURES: tuple[str, ...] = (
    "sarcasm_score",
    "emoji_density",
    "intensifier_score",
    "punctuation_density",
    "caps_ratio",
)


# --------------------------------------------------------------------------- #
# Expert output wrappers
# --------------------------------------------------------------------------- #
def _validate_probs(probs: np.ndarray, name: str) -> np.ndarray:
    """Check a probability matrix is 2-D, finite, and row-normalised."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"{name}: expected [n_samples, n_classes], got {probs.shape}.")
    if not np.isfinite(probs).all():
        raise ValueError(f"{name}: contains NaN/Inf values.")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(
            f"{name}: rows do not sum to 1 (min={row_sums.min():.4f}, "
            f"max={row_sums.max():.4f}). Are these logits instead of probabilities?"
        )
    return probs


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy (natural log) of a probability matrix."""
    clipped = np.clip(probs, _EPS, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


@dataclass(frozen=True)
class ExpertOutputs:
    """Derived views over one expert's probability matrix for one split."""

    name: str
    probs: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "probs", _validate_probs(self.probs, self.name))

    @property
    def preds(self) -> np.ndarray:
        """Hard class predictions."""
        return self.probs.argmax(axis=1)

    @property
    def confidence(self) -> np.ndarray:
        """Softmax confidence: the maximum class probability."""
        return self.probs.max(axis=1)

    @property
    def entropy(self) -> np.ndarray:
        """Shannon entropy of the predictive distribution."""
        return shannon_entropy(self.probs)


# --------------------------------------------------------------------------- #
# Historical residual estimation
# --------------------------------------------------------------------------- #
class ResidualEstimator:
    """Leak-free per-sample residual estimate from historical predictions.

    During expert training we can observe each expert's *residual* on every
    historical (training-split) sample::

        residual_i = 1 - p_expert(y_i | x_i)          # in [0, 1]

    Residuals of unseen samples are unknown at inference time, so we estimate
    them: historical residuals are grouped into quantile bins of the expert's
    softmax confidence, and a new sample receives the mean historical residual
    of its confidence bin. This preserves the per-sample dynamics the gate
    needs ("this expert tends to be wrong when it is this unsure") without
    touching evaluation labels.
    """

    def __init__(self, n_bins: int = 10) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2.")
        self.n_bins = n_bins
        self.bin_edges_: np.ndarray | None = None
        self.bin_means_: np.ndarray | None = None
        self.global_mean_: float | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "ResidualEstimator":
        """Learn the confidence -> expected-residual lookup table.

        Args:
            probs: Historical probability matrix ``[n, n_classes]``.
            labels: Historical integer labels aligned with ``probs``.
        """
        probs = _validate_probs(probs, "residual-history probs")
        labels = np.asarray(labels, dtype=np.int64)
        if len(probs) != len(labels):
            raise ValueError("probs and labels are misaligned.")

        confidence = probs.max(axis=1)
        residuals = 1.0 - probs[np.arange(len(labels)), labels]
        self.global_mean_ = float(residuals.mean())

        # Quantile edges give evenly populated bins even for peaked confidence
        # distributions; duplicates collapse when an expert is uniformly sure.
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.unique(np.quantile(confidence, quantiles))
        if len(edges) < 3:  # degenerate: (almost) constant confidence
            self.bin_edges_ = np.array([confidence.min(), confidence.max()])
            self.bin_means_ = np.array([self.global_mean_])
            return self

        inner_edges = edges[1:-1]
        bin_ids = np.searchsorted(inner_edges, confidence, side="right")
        n_actual_bins = len(inner_edges) + 1
        means = np.full(n_actual_bins, self.global_mean_)
        for b in range(n_actual_bins):
            mask = bin_ids == b
            if mask.any():
                means[b] = residuals[mask].mean()

        self.bin_edges_ = inner_edges
        self.bin_means_ = means
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Return the expected residual for each row of ``probs``."""
        if self.bin_means_ is None:
            raise RuntimeError("ResidualEstimator.transform called before fit().")
        probs = _validate_probs(probs, "residual-query probs")
        if len(self.bin_means_) == 1:
            return np.full(len(probs), self.bin_means_[0])
        confidence = probs.max(axis=1)
        bin_ids = np.searchsorted(self.bin_edges_, confidence, side="right")
        return self.bin_means_[bin_ids]

    # -- (de)serialisation for embedding in the torch checkpoint ------------
    def state_dict(self) -> dict[str, object]:
        return {
            "n_bins": self.n_bins,
            "bin_edges": None if self.bin_edges_ is None else self.bin_edges_.tolist(),
            "bin_means": None if self.bin_means_ is None else self.bin_means_.tolist(),
            "global_mean": self.global_mean_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ResidualEstimator":
        est = cls(n_bins=int(state["n_bins"]))
        est.bin_edges_ = np.asarray(state["bin_edges"], dtype=np.float64)
        est.bin_means_ = np.asarray(state["bin_means"], dtype=np.float64)
        est.global_mean_ = state["global_mean"]
        return est


# --------------------------------------------------------------------------- #
# Feature scaling (minimal, checkpoint-friendly standard scaler)
# --------------------------------------------------------------------------- #
class StandardScaler:
    """Zero-mean / unit-variance scaler with trivial torch-checkpoint export."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < _EPS] = 1.0  # constant columns pass through untouched
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler.transform called before fit().")
        return (X - self.mean_) / self.scale_

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "StandardScaler":
        scaler = cls()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
        return scaler


# --------------------------------------------------------------------------- #
# Fusion split container
# --------------------------------------------------------------------------- #
@dataclass
class FusionSplit:
    """Row-aligned fusion inputs for one data split."""

    name: str
    features: np.ndarray  # [n, d] unscaled meta-features
    kan_probs: np.ndarray  # [n, C]
    xgb_probs: np.ndarray  # [n, C]
    labels: np.ndarray  # [n]
    feature_names: list[str] = field(default_factory=list)
    source_rows: np.ndarray | None = None  # indices into the original split CSV
    texts: list[str] | None = None
    embeddings: np.ndarray | None = None  # [n, D] DeBERTa embeddings (for fallback OOD)

    def __len__(self) -> int:
        return len(self.labels)

    def residual_column(self, name: str) -> np.ndarray | None:
        """Return a meta-feature column by name (e.g. ``kan_expected_residual``)."""
        if name not in self.feature_names:
            return None
        return self.features[:, self.feature_names.index(name)]


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #
class FusionFeatureBuilder:
    """Assemble (and validate) the fusion meta-feature matrix for any split.

    Typical usage::

        builder = FusionFeatureBuilder(n_residual_bins=10)
        builder.fit_residuals(kan_train_probs, xgb_train_probs, y_train)
        val_split  = builder.build("val", val_df, val_emb, kan_val, xgb_val)
        test_split = builder.build("test", test_df, test_emb, kan_test, xgb_test)

    ``fit_residuals`` must see only *historical* data (the experts' training
    split); ``build`` may then be called on any split without label leakage.
    """

    def __init__(self, n_residual_bins: int = 10, text_column: str = "text") -> None:
        self.text_column = text_column
        # The gate only consumes the front lexical columns (LINGUISTIC_FEATURES),
        # so skip the heavy spaCy dependency pass here; the XGBoost expert still
        # gets the full feature set from its own FeatureExtractor instance.
        self._extractor = FeatureExtractor(use_spacy_dep=False)
        self._extractor.fit([])
        self._ling_idx = {
            name: self._extractor.feature_names_.index(name)
            for name in LINGUISTIC_FEATURES
        }
        self._neg_idx = self._extractor.feature_names_.index("negation_count")
        self.kan_residuals = ResidualEstimator(n_bins=n_residual_bins)
        self.xgb_residuals = ResidualEstimator(n_bins=n_residual_bins)
        self._residuals_fitted = False

    # ------------------------------------------------------------------ fit
    def fit_residuals(
        self,
        kan_train_probs: np.ndarray,
        xgb_train_probs: np.ndarray,
        train_labels: np.ndarray,
    ) -> "FusionFeatureBuilder":
        """Fit both residual estimators on the experts' historical predictions."""
        self.kan_residuals.fit(kan_train_probs, train_labels)
        self.xgb_residuals.fit(xgb_train_probs, train_labels)
        self._residuals_fitted = True
        return self

    # ---------------------------------------------------------------- build
    def build(
        self,
        name: str,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        kan_probs: np.ndarray,
        xgb_probs: np.ndarray,
        aux_fail: np.ndarray | None = None,
    ) -> FusionSplit:
        """Build the fusion feature matrix for one split.

        Args:
            name: Split name used in logs and error messages.
            df: Subsampled split dataframe; must contain ``text_column``,
                ``label``, and (optionally) ``source_row``.
            embeddings: DeBERTa embeddings aligned with ``df`` rows.
            kan_probs: KAN expert probabilities aligned with ``df`` rows.
            xgb_probs: XGBoost expert probabilities aligned with ``df`` rows.
            aux_fail: Optional ``[n]`` per-sample signal in ``[0, 1]`` — the
                KAN aux head's estimate of P(XGBoost fails). When given it is
                appended as an extra gate feature (``kan_pred_xgb_fail``);
                ``None`` reproduces the original feature set exactly.

        Returns:
            A :class:`FusionSplit` with unscaled features.
        """
        if not self._residuals_fitted:
            raise RuntimeError("Call fit_residuals() before build().")

        kan = ExpertOutputs("kan", kan_probs)
        xgb = ExpertOutputs("xgb", xgb_probs)
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = df["label"].to_numpy(dtype=np.int64)

        lengths = {
            "dataframe": len(df),
            "embeddings": len(embeddings),
            "kan_probs": len(kan.probs),
            "xgb_probs": len(xgb.probs),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{name}: misaligned fusion inputs: {lengths}")
        if kan.probs.shape[1] != xgb.probs.shape[1]:
            raise ValueError(
                f"{name}: experts disagree on class count "
                f"({kan.probs.shape[1]} vs {xgb.probs.shape[1]})."
            )

        n_classes = kan.probs.shape[1]
        columns: list[tuple[str, np.ndarray]] = []

        # --- Expert beliefs -------------------------------------------------
        for expert in (kan, xgb):
            for c in range(n_classes):
                columns.append((f"{expert.name}_prob_{c}", expert.probs[:, c]))
            columns.append((f"{expert.name}_pred", expert.preds.astype(np.float64)))
            columns.append((f"{expert.name}_confidence", expert.confidence))
            columns.append((f"{expert.name}_entropy", expert.entropy))

        # --- Cross-model signals --------------------------------------------
        prob_delta = kan.probs - xgb.probs
        for c in range(n_classes):
            columns.append((f"prob_diff_class_{c}", prob_delta[:, c]))
        columns.append(("prob_diff_l1", np.abs(prob_delta).sum(axis=1)))
        columns.append(("agreement", (kan.preds == xgb.preds).astype(np.float64)))
        columns.append(("kl_kan_xgb", self._kl(kan.probs, xgb.probs)))
        columns.append(("kl_xgb_kan", self._kl(xgb.probs, kan.probs)))
        columns.append(("confidence_diff", kan.confidence - xgb.confidence))

        # --- Linguistic cues -------------------------------------------------
        texts = df[self.text_column].fillna("").astype(str).tolist()
        ling = self._extractor.transform(texts)
        for feat_name, idx in self._ling_idx.items():
            columns.append((feat_name, ling[:, idx]))
        columns.append(("negation_indicator", (ling[:, self._neg_idx] > 0).astype(np.float64)))

        # --- Embedding statistics --------------------------------------------
        columns.append(("embedding_l2_norm", np.linalg.norm(embeddings, axis=1)))
        columns.append(("embedding_variance", embeddings.var(axis=1)))

        # --- Residual estimates ----------------------------------------------
        kan_res = self.kan_residuals.transform(kan.probs)
        xgb_res = self.xgb_residuals.transform(xgb.probs)
        columns.append(("kan_expected_residual", kan_res))
        columns.append(("xgb_expected_residual", xgb_res))
        columns.append(("expected_residual_gap", kan_res - xgb_res))

        # --- Optional learned competence signal ------------------------------
        # The KAN aux head's P(XGBoost fails), a per-sample estimate that is
        # complementary to the confidence-binned residual proxies above.
        if aux_fail is not None:
            aux_fail = np.asarray(aux_fail, dtype=np.float64).reshape(-1)
            if len(aux_fail) != len(labels):
                raise ValueError(
                    f"{name}: aux_fail ({len(aux_fail)}) misaligned with rows ({len(labels)})."
                )
            columns.append(("kan_pred_xgb_fail", aux_fail))

        names = [c[0] for c in columns]
        features = np.column_stack([c[1] for c in columns]).astype(np.float64)
        if not np.isfinite(features).all():
            bad = [names[j] for j in np.unique(np.argwhere(~np.isfinite(features))[:, 1])]
            raise ValueError(f"{name}: non-finite fusion features in columns {bad}")

        return FusionSplit(
            name=name,
            features=features,
            kan_probs=kan.probs,
            xgb_probs=xgb.probs,
            labels=labels,
            feature_names=names,
            source_rows=(
                df["source_row"].to_numpy() if "source_row" in df.columns else None
            ),
            texts=texts,
            embeddings=embeddings,
        )

    @staticmethod
    def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Row-wise KL(p || q) with numerical clipping."""
        p_c = np.clip(p, _EPS, 1.0)
        q_c = np.clip(q, _EPS, 1.0)
        return (p_c * (np.log(p_c) - np.log(q_c))).sum(axis=1)


# --------------------------------------------------------------------------- #
# Torch dataloader helper
# --------------------------------------------------------------------------- #
def make_dataloader(
    features: np.ndarray,
    kan_probs: np.ndarray,
    xgb_probs: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Wrap scaled features + expert probabilities + labels for the gate.

    Each batch yields ``(features, kan_probs, xgb_probs, labels)`` -- the gate
    needs the expert probabilities alongside the features because the loss is
    computed on the *fused* distribution, not on the features themselves.
    """
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(kan_probs, dtype=torch.float32),
        torch.as_tensor(xgb_probs, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


# --------------------------------------------------------------------------- #
# Input loading for the standard on-disk layout
# --------------------------------------------------------------------------- #
def load_fusion_inputs(
    inputs_dir: Path | str = "data/fusion_inputs",
    kan_dir: Path | str = "models/kan_subset",
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, dict[str, object]]:
    """Load the row-aligned artifacts produced by ``scripts/prepare_fusion_inputs.py``.

    Returns:
        Mapping ``split -> {"df", "embeddings", "kan_probs", "xgb_probs"}``.
    """
    inputs_dir, kan_dir = Path(inputs_dir), Path(kan_dir)
    out: dict[str, dict[str, object]] = {}
    for split in splits:
        paths = {
            "df": inputs_dir / "splits" / f"{split}.csv",
            "embeddings": inputs_dir / "embeddings" / f"{split}_embeddings.npy",
            "xgb_probs": inputs_dir / "xgboost" / f"{split}_probs.npy",
            "kan_probs": kan_dir / f"{split}_probs.npy",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing fusion inputs:\n  " + "\n  ".join(missing) + "\n"
                "Run scripts/prepare_fusion_inputs.py and retrain the KAN expert "
                "(src/models/kan/train.py --embeddings-dir data/fusion_inputs/embeddings "
                "--output-dir models/kan_subset) first."
            )
        out[split] = {
            "df": pd.read_csv(paths["df"]),
            "embeddings": np.load(paths["embeddings"]),
            "kan_probs": np.load(paths["kan_probs"]),
            "xgb_probs": np.load(paths["xgb_probs"]),
        }
        # Optional KAN aux competence signal (present only when the KAN was
        # trained with --aux-failure-head and the signal was exported). Loaded
        # when available so the gate can use it; absent keeps prior behaviour.
        aux_path = kan_dir / f"{split}_xgb_fail_prob.npy"
        if aux_path.exists():
            out[split]["aux_fail"] = np.load(aux_path)
    return out
