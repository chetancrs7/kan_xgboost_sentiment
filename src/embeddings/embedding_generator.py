"""Reusable DeBERTa-v3 sentence-embedding generator.

``EmbeddingGenerator`` wraps a HuggingFace encoder and turns raw text into
fixed-size sentence vectors via masked mean pooling over the last hidden
state. It is deliberately model-agnostic in its interface so the same
embeddings can feed the KAN, XGBoost, and hybrid ensemble models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from config import EmbeddingConfig, set_seed

if TYPE_CHECKING:  # Avoid importing torch at module load for path-only uses.
    import torch

logger = logging.getLogger("embeddings.generator")


class EmbeddingGenerator:
    """Encode text into mean-pooled DeBERTa-v3 sentence embeddings."""

    def __init__(self, config: EmbeddingConfig) -> None:
        # Import heavy deps lazily so importing this module stays cheap and
        # only fails when a generator is actually constructed.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.config = config

        set_seed(config.seed)
        self.device = torch.device(self._resolve_device(config.device))

        logger.info("Loading tokenizer and model '%s' ...", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()  # Disable dropout for deterministic inference.

        self.embedding_dim: int = self.model.config.hidden_size
        logger.info(
            "Model ready on %s | embedding_dim=%d | max_length=%d | batch_size=%d",
            self.device,
            self.embedding_dim,
            config.max_length,
            config.batch_size,
        )

    @staticmethod
    def _resolve_device(requested: str | None) -> str:
        """Pick the requested device, defaulting to CUDA when available."""
        import torch

        if requested:
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _mean_pool(
        last_hidden_state: "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Average token vectors, ignoring padding via the attention mask."""
        # mask: (batch, seq_len) -> (batch, seq_len, 1) broadcast over hidden dim.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # Guard against empty masks.
        return summed / counts

    def generate(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts and return an ``(n, embedding_dim)`` array."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        torch = self._torch
        batch_size = self.config.batch_size
        batches = range(0, len(texts), batch_size)
        embeddings: list[np.ndarray] = []

        progress = tqdm(
            batches,
            total=(len(texts) + batch_size - 1) // batch_size,
            desc="Embedding",
            unit="batch",
            disable=not show_progress,
        )

        with torch.no_grad():
            for start in progress:
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**encoded)
                pooled = self._mean_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )
                embeddings.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(embeddings)
