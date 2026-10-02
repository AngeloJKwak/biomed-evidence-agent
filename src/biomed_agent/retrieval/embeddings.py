"""Embedding backends.

The default is a local Hugging Face sentence-transformers model (BGE-small), so
retrieval runs without any embedding API. `HashingEmbedder` is a dependency-free
stand-in used by tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    # BGE models are trained with this instruction prefix on queries (not passages).
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import; keep lazy

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._query_prefix = self._BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(self._query_prefix + text, normalize_embeddings=True)
        return vec.tolist()


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. Good enough for tests, not for real use."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self._dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
