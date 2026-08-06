"""Embedding backends.

`HashEmbedding` is not semantic — it is a hashed bag-of-character-ngrams that
happens to satisfy the vector interface. It exists so the RAG pipeline, the
vector store and the retrieval tool can be developed and tested with no model
download. It retrieves on lexical overlap, which is enough to prove plumbing and
nothing more. Production uses `SentenceTransformerEmbedding` with BGE-M3.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbedding:
    name = "hash"
    # Hashed n-grams spread signal thinly across the vector, so even an exact
    # topical match lands around 0.08-0.20. Judged on the same scale as a real
    # embedder it would return nothing at all.
    similarity_floor = 0.03

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def start(self) -> None:
        return None

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            # Two hashes per token: one for the term, one for a bigram of it,
            # which gives a little robustness to inflection.
            for form in (token, token[:4]):
                digest = hashlib.blake2b(form.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    async def close(self) -> None:
        return None


class SentenceTransformerEmbedding:
    name = "sentence_transformers"
    similarity_floor = 0.25

    def __init__(self, model: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self._model_name = model
        self._device = device
        self._model: Any = None
        self.dim = 0

    async def start(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = await asyncio.to_thread(
            SentenceTransformer, self._model_name, device=self._device
        )
        self.dim = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("SentenceTransformerEmbedding.start() was not awaited")
        vectors = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,  # lets the store use plain dot product
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    async def close(self) -> None:
        self._model = None
