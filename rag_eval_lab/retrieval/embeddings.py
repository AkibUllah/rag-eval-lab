"""Embedding providers: offline hashed n-gram TF-IDF plus optional HTTP adapters.

``HashedTfidfEmbedder`` is the default. It is not semantic: it maps word unigrams,
word bigrams and character n-grams into a fixed-size signed hash space with TF-IDF
weights. Character n-grams give it partial credit for inflected forms ("refunded",
"রিফান্ডের") that exact-term BM25 misses, which is the property the hybrid relies on.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Protocol

import httpx
import numpy as np

from rag_eval_lab.text import tokenize


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingError(RuntimeError):
    """The embeddings backend returned an error or an unexpected payload."""


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashedTfidfEmbedder:
    name = "hashed-tfidf"

    def __init__(
        self,
        dim: int = 8192,
        char_ngrams: tuple[int, int] = (3, 5),
        char_weight: float = 0.5,
    ) -> None:
        if dim & (dim - 1):
            raise ValueError("dim must be a power of two")
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.char_weight = char_weight
        self._idf = np.ones(dim, dtype=np.float32)
        self._fitted = False

    def features(self, text: str) -> Counter[str]:
        terms = tokenize(text)
        feats: Counter[str] = Counter()
        for term in terms:
            feats["w:" + term] += 1
        for a, b in zip(terms, terms[1:], strict=False):
            feats[f"b:{a}_{b}"] += 1
        lo, hi = self.char_ngrams
        for term in terms:
            padded = f"<{term}>"
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    feats["c:" + padded[i : i + n]] += 1
        return feats

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = zlib.crc32(feature.encode("utf-8"))
        return h & (self.dim - 1), (1.0 if (h >> 31) & 1 else -1.0)

    def fit(self, texts: list[str]) -> None:
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            buckets = {self._bucket(f)[0] for f in self.features(text)}
            df[list(buckets)] += 1
        n = len(texts)
        self._idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self.features(text).items():
                idx, sign = self._bucket(feat)
                weight = self.char_weight if feat.startswith("c:") else 1.0
                out[row, idx] += sign * weight * (1 + math.log(tf)) * self._idf[idx]
        return _l2_normalise(out)


class _HttpEmbedder:
    name = "http"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
        batch_size: int = 64,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)

    def fit(self, texts: list[str]) -> None:
        """Remote models are pre-trained; nothing to fit."""

    def _batch(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self._batch(texts[i : i + self.batch_size]))
        if len(rows) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(rows)}")
        return _l2_normalise(np.asarray(rows, dtype=np.float32))

    def _post(self, path: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
        try:
            resp = self._client.post(f"{self.base_url}{path}", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"{self.name} request failed: {exc}") from exc


class OpenAICompatibleEmbedder(_HttpEmbedder):
    """Any server implementing ``POST /embeddings`` (OpenAI, vLLM, LM Studio, LiteLLM)."""

    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "", **kwargs) -> None:
        super().__init__(base_url, model, **kwargs)
        self.api_key = api_key

    def _batch(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        body = self._post("/embeddings", {"model": self.model, "input": texts}, headers)
        try:
            data = sorted(body["data"], key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("malformed /embeddings response") from exc


class OllamaEmbedder(_HttpEmbedder):
    """Ollama ``POST /api/embed`` (e.g. ``nomic-embed-text``, ``bge-m3`` for Bangla)."""

    name = "ollama"

    def _batch(self, texts: list[str]) -> list[list[float]]:
        body = self._post("/api/embed", {"model": self.model, "input": texts})
        try:
            return body["embeddings"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("malformed /api/embed response") from exc
