"""Retrievers and the factory that builds them by name."""

from __future__ import annotations

from collections.abc import Callable

from rag_eval_lab.retrieval.base import NotIndexedError, Retriever, ScoredChunk
from rag_eval_lab.retrieval.bm25 import BM25Retriever
from rag_eval_lab.retrieval.dense import DenseRetriever
from rag_eval_lab.retrieval.embeddings import Embedder, HashedTfidfEmbedder
from rag_eval_lab.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_eval_lab.retrieval.rerank import LexicalReranker, RerankingRetriever
from rag_eval_lab.retrieval.store import InMemoryVectorStore, VectorStore

RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid-rerank")


def build_retriever(
    name: str,
    embedder_factory: Callable[[], Embedder] | None = None,
    store_factory: Callable[[], VectorStore] | None = None,
) -> Retriever:
    """Build a retriever by name. Factories let callers swap in remote embeddings/pgvector."""
    make_embedder = embedder_factory or HashedTfidfEmbedder
    make_store = store_factory or InMemoryVectorStore

    def dense() -> DenseRetriever:
        return DenseRetriever(make_embedder(), make_store())

    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return dense()
    if name == "hybrid":
        return HybridRetriever([BM25Retriever(), dense()])
    if name == "hybrid-rerank":
        return RerankingRetriever(HybridRetriever([BM25Retriever(), dense()]), LexicalReranker())
    raise ValueError(f"unknown retriever {name!r}; choose from {', '.join(RETRIEVERS)}")


__all__ = [
    "RETRIEVERS",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LexicalReranker",
    "NotIndexedError",
    "RerankingRetriever",
    "Retriever",
    "ScoredChunk",
    "build_retriever",
    "reciprocal_rank_fusion",
]
