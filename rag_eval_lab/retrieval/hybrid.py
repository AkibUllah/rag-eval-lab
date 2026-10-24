"""Hybrid retrieval with reciprocal rank fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections import defaultdict

from rag_eval_lab.documents import Chunk
from rag_eval_lab.retrieval.base import Retriever, ScoredChunk


def reciprocal_rank_fusion(rankings: list[list[ScoredChunk]], k_rrf: int = 60) -> list[ScoredChunk]:
    """Fuse ranked lists by sum(1 / (k_rrf + rank)). Only ranks matter, never raw scores,
    so BM25 scores (unbounded) and cosine similarities (-1..1) combine without tuning."""
    fused: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            fused[cid] += 1.0 / (k_rrf + rank)
            chunks[cid] = hit.chunk
            first_seen.setdefault(cid, len(first_seen))
    order = sorted(fused, key=lambda cid: (-fused[cid], first_seen[cid]))
    return [ScoredChunk(chunks[cid], round(fused[cid], 6)) for cid in order]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], k_rrf: int = 60, depth: int = 50) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        self.depth = depth

    def index(self, chunks: list[Chunk]) -> None:
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        rankings = [r.search(query, max(k, self.depth)) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.k_rrf)[:k]
