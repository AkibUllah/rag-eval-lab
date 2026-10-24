"""Second-stage reranking of a candidate list.

``LexicalReranker`` is a deterministic, dependency-free stand-in for a cross-encoder: it
rewards candidates that cover more of the distinct query terms and contain query
bigrams in order, blended with the first-stage rank. Any object with the same
``rerank`` signature (e.g. a hosted cross-encoder) can be dropped in.
"""

from __future__ import annotations

from typing import Protocol

from rag_eval_lab.documents import Chunk
from rag_eval_lab.retrieval.base import Retriever, ScoredChunk
from rag_eval_lab.text import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


class LexicalReranker:
    def __init__(self, coverage: float = 0.5, bigrams: float = 0.2, prior: float = 0.3) -> None:
        self.weights = (coverage, bigrams, prior)

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        q_terms = tokenize(query)
        q_set = set(q_terms)
        q_bigrams = set(zip(q_terms, q_terms[1:], strict=False))
        w_cov, w_bi, w_prior = self.weights
        n = len(candidates)
        rescored: list[tuple[float, int, ScoredChunk]] = []
        for rank, cand in enumerate(candidates):
            terms = tokenize(cand.chunk.search_text)
            coverage = len(q_set & set(terms)) / len(q_set) if q_set else 0.0
            bigram = (
                len(q_bigrams & set(zip(terms, terms[1:], strict=False))) / len(q_bigrams)
                if q_bigrams
                else 0.0
            )
            prior = 1.0 - rank / n if n else 0.0
            score = w_cov * coverage + w_bi * bigram + w_prior * prior
            rescored.append((score, rank, cand))
        rescored.sort(key=lambda item: (-item[0], item[1]))
        return [ScoredChunk(c.chunk, round(s, 6)) for s, _, c in rescored[:k]]


class RerankingRetriever:
    """Wrap a retriever: fetch ``depth`` candidates, rerank, keep ``k``."""

    def __init__(self, base: Retriever, reranker: Reranker, depth: int = 20) -> None:
        self.base = base
        self.reranker = reranker
        self.depth = depth
        self.name = f"{base.name}-rerank"

    def index(self, chunks: list[Chunk]) -> None:
        self.base.index(chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        candidates = self.base.search(query, max(k, self.depth))
        return self.reranker.rerank(query, candidates, k)
