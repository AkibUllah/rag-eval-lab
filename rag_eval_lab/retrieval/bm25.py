"""Okapi BM25 over the bilingual tokenizer, with an inverted index (no external deps)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from rag_eval_lab.documents import Chunk
from rag_eval_lab.retrieval.base import NotIndexedError, ScoredChunk
from rag_eval_lab.text import tokenize


class BM25Retriever:
    """Classic BM25 with the non-negative (Lucene) IDF variant.

    ``k1`` controls term-frequency saturation, ``b`` the document-length normalisation.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for i, chunk in enumerate(chunks):
            terms = tokenize(chunk.search_text)
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                postings[term].append((i, tf))
        n = len(chunks)
        self._chunks = list(chunks)
        self._postings = dict(postings)
        self._lengths = lengths
        self._avgdl = (sum(lengths) / n) if n else 0.0
        self._idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def scores(self, query: str) -> dict[int, float]:
        if not self._chunks:
            raise NotIndexedError("BM25Retriever.search() called before index()")
        acc: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = 1 - self.b + self.b * self._lengths[i] / (self._avgdl or 1.0)
                acc[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return acc

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        acc = self.scores(query)
        ranked = sorted(acc.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [ScoredChunk(self._chunks[i], round(s, 6)) for i, s in ranked]
