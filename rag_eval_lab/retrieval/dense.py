"""Dense retrieval: an embedder plus a vector store."""

from __future__ import annotations

from rag_eval_lab.documents import Chunk
from rag_eval_lab.retrieval.base import NotIndexedError, ScoredChunk
from rag_eval_lab.retrieval.embeddings import Embedder, HashedTfidfEmbedder
from rag_eval_lab.retrieval.store import InMemoryVectorStore, VectorStore


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder | None = None, store: VectorStore | None = None) -> None:
        self.embedder = embedder or HashedTfidfEmbedder()
        self.store = store or InMemoryVectorStore()
        self._by_id: dict[str, Chunk] = {}

    def index(self, chunks: list[Chunk]) -> None:
        texts = [c.search_text for c in chunks]
        self.embedder.fit(texts)
        self.store.clear()
        if chunks:
            self.store.add([c.id for c in chunks], self.embedder.embed(texts))
        self._by_id = {c.id: c for c in chunks}

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not self._by_id:
            raise NotIndexedError("DenseRetriever.search() called before index()")
        vector = self.embedder.embed([query])[0]
        return [
            ScoredChunk(self._by_id[cid], round(score, 6))
            for cid, score in self.store.search(vector, k)
            if cid in self._by_id
        ]
