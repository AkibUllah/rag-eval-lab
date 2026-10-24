"""Wiring: documents -> chunks -> retriever -> generator, plus a cached knowledge base."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from rag_eval_lab.chunking import Chunker, chunk_documents, get_chunker
from rag_eval_lab.config import Settings
from rag_eval_lab.documents import Chunk, Document, load_directory
from rag_eval_lab.generation.base import Answer, Generator
from rag_eval_lab.generation.extractive import ExtractiveAnswerer
from rag_eval_lab.generation.llm import LLMAnswerer, build_provider
from rag_eval_lab.retrieval import Retriever, ScoredChunk, build_retriever
from rag_eval_lab.retrieval.embeddings import (
    Embedder,
    HashedTfidfEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
)
from rag_eval_lab.retrieval.store import InMemoryVectorStore, PgVectorStore, VectorStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: Answer
    hits: list[ScoredChunk]
    retrieval_ms: float
    generation_ms: float


class RAGPipeline:
    def __init__(
        self, chunker: Chunker, retriever: Retriever, generator: Generator, top_k: int = 5
    ) -> None:
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.chunks: list[Chunk] = []

    def ingest(self, docs: list[Document]) -> list[Chunk]:
        self.chunks = chunk_documents(docs, self.chunker)
        self.retriever.index(self.chunks)
        return self.chunks

    def retrieve(self, question: str, k: int | None = None) -> list[ScoredChunk]:
        return self.retriever.search(question, k or self.top_k)

    def ask(self, question: str, k: int | None = None) -> AskResult:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, hits)
        t2 = time.perf_counter()
        return AskResult(question, answer, hits, (t1 - t0) * 1000, (t2 - t1) * 1000)


def build_embedder_factory(settings: Settings) -> Callable[[], Embedder]:
    if settings.embeddings_provider == "openai":
        return lambda: OpenAICompatibleEmbedder(
            settings.embeddings_url(), settings.embeddings_model, settings.embeddings_api_key
        )
    if settings.embeddings_provider == "ollama":
        return lambda: OllamaEmbedder(settings.embeddings_url(), settings.embeddings_model)
    return HashedTfidfEmbedder


def build_store_factory(settings: Settings, table: str = "rag_chunks") -> Callable[[], VectorStore]:
    """One table per pipeline configuration so cached pipelines never share rows."""
    if settings.vector_store == "pgvector":
        if not settings.database_url:
            raise ValueError("RAG_VECTOR_STORE=pgvector requires DATABASE_URL")
        return lambda: PgVectorStore(settings.database_url, table=table)
    return InMemoryVectorStore


def build_generator(settings: Settings) -> Generator:
    if settings.llm_provider == "fake":
        return ExtractiveAnswerer()
    provider = build_provider(
        settings.llm_provider, settings.llm_url(), settings.llm_model, settings.llm_api_key
    )
    return LLMAnswerer(provider)


def build_pipeline(
    settings: Settings, chunker: str | None = None, retriever: str | None = None
) -> RAGPipeline:
    chunker = chunker or settings.chunker
    retriever = retriever or settings.retriever
    table = f"rag_{chunker}_{retriever}".replace("-", "_")
    return RAGPipeline(
        get_chunker(chunker),
        build_retriever(
            retriever,
            build_embedder_factory(settings),
            build_store_factory(settings, table),
        ),
        build_generator(settings),
        settings.top_k,
    )


class KnowledgeBase:
    """Holds the document set and lazily builds one indexed pipeline per configuration.

    Re-ingesting invalidates every cached pipeline; the next request re-indexes. A lock
    keeps concurrent requests from indexing the same configuration twice.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._docs: dict[str, Document] = {}
        self._pipelines: dict[tuple[str, str], RAGPipeline] = {}
        self._lock = threading.Lock()

    @property
    def documents(self) -> list[Document]:
        return list(self._docs.values())

    def load_corpus(self) -> int:
        docs = load_directory(self.settings.corpus_dir)
        self.add(docs, replace=True)
        log.info("loaded %d documents from %s", len(docs), self.settings.corpus_dir)
        return len(docs)

    def add(self, docs: list[Document], replace: bool = False) -> None:
        with self._lock:
            if replace:
                self._docs = {}
            for doc in docs:
                self._docs[doc.id] = doc
            self._pipelines = {}

    def pipeline(self, chunker: str | None = None, retriever: str | None = None) -> RAGPipeline:
        key = (chunker or self.settings.chunker, retriever or self.settings.retriever)
        with self._lock:
            pipe = self._pipelines.get(key)
            if pipe is None:
                pipe = build_pipeline(self.settings, *key)
                pipe.ingest(list(self._docs.values()))
                self._pipelines[key] = pipe
            return pipe
