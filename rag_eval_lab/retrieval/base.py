"""Retriever protocol and result type shared by every retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_eval_lab.documents import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]: ...


class NotIndexedError(RuntimeError):
    """search() was called before index()."""
