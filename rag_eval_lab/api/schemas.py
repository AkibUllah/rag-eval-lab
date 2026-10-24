"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChunkerName = Literal["fixed", "recursive", "sentence"]
RetrieverName = Literal["bm25", "dense", "hybrid", "hybrid-rerank"]


class DocumentIn(BaseModel):
    id: str | None = Field(default=None, max_length=80, description="defaults to a slug of title")
    title: str | None = Field(default=None, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    lang: Literal["en", "bn"] | None = None


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=100)
    replace: bool = Field(default=False, description="drop existing documents first")


class IngestResponse(BaseModel):
    ingested: list[str]
    documents_total: int


class DocumentOut(BaseModel):
    id: str
    title: str
    lang: str
    source: str
    characters: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20)
    chunker: ChunkerName | None = None
    retriever: RetrieverName | None = None


class HitOut(BaseModel):
    id: str
    doc: str
    title: str
    heading: str
    score: float
    text: str
    cited: bool


class Timings(BaseModel):
    retrieval_ms: float
    generation_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    abstained: bool
    generator: str
    chunker: str
    retriever: str
    hits: list[HitOut]
    timings: Timings


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    documents: int
    chunker: str
    retriever: str
    llm_provider: str
    embeddings_provider: str
    vector_store: str
