"""FastAPI application: ingest documents, ask questions, read evaluation reports."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_eval_lab import __version__
from rag_eval_lab.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    HealthOut,
    HitOut,
    IngestRequest,
    IngestResponse,
    Timings,
)
from rag_eval_lab.config import PROJECT_ROOT, Settings, get_settings
from rag_eval_lab.documents import LoaderError, load_bytes, make_document, slugify
from rag_eval_lab.evaluation.runner import latest
from rag_eval_lab.generation.llm import LLMError
from rag_eval_lab.pipeline import KnowledgeBase
from rag_eval_lab.retrieval.embeddings import EmbeddingError

log = logging.getLogger("rag_eval_lab.api")
WEB_DIR = PROJECT_ROOT / "web"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        kb = KnowledgeBase(settings)
        if settings.autoload_corpus and settings.corpus_dir.is_dir():
            kb.load_corpus()
        app.state.kb = kb
        yield

    app = FastAPI(
        title="RAG Eval Lab",
        version=__version__,
        description=(
            "Retrieval-augmented Q&A over a document set, with cited answers and an "
            "evaluation harness. Offline by default: BM25 + hashed n-gram vectors and an "
            "extractive answerer; LLM and embedding providers are optional."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_log(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        log.info(
            "%s %s %s %.1fms id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
        return response

    @app.exception_handler(LoaderError)
    async def loader_error(_: Request, exc: LoaderError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(LLMError)
    @app.exception_handler(EmbeddingError)
    async def upstream_error(_: Request, exc: Exception) -> JSONResponse:
        log.warning("upstream provider error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": "upstream provider error"})

    def kb_of(request: Request) -> KnowledgeBase:
        return request.app.state.kb

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health(request: Request) -> HealthOut:
        return HealthOut(
            status="ok",
            version=__version__,
            documents=len(kb_of(request).documents),
            chunker=settings.chunker,
            retriever=settings.retriever,
            llm_provider=settings.llm_provider,
            embeddings_provider=settings.embeddings_provider,
            vector_store=settings.vector_store,
        )

    @app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
    def documents(request: Request) -> list[DocumentOut]:
        return [
            DocumentOut(
                id=d.id, title=d.title, lang=d.lang, source=d.source, characters=len(d.text)
            )
            for d in kb_of(request).documents
        ]

    @app.post("/ingest", response_model=IngestResponse, tags=["documents"])
    def ingest(body: IngestRequest, request: Request) -> IngestResponse:
        docs = [
            make_document(
                d.text,
                doc_id=slugify(d.id or d.title or f"doc-{i + 1}"),
                title=d.title,
                lang=d.lang,
                source="api",
            )
            for i, d in enumerate(body.documents)
        ]
        kb = kb_of(request)
        kb.add(docs, replace=body.replace)
        return IngestResponse(ingested=[d.id for d in docs], documents_total=len(kb.documents))

    @app.post("/ingest/files", response_model=IngestResponse, tags=["documents"])
    async def ingest_files(
        request: Request,
        files: list[UploadFile] = File(..., description=".md, .txt or .pdf"),
        replace: bool = Form(False),
    ) -> IngestResponse:
        docs = []
        for upload in files:
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(
                    413, f"{upload.filename} exceeds {settings.max_upload_bytes} bytes"
                )
            docs.append(load_bytes(upload.filename or "upload.txt", data))
        kb = kb_of(request)
        kb.add(docs, replace=replace)
        return IngestResponse(ingested=[d.id for d in docs], documents_total=len(kb.documents))

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        kb = kb_of(request)
        if not kb.documents:
            raise HTTPException(409, "no documents ingested yet")
        pipe = kb.pipeline(body.chunker, body.retriever)
        result = pipe.ask(body.question, body.k)
        cited = set(result.answer.citations)
        return AskResponse(
            question=body.question,
            answer=result.answer.text,
            citations=result.answer.citations,
            abstained=result.answer.abstained,
            generator=result.answer.generator,
            chunker=pipe.chunker.name,
            retriever=pipe.retriever.name,
            hits=[
                HitOut(
                    id=h.chunk.id,
                    doc=h.chunk.doc_id,
                    title=h.chunk.title,
                    heading=h.chunk.heading,
                    score=h.score,
                    text=h.chunk.text,
                    cited=h.chunk.id in cited,
                )
                for h in result.hits
            ],
            timings=Timings(
                retrieval_ms=round(result.retrieval_ms, 3),
                generation_ms=round(result.generation_ms, 3),
            ),
        )

    @app.get("/eval/latest", tags=["evaluation"])
    def eval_latest() -> dict[str, Any]:
        """The most recent ``rag-eval run`` report (per-question detail included)."""
        report = latest(settings.runs_dir, "run")
        if report is None:
            raise HTTPException(404, "no evaluation run yet; run `rag-eval run`")
        return report

    @app.get("/eval/latest/compare", tags=["evaluation"])
    def eval_latest_compare() -> dict[str, Any]:
        """The most recent ``rag-eval compare`` grid."""
        report = latest(settings.runs_dir, "compare")
        if report is None:
            raise HTTPException(404, "no comparison yet; run `rag-eval compare`")
        return report

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
