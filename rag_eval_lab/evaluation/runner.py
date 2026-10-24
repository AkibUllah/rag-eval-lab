"""Evaluation runs: one configuration, or a grid compared side by side."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval_lab.config import Settings
from rag_eval_lab.documents import Document
from rag_eval_lab.evaluation import metrics as m
from rag_eval_lab.evaluation.dataset import GoldenItem, covers, resolve
from rag_eval_lab.generation.base import Generator
from rag_eval_lab.pipeline import build_pipeline
from rag_eval_lab.text import word_count

METRIC_LABELS: dict[str, str] = {
    "recall@k": "Recall@{k}",
    "mrr": "MRR",
    "ndcg@k": "nDCG@{k}",
    "hit@1": "Hit@1",
    "hit@k": "Hit@{k}",
    "citation_precision": "Cit. prec",
    "f1": "Answer F1",
    "faithfulness": "Faithful",
    "abstain_accuracy": "Abstain acc",
}
HIGHER_IS_BETTER = tuple(METRIC_LABELS)


@dataclass
class QuestionResult:
    id: str
    question: str
    tags: list[str]
    answerable: bool
    retrieved: list[str]
    relevant: list[str]
    recall: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    hit1: float | None
    hitk: float | None
    answer: str
    citations: list[str]
    abstained: bool
    citation_precision: float | None
    f1: float | None
    exact_match: float | None
    faithfulness: float | None
    context_words: int
    retrieval_ms: float


@dataclass
class EvalRun:
    chunker: str
    retriever: str
    generator: str
    k: int
    n_chunks: int
    metrics: dict[str, float | None]
    by_tag: dict[str, dict[str, float | None]]
    questions: list[QuestionResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    environment: dict[str, str] = field(
        default_factory=lambda: {"python": platform.python_version()}
    )

    @property
    def label(self) -> str:
        return f"{self.chunker}/{self.retriever}"

    def to_dict(self, include_questions: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_questions:
            data.pop("questions")
        return data

    def misses(self) -> list[QuestionResult]:
        return [q for q in self.questions if q.answerable and not q.hitk]


def _aggregate(results: list[QuestionResult]) -> dict[str, float | None]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    return {
        "questions": float(len(results)),
        "recall@k": m.mean([r.recall for r in answerable]),
        "mrr": m.mean([r.reciprocal_rank for r in answerable]),
        "ndcg@k": m.mean([r.ndcg for r in answerable]),
        "hit@1": m.mean([r.hit1 for r in answerable]),
        "hit@k": m.mean([r.hitk for r in answerable]),
        "citation_precision": m.mean([r.citation_precision for r in answerable]),
        "f1": m.mean([r.f1 for r in answerable]),
        "exact_match": m.mean([r.exact_match for r in answerable]),
        "faithfulness": m.mean([r.faithfulness for r in results]),
        "answered_rate": m.mean([0.0 if r.abstained else 1.0 for r in answerable]),
        "abstain_accuracy": m.mean([1.0 if r.abstained else 0.0 for r in unanswerable]),
        "context_words": m.mean([float(r.context_words) for r in answerable]),
        "latency_p50_ms": m.percentile([r.retrieval_ms for r in results], 50),
        "latency_p95_ms": m.percentile([r.retrieval_ms for r in results], 95),
    }


def run_eval(
    docs: list[Document],
    golden: list[GoldenItem],
    settings: Settings,
    chunker: str,
    retriever: str,
    k: int = 5,
    generator: Generator | None = None,
    support: m.SupportFn | None = None,
) -> EvalRun:
    items = resolve(golden, docs)
    pipe = build_pipeline(settings, chunker, retriever)
    if generator is not None:
        pipe.generator = generator
    chunks = pipe.ingest(docs)

    results: list[QuestionResult] = []
    for item in items:
        t0 = time.perf_counter()
        hits = pipe.retrieve(item.question, k)
        elapsed = (time.perf_counter() - t0) * 1000
        answer = pipe.generator.generate(item.question, hits)
        ranked = [h.chunk for h in hits]
        cov = m.coverage_by_rank(ranked, item.evidence)
        relevant = [c.id for c, found in zip(ranked, cov, strict=True) if found]
        n_ev = len(item.evidence)
        contexts = {c.id: c for c in ranked}
        if item.answerable:
            # Relevance for citations is judged against all chunks, not only retrieved ones.
            relevant_all = {c.id for c in chunks if any(covers(c, ev) for ev in item.evidence)}
            results.append(
                QuestionResult(
                    id=item.id,
                    question=item.question,
                    tags=item.tags,
                    answerable=True,
                    retrieved=[c.id for c in ranked],
                    relevant=relevant,
                    recall=m.recall_at_k(cov, n_ev, k),
                    reciprocal_rank=m.reciprocal_rank(cov),
                    ndcg=m.ndcg_at_k(cov, n_ev, k),
                    hit1=m.hit_at_k(cov, 1),
                    hitk=m.hit_at_k(cov, k),
                    answer=answer.text,
                    citations=answer.citations,
                    abstained=answer.abstained,
                    citation_precision=m.citation_precision(answer, relevant_all),
                    f1=m.token_f1(answer.text, item.answer) if not answer.abstained else 0.0,
                    exact_match=m.exact_match(answer.text, item.answer),
                    faithfulness=m.faithfulness(answer, contexts, support),
                    context_words=sum(word_count(c.text) for c in ranked),
                    retrieval_ms=round(elapsed, 3),
                )
            )
        else:
            results.append(
                QuestionResult(
                    id=item.id,
                    question=item.question,
                    tags=item.tags,
                    answerable=False,
                    retrieved=[c.id for c in ranked],
                    relevant=[],
                    recall=None,
                    reciprocal_rank=None,
                    ndcg=None,
                    hit1=None,
                    hitk=None,
                    answer=answer.text,
                    citations=answer.citations,
                    abstained=answer.abstained,
                    citation_precision=None,
                    f1=None,
                    exact_match=None,
                    faithfulness=m.faithfulness(answer, contexts, support),
                    context_words=sum(word_count(c.text) for c in ranked),
                    retrieval_ms=round(elapsed, 3),
                )
            )

    tags = sorted({t for r in results for t in r.tags} | {"en"})
    by_tag = {}
    for tag in tags:
        subset = [r for r in results if (tag in r.tags) or (tag == "en" and "bn" not in r.tags)]
        by_tag[tag] = {
            key: _aggregate(subset)[key] for key in ("questions", "recall@k", "mrr", "hit@k", "f1")
        }
    return EvalRun(
        chunker=chunker,
        retriever=retriever,
        generator=pipe.generator.name,
        k=k,
        n_chunks=len(chunks),
        metrics=_aggregate(results),
        by_tag=by_tag,
        questions=results,
    )


def compare(
    docs: list[Document],
    golden: list[GoldenItem],
    settings: Settings,
    chunkers: list[str],
    retrievers: list[str],
    k: int = 5,
) -> list[EvalRun]:
    return [run_eval(docs, golden, settings, c, r, k) for c in chunkers for r in retrievers]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def markdown_table(runs: list[EvalRun]) -> str:
    """Comparison table; with several runs, the best value in each quality column is bold."""
    if not runs:
        return ""
    k = runs[0].k
    cols = list(METRIC_LABELS)
    best = {
        c: max((r.metrics[c] for r in runs if r.metrics[c] is not None), default=None) for c in cols
    }
    header = [
        "Chunker",
        "Retriever",
        "Chunks",
        *[METRIC_LABELS[c].format(k=k) for c in cols],
        "Ctx words",
        "p50 ms",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * 3 + ["---:"] * (len(cols) + 2)) + "|",
    ]
    for r in runs:
        cells = [r.chunker, r.retriever, str(r.n_chunks)]
        for c in cols:
            v = r.metrics[c]
            text = _fmt(v)
            if len(runs) > 1 and v is not None and best[c] is not None and abs(v - best[c]) < 1e-9:
                text = f"**{text}**"
            cells.append(text)
        cells.append(_fmt(r.metrics["context_words"], 0))
        cells.append(_fmt(r.metrics["latency_p50_ms"], 2))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tag_table(run: EvalRun, tags: list[str] | None = None) -> str:
    k = run.k
    tags = tags or list(run.by_tag)
    lines = [
        f"| Slice | Questions | Recall@{k} | MRR | Hit@{k} | Answer F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in tags:
        if tag not in run.by_tag:
            continue
        s = run.by_tag[tag]
        lines.append(
            f"| {tag} | {int(s['questions'] or 0)} | {_fmt(s['recall@k'])} | {_fmt(s['mrr'])} "
            f"| {_fmt(s['hit@k'])} | {_fmt(s['f1'])} |"
        )
    return "\n".join(lines)


def save_run(run: EvalRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = runs_dir / f"run-{run.chunker}-{run.retriever}-{stamp}.json"
    path.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_comparison(runs: list[EvalRun], runs_dir: Path) -> tuple[Path, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    json_path = runs_dir / f"compare-{stamp}.json"
    md_path = runs_dir / f"compare-{stamp}.md"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "k": runs[0].k if runs else None,
        "runs": [r.to_dict(include_questions=False) for r in runs],
        "markdown": markdown_table(runs),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(payload["markdown"] + "\n", encoding="utf-8")
    return json_path, md_path


def latest(runs_dir: Path, prefix: str) -> dict[str, Any] | None:
    """Newest ``<prefix>-*.json`` in ``runs_dir`` by modification time, or ``None``."""
    if not runs_dir.is_dir():
        return None
    files = sorted(runs_dir.glob(f"{prefix}-*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))
