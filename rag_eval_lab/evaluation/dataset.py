"""Golden set loading and evidence resolution.

Each golden item names its evidence as ``{"doc": id, "evidence": "verbatim passage"}``
instead of chunk ids. Evidence is resolved to a character span once, and a chunk is
relevant when it covers at least ``MIN_COVERAGE`` of that span. This keeps one golden
set valid for every chunker (ids like ``fees#3`` differ between strategies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_eval_lab.documents import Chunk, Document

MIN_COVERAGE = 0.5


class GoldenSetError(ValueError):
    """The golden set references a missing document or evidence that is not in it."""


@dataclass(frozen=True)
class Evidence:
    doc: str
    text: str
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    evidence: list[Evidence]
    answer: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)

    @property
    def lang(self) -> str:
        return "bn" if "bn" in self.tags else "en"


def load_golden(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            item = GoldenItem(
                id=raw["id"],
                question=raw["question"],
                evidence=[Evidence(r["doc"], r["evidence"]) for r in raw.get("relevant", [])],
                answer=raw.get("answer", ""),
                tags=list(raw.get("tags", [])),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise GoldenSetError(f"{path}:{lineno}: invalid golden item ({exc})") from exc
        if item.id in seen:
            raise GoldenSetError(f"{path}:{lineno}: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    return items


def resolve(items: list[GoldenItem], docs: list[Document]) -> list[GoldenItem]:
    """Attach character spans to every evidence passage; fail loudly on drift."""
    by_id = {d.id: d for d in docs}
    out: list[GoldenItem] = []
    for item in items:
        resolved: list[Evidence] = []
        for ev in item.evidence:
            doc = by_id.get(ev.doc)
            if doc is None:
                raise GoldenSetError(f"{item.id}: unknown document {ev.doc!r}")
            start = doc.text.find(ev.text)
            if start < 0:
                raise GoldenSetError(f"{item.id}: evidence not found in {ev.doc}: {ev.text[:60]!r}")
            resolved.append(Evidence(ev.doc, ev.text, start, start + len(ev.text)))
        out.append(GoldenItem(item.id, item.question, resolved, item.answer, item.tags))
    return out


def covers(chunk: Chunk, ev: Evidence, min_coverage: float = MIN_COVERAGE) -> bool:
    if chunk.doc_id != ev.doc or ev.end <= ev.start:
        return False
    overlap = min(chunk.end, ev.end) - max(chunk.start, ev.start)
    return overlap > 0 and overlap / (ev.end - ev.start) >= min_coverage
