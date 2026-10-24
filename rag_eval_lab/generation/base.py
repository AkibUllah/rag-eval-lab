"""Answer type and generator protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from rag_eval_lab.retrieval.base import ScoredChunk

ABSTAIN_TEXT = "I could not find this in the knowledge base."
CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]*#\d+)\]")


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    generator: str = ""


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Answer: ...


def extract_citations(text: str, allowed: set[str] | None = None) -> list[str]:
    """Citation ids in order of first appearance, optionally restricted to ``allowed``."""
    seen: list[str] = []
    for cid in CITATION_RE.findall(text):
        if cid not in seen and (allowed is None or cid in allowed):
            seen.append(cid)
    return seen


def strip_citations(text: str) -> str:
    return re.sub(r"\s*" + CITATION_RE.pattern, "", text).strip()
