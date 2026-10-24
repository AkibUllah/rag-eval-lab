"""Offline answerer: picks the best-supported sentences from the retrieved chunks.

This is the deterministic "fake LLM" used by tests, the default demo and the evaluation
baseline. It never invents text, so faithfulness is high by construction; what it does
measure honestly is whether retrieval put the answer in front of the generator and
whether citations point at the right chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_eval_lab.generation.base import ABSTAIN_TEXT, Answer
from rag_eval_lab.retrieval.base import ScoredChunk
from rag_eval_lab.text import split_sentences, tokenize

_MARKUP_RE = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)|\*\*|__|`")


def clean_sentence(sentence: str) -> str:
    return _MARKUP_RE.sub("", sentence).strip()


class ExtractiveAnswerer:
    """Score every sentence of the top chunks against the question and quote the best.

    A query term counts fully when it appears in the sentence, ``context_weight`` when it
    only appears in the chunk's title/heading or neighbouring text (so "Wrong PIN
    attempts" as a heading still helps the sentence below it). Terms are IDF-weighted
    across the candidate sentences, and earlier-ranked chunks get a small prior. If the
    best sentence covers less than ``min_coverage`` of the question, it abstains.
    """

    name = "extractive"

    def __init__(
        self,
        max_contexts: int = 3,
        max_sentences: int = 2,
        min_coverage: float = 0.4,
        context_weight: float = 0.4,
        rank_prior: float = 0.08,
    ) -> None:
        self.max_contexts = max_contexts
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage
        self.context_weight = context_weight
        self.rank_prior = rank_prior

    def _candidates(self, contexts: list[ScoredChunk]) -> list[_Candidate]:
        out: list[_Candidate] = []
        for rank, hit in enumerate(contexts[: self.max_contexts]):
            around = set(tokenize(hit.chunk.search_text)) | set(tokenize(hit.chunk.text))
            for sentence in split_sentences(hit.chunk.text):
                if sentence.lstrip().startswith("#"):
                    continue
                text = clean_sentence(sentence)
                if text:
                    out.append(_Candidate(text, hit.chunk.id, set(tokenize(text)), around, rank))
        return out

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Answer:
        q_terms = set(tokenize(question))
        candidates = self._candidates(contexts)
        if not q_terms or not candidates:
            return Answer(ABSTAIN_TEXT, abstained=True, generator=self.name)

        # Query terms found in many candidate sentences ("shapla", "pay") carry less
        # signal than rare ones ("dormant", "ডিপিএস"); terms found nowhere weigh most.
        df = Counter(t for c in candidates for t in c.terms & q_terms)
        n = len(candidates)
        weight = {t: math.log(1 + n / (1 + df.get(t, 0))) for t in q_terms}
        total = sum(weight.values()) or 1.0

        def coverage(c: _Candidate) -> float:
            got = 0.0
            for t in q_terms:
                if t in c.terms:
                    got += weight[t]
                elif t in c.context:
                    got += self.context_weight * weight[t]
            return got / total

        def score(c: _Candidate) -> float:
            return coverage(c) - self.rank_prior * c.rank

        ranked = sorted(candidates, key=score, reverse=True)
        if coverage(ranked[0]) < self.min_coverage:
            return Answer(ABSTAIN_TEXT, abstained=True, generator=self.name)

        best = score(ranked[0])
        picked = [ranked[0]]
        covered = ranked[0].terms & q_terms
        for cand in ranked[1:]:
            if len(picked) >= self.max_sentences:
                break
            new_terms = (cand.terms & q_terms) - covered
            if score(cand) >= 0.85 * best and new_terms:
                picked.append(cand)
                covered |= new_terms
        text = " ".join(f"{c.text} [{c.chunk_id}]" for c in picked)
        citations = list(dict.fromkeys(c.chunk_id for c in picked))
        return Answer(text, citations=citations, generator=self.name)


@dataclass(frozen=True)
class _Candidate:
    text: str
    chunk_id: str
    terms: set[str]
    context: set[str]
    rank: int
