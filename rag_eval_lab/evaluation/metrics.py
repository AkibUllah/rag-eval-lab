"""Retrieval and answer metrics.

Retrieval metrics are computed over *evidence passages*, not chunk ids, so two chunks
that both contain the same passage count once (otherwise overlapping fixed-size
windows would inflate recall and nDCG).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from rag_eval_lab.documents import Chunk
from rag_eval_lab.evaluation.dataset import Evidence, covers
from rag_eval_lab.generation.base import CITATION_RE, Answer, strip_citations
from rag_eval_lab.text import overlap_ratio, raw_tokens, tokenize


def coverage_by_rank(chunks: Sequence[Chunk], evidence: Sequence[Evidence]) -> list[set[int]]:
    """For each ranked chunk, the indices of evidence passages it covers."""
    return [{i for i, ev in enumerate(evidence) if covers(c, ev)} for c in chunks]


def recall_at_k(cov: list[set[int]], n_evidence: int, k: int) -> float:
    if n_evidence == 0:
        return 0.0
    return len(set().union(*cov[:k])) / n_evidence if cov[:k] else 0.0


def hit_at_k(cov: list[set[int]], k: int) -> float:
    return 1.0 if any(cov[:k]) else 0.0


def reciprocal_rank(cov: list[set[int]]) -> float:
    for rank, found in enumerate(cov, start=1):
        if found:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(cov: list[set[int]], n_evidence: int, k: int) -> float:
    """Binary-gain nDCG where a chunk only gains if it covers a not-yet-seen passage."""
    if n_evidence == 0:
        return 0.0
    seen: set[int] = set()
    dcg = 0.0
    for rank, found in enumerate(cov[:k], start=1):
        new = found - seen
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_evidence, k) + 1))
    return dcg / ideal


def _answer_tokens(text: str) -> list[str]:
    return tokenize(strip_citations(text))


def exact_match(prediction: str, reference: str) -> float:
    norm = lambda s: " ".join(raw_tokens(strip_citations(s)))  # noqa: E731
    return 1.0 if reference and norm(prediction) == norm(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(_answer_tokens(prediction)), Counter(_answer_tokens(reference))
    common = sum((pred & ref).values())
    if not pred or not ref or common == 0:
        return 0.0
    precision, recall = common / sum(pred.values()), common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def citation_precision(answer: Answer, relevant_ids: set[str]) -> float | None:
    """Share of cited chunks that cover golden evidence. ``None`` if nothing was cited."""
    if not answer.citations:
        return None
    return sum(1 for c in answer.citations if c in relevant_ids) / len(answer.citations)


def claims(text: str) -> list[tuple[str, list[str]]]:
    """Split an answer into (claim text, citation ids) pairs.

    A claim is the text before a run of citation markers; trailing uncited text becomes
    a claim with no citations.
    """
    out: list[tuple[str, list[str]]] = []
    pos = 0
    pending: tuple[str, list[str]] | None = None
    for m in CITATION_RE.finditer(text):
        between = text[pos : m.start()].strip(" .,;:\n")
        if between:
            pending = (between, [m.group(1)])
            out.append(pending)
        elif pending is not None and m.group(1) not in pending[1]:
            pending[1].append(m.group(1))
        pos = m.end()
    tail = re.sub(r"^[\s.,;:]+", "", text[pos:]).strip()
    if tail:
        out.append((tail, []))
    return out


SupportFn = Callable[[str, str], bool]


def overlap_support(threshold: float = 0.6) -> SupportFn:
    def supported(claim: str, evidence: str) -> bool:
        return overlap_ratio(tokenize(claim), tokenize(evidence)) >= threshold

    return supported


def faithfulness(
    answer: Answer, contexts: dict[str, Chunk], support: SupportFn | None = None
) -> float | None:
    """Share of claims supported by the chunks they cite.

    Uncited claims are checked against all retrieved context (and usually fail, which is
    the point). Default support is token overlap >= 0.6; pass an LLM judge instead.
    """
    if answer.abstained:
        return None
    check = support or overlap_support()
    parts = claims(answer.text)
    if not parts:
        return None
    ok = 0
    for claim, cited in parts:
        ids = [c for c in cited if c in contexts] or list(contexts)
        evidence = "\n".join(contexts[c].text for c in ids)
        ok += 1 if evidence and check(claim, evidence) else 0
    return ok / len(parts)


def mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[idx]
