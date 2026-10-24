"""Three chunking strategies that all emit character-offset spans into the source text.

Spans are what make the evaluation chunker-independent: the golden set marks evidence
as a passage of the document, and a chunk counts as relevant when its span covers that
passage. Chunk ids (``doc#n``) are therefore free to differ between strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_eval_lab.documents import Chunk, Document
from rag_eval_lab.text import sentence_spans

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


@dataclass(frozen=True)
class Section:
    heading: str  # "Title > Sub heading"
    start: int  # start of the heading line (or 0)
    body_start: int
    end: int


def sections(doc: Document) -> list[Section]:
    """Split a Markdown document into heading-delimited sections with heading paths."""
    text = doc.text
    matches = list(_HEADING_RE.finditer(text))
    out: list[Section] = []
    if not matches or matches[0].start() > 0:
        first_end = matches[0].start() if matches else len(text)
        if text[:first_end].strip():
            out.append(Section(heading="", start=0, body_start=0, end=first_end))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append(
            Section(
                heading=" > ".join(t for _, t in stack),
                start=m.start(),
                body_start=min(m.end() + 1, end),
                end=end,
            )
        )
    return out


def heading_at(secs: list[Section], pos: int) -> str:
    current = ""
    for s in secs:
        if s.start <= pos:
            current = s.heading
        else:
            break
    return current


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _words(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(m.start() + start, m.end() + start) for m in _WORD_RE.finditer(text[start:end])]


def _search_text(doc: Document, heading: str, core: str) -> str:
    """Title + heading path + body. Heading lines inside the body are dropped so a
    section heading is not counted twice."""
    core = "\n".join(line for line in core.splitlines() if not line.lstrip().startswith("#"))
    parts = [doc.title]
    if heading and heading != doc.title:
        parts.append(heading.replace(f"{doc.title} > ", ""))
    parts.append(core)
    return "\n".join(parts)


def _make(doc: Document, index: int, start: int, end: int, heading: str, core: str | None = None):
    text = doc.text[start:end]
    return Chunk(
        doc_id=doc.id,
        index=index,
        text=text,
        start=start,
        end=end,
        search_text=_search_text(doc, heading, core if core is not None else text),
        title=doc.title,
        heading=heading,
        lang=doc.lang,
    )


class FixedSizeChunker:
    """Sliding window of ``size`` words with ``overlap`` words shared between neighbours."""

    name = "fixed"

    def __init__(self, size: int = 80, overlap: int = 20) -> None:
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("require size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        words = _words(doc.text, 0, len(doc.text))
        secs = sections(doc)
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        i = 0
        while i < len(words):
            window = words[i : i + self.size]
            start, end = window[0][0], window[-1][1]
            chunks.append(_make(doc, len(chunks), start, end, heading_at(secs, start)))
            if i + self.size >= len(words):
                break
            i += step
        return chunks


class RecursiveChunker:
    """Structure-aware: sections by heading, then paragraphs, then sentences.

    Pieces are merged greedily up to ``max_words`` without crossing a section boundary,
    so a chunk is always "one topic under one heading".
    """

    name = "recursive"

    def __init__(self, max_words: int = 120) -> None:
        if max_words < 10:
            raise ValueError("max_words must be >= 10")
        self.max_words = max_words

    def _pieces(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Paragraph spans, with oversized paragraphs broken into sentence spans."""
        pieces: list[tuple[int, int]] = []
        pos = start
        bounds = [
            (m.start() + start, m.end() + start) for m in _PARAGRAPH_RE.finditer(text[start:end])
        ]
        for gap_start, gap_end in [*bounds, (end, end)]:
            p_start, p_end = _trim(text, pos, gap_start)
            pos = gap_end
            if p_start >= p_end:
                continue
            if len(_words(text, p_start, p_end)) <= self.max_words:
                pieces.append((p_start, p_end))
            else:
                pieces.extend(sentence_spans(text[p_start:p_end], offset=p_start))
        return pieces

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        chunks: list[Chunk] = []
        for sec in sections(doc):
            s_start, s_end = _trim(text, sec.start, sec.end)
            if s_start >= s_end or (sec.heading and not text[sec.body_start : sec.end].strip()):
                continue  # empty, or a heading with no body (e.g. directly followed by H3)
            if len(_words(text, s_start, s_end)) <= self.max_words:
                chunks.append(_make(doc, len(chunks), s_start, s_end, sec.heading))
                continue
            group_start: int | None = None
            group_end = 0
            count = 0
            for p_start, p_end in self._pieces(text, sec.body_start, sec.end):
                n = len(_words(text, p_start, p_end))
                if group_start is not None and count + n > self.max_words:
                    chunks.append(_make(doc, len(chunks), group_start, group_end, sec.heading))
                    group_start, count = None, 0
                if group_start is None:
                    group_start = p_start
                group_end = p_end
                count += n
            if group_start is not None:
                chunks.append(_make(doc, len(chunks), group_start, group_end, sec.heading))
        return chunks


class SentenceWindowChunker:
    """Index single sentences, hand the generator the sentence plus ``window`` neighbours.

    Retrieval precision comes from matching one sentence; answer context comes from the
    window. Windows never cross a section boundary.
    """

    name = "sentence"

    def __init__(self, window: int = 1) -> None:
        if window < 0:
            raise ValueError("window must be >= 0")
        self.window = window

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        chunks: list[Chunk] = []
        for sec in sections(doc):
            spans = [
                (s, e)
                for s, e in sentence_spans(text[sec.body_start : sec.end], offset=sec.body_start)
                if not text[s:e].startswith("#")
            ]
            for i, (s, e) in enumerate(spans):
                lo = spans[max(0, i - self.window)][0]
                hi = spans[min(len(spans) - 1, i + self.window)][1]
                chunks.append(_make(doc, len(chunks), lo, hi, sec.heading, core=text[s:e]))
        return chunks


CHUNKERS: dict[str, type] = {
    FixedSizeChunker.name: FixedSizeChunker,
    RecursiveChunker.name: RecursiveChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
}


def get_chunker(name: str, **kwargs: int) -> Chunker:
    try:
        return CHUNKERS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; choose from {', '.join(CHUNKERS)}") from None


def chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    return [c for d in docs for c in chunker.chunk(d)]
