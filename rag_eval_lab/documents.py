"""Document model and loaders for Markdown, plain text and PDF (text layer only)."""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from rag_eval_lab.text import is_bangla

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ID_RE = re.compile(r"[^a-z0-9-]+")


class LoaderError(ValueError):
    """Raised when a file cannot be turned into a document."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    lang: str = "en"
    source: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. ``start``/``end`` are character offsets into the document text.

    ``text`` is what the generator sees (for sentence-window chunks: the window), while
    ``search_text`` is what the retrievers index (title + heading + the core passage).
    """

    doc_id: str
    index: int
    text: str
    start: int
    end: int
    search_text: str
    title: str = ""
    heading: str = ""
    lang: str = "en"

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def slugify(value: str) -> str:
    slug = _ID_RE.sub("-", value.lower()).strip("-")
    return slug or "document"


def clean_text(text: str) -> str:
    """NFC, unified newlines, trailing spaces stripped. Offsets are computed after this."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip("\n") + "\n"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Minimal ``key: value`` front matter (no YAML dependency needed for flat metadata)."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end() :]


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def make_document(
    raw: str,
    *,
    doc_id: str,
    title: str | None = None,
    lang: str | None = None,
    source: str = "",
) -> Document:
    meta, body = parse_front_matter(raw)
    text = clean_text(body)
    if not text.strip():
        raise LoaderError(f"document {doc_id!r} is empty")
    resolved_title = title or meta.pop("title", None) or _first_heading(text) or doc_id
    resolved_lang = lang or meta.pop("lang", None) or ("bn" if is_bangla(text) else "en")
    meta.pop("title", None)
    meta.pop("lang", None)
    return Document(
        id=slugify(meta.pop("id", doc_id)),
        title=resolved_title,
        text=text,
        lang=resolved_lang,
        source=source,
        metadata=meta,
    )


def pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise LoaderError("PDF support needs the optional extra: pip install '.[pdf]'") from exc
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def load_bytes(name: str, data: bytes) -> Document:
    """Turn an uploaded file into a document, choosing the loader by extension."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise LoaderError(f"unsupported file type {suffix or '(none)'}; use .md, .txt or .pdf")
    if suffix == ".pdf":
        raw = pdf_to_text(data)
    else:
        try:
            raw = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise LoaderError(f"{name} is not valid UTF-8") from exc
    return make_document(raw, doc_id=slugify(Path(name).stem), source=name)


def load_path(path: str | Path) -> Document:
    p = Path(path)
    return load_bytes(p.name, p.read_bytes())


def load_directory(directory: str | Path) -> list[Document]:
    root = Path(directory)
    if not root.is_dir():
        raise LoaderError(f"corpus directory not found: {root}")
    docs = [
        load_path(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    ids = [d.id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise LoaderError(f"duplicate document ids: {', '.join(sorted(dupes))}")
    return docs
