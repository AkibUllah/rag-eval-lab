"""Bilingual (English + Bangla) text normalisation, tokenisation and sentence splitting.

Everything that turns text into terms lives here so the indexers, the metrics and the
browser demo (``web/bm25.js`` mirrors this module) agree on what a "token" is.

Bangla needs special care: vowel signs (``া``, ``ি`` ...) and the hasanta are Unicode
combining marks, so ``\\w`` alone would split ``টাকা`` into pieces. Tokens are therefore
matched with an explicit Bengali block range, digits are folded to ASCII so ``৫,০০০``
and ``5,000`` produce the same term, and a light suffix stripper removes the most common
inflections (``-এর``, ``-কে``, ``-টি``, ``-গুলো`` ...).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)

# A token is a run of ASCII letters/digits or Bengali-block code points (letters, vowel
# signs, hasanta, nukta), excluding the currency signs ৲/৳. Digit-group separators are
# removed beforehand, for both 25,000 and the South Asian 1,00,000 grouping.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0980-\u09f1\u09f4-\u09ff]+")
_NUM_SEP_RE = re.compile(r"(?<=\d),(?=\d{2,3}\b)")
_BANGLA_RE = re.compile(r"[\u0980-\u09ff]")
_SENTENCE_RE = re.compile(r"(?<=[.!?\u0964])\s+|\n\s*\n|\n(?=\s*(?:[-*]\s|\d+\.\s|#))")

EN_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being below
    between both but by can could did do does doing during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me
    more most my no nor not now of on once only or other our ours over own same she
    should so some such than that the their theirs them then there these they this those
    through to too under until very was we were what when where which while who whom why
    will with would you your yours i'm can't don't much many need want get ever still
    please tell know
    """.split()
)

BN_STOPWORDS = frozenset(
    unicodedata.normalize("NFC", w)
    for w in """
    এবং ও বা কি কী কত কোন কোনো কীভাবে কিভাবে কেন কখন কোথায় যে যদি তবে না নয় হয় হবে হলে
    হয়ে করে করা করতে করুন করবেন আমি আমার আমাদের আপনি আপনার আপনাকে তিনি তার এই ওই সেই এটি
    একটি জন্য থেকে পর্যন্ত মধ্যে দিয়ে নিয়ে সাথে সঙ্গে আছে নেই যায় যাবে পারে পারবেন পারি
    """.split()
)

# Longest first; applied once per token, only when a stem of >= 2 code points remains.
BN_SUFFIXES = tuple(
    sorted(
        (
            unicodedata.normalize("NFC", s)
            for s in """
            গুলোতে গুলোর গুলো গুলি দেরকে দের েরা রা টির টিতে টি টার টা কে তে য় েতে ের র ে
            """.split()
        ),
        key=len,
        reverse=True,
    )
)


def normalize(text: str) -> str:
    """Canonical form used before tokenising: NFC, no zero-width marks, ASCII digits."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    return text.translate(BANGLA_DIGITS)


def is_bangla(text: str, threshold: float = 0.3) -> bool:
    """True when at least ``threshold`` of the letters are in the Bengali block."""
    letters = [c for c in text if c.isalpha() or _BANGLA_RE.match(c)]
    if not letters:
        return False
    bn = sum(1 for c in letters if _BANGLA_RE.match(c))
    return bn / len(letters) >= threshold


def stem_en(token: str) -> str:
    """Tiny suffix stripper: enough to match plurals and -ing/-ed forms, nothing clever."""
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and token[-3] in "sxz" and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def stem_bn(token: str) -> str:
    for suffix in BN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token


def raw_tokens(text: str) -> list[str]:
    """Lower-cased word tokens with digits folded, before stopword removal and stemming."""
    text = _NUM_SEP_RE.sub("", normalize(text).lower())
    return _TOKEN_RE.findall(text)


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Search terms for BM25, the hashed vectoriser and the overlap metrics."""
    out: list[str] = []
    for tok in raw_tokens(text):
        bangla = tok[0] >= "\u0980"
        if drop_stopwords and (tok in (BN_STOPWORDS if bangla else EN_STOPWORDS)):
            continue
        if stem:
            tok = stem_bn(tok) if bangla else stem_en(tok)
        out.append(tok)
    return out


def split_sentences(text: str) -> list[str]:
    """Split on ., !, ?, the Bangla danda (।), blank lines and list/heading line starts."""
    parts = (p.strip() for p in _SENTENCE_RE.split(text))
    return [p for p in parts if p]


def sentence_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Character spans (start, end) of :func:`split_sentences` output within ``text``."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, pos)
        if start < 0:  # pragma: no cover - split output always comes from text
            continue
        end = start + len(sentence)
        spans.append((offset + start, offset + end))
        pos = end
    return spans


def word_count(text: str) -> int:
    return len(raw_tokens(text))


def overlap_ratio(tokens: Iterable[str], reference: Iterable[str]) -> float:
    """Share of distinct ``tokens`` that also appear in ``reference`` (0 when empty)."""
    toks = set(tokens)
    if not toks:
        return 0.0
    return len(toks & set(reference)) / len(toks)
