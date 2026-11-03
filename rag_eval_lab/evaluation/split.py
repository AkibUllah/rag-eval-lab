"""Deterministic, stratified dev/test assignment for the golden set.

Questions are grouped into strata (unanswerable, Bangla, and English by multi-document,
multi-evidence, paraphrase, lexical or other wording) so both splits see every kind of
question. Inside a stratum the order comes from a seeded hash of the question id, not
from file position, and splits that are already stored are kept: adding questions only
assigns the new ones, filling whichever split is below its target share, and never moves
an existing question from dev to test or back.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from rag_eval_lab.evaluation.dataset import SPLITS, GoldenItem

DEFAULT_SEED = "rag-eval-lab"
TEST_FRACTION = 0.3


def stratum(item: GoldenItem) -> str:
    if not item.answerable:
        return "unanswerable"
    if item.lang == "bn":
        return "bn"
    for tag in ("multi-doc", "multi-evidence", "paraphrase", "lexical"):
        if tag in item.tags:
            return f"en/{tag}"
    return "en/other"


def split_hash(item_id: str, seed: str = DEFAULT_SEED) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{item_id}".encode()).digest()[:8], "big")


def target_test_size(n: int, test_fraction: float) -> int:
    """Test questions for a stratum of ``n``: ``n * fraction`` rounded half up."""
    return math.floor(n * test_fraction + 0.5)


def assign_splits(
    items: Iterable[GoldenItem],
    test_fraction: float = TEST_FRACTION,
    seed: str = DEFAULT_SEED,
    keep_existing: bool = True,
) -> dict[str, str]:
    """Map every item id to ``dev`` or ``test``.

    With ``keep_existing`` (the default), items that already have a split keep it and
    only unassigned items are placed; the result does not depend on input order.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    groups: dict[str, list[GoldenItem]] = defaultdict(list)
    for item in items:
        groups[stratum(item)].append(item)

    out: dict[str, str] = {}
    for members in groups.values():
        fixed = [i for i in members if keep_existing and i.split in SPLITS]
        new = sorted(
            (i for i in members if not (keep_existing and i.split in SPLITS)),
            key=lambda i: (split_hash(i.id, seed), i.id),
        )
        have = sum(1 for i in fixed if i.split == "test")
        n_test = min(max(target_test_size(len(members), test_fraction) - have, 0), len(new))
        for i in fixed:
            out[i.id] = i.split  # type: ignore[assignment]
        for rank, i in enumerate(new):
            out[i.id] = "test" if rank < n_test else "dev"
    return out


def split_summary(items: Iterable[GoldenItem]) -> dict[str, Counter[str]]:
    """Dev/test counts per stratum, per language and per tag."""
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        split = item.split or "unassigned"
        summary["all"][split] += 1
        summary[f"stratum:{stratum(item)}"][split] += 1
        summary[f"lang:{item.lang}"][split] += 1
        for tag in item.tags:
            summary[f"tag:{tag}"][split] += 1
    return dict(summary)
