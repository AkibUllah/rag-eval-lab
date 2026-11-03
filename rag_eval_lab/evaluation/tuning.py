"""Threshold sweeps that may only see the dev split.

Choosing a setting on the questions it is later reported on makes the reported number
optimistic, so ``sweep_min_coverage`` refuses any question that is in the test split or
has no split at all (an unassigned question may end up in test).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval_lab.config import Settings
from rag_eval_lab.documents import Document
from rag_eval_lab.evaluation.dataset import GoldenItem, GoldenSetError, select_split
from rag_eval_lab.evaluation.runner import EvalRun, run_eval
from rag_eval_lab.generation.extractive import ExtractiveAnswerer


class TestSplitLeakError(GoldenSetError):
    """A tuning run was asked to look at test (or unassigned) questions."""

    __test__ = False  # not a pytest test class despite the name


def ensure_tunable(items: list[GoldenItem]) -> None:
    leaked = [i.id for i in items if i.split != "dev"]
    if leaked:
        raise TestSplitLeakError(
            f"refusing to tune on {len(leaked)} questions outside the dev split "
            f"(first: {leaked[0]}); tune with --split dev and report on test"
        )


@dataclass
class SweepPoint:
    min_coverage: float
    run: EvalRun

    @property
    def balanced_accuracy(self) -> float | None:
        """Mean of answered rate (answerable) and abstention accuracy (unanswerable)."""
        parts = [self.run.metrics["answered_rate"], self.run.metrics["abstain_accuracy"]]
        present = [p for p in parts if p is not None]
        return sum(present) / len(present) if present else None


def sweep_min_coverage(
    docs: list[Document],
    golden: list[GoldenItem],
    settings: Settings,
    chunker: str,
    retriever: str,
    thresholds: list[float],
    k: int = 5,
    split: str = "dev",
) -> tuple[list[SweepPoint], SweepPoint]:
    """Evaluate the extractive answerer at each threshold; return all points and the best.

    Best = highest balanced accuracy, then answer F1, then the lower threshold.
    """
    if not thresholds:
        raise ValueError("no thresholds to sweep")
    ensure_tunable(select_split(golden, split))
    points = [
        SweepPoint(
            t,
            run_eval(
                docs,
                golden,
                settings,
                chunker,
                retriever,
                k,
                generator=ExtractiveAnswerer(min_coverage=t),
                split=split,
            ),
        )
        for t in sorted(set(thresholds))
    ]
    best = max(
        points,
        key=lambda p: (p.balanced_accuracy or 0.0, p.run.metrics["f1"] or 0.0, -p.min_coverage),
    )
    return points, best


def sweep_table(points: list[SweepPoint], best: SweepPoint) -> str:
    lines = [
        "| min_coverage | Answered | Abstain acc | Balanced | Answer F1 | Cit. prec |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for p in points:
        mt = p.run.metrics
        cells = [
            f"{p.min_coverage:g}",
            *(
                "–" if v is None else f"{v:.3f}"
                for v in (
                    mt["answered_rate"],
                    mt["abstain_accuracy"],
                    p.balanced_accuracy,
                    mt["f1"],
                    mt["citation_precision"],
                )
            ),
        ]
        if p is best:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
