#!/usr/bin/env python3
"""Assign every golden question to the dev or test split and store it in the file.

    python scripts/split_golden.py                 # place unassigned questions, keep the rest
    python scripts/split_golden.py --check         # exit 1 if any question has no split
    python scripts/split_golden.py --reassign      # recompute every split (moves questions!)

The assignment is stratified and seeded (see rag_eval_lab/evaluation/split.py), so the
same file always produces the same split, and new questions never move existing ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_eval_lab.config import PROJECT_ROOT
from rag_eval_lab.evaluation.dataset import load_golden
from rag_eval_lab.evaluation.split import (
    DEFAULT_SEED,
    TEST_FRACTION,
    assign_splits,
    split_summary,
)


def print_summary(summary: dict) -> None:
    print(f"{'slice':<26}{'dev':>5}{'test':>6}{'test %':>8}")
    for key in sorted(summary, key=lambda k: (k != "all", k)):
        counts = summary[key]
        total = sum(counts.values())
        test = counts.get("test", 0)
        line = f"{key:<26}{counts.get('dev', 0):>5}{test:>6}{test / total:>8.0%}"
        if counts.get("unassigned"):
            line += f"  ({counts['unassigned']} unassigned)"
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data" / "golden.jsonl")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--check", action="store_true", help="only verify, do not write")
    parser.add_argument("--reassign", action="store_true", help="ignore stored splits")
    args = parser.parse_args(argv)

    items = load_golden(args.golden)
    if args.check:
        print_summary(split_summary(items))
        missing = [i.id for i in items if i.split is None]
        if missing:
            print(f"\n{len(missing)} questions have no split: {', '.join(missing)}")
            return 1
        return 0

    splits = assign_splits(items, args.test_fraction, args.seed, keep_existing=not args.reassign)
    lines = []
    for line in args.golden.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["split"] = splits[raw["id"]]
        lines.append(json.dumps(raw, ensure_ascii=False))
    args.golden.write_text("\n".join(lines) + "\n", encoding="utf-8")
    changed = sum(1 for i in items if i.split != splits[i.id])
    print(f"wrote {args.golden}: {changed} of {len(items)} questions assigned\n")
    print_summary(split_summary(load_golden(args.golden)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
