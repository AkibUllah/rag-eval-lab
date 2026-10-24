"""``rag-eval`` command line: run, compare, ask, export-index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_eval_lab.chunking import CHUNKERS, chunk_documents, get_chunker
from rag_eval_lab.config import Settings
from rag_eval_lab.documents import load_directory
from rag_eval_lab.evaluation.dataset import load_golden
from rag_eval_lab.evaluation.runner import (
    compare,
    markdown_table,
    run_eval,
    save_comparison,
    save_run,
    tag_table,
)
from rag_eval_lab.generation.extractive import ExtractiveAnswerer
from rag_eval_lab.generation.llm import LLMJudge, build_provider
from rag_eval_lab.pipeline import build_pipeline
from rag_eval_lab.retrieval import RETRIEVERS
from rag_eval_lab.text import BN_STOPWORDS, BN_SUFFIXES, EN_STOPWORDS, tokenize


def _csv(value: str, allowed: tuple[str, ...] | list[str]) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown: {', '.join(bad)} (choose from {', '.join(allowed)})"
        )
    return items


def _settings(args: argparse.Namespace) -> Settings:
    overrides = {}
    if getattr(args, "corpus", None):
        overrides["corpus_dir"] = Path(args.corpus)
    if getattr(args, "golden", None):
        overrides["golden_path"] = Path(args.golden)
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = Path(args.runs_dir)
    return Settings(**overrides)


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    docs = load_directory(settings.corpus_dir)
    golden = load_golden(settings.golden_path)
    support = None
    if args.judge:
        if settings.llm_provider == "fake":
            print("--judge needs RAG_LLM_PROVIDER set to a real provider", file=sys.stderr)
            return 2
        provider = build_provider(
            settings.llm_provider, settings.llm_url(), settings.llm_model, settings.llm_api_key
        )
        support = LLMJudge(provider).supported
    generator = None
    if args.min_coverage is not None:
        if settings.llm_provider != "fake":
            print("--min-coverage applies to the offline extractive answerer only", file=sys.stderr)
            return 2
        generator = ExtractiveAnswerer(min_coverage=args.min_coverage)
    run = run_eval(
        docs,
        golden,
        settings,
        args.chunker,
        args.retriever,
        args.k,
        generator=generator,
        support=support,
    )
    print(markdown_table([run]))
    mt = run.metrics
    print(
        f"\nanswered {mt['answered_rate'] or 0:.1%} of answerable questions, "
        f"abstained on {mt['abstain_accuracy'] or 0:.1%} of unanswerable ones "
        f"(generator: {run.generator})\n"
    )
    print(tag_table(run))
    if args.show_misses:
        misses = run.misses()
        print(f"\nMisses (no evidence in top {run.k}): {len(misses)}")
        for q in misses:
            print(f"  {q.id} {q.question}  ->  {', '.join(q.retrieved[:3])}")
    if not args.no_save:
        print(f"\nsaved {save_run(run, settings.runs_dir)}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    settings = _settings(args)
    docs = load_directory(settings.corpus_dir)
    golden = load_golden(settings.golden_path)
    runs = compare(docs, golden, settings, args.chunkers, args.retrievers, args.k)
    table = markdown_table(runs)
    print(f"{len(docs)} documents, {len(golden)} questions, k={args.k}\n")
    print(table)
    best = max(runs, key=lambda r: (r.metrics["ndcg@k"] or 0, r.metrics["recall@k"] or 0))
    print(f"\nbest by nDCG@{args.k}: {best.label}\n")
    print(tag_table(best))
    json_path, md_path = save_comparison(runs, settings.runs_dir)
    print(f"\nsaved {json_path} and {md_path}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipe = build_pipeline(settings, args.chunker, args.retriever)
    pipe.ingest(load_directory(settings.corpus_dir))
    result = pipe.ask(args.question, args.k)
    print(result.answer.text)
    print()
    for rank, hit in enumerate(result.hits, start=1):
        snippet = " ".join(hit.chunk.text.split())[:110]
        print(f"{rank}. [{hit.chunk.id}] {hit.score:.4f}  {snippet}")
    return 0


def cmd_export_index(args: argparse.Namespace) -> int:
    """Pre-tokenised chunks for the browser demo (client-side BM25 in web/bm25.js)."""
    settings = _settings(args)
    docs = load_directory(settings.corpus_dir)
    chunks = chunk_documents(docs, get_chunker(args.chunker))
    payload = {
        "chunker": args.chunker,
        "lexicon": {
            "en_stopwords": sorted(EN_STOPWORDS),
            "bn_stopwords": sorted(BN_STOPWORDS),
            "bn_suffixes": list(BN_SUFFIXES),
        },
        "documents": [{"id": d.id, "title": d.title, "lang": d.lang} for d in docs],
        "chunks": [
            {
                "id": c.id,
                "doc": c.doc_id,
                "title": c.title,
                "heading": c.heading,
                "text": c.text,
                "terms": tokenize(c.search_text),
            }
            for c in chunks
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(chunks)} chunks from {len(docs)} documents)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", help="corpus directory (default: data/corpus)")
    common.add_argument("--golden", help="golden set JSONL (default: data/golden.jsonl)")
    common.add_argument("--runs-dir", help="where run reports are written (default: runs/)")
    common.add_argument("--k", type=int, default=5, help="retrieval depth (default 5)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="evaluate one configuration")
    run.add_argument("--chunker", choices=list(CHUNKERS), default="recursive")
    run.add_argument("--retriever", choices=RETRIEVERS, default="hybrid-rerank")
    run.add_argument("--judge", action="store_true", help="LLM-judge faithfulness (needs provider)")
    run.add_argument(
        "--min-coverage", type=float, help="abstention threshold of the extractive answerer"
    )
    run.add_argument("--show-misses", action="store_true", help="list questions with no hit")
    run.add_argument("--no-save", action="store_true", help="do not write a run report")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", parents=[common], help="evaluate a grid, print a table")
    cmp_.add_argument("--chunkers", type=lambda v: _csv(v, list(CHUNKERS)), default=list(CHUNKERS))
    cmp_.add_argument("--retrievers", type=lambda v: _csv(v, RETRIEVERS), default=list(RETRIEVERS))
    cmp_.add_argument("--markdown", help="also write the table to this file")
    cmp_.add_argument("--json", help="also write the comparison JSON to this file")
    cmp_.set_defaults(func=cmd_compare)

    ask = sub.add_parser("ask", parents=[common], help="answer one question from the corpus")
    ask.add_argument("question")
    ask.add_argument("--chunker", choices=list(CHUNKERS), default="recursive")
    ask.add_argument("--retriever", choices=RETRIEVERS, default="hybrid-rerank")
    ask.set_defaults(func=cmd_ask)

    exp = sub.add_parser("export-index", parents=[common], help="write the browser demo index")
    exp.add_argument("output")
    exp.add_argument("--chunker", choices=list(CHUNKERS), default="recursive")
    exp.set_defaults(func=cmd_export_index)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
