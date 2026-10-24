import json

import pytest

from rag_eval_lab.evaluation.dataset import (
    Evidence,
    GoldenItem,
    GoldenSetError,
    load_golden,
    resolve,
)
from rag_eval_lab.evaluation.runner import (
    compare,
    latest,
    markdown_table,
    run_eval,
    save_comparison,
    save_run,
    tag_table,
)


def test_golden_set_shape(golden):
    assert len(golden) == 70
    assert sum(1 for g in golden if not g.answerable) == 4
    assert sum(1 for g in golden if g.lang == "bn") == 12
    assert len({g.id for g in golden}) == len(golden)


def test_every_golden_evidence_resolves_to_a_span(corpus, golden):
    resolved = resolve(golden, corpus)
    by_id = {d.id: d for d in corpus}
    for item in resolved:
        for ev in item.evidence:
            assert by_id[ev.doc].text[ev.start : ev.end] == ev.text


def test_resolve_fails_loudly_on_drift(corpus):
    bad = [GoldenItem("x", "q", [Evidence("cash-in", "this sentence is not in the doc")])]
    with pytest.raises(GoldenSetError, match="evidence not found"):
        resolve(bad, corpus)
    with pytest.raises(GoldenSetError, match="unknown document"):
        resolve([GoldenItem("y", "q", [Evidence("nope", "x")])], corpus)


def test_load_golden_rejects_bad_lines(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"id": "a", "question": "q"}\n{"id": "a", "question": "q2"}\n')
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden(path)
    path.write_text("{not json}\n")
    with pytest.raises(GoldenSetError, match="invalid"):
        load_golden(path)


def test_run_eval_produces_sane_bounded_metrics(corpus, golden, settings):
    run = run_eval(corpus, golden, settings, "recursive", "bm25", k=5)
    for key in ("recall@k", "mrr", "ndcg@k", "hit@1", "hit@k", "citation_precision", "f1"):
        assert 0.0 <= run.metrics[key] <= 1.0
    # Regression guard: BM25 over section chunks finds evidence for >= 90% of questions.
    assert run.metrics["hit@k"] >= 0.9
    assert run.metrics["questions"] == 70
    assert {"bn", "en", "lexical", "paraphrase"} <= set(run.by_tag)
    assert all(len(q.retrieved) <= 5 for q in run.questions)


def test_markdown_table_marks_best_values(corpus, golden, settings):
    runs = compare(corpus, golden[:20], settings, ["recursive"], ["bm25", "dense"], k=3)
    table = markdown_table(runs)
    lines = table.splitlines()
    assert lines[0].startswith("| Chunker | Retriever | Chunks | Recall@3")
    assert len(lines) == 4 and "**" in table
    assert "| Slice |" in tag_table(runs[0])


def test_reports_are_saved_and_latest_is_found(corpus, golden, settings):
    assert latest(settings.runs_dir, "run") is None
    run = run_eval(corpus, golden[:5], settings, "sentence", "dense", k=3)
    path = save_run(run, settings.runs_dir)
    data = json.loads(path.read_text())
    assert data["chunker"] == "sentence" and len(data["questions"]) == 5
    assert latest(settings.runs_dir, "run")["retriever"] == "dense"
    json_path, md_path = save_comparison([run], settings.runs_dir)
    assert latest(settings.runs_dir, "compare")["runs"][0]["chunker"] == "sentence"
    assert md_path.read_text().startswith("| Chunker")
