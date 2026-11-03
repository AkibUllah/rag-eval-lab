import dataclasses

import pytest

from rag_eval_lab.cli import main
from rag_eval_lab.evaluation import tuning
from rag_eval_lab.evaluation.dataset import GoldenSetError
from rag_eval_lab.evaluation.tuning import (
    TestSplitLeakError,
    ensure_tunable,
    sweep_min_coverage,
    sweep_table,
)


@pytest.fixture(scope="module")
def dev_sample(golden):
    dev = [g for g in golden if g.split == "dev"]
    return [g for g in dev if g.answerable][:12] + [g for g in dev if not g.answerable]


def test_guard_refuses_test_and_unassigned_questions(golden):
    dev = [g for g in golden if g.split == "dev"]
    ensure_tunable(dev)
    with pytest.raises(TestSplitLeakError, match="outside the dev split"):
        ensure_tunable(golden)
    unassigned = [dataclasses.replace(dev[0], split=None)]
    with pytest.raises(TestSplitLeakError, match=f"first: {dev[0].id}"):
        ensure_tunable(unassigned)
    assert issubclass(TestSplitLeakError, GoldenSetError)


def test_sweep_refuses_test_before_running_anything(monkeypatch, corpus, golden, settings):
    def fail(*args, **kwargs):
        raise AssertionError("run_eval must not be called")

    monkeypatch.setattr(tuning, "run_eval", fail)
    for split in ("test", "all"):
        with pytest.raises(TestSplitLeakError):
            sweep_min_coverage(corpus, golden, settings, "recursive", "bm25", [0.4], split=split)
    with pytest.raises(ValueError, match="no thresholds"):
        sweep_min_coverage(corpus, golden, settings, "recursive", "bm25", [])


def test_sweep_orders_points_and_picks_the_balanced_best(corpus, dev_sample, settings):
    points, best = sweep_min_coverage(
        corpus, dev_sample, settings, "recursive", "bm25", [0.6, 0.3, 0.3, 0.9]
    )
    assert [p.min_coverage for p in points] == [0.3, 0.6, 0.9]
    assert all(p.run.split == "dev" and p.run.metrics["questions"] == 15 for p in points)
    answered = [p.run.metrics["answered_rate"] for p in points]
    assert answered == sorted(answered, reverse=True)  # a higher bar answers less
    key = [(p.balanced_accuracy, p.run.metrics["f1"], -p.min_coverage) for p in points]
    assert best is points[key.index(max(key))]
    table = sweep_table(points, best)
    assert table.splitlines()[0].startswith("| min_coverage | Answered | Abstain acc")
    assert table.count("**") == 12  # every cell of the best row, and only it


def test_balanced_accuracy_ignores_missing_halves(corpus, dev_sample, settings):
    answerable_only = [g for g in dev_sample if g.answerable]
    points, best = sweep_min_coverage(
        corpus, answerable_only, settings, "recursive", "bm25", [0.3, 0.9]
    )
    assert best.min_coverage == 0.3
    assert best.balanced_accuracy == best.run.metrics["answered_rate"]


def test_tune_cli_sweeps_dev_and_refuses_test(capsys, monkeypatch):
    assert main(["tune", "--retriever", "bm25", "--min-coverage", "0.4,0.5"]) == 0
    out = capsys.readouterr().out
    assert "on split dev (49 questions)" in out and "rag-eval run --split test" in out
    assert main(["tune", "--retriever", "bm25", "--split", "test"]) == 2
    assert "refusing to tune" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["tune", "--min-coverage", "0.4,abc"])
    with pytest.raises(SystemExit):
        main(["tune", "--min-coverage", "1.5"])
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    assert main(["tune"]) == 2
    assert "extractive answerer only" in capsys.readouterr().err
