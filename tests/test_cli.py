import json

import pytest

from rag_eval_lab.cli import main


def test_run_prints_table_and_misses(capsys, tmp_path):
    assert (
        main(
            [
                "run",
                "--chunker",
                "recursive",
                "--retriever",
                "bm25",
                "--no-save",
                "--show-misses",
                "--runs-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "| recursive | bm25 |" in out and "Misses" in out and "| Slice |" in out


def test_compare_writes_markdown_and_json(capsys, tmp_path):
    md, js = tmp_path / "table.md", tmp_path / "cmp.json"
    assert (
        main(
            [
                "compare",
                "--chunkers",
                "recursive,sentence",
                "--retrievers",
                "bm25",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--markdown",
                str(md),
                "--json",
                str(js),
            ]
        )
        == 0
    )
    assert md.read_text().count("\n") == 4
    assert len(json.loads(js.read_text())["runs"]) == 2
    assert "best by nDCG@5" in capsys.readouterr().out


def test_export_index_contains_lexicon_and_terms(tmp_path):
    out = tmp_path / "index.json"
    assert main(["export-index", str(out), "--chunker", "recursive"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["chunker"] == "recursive" and len(data["documents"]) == 29
    assert {"en_stopwords", "bn_stopwords", "bn_suffixes"} <= set(data["lexicon"])
    assert all(c["terms"] for c in data["chunks"])


def test_ask_prints_answer_and_ranked_hits(capsys):
    assert main(["ask", "How many cards can I link?", "--retriever", "bm25"]) == 0
    out = capsys.readouterr().out
    assert "Up to 3 cards" in out and "1. [card-linking#" in out


def test_run_on_test_split_reports_split_and_intervals(capsys, tmp_path):
    args = ["run", "--retriever", "bm25", "--split", "test", "--runs-dir", str(tmp_path)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("split: test (21 questions), k=5")
    assert "| Metric | Value | 95% CI |" in out
    saved = json.loads(next(tmp_path.glob("run-*.json")).read_text(encoding="utf-8"))
    assert saved["split"] == "test" and len(saved["questions"]) == 21


def test_compare_on_dev_split_states_it(capsys, tmp_path):
    js = tmp_path / "cmp.json"
    args = ["compare", "--chunkers", "recursive", "--retrievers", "bm25,dense", "--split", "dev"]
    assert main([*args, "--runs-dir", str(tmp_path / "runs"), "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "29 documents, 49 questions (split: dev), k=5" in out
    assert "choose configurations with --split dev" not in out
    assert json.loads(js.read_text())["split"] == "dev"
    args = ["compare", "--chunkers", "recursive", "--retrievers", "bm25", "--split", "test"]
    assert main([*args, "--runs-dir", str(tmp_path / "runs")]) == 0
    assert "choose configurations with --split dev" in capsys.readouterr().out


def test_split_flags_fail_cleanly_on_unsplit_golden_set(capsys, tmp_path):
    golden = tmp_path / "golden.jsonl"
    row = {"id": "a", "question": "How many cards can I link?", "relevant": []}
    golden.write_text(json.dumps(row) + "\n", encoding="utf-8")
    common = ["--golden", str(golden), "--runs-dir", str(tmp_path), "--split", "dev"]
    assert main(["run", "--no-save", *common]) == 2
    assert "have no split" in capsys.readouterr().err
    assert main(["compare", "--retrievers", "bm25", *common]) == 2
    assert "have no split" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["run", "--split", "holdout"])
