import json

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
