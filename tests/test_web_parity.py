"""The browser demo re-implements the tokenizer in JS; this keeps the two in lockstep."""

import json
import shutil
import subprocess

import pytest

from rag_eval_lab.config import PROJECT_ROOT
from rag_eval_lab.text import BN_STOPWORDS, BN_SUFFIXES, EN_STOPWORDS, split_sentences, tokenize

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_tokenizer_matches_python_on_the_whole_corpus(corpus, golden):
    texts = (
        [d.text for d in corpus] + [g.question for g in golden] + ["৳১,৫০,০০০ and 25,000 BDT\u200c"]
    )
    payload = {
        "lexicon": {
            "en_stopwords": sorted(EN_STOPWORDS),
            "bn_stopwords": sorted(BN_STOPWORDS),
            "bn_suffixes": list(BN_SUFFIXES),
        },
        "texts": texts,
    }
    proc = subprocess.run(
        [NODE, str(PROJECT_ROOT / "web" / "tests" / "tokenize-cli.mjs")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    js = json.loads(proc.stdout)
    for text, out in zip(texts, js, strict=True):
        assert out["tokens"] == tokenize(text), text[:60]
        assert out["sentences"] == split_sentences(text), text[:60]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_demo_returns_the_same_chunks_and_answers_as_python(tmp_path, corpus, golden):
    from rag_eval_lab.cli import main
    from rag_eval_lab.config import Settings
    from rag_eval_lab.pipeline import build_pipeline

    index = tmp_path / "index.json"
    assert main(["export-index", str(index), "--chunker", "recursive"]) == 0
    questions = [g.question for g in golden]
    proc = subprocess.run(
        [NODE, str(PROJECT_ROOT / "web" / "tests" / "pipeline-cli.mjs"), str(index)],
        input=json.dumps(questions),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    browser = json.loads(proc.stdout)
    pipe = build_pipeline(Settings(_env_file=None), "recursive", "bm25")
    pipe.ingest(corpus)
    for question, js in zip(questions, browser, strict=True):
        result = pipe.ask(question, k=5)
        assert js["ids"] == [h.chunk.id for h in result.hits], question
        assert js["answer"] == result.answer.text, question
