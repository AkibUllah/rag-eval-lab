from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_lab.config import PROJECT_ROOT, Settings
from rag_eval_lab.documents import Document, load_directory, make_document
from rag_eval_lab.evaluation.dataset import load_golden

CORPUS = PROJECT_ROOT / "data" / "corpus"
GOLDEN = PROJECT_ROOT / "data" / "golden.jsonl"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, app_env="test", runs_dir=tmp_path / "runs", log_level="WARNING")


@pytest.fixture(scope="session")
def corpus() -> list[Document]:
    return load_directory(CORPUS)


@pytest.fixture(scope="session")
def golden():
    return load_golden(GOLDEN)


@pytest.fixture
def small_doc() -> Document:
    raw = """---
title: Cash out charges
lang: en
---
# Cash out charges

Cash out converts wallet balance to cash. The charge depends on the channel.

## Agents

At an agent using the app the charge is 1.39%. Using USSD the charge is 1.49%.

## ATM

At a partner bank ATM the charge is 1.5% with a minimum of BDT 10.
"""
    return make_document(raw, doc_id="cash-out")
