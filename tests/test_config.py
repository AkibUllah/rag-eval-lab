import pytest

from rag_eval_lab.config import Settings
from rag_eval_lab.pipeline import build_generator, build_pipeline, build_store_factory


@pytest.mark.parametrize(
    "raw",
    [
        "postgres://u:p@h:5432/db",
        "postgresql+psycopg://u:p@h:5432/db",
        "postgresql://u:p@h:5432/db",
    ],
)
def test_database_url_is_normalised_for_psycopg(raw):
    assert Settings(_env_file=None, database_url=raw).database_url == "postgresql://u:p@h:5432/db"


def test_defaults_are_offline():
    s = Settings(_env_file=None)
    assert (s.llm_provider, s.embeddings_provider, s.vector_store) == ("fake", "hashed", "memory")
    assert build_generator(s).name == "extractive"
    assert s.llm_url() == ""


def test_provider_urls_default_per_provider(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("RAG_LLM_MODEL", "test-model")
    s = Settings(_env_file=None)
    assert s.llm_url() == "https://api.anthropic.com"
    assert build_generator(s).name == "llm:anthropic"


def test_pgvector_requires_database_url():
    s = Settings(_env_file=None, vector_store="pgvector")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        build_store_factory(s)
    pipe = build_pipeline(Settings(_env_file=None), "fixed", "bm25")
    assert pipe.chunker.name == "fixed" and pipe.retriever.name == "bm25"
