"""Settings from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=list)

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    golden_path: Path = PROJECT_ROOT / "data" / "golden.jsonl"
    runs_dir: Path = PROJECT_ROOT / "runs"
    autoload_corpus: bool = True

    chunker: Literal["fixed", "recursive", "sentence"] = "recursive"
    retriever: Literal["bm25", "dense", "hybrid", "hybrid-rerank"] = "hybrid-rerank"
    top_k: int = Field(default=5, ge=1, le=50)

    llm_provider: Literal["fake", "openai", "anthropic", "ollama"] = "fake"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    embeddings_provider: Literal["hashed", "openai", "ollama"] = "hashed"
    embeddings_base_url: str = ""
    embeddings_model: str = ""
    embeddings_api_key: str = ""

    vector_store: Literal["memory", "pgvector"] = "memory"
    database_url: str = Field(
        default="", validation_alias=AliasChoices("DATABASE_URL", "RAG_DATABASE_URL")
    )

    max_upload_bytes: int = 2_000_000

    @field_validator("database_url")
    @classmethod
    def _normalise_dsn(cls, value: str) -> str:
        # Render/Heroku hand out postgres://; SQLAlchemy-style driver suffixes are not
        # understood by psycopg. Both normalise to plain postgresql://.
        for prefix in ("postgres://", "postgresql+psycopg://", "postgresql+asyncpg://"):
            if value.startswith(prefix):
                return "postgresql://" + value[len(prefix) :]
        return value

    def llm_url(self) -> str:
        return self.llm_base_url or DEFAULT_BASE_URLS.get(self.llm_provider, "")

    def embeddings_url(self) -> str:
        return self.embeddings_base_url or DEFAULT_BASE_URLS.get(self.embeddings_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
