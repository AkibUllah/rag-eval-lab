"""Vector store protocol with an in-memory numpy store and an optional pgvector adapter."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class VectorStore(Protocol):
    def clear(self) -> None: ...

    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class InMemoryVectorStore:
    """Brute-force cosine similarity over L2-normalised rows; fine to ~100k chunks."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids = []
        self._matrix = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._ids.extend(ids)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None or k <= 0:
            return []
        sims = self._matrix @ np.asarray(vector, dtype=np.float32)
        k = min(k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(self._ids[i], float(sims[i])) for i in top]


_TABLE_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in np.asarray(vector, dtype=np.float32)) + "]"


class PgVectorStore:
    """PostgreSQL + pgvector. Needs ``pip install '.[pgvector]'`` and the ``vector`` extension.

    The table is (re)created on the first ``add`` after ``clear`` with the dimension of
    the incoming vectors, so switching embedders never leaves a mismatched column.
    Cosine distance (``<=>``) is converted to similarity. ``connect`` is injectable so
    the SQL can be exercised without a database in tests.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "rag_chunks",
        connect: Callable[[str], Any] | None = None,
    ) -> None:
        if not _TABLE_RE.match(table):
            raise ValueError(f"invalid table name {table!r}")
        self.dsn = dsn
        self.table = table
        self._connect = connect or self._default_connect
        self._conn: Any = None
        self._ready = False

    @staticmethod
    def _default_connect(dsn: str) -> Any:  # pragma: no cover - needs a database
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("pgvector store needs: pip install '.[pgvector]'") from exc
        return psycopg.connect(dsn, autocommit=True)

    @property
    def conn(self) -> Any:
        if self._conn is None:
            self._conn = self._connect(self.dsn)
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        return self._conn

    def clear(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
        self._ready = False

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if not ids:
            return
        dim = int(np.asarray(vectors).shape[1])
        rows = [(i, _vector_literal(v)) for i, v in zip(ids, vectors, strict=True)]
        with self.conn.cursor() as cur:
            if not self._ready:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.table} "
                    f"(id text PRIMARY KEY, embedding vector({dim}) NOT NULL)"
                )
                self._ready = True
            cur.executemany(
                f"INSERT INTO {self.table} (id, embedding) VALUES (%s, %s::vector) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                rows,
            )

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ready or k <= 0:
            return []
        literal = _vector_literal(vector)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) AS score FROM {self.table} "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (literal, literal, int(k)),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]
