import httpx
import numpy as np
import pytest

from rag_eval_lab.chunking import RecursiveChunker, chunk_documents
from rag_eval_lab.documents import make_document
from rag_eval_lab.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    InMemoryVectorStore,
    LexicalReranker,
    NotIndexedError,
    ScoredChunk,
    build_retriever,
    reciprocal_rank_fusion,
)
from rag_eval_lab.retrieval.embeddings import (
    EmbeddingError,
    HashedTfidfEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
)
from rag_eval_lab.retrieval.store import PgVectorStore


@pytest.fixture(scope="module")
def chunks(corpus):
    return chunk_documents(corpus, RecursiveChunker())


def test_bm25_finds_the_right_section(chunks):
    bm25 = BM25Retriever()
    bm25.index(chunks)
    top = bm25.search("wrong PIN attempts locked", k=3)
    assert top[0].chunk.doc_id == "pin-security"
    assert top[0].score >= top[1].score


def test_bm25_handles_bangla_queries(chunks):
    bm25 = BM25Retriever()
    bm25.index(chunks)
    assert bm25.search("এটিএম থেকে ক্যাশ আউট চার্জ", k=1)[0].chunk.doc_id == "bn-cash-out"


def test_bm25_unknown_terms_and_unindexed():
    with pytest.raises(NotIndexedError):
        BM25Retriever().search("anything")
    bm25 = BM25Retriever()
    bm25.index(RecursiveChunker().chunk(make_document("# A\n\nalpha beta", doc_id="a")))
    assert bm25.search("zzzz") == []


def test_hashed_embedder_is_normalised_and_rewards_shared_ngrams():
    emb = HashedTfidfEmbedder(dim=1024)
    texts = ["refunds are issued by the merchant", "cash out at an agent", "PIN reset steps"]
    emb.fit(texts)
    vecs = emb.embed(texts + ["refunded by merchants"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    sims = vecs[:3] @ vecs[3]
    assert int(np.argmax(sims)) == 0
    assert np.array_equal(emb.embed(["same text"]), emb.embed(["same text"]))
    with pytest.raises(ValueError):
        HashedTfidfEmbedder(dim=1000)


def test_in_memory_store_returns_top_k_in_order():
    store = InMemoryVectorStore()
    store.add(["a", "b", "c"], np.eye(3, dtype=np.float32))
    assert [i for i, _ in store.search(np.array([0.1, 0.9, 0.3]), 2)] == ["b", "c"]
    assert len(store) == 3
    store.clear()
    assert store.search(np.array([1.0, 0, 0]), 2) == []
    with pytest.raises(ValueError):
        store.add(["x"], np.zeros((2, 3)))


def test_dense_retriever_finds_inflected_forms(chunks):
    dense = DenseRetriever()
    dense.index(chunks)
    assert dense.search("dormancy fee", k=3)[0].chunk.doc_id == "dormant-accounts"


def test_reciprocal_rank_fusion_math(small_doc):
    a, b, c = RecursiveChunker().chunk(small_doc)
    fused = reciprocal_rank_fusion(
        [[ScoredChunk(a, 9), ScoredChunk(b, 5)], [ScoredChunk(b, 0.9), ScoredChunk(c, 0.1)]],
        k_rrf=60,
    )
    assert [h.chunk.id for h in fused] == [b.id, a.id, c.id]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)


def test_hybrid_and_rerank_factory(chunks):
    hybrid = build_retriever("hybrid")
    assert isinstance(hybrid, HybridRetriever)
    rerank = build_retriever("hybrid-rerank")
    assert rerank.name == "hybrid-rerank"
    rerank.index(chunks)
    assert rerank.search("government incentive on remittance", k=2)[0].chunk.doc_id == "remittance"
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("colbert")


def test_lexical_reranker_promotes_full_query_coverage(small_doc):
    a, b, c = RecursiveChunker().chunk(small_doc)
    cands = [ScoredChunk(a, 3.0), ScoredChunk(c, 2.0), ScoredChunk(b, 1.0)]
    out = LexicalReranker().rerank("agent app USSD charge", cands, k=3)
    assert out[0].chunk.id == b.id


class _FakeCursor:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))

    def executemany(self, sql, rows):
        self.log.append((sql, list(rows)))

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, rows):
        self.log: list = []
        self.rows = rows

    def cursor(self):
        return _FakeCursor(self.log, self.rows)


def test_pgvector_store_issues_parameterised_sql():
    conn = _FakeConn(rows=[("fees#1", 0.93)])
    store = PgVectorStore("postgresql://x", table="rag_test", connect=lambda dsn: conn)
    assert store.search(np.ones(3), 2) == []  # nothing indexed yet
    store.clear()
    store.add(["fees#1", "fees#2"], np.eye(2, 3, dtype=np.float32))
    assert store.search(np.array([1.0, 0, 0]), 2) == [("fees#1", 0.93)]
    sqls = [sql for sql, _ in conn.log]
    assert sqls[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert "DROP TABLE IF EXISTS rag_test" in sqls
    assert any("vector(3)" in s for s in sqls)
    insert = next(p for s, p in conn.log if s.startswith("INSERT"))
    assert insert[0] == ("fees#1", "[1,0,0]")
    select_sql, params = conn.log[-1]
    assert "<=>" in select_sql and params[2] == 2
    with pytest.raises(ValueError):
        PgVectorStore("postgresql://x", table="bad;drop")


def test_openai_and_ollama_embedders_over_mock_transport():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 2.0]},
                        {"index": 0, "embedding": [3.0, 0.0]},
                    ]
                },
            )
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 1.0], [0.0, 1.0]]})
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    oa = OpenAICompatibleEmbedder("https://api.test/v1", "text-embed", "sk-test", client=client)
    vecs = oa.embed(["a", "b"])
    assert vecs.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    ol = OllamaEmbedder("http://ollama.test", "bge-m3", client=client)
    assert ol.embed(["x", "y"]).shape == (2, 2)
    broken = OllamaEmbedder("http://ollama.test/broken", "m", client=client)
    with pytest.raises(EmbeddingError):
        broken.embed(["x"])
