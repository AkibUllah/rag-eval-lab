import json

import pytest
from fastapi.testclient import TestClient

from rag_eval_lab.api.main import create_app
from rag_eval_lab.evaluation.runner import run_eval, save_run


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 29
    assert body["llm_provider"] == "fake"
    assert body["retriever"] == "hybrid-rerank"


def test_documents_lists_the_corpus(client):
    docs = client.get("/documents").json()
    assert {"bn-cash-out", "cash-out-fees"} <= {d["id"] for d in docs}


def test_ask_returns_answer_with_cited_hits(client):
    res = client.post("/ask", json={"question": "What happens after three wrong PIN attempts?"})
    assert res.status_code == 200
    body = res.json()
    assert not body["abstained"]
    assert "24 hours" in body["answer"]
    cited = [h for h in body["hits"] if h["cited"]]
    assert cited and all(h["id"] in body["citations"] for h in cited)
    assert body["hits"][0]["doc"] == "pin-security"
    assert res.headers["x-request-id"]


def test_ask_with_overrides_and_validation(client):
    body = client.post(
        "/ask", json={"question": "ক্যাশ আউট চার্জ", "k": 2, "chunker": "sentence", "retriever": "bm25"}
    ).json()
    assert body["chunker"] == "sentence" and body["retriever"] == "bm25"
    assert len(body["hits"]) == 2 and body["hits"][0]["doc"] == "bn-cash-out"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "retriever": "colbert"}).status_code == 422


def test_ingest_json_makes_new_content_answerable(client):
    res = client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "title": "Eid holiday hours",
                    "text": "# Eid holiday hours\n\n"
                    "Service points are closed for three days during Eid.",
                }
            ]
        },
    )
    assert res.status_code == 200
    assert res.json() == {"ingested": ["eid-holiday-hours"], "documents_total": 30}
    body = client.post("/ask", json={"question": "Are service points closed during Eid?"}).json()
    assert body["citations"][0].startswith("eid-holiday-hours#")


def test_ingest_replace_and_empty_kb(client):
    client.post(
        "/ingest", json={"replace": True, "documents": [{"id": "only", "text": "Only doc."}]}
    )
    assert client.get("/health").json()["documents"] == 1


def test_ingest_files_accepts_markdown_and_rejects_unknown_types(client):
    files = [
        ("files", ("agent-faq.md", b"# Agent FAQ\n\nAgents cannot reset PINs.", "text/markdown"))
    ]
    res = client.post("/ingest/files", files=files)
    assert res.status_code == 200 and res.json()["ingested"] == ["agent-faq"]
    bad = client.post(
        "/ingest/files", files=[("files", ("x.docx", b"PK..", "application/octet-stream"))]
    )
    assert bad.status_code == 422


def test_ask_without_documents_is_a_conflict(settings):
    settings.autoload_corpus = False
    with TestClient(create_app(settings)) as c:
        assert c.post("/ask", json={"question": "anything"}).status_code == 409


def test_eval_endpoints(client, settings, corpus, golden):
    assert client.get("/eval/latest").status_code == 404
    assert client.get("/eval/latest/compare").status_code == 404
    save_run(run_eval(corpus, golden[:3], settings, "recursive", "bm25", k=3), settings.runs_dir)
    body = client.get("/eval/latest").json()
    assert body["chunker"] == "recursive" and len(body["questions"]) == 3


def test_static_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "RAG Eval Lab" in res.text
    assert client.get("/bm25.js").status_code == 200
    assert json.loads(client.get("/openapi.json").text)["info"]["title"] == "RAG Eval Lab"
