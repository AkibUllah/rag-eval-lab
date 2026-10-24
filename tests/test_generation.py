import json

import httpx
import pytest

from rag_eval_lab.chunking import RecursiveChunker
from rag_eval_lab.generation.base import ABSTAIN_TEXT, extract_citations, strip_citations
from rag_eval_lab.generation.extractive import ExtractiveAnswerer
from rag_eval_lab.generation.llm import (
    AnthropicMessages,
    LLMAnswerer,
    LLMError,
    LLMJudge,
    OllamaChat,
    OpenAICompatibleChat,
    ScriptedLLM,
    build_prompt,
    build_provider,
)
from rag_eval_lab.retrieval import ScoredChunk


@pytest.fixture
def hits(small_doc):
    return [ScoredChunk(c, 1.0 - i / 10) for i, c in enumerate(RecursiveChunker().chunk(small_doc))]


def test_extractive_answer_quotes_and_cites_the_best_sentence(hits):
    answer = ExtractiveAnswerer().generate("What is the ATM charge and minimum?", hits)
    assert not answer.abstained
    assert answer.citations == ["cash-out#2"]
    assert answer.text.startswith("At a partner bank ATM the charge is 1.5%")
    assert answer.text.endswith("[cash-out#2]")


def test_extractive_answer_abstains_on_unrelated_question(hits):
    answer = ExtractiveAnswerer().generate("Do you offer cryptocurrency trading?", hits)
    assert answer.abstained and answer.text == ABSTAIN_TEXT and answer.citations == []
    assert ExtractiveAnswerer().generate("anything", []).abstained


def test_citation_helpers():
    text = "Fee is 1.5% [fees#2]. Minimum BDT 10 [fees#2][atm#0]. Odd [Bad#x]"
    assert extract_citations(text) == ["fees#2", "atm#0"]
    assert extract_citations(text, allowed={"atm#0"}) == ["atm#0"]
    assert strip_citations("Fee is 1.5% [fees#2].") == "Fee is 1.5%."


def test_llm_answerer_keeps_only_citations_it_was_shown(hits):
    llm = ScriptedLLM(["The ATM charge is 1.5% [cash-out#2] (see also [made-up#9])."])
    answer = LLMAnswerer(llm).generate("ATM charge?", hits)
    assert answer.citations == ["cash-out#2"]
    assert answer.generator == "llm:scripted"
    system, prompt = llm.calls[0]
    assert "[cash-out#2]" in prompt and "Cite every sentence" in system


def test_llm_answerer_detects_abstention(hits):
    answer = LLMAnswerer(ScriptedLLM([ABSTAIN_TEXT])).generate("Crypto?", hits)
    assert answer.abstained


def _transport(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        path = request.url.path
        if path.endswith("/chat/completions"):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "oa:" + body["model"]}}]}
            )
        if path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "an"}, {"type": "text", "text": "thropic"}]
                },
            )
        if path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "ollama"}})
        return httpx.Response(503, json={"error": "down"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_providers_build_correct_requests():
    captured: list[httpx.Request] = []
    client = _transport(captured)
    assert (
        OpenAICompatibleChat("https://api.test/v1", "gpt-x", "sk-1", client=client).complete(
            "s", "p"
        )
        == "oa:gpt-x"
    )
    assert (
        AnthropicMessages("https://anthropic.test", "model-x", "ak-1", client=client).complete(
            "s", "p"
        )
        == "anthropic"
    )
    assert OllamaChat("http://ollama.test", "llama3", client=client).complete("s", "p") == "ollama"

    oa, an, ol = captured
    assert oa.headers["authorization"] == "Bearer sk-1"
    assert json.loads(oa.content)["messages"][0] == {"role": "system", "content": "s"}
    assert an.headers["x-api-key"] == "ak-1" and an.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(an.content)["system"] == "s"
    assert json.loads(ol.content)["stream"] is False


def test_provider_errors_are_wrapped():
    client = _transport([])
    with pytest.raises(LLMError):
        OllamaChat("http://ollama.test/down", "m", client=client).complete("s", "p")


def test_build_provider_validation():
    assert isinstance(build_provider("ollama", "http://x", "llama3"), OllamaChat)
    with pytest.raises(ValueError, match="RAG_LLM_MODEL"):
        build_provider("openai", "http://x", "")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_provider("palm", "http://x", "m")


def test_judge_and_prompt(hits):
    judge = LLMJudge(ScriptedLLM(["YES", "no, unsupported"]))
    assert judge.supported("claim", "evidence") is True
    assert judge.supported("claim", "evidence") is False
    assert build_prompt("Q?", hits[:1]).endswith("Question: Q?\nAnswer:")
