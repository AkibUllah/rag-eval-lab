"""Optional LLM providers (plain httpx) and the LLM-backed answerer / judge.

Nothing here is needed offline. Providers share one tiny protocol, ``complete(system,
prompt) -> str``, so the answerer and judge do not care which backend is configured.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

import httpx

from rag_eval_lab.generation.base import ABSTAIN_TEXT, Answer, extract_citations
from rag_eval_lab.retrieval.base import ScoredChunk


class LLMError(RuntimeError):
    """The provider returned an error status or an unexpected payload."""


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class _HttpProvider:
    name = "http"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        timeout: float = 60.0,
        max_tokens: int = 512,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout)

    def _post(self, path: str, payload: dict, headers: dict[str, str]) -> dict:
        try:
            resp = self._client.post(f"{self.base_url}{path}", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc


class OpenAICompatibleChat(_HttpProvider):
    """``POST /chat/completions`` on OpenAI or any compatible server (vLLM, LiteLLM...)."""

    name = "openai"

    def complete(self, system: str, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            headers,
        )
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("malformed chat completion response") from exc


class AnthropicMessages(_HttpProvider):
    """Anthropic Messages API (``POST /v1/messages``)."""

    name = "anthropic"
    api_version = "2023-06-01"

    def complete(self, system: str, prompt: str) -> str:
        body = self._post(
            "/v1/messages",
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"x-api-key": self.api_key, "anthropic-version": self.api_version},
        )
        try:
            return "".join(b.get("text", "") for b in body["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError("malformed messages response") from exc


class OllamaChat(_HttpProvider):
    """Ollama ``POST /api/chat`` with streaming disabled."""

    name = "ollama"

    def complete(self, system: str, prompt: str) -> str:
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            {},
        )
        try:
            return body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("malformed /api/chat response") from exc


class ScriptedLLM:
    """Deterministic provider for tests: replays canned replies or calls a function."""

    name = "scripted"

    def __init__(self, replies: Iterable[str] | Callable[[str, str], str]) -> None:
        self._fn = replies if callable(replies) else None
        self._replies = [] if callable(replies) else list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if self._fn is not None:
            return self._fn(system, prompt)
        if not self._replies:
            raise LLMError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


ANSWER_SYSTEM = (
    "You answer customer questions using only the numbered context passages. "
    "Cite every sentence with the passage id in square brackets, e.g. [fees#2]. "
    f'If the passages do not contain the answer, reply exactly: "{ABSTAIN_TEXT}" '
    "Answer in the language of the question. Be brief."
)


def build_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    blocks = [f"[{hit.chunk.id}] ({hit.chunk.title})\n{hit.chunk.text.strip()}" for hit in contexts]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


class LLMAnswerer:
    def __init__(self, provider: LLMProvider, max_contexts: int = 5) -> None:
        self.provider = provider
        self.max_contexts = max_contexts
        self.name = f"llm:{provider.name}"

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Answer:
        used = contexts[: self.max_contexts]
        text = self.provider.complete(ANSWER_SYSTEM, build_prompt(question, used)).strip()
        abstained = ABSTAIN_TEXT.lower().rstrip(".") in text.lower()
        citations = extract_citations(text, {hit.chunk.id for hit in used})
        return Answer(text, citations=citations, abstained=abstained, generator=self.name)


JUDGE_SYSTEM = (
    "You check whether a claim is fully supported by the evidence. "
    "Reply with exactly one word: YES or NO."
)


class LLMJudge:
    """Optional LLM faithfulness judge: one YES/NO call per answer sentence."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def supported(self, claim: str, evidence: str) -> bool:
        reply = self.provider.complete(JUDGE_SYSTEM, f"Evidence:\n{evidence}\n\nClaim: {claim}")
        return reply.strip().upper().startswith("YES")


def build_provider(kind: str, base_url: str, model: str, api_key: str = "") -> LLMProvider:
    providers: dict[str, type[_HttpProvider]] = {
        "openai": OpenAICompatibleChat,
        "anthropic": AnthropicMessages,
        "ollama": OllamaChat,
    }
    try:
        cls = providers[kind]
    except KeyError:
        raise ValueError(
            f"unknown LLM provider {kind!r}; choose from fake, {', '.join(providers)}"
        ) from None
    if not model:
        raise ValueError(f"RAG_LLM_MODEL is required for provider {kind!r}")
    return cls(base_url, model, api_key)
