"""Tests for the LLM router: provider selection, fallback, 429 cooldown."""
from __future__ import annotations

import os
import time
from typing import Optional

import httpx
import pytest

from router import LLMRouter, _RateLimited


@pytest.fixture
def clear_env(monkeypatch):
    """Strip every provider env var so tests start from a clean slate."""
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    yield


def _mock_transport(responder):
    """Build an httpx MockTransport whose handler is `responder(request)`."""
    return httpx.MockTransport(responder)


@pytest.mark.asyncio
async def test_no_keys_means_no_providers(clear_env):
    r = LLMRouter()
    # Patch the AsyncClient with a transport that always 404s for the ollama probe.
    r._client = httpx.AsyncClient(transport=_mock_transport(lambda req: httpx.Response(404)))
    await r.init()
    assert not r.has_any_free()
    assert r.providers == []
    await r.close()


@pytest.mark.asyncio
async def test_groq_is_picked_when_only_key(monkeypatch, clear_env):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(lambda req: httpx.Response(404)))
    await r.init()
    assert r.has_any_free()
    assert r.active_provider == "groq"
    await r.close()


@pytest.mark.asyncio
async def test_429_puts_provider_on_cooldown(monkeypatch, clear_env):
    """When provider returns 429, it should be cooled down so subsequent
    calls skip straight to the next provider."""
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")

    calls: list[str] = []
    def responder(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if "groq.com" in host:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": "rate"})
        if "openrouter.ai" in host:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}]
            })
        return httpx.Response(404)

    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    out = await r.compress("summarise this")
    assert out == "hi"
    assert r.active_provider == "openrouter"
    # groq was tried, hit 429, then openrouter served the response
    assert calls.count("api.groq.com") == 1

    # Second call: groq should be skipped immediately due to cooldown
    out2 = await r.compress("again")
    assert out2 == "hi"
    assert calls.count("api.groq.com") == 1   # no second attempt
    assert calls.count("openrouter.ai") == 2

    status = r.provider_status()
    assert status["groq"]["cooldown_s"] > 0
    await r.close()


@pytest.mark.asyncio
async def test_all_providers_failing_raises(monkeypatch, clear_env):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    def responder(request):
        return httpx.Response(500, json={"error": "boom"})
    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        await r.compress("x")
    await r.close()


@pytest.mark.asyncio
async def test_ollama_is_opt_in_by_default(monkeypatch, clear_env):
    """A running Ollama server should NOT be auto-added to providers
    unless the user explicitly opts in via use_ollama=True."""
    def responder(request):
        # Pretend Ollama is up and healthy at localhost:11434
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1"}]})
        return httpx.Response(404)

    r = LLMRouter()  # use_ollama defaults to False
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    assert not any(p["name"] == "ollama" for p in r.providers), \
        "Ollama must be opt-in to avoid silent slow-fallback surprises"
    await r.close()


@pytest.mark.asyncio
async def test_ollama_added_when_explicitly_enabled(monkeypatch, clear_env):
    """Pass use_ollama=True to opt in."""
    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "x"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "yo"}})
        return httpx.Response(404)

    r = LLMRouter(use_ollama=True)
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    assert any(p["name"] == "ollama" for p in r.providers)
    out = await r.compress("hi")
    assert out == "yo"
    await r.close()


@pytest.mark.asyncio
async def test_ollama_host_env_alone_does_not_opt_in(monkeypatch, clear_env):
    """Setting OLLAMA_HOST is NOT enough — only the use_ollama flag enables it."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("ORCH_USE_OLLAMA", "1")  # still ignored

    def responder(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    r = LLMRouter()  # use_ollama=False by default
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    assert not any(p["name"] == "ollama" for p in r.providers)
    await r.close()
