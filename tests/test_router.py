"""Tests for the LLM router: provider selection, fallback, 429 cooldown."""
from __future__ import annotations

import os
import time
from typing import Optional

import httpx
import pytest

from agentorchestr.router import LLMRouter, _RateLimited


@pytest.fixture
def clear_env(monkeypatch):
    """Strip every provider env var so tests start from a clean slate."""
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
              "CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_HOST"):
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
async def test_ollama_is_no_longer_a_provider(monkeypatch, clear_env):
    """Local-LLM support has been removed.  Even if OLLAMA_HOST is set,
    the router must NOT instantiate any ollama provider."""
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(lambda req: httpx.Response(404)))
    await r.init()
    assert not any(p["name"] == "ollama" for p in r.providers)
    # The PROVIDERS list itself must no longer mention ollama.
    from agentorchestr.router import PROVIDERS
    assert not any(p["name"] == "ollama" for p in PROVIDERS)
    await r.close()


@pytest.mark.asyncio
async def test_anthropic_call_marks_system_prompt_cacheable(monkeypatch, clear_env):
    """When ANTHROPIC_API_KEY is set, the router uses the native messages
    endpoint and tags the system prompt with cache_control={type:ephemeral}.
    Saves ~90% on input tokens for repeated supervisor/worker calls."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    captured: dict = {}
    def responder(request):
        if "anthropic.com" in request.url.host:
            import json as _json
            captured["payload"] = _json.loads(request.content.decode())
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "hello-claude"}],
            })
        return httpx.Response(404)

    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    out = await r.compress("hi")
    assert out == "hello-claude"
    assert r.active_provider == "anthropic"
    # The system prompt MUST arrive as a list of blocks with cache_control.
    sys_field = captured["payload"]["system"]
    assert isinstance(sys_field, list)
    assert sys_field[0]["cache_control"] == {"type": "ephemeral"}
    # Anthropic auth header
    assert captured["headers"].get("x-api-key") == "sk-test"
    assert captured["headers"].get("anthropic-version") == "2023-06-01"
    await r.close()


@pytest.mark.asyncio
async def test_openrouter_marks_system_prompt_cacheable(monkeypatch, clear_env):
    """OpenRouter forwards cache_control to Anthropic models — keep it on."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")

    captured: dict = {}
    def responder(request):
        if "openrouter.ai" in request.url.host:
            import json as _json
            captured["payload"] = _json.loads(request.content.decode())
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            })
        return httpx.Response(404)

    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    out = await r.compress("hi")
    assert out == "ok"
    # System content should be a list-of-blocks form, not a bare string.
    sys_field = captured["payload"]["messages"][0]["content"]
    assert isinstance(sys_field, list)
    assert sys_field[0]["cache_control"] == {"type": "ephemeral"}
    await r.close()


@pytest.mark.asyncio
async def test_groq_does_not_mark_cache_control(monkeypatch, clear_env):
    """Groq doesn't support cache_control — system must stay a plain string."""
    monkeypatch.setenv("GROQ_API_KEY", "gk-test")

    captured: dict = {}
    def responder(request):
        if "groq.com" in request.url.host:
            import json as _json
            captured["payload"] = _json.loads(request.content.decode())
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            })
        return httpx.Response(404)

    r = LLMRouter()
    r._client = httpx.AsyncClient(transport=_mock_transport(responder))
    await r.init()
    await r.compress("hi")
    sys_field = captured["payload"]["messages"][0]["content"]
    assert isinstance(sys_field, str), "Groq must receive system as a plain string"
    await r.close()
