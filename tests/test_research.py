"""Tests for research.py — fully mocked httpx + ddgs."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Iterator

import httpx
import pytest

from agentorchestr import research


# ── helpers ─────────────────────────────────────────────────────────────

class _FakeDDGS:
    """Stub the ddgs context-manager + .text() iterator."""
    def __init__(self, hits: list[dict]):
        self._hits = hits
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def text(self, query: str, max_results: int = 5) -> Iterator[dict]:
        for h in self._hits[:max_results]:
            yield h


def _mock_transport(responder):
    return httpx.MockTransport(responder)


# ── search ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_no_ddgs_returns_empty(monkeypatch):
    monkeypatch.setattr(research, "HAS_DDGS", False)
    out = await research.search("anything", k=5)
    assert out == []


@pytest.mark.asyncio
async def test_search_returns_results(monkeypatch):
    hits = [
        {"title": "A title", "body": "snippet here", "href": "https://a.example"},
        {"title": "B", "body": "more", "href": "https://b.example"},
    ]
    monkeypatch.setattr(research, "HAS_DDGS", True)
    monkeypatch.setattr(research, "DDGS", lambda: _FakeDDGS(hits))
    out = await research.search("python jwt", k=5)
    assert len(out) == 2
    assert out[0].title == "A title"
    assert out[0].url == "https://a.example"


@pytest.mark.asyncio
async def test_search_empty_query_returns_empty(monkeypatch):
    monkeypatch.setattr(research, "HAS_DDGS", True)
    monkeypatch.setattr(research, "DDGS", lambda: _FakeDDGS([]))
    assert await research.search("", k=5) == []
    assert await research.search("   ", k=5) == []


@pytest.mark.asyncio
async def test_search_swallows_ddgs_exceptions(monkeypatch):
    """If ddgs raises (network down, parsing change), return [] instead of crashing."""
    class _Boom:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **kw): raise RuntimeError("boom")
    monkeypatch.setattr(research, "HAS_DDGS", True)
    monkeypatch.setattr(research, "DDGS", lambda: _Boom())
    assert await research.search("x", k=5) == []


# ── fetch ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_url_strips_html():
    def responder(req):
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body>  <h1>Title</h1>\n  <p>Para1</p>  </body></html>",
        )
    client = httpx.AsyncClient(transport=_mock_transport(responder))
    try:
        text = await research.fetch_url("https://example.com", client=client)
    finally:
        await client.aclose()
    assert "Title" in text
    assert "Para1" in text
    assert "<h1>" not in text


@pytest.mark.asyncio
async def test_fetch_url_4xx_returns_empty():
    def responder(req):
        return httpx.Response(404, text="not found")
    client = httpx.AsyncClient(transport=_mock_transport(responder))
    try:
        assert await research.fetch_url("https://example.com", client=client) == ""
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_url_network_error_returns_empty():
    def responder(req):
        raise httpx.ConnectError("dns")
    client = httpx.AsyncClient(transport=_mock_transport(responder))
    try:
        assert await research.fetch_url("https://example.com", client=client) == ""
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_url_truncates_to_max_chars():
    payload = "x" * 100_000
    def responder(req):
        return httpx.Response(200, headers={"content-type": "text/plain"}, text=payload)
    client = httpx.AsyncClient(transport=_mock_transport(responder))
    try:
        text = await research.fetch_url("https://example.com", max_chars=2000, client=client)
    finally:
        await client.aclose()
    assert len(text) <= 2000


# ── research (combined) ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_research_returns_payload(monkeypatch):
    hits = [
        {"title": "JWT spec", "body": "RFC 7519", "href": "https://example.com/spec"},
        {"title": "PyJWT docs", "body": "guide", "href": "https://example.com/pyjwt"},
        {"title": "Flask-JWT", "body": "ext", "href": "https://example.com/flask"},
    ]
    monkeypatch.setattr(research, "HAS_DDGS", True)
    monkeypatch.setattr(research, "DDGS", lambda: _FakeDDGS(hits))

    # Capture the URLs we fetch and return canned bodies.
    fetched: list[str] = []
    async def _fake_fetch(url, *, max_chars=4000, timeout=8.0, client=None):
        fetched.append(url)
        return f"BODY[{url}]"
    monkeypatch.setattr(research, "fetch_url", _fake_fetch)

    payload = await research.research("python jwt", k=5, fetch_top_n=2)
    assert payload.query == "python jwt"
    assert len(payload.results) == 3
    # Only top 2 are body-fetched.
    assert len(payload.fetched_bodies) == 2
    assert "BODY[https://example.com/spec]" in payload.fetched_bodies.values()


@pytest.mark.asyncio
async def test_research_empty_query():
    payload = await research.research("", k=5)
    assert payload.error == "empty query"
    assert payload.results == []


@pytest.mark.asyncio
async def test_research_no_results(monkeypatch):
    monkeypatch.setattr(research, "HAS_DDGS", True)
    monkeypatch.setattr(research, "DDGS", lambda: _FakeDDGS([]))
    payload = await research.research("nothing matches", k=5)
    assert payload.results == []
    assert payload.error  # populated


def test_render_markdown_shape():
    payload = research.ResearchPayload(
        query="jwt 2026",
        fetched_at=1.0,
        results=[
            research.SearchResult(title="A", snippet="snip", url="https://a"),
        ],
        fetched_bodies={"https://a": "alpha body"},
    )
    md = payload.render_markdown(max_chars_per_body=20)
    assert "## Web research: jwt 2026" in md
    assert "### 1. A" in md
    assert "https://a" in md
    assert "alpha body" in md


def test_render_markdown_no_results():
    payload = research.ResearchPayload(query="x", fetched_at=1.0)
    md = payload.render_markdown()
    assert "no results" in md
