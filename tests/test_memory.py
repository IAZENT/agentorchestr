"""Tests for the memory federation.

We exercise FTS5 only (the always-on path).  sqlite-vec is monkeypatched
to "unavailable" so the test runs deterministically on any host.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

import memory.federation as fed
from memory import MemoryFederation, MemoryHit


@pytest.fixture
def disable_vec(monkeypatch):
    """Force the FTS-only code path so the tests don't depend on sqlite-vec."""
    monkeypatch.setattr(fed, "HAS_SQLITE_VEC", False)
    monkeypatch.setattr(fed, "HAS_FASTEMBED", False)


@pytest.fixture
async def fed_under_test(tmp_path, disable_vec, monkeypatch):
    # Force a project-local global dir so we don't pollute ~/.orch
    global_dir = tmp_path / "user-global"
    monkeypatch.setattr(fed, "GLOBAL_DIR", global_dir)
    project_root = tmp_path / "proj"
    project_root.mkdir()
    f = MemoryFederation(project_root)
    await f.init()
    yield f
    f.close()


@pytest.mark.asyncio
async def test_init_creates_directories_and_db(tmp_path, disable_vec, monkeypatch):
    monkeypatch.setattr(fed, "GLOBAL_DIR", tmp_path / "g")
    project = tmp_path / "p"
    project.mkdir()
    f = MemoryFederation(project)
    await f.init()
    try:
        assert (project / ".orch" / "memory" / "topics").is_dir()
        assert (project / ".orch" / "memory" / "episodes").is_dir()
        assert (project / ".orch" / "memory" / "index.sqlite").exists()
    finally:
        f.close()


@pytest.mark.asyncio
async def test_static_context_loads_project_md(fed_under_test, tmp_path):
    f = fed_under_test
    (f.project_root / "PROJECT.md").write_text(
        "# This Project\n\nUses Python 3.13 and asyncio."
    )
    (f.project_root / "CONVENTIONS.md").write_text(
        "Use snake_case for functions."
    )
    out = await f.load_static_context()
    assert "PROJECT.md" in out
    assert "Python 3.13" in out
    assert "snake_case" in out


@pytest.mark.asyncio
async def test_index_topic_and_retrieve_with_fts(fed_under_test):
    f = fed_under_test
    await f.index_topic("auth", "JWT-based authentication using PyJWT, 24h expiry, HS256.")
    await f.index_topic("db", "Postgres 16 with sqlx for connection pooling.")
    await f.index_topic("perf", "Avoid N+1 queries; use joinedload.")

    hits = await f.retrieve("authentication tokens", k=5)
    assert hits, "FTS should match 'authentication' against the auth topic"
    assert any(h.title == "auth" for h in hits)
    # Sources must include 'fts' since vec is disabled.
    auth = next(h for h in hits if h.title == "auth")
    assert "fts" in auth.sources
    assert auth.score > 0


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty(fed_under_test):
    assert await fed_under_test.retrieve("") == []
    assert await fed_under_test.retrieve("   ") == []


@pytest.mark.asyncio
async def test_recency_boost_prefers_newer_note(fed_under_test, monkeypatch):
    """Two equally relevant notes: the newer one ranks higher."""
    f = fed_under_test
    # Index an older note first, then a newer one. Manually set updated_at
    # on the older one to force the gap.
    await f.index_topic("alpha", "Apple banana cherry.")
    # Backdate alpha by 90 days.
    f._db.execute("UPDATE notes SET updated_at = ? WHERE title = 'alpha'",
                  (time.time() - 90 * 86400,))
    f._db.commit()
    await f.index_topic("beta", "Apple banana cherry.")

    hits = await f.retrieve("apple banana", k=5)
    titles = [h.title for h in hits]
    # Both should hit; beta (recent) should outrank alpha.
    assert "beta" in titles and "alpha" in titles
    assert titles.index("beta") < titles.index("alpha")


@pytest.mark.asyncio
async def test_index_episode_persists_markdown(fed_under_test):
    f = fed_under_test
    await f.index_episode("abc12345", "## Session abc12345\n\nFixed flaky test in user_login.")
    md = f.project_dir / "episodes" / "session_abc12345.md"
    assert md.exists()
    hits = await f.retrieve("flaky test login", k=5)
    assert any("abc12345" in h.title for h in hits)


@pytest.mark.asyncio
async def test_sanitize_fts_query_handles_punctuation():
    """A query with parens / quotes must not break the FTS5 parser."""
    out = fed._sanitize_fts_query('what about "JWT" (auth) for /v1/login?')
    # Should yield prefix-matched OR query, no quotes/parens
    assert '"' not in out and "(" not in out
    assert "*" in out
    assert "OR" in out


@pytest.mark.asyncio
async def test_safe_filename_drops_unsafe_chars():
    assert fed._safe_filename("my topic / has: weird?? chars!") == "my-topic-has-weird-chars"
    assert fed._safe_filename("") == "untitled"


@pytest.mark.asyncio
async def test_recency_score_decays():
    now = 1_000_000_000
    fresh = fed._recency_score(now, now=now)
    one_halflife = fed._recency_score(now - 30 * 86400, now=now)
    two_halflives = fed._recency_score(now - 60 * 86400, now=now)
    assert fresh == pytest.approx(1.0)
    assert one_halflife == pytest.approx(0.5, rel=1e-3)
    assert two_halflives == pytest.approx(0.25, rel=1e-3)


@pytest.mark.asyncio
async def test_cache_research_roundtrip(fed_under_test):
    f = fed_under_test
    payload = {
        "query": "fastapi 0.115",
        "fetched_at": 1_000_000_000.0,
        "results": [{"title": "Docs", "snippet": "FastAPI", "url": "https://fastapi.tiangolo.com", "source": "duckduckgo"}],
        "fetched_bodies": {"https://fastapi.tiangolo.com": "lorem ipsum guide"},
        "error": None,
    }
    await f.cache_research("fastapi 0.115", payload, ttl_hours=24)
    cached = await f.get_cached_research("fastapi 0.115")
    assert cached is not None
    assert cached["query"] == "fastapi 0.115"
    assert cached["fetched_bodies"]["https://fastapi.tiangolo.com"].startswith("lorem")


@pytest.mark.asyncio
async def test_cache_research_ttl_expires(fed_under_test):
    f = fed_under_test
    payload = {"query": "stale", "fetched_at": 0.0, "results": [], "fetched_bodies": {}, "error": None}
    # Negative TTL forces immediate expiry.
    await f.cache_research("stale", payload, ttl_hours=-1)
    assert await f.get_cached_research("stale") is None


@pytest.mark.asyncio
async def test_cache_research_miss_returns_none(fed_under_test):
    assert await fed_under_test.get_cached_research("never-asked") is None


@pytest.mark.asyncio
async def test_cache_research_is_retrievable_via_fts(fed_under_test):
    """Research notes are mirrored into the FTS index so future workers
    surface them organically by topic match — that's the whole point."""
    f = fed_under_test
    payload = {
        "query": "JWT authentication PyJWT",
        "fetched_at": 1_000_000_000.0,
        "results": [{"title": "PyJWT", "snippet": "RFC 7519", "url": "https://pyjwt.example", "source": "duckduckgo"}],
        "fetched_bodies": {},
        "error": None,
    }
    await f.cache_research("JWT authentication PyJWT", payload, ttl_hours=24)
    hits = await f.retrieve("authentication tokens", k=5)
    assert any(h.kind == "research" for h in hits), \
        "research notes must surface via FTS retrieval"


@pytest.mark.asyncio
async def test_cache_research_empty_query_is_noop(fed_under_test):
    f = fed_under_test
    await f.cache_research("", {"query": "", "fetched_at": 0.0, "results": [], "fetched_bodies": {}, "error": None}, ttl_hours=24)
    # No file should be written.
    assert not (f.project_dir / "research").exists() or \
           not list((f.project_dir / "research").glob("*.md"))
