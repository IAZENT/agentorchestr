"""Tests for WorkerPool — pure-logic pieces (sentinel detection, agent
argv) plus a small async test against the in-process StateStore.

We don't actually start tmux in tests; instead we instantiate WorkerPool
directly and exercise the methods that don't require a live pane."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import worker_pool
from state_store import StateStore
from worker_pool import Worker, WorkerPool, _BLOCKED_RE, _DONE_RE, _FAILED_RE


@pytest.fixture
async def pool(tmp_path):
    db = tmp_path / "state.db"
    store = StateStore(db_path=str(db))
    await store.init()
    # FK requires a parent session row before we can upsert workers
    await store.save_session(
        "abcd1234",
        {"goal": "test", "plan": {}, "agents": [], "status": "active"},
    )
    p = WorkerPool(str(tmp_path), "abcd1234", store)
    yield p
    await store.close()


# ── sentinel regexes ───────────────────────────────────────────────────

def test_done_regex_matches_one_line():
    assert _DONE_RE.search("blah\nWORKER_DONE: implemented foo()\n").group(1) == "implemented foo()"


def test_blocked_regex_matches():
    m = _BLOCKED_RE.search("WORKER_BLOCKED: cannot find db creds\nmore output")
    assert m and "db creds" in m.group(1)


def test_failed_regex_matches():
    assert _FAILED_RE.search("...\nWORKER_FAILED: tests crashed").group(1).startswith("tests")


def test_no_sentinel_returns_no_match():
    assert _DONE_RE.search("the agent is still working") is None


# ── agent argv builder ────────────────────────────────────────────────

def test_argv_for_kiro_cli(tmp_path):
    p = WorkerPool(str(tmp_path), "x", store=None)  # type: ignore[arg-type]
    argv = p._agent_argv({"name": "kiro", "cmd": "kiro-cli"}, "/tmp/p.txt")
    # Lands as a bash -lc invocation that cats the prompt file
    assert argv[0] == "bash"
    assert "kiro-cli" in argv[-1]
    assert "chat" in argv[-1]
    assert "--trust-all-tools" in argv[-1]
    assert "/tmp/p.txt" in argv[-1]


def test_argv_for_claude(tmp_path):
    p = WorkerPool(str(tmp_path), "x", store=None)  # type: ignore[arg-type]
    argv = p._agent_argv({"name": "claude", "cmd": "claude"}, "/tmp/p.txt")
    assert "--dangerously-skip-permissions" in argv[-1]


def test_argv_for_aider_uses_message_file(tmp_path):
    p = WorkerPool(str(tmp_path), "x", store=None)  # type: ignore[arg-type]
    argv = p._agent_argv({"name": "aider", "cmd": "aider"}, "/tmp/p.txt")
    # Aider has a real flag for prompt-file input
    assert "--message-file" in argv[-1]
    assert "/tmp/p.txt" in argv[-1]


def test_unknown_agent_falls_back_to_positional(tmp_path):
    p = WorkerPool(str(tmp_path), "x", store=None)  # type: ignore[arg-type]
    argv = p._agent_argv({"name": "unknown-cli", "cmd": "weird"}, "/tmp/p.txt")
    assert argv[0] == "bash"
    assert "weird" in argv[-1]
    assert "/tmp/p.txt" in argv[-1]


# ── completion detection: bypass tmux by stubbing read_output ─────────

@pytest.mark.asyncio
async def test_detect_completion_done(pool, monkeypatch):
    pool._workers["w01"] = Worker(
        id="w01", session_id="abcd1234", perspective="implementer",
        agent={"name": "openclaude"}, worktree="/tmp", branch="b",
        pane=object(), pane_id="%1", task="x",
    )
    monkeypatch.setattr(
        WorkerPool, "read_output", lambda self, wid, tail_lines=80: "doing things\nWORKER_DONE: created foo.py"
    )
    state, summary = pool.detect_completion("w01")
    assert state == "done"
    assert "foo.py" in summary


@pytest.mark.asyncio
async def test_wait_until_done_persists_state(pool, monkeypatch):
    pool._workers["w01"] = Worker(
        id="w01", session_id="abcd1234", perspective="tester",
        agent={"name": "openclaude"}, worktree="/tmp", branch="b",
        pane=object(), pane_id="%1", task="x",
    )
    # Simulate output transitioning to WORKER_FAILED on second poll
    calls = {"n": 0}
    def fake_read(self, wid, tail_lines=80):
        calls["n"] += 1
        return "running" if calls["n"] < 2 else "WORKER_FAILED: 3 tests broken"
    monkeypatch.setattr(WorkerPool, "read_output", fake_read)

    state, summary = await pool.wait_until_done("w01", timeout=2.0, poll_interval=0.05)
    assert state == "failed"
    assert "3 tests" in summary
    # Verify persisted to state store
    rows = await pool.store.get_workers("abcd1234")
    assert rows[0]["state"] == "failed"
    assert "3 tests" in rows[0]["summary"]


@pytest.mark.asyncio
async def test_wait_until_done_times_out(pool, monkeypatch):
    pool._workers["w01"] = Worker(
        id="w01", session_id="abcd1234", perspective="implementer",
        agent={"name": "openclaude"}, worktree="/tmp", branch="b",
        pane=object(), pane_id="%1", task="x",
    )
    monkeypatch.setattr(WorkerPool, "read_output", lambda self, wid, tail_lines=80: "still chugging")
    state, summary = await pool.wait_until_done("w01", timeout=0.2, poll_interval=0.05)
    assert state == "timeout"
    # Timeout MUST persist to the store so --resume / dashboard see the row.
    rows = await pool.store.get_workers("abcd1234")
    assert rows[0]["state"] == "timeout"
    assert "timeout" in rows[0]["summary"]


@pytest.mark.asyncio
async def test_kill_worker_persists(pool, monkeypatch):
    """kill_worker must update the DB so the dashboard / --resume see the
    'killed' state instead of leaving it stuck at 'running' forever."""
    class FakePane:
        sent: list[tuple[str, bool]] = []
        def send_keys(self, msg, enter=True, suppress_history=False):
            self.sent.append((msg, enter))

    pool._workers["w01"] = Worker(
        id="w01", session_id="abcd1234", perspective="implementer",
        agent={"name": "openclaude"}, worktree="/tmp", branch="b",
        pane=FakePane(), pane_id="%1", task="x", state="running",
    )
    # Pre-flight: row must reflect 'running' before kill.
    await pool.store.upsert_worker(
        "abcd1234",
        {
            "worker_id": "w01", "perspective": "implementer", "agent_name": "openclaude",
            "worktree": "/tmp", "branch": "b", "pane_id": "%1",
            "task": "x", "state": "running", "summary": "",
            "started_at": 1.0, "finished_at": None,
        },
    )

    await pool.kill_worker("w01")
    rows = await pool.store.get_workers("abcd1234")
    assert rows[0]["state"] == "killed"
    assert rows[0]["summary"]  # non-empty
    assert rows[0]["finished_at"] is not None


@pytest.mark.asyncio
async def test_fetch_memory_context_no_memory(pool):
    """When pool.memory is None, retrieval is a no-op."""
    assert pool.memory is None
    assert await pool._fetch_memory_context("anything") == ""


@pytest.mark.asyncio
async def test_fetch_memory_context_formats_hits(pool):
    """Stub a memory federation that returns 2 hits and verify formatting."""
    from memory.federation import MemoryHit

    class _StubMem:
        async def retrieve(self, query, k=3):
            return [
                MemoryHit(id="1", tier="project", kind="topic",
                          title="auth notes", body="JWT setup details", sources=["fts"]),
                MemoryHit(id="2", tier="global", kind="static",
                          title="conventions", body="snake_case everywhere",
                          sources=["fts", "vector"]),
            ]

    pool.memory = _StubMem()
    out = await pool._fetch_memory_context("authentication")
    assert "## Retrieved memories" in out
    assert "auth notes" in out
    assert "JWT setup details" in out
    assert "conventions" in out
    # source flavour is annotated in the header
    assert "fts, vector" in out


@pytest.mark.asyncio
async def test_fetch_memory_context_swallows_errors(pool):
    """A broken memory backend must NOT crash worker spawning."""
    class _BrokenMem:
        async def retrieve(self, query, k=3):
            raise RuntimeError("vector index corrupt")
    pool.memory = _BrokenMem()
    assert await pool._fetch_memory_context("x") == ""


def test_fetch_skills_context_no_skills(pool):
    assert pool.skills is None
    assert pool._fetch_skills_context("anything", []) == ""


def test_fetch_skills_context_returns_render(pool):
    """When skills.matching returns hits, _fetch_skills_context renders them."""
    class _StubSkills:
        def matching(self, task, file_scope):
            return ["jwt-skill"]  # opaque — render() decides format
        def render(self, skills):
            assert skills == ["jwt-skill"]
            return "## Active skills\n\n### jwt-skill\nuse PyJWT"
    pool.skills = _StubSkills()
    out = pool._fetch_skills_context("set up jwt auth", ["src/auth.py"])
    assert "Active skills" in out
    assert "jwt-skill" in out


def test_fetch_skills_context_swallows_errors(pool):
    class _Broken:
        def matching(self, task, file_scope):
            raise RuntimeError("oops")
    pool.skills = _Broken()
    assert pool._fetch_skills_context("x", []) == ""


@pytest.mark.asyncio
async def test_fetch_memory_context_renders_research_envelope(pool):
    """When a hit's kind=='research', the JSON envelope must be unwrapped
    into the human-friendly markdown so the worker doesn't see raw JSON."""
    import json as _json
    from memory.federation import MemoryHit

    payload = {
        "query": "fastapi 0.115 changes",
        "fetched_at": 1.0,
        "results": [
            {"title": "FastAPI 0.115 release", "snippet": "Adds X", "url": "https://fastapi.tiangolo.com/release-notes", "source": "duckduckgo"}
        ],
        "fetched_bodies": {"https://fastapi.tiangolo.com/release-notes": "Body of release notes"},
        "error": None,
    }
    body = (
        "<!--agentorchestr-research v1\n"
        "query=fastapi 0.115 changes\n"
        "fetched_at=1.0\n"
        "ttl_s=86400\n-->\n"
        f"# Research: fastapi 0.115\n\n```json\n{_json.dumps(payload)}\n```\n"
    )

    class _StubMem:
        async def retrieve(self, query, k=3):
            return [MemoryHit(
                id="r1", tier="project", kind="research",
                title="research: fastapi 0.115 changes", body=body, sources=["fts"],
            )]
    pool.memory = _StubMem()
    out = await pool._fetch_memory_context("anything")
    # The raw JSON envelope must NOT appear; the rendered form does.
    assert "<!--agentorchestr-research" not in out
    assert "Web research: fastapi 0.115 changes" in out
    assert "FastAPI 0.115 release" in out
    assert "https://fastapi.tiangolo.com/release-notes" in out
