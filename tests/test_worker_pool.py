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
