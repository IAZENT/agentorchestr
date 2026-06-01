"""Tests for worker completion-sentinel detection.

All tests are pure unit tests — no tmux, no subprocesses.
They cover both detection paths (flag file and pane regex) and the
anchored-regex hardening that prevents quoted sentinels from triggering.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agentorchestr.state_store import StateStore
from agentorchestr.worker_pool import (
    Worker, WorkerPool,
    _DONE_RE, _BLOCKED_RE, _FAILED_RE,
)


def _pool_with_worker(tmp_path: Path):
    store = StateStore(db_path=str(tmp_path / "state.db"))
    pool = WorkerPool.__new__(WorkerPool)
    pool.project_root = tmp_path
    pool.agentorchestr_session_id = "s"
    pool.store = store
    pool._workers = {}
    pool._counter = 0
    pool._tmp_root = tmp_path / "tmp"
    pool._tmp_root.mkdir(parents=True, exist_ok=True)
    pool.memory = None
    pool.skills = None
    pool._tmux_session = None
    wt = tmp_path / "wt"
    wt.mkdir()
    w = Worker(id="w01", session_id="s", perspective="implementer",
               agent={"name": "fake", "cmd": "fake"},
               worktree=str(wt), branch="", pane=None, pane_id="%1", task="t")
    pool._workers["w01"] = w
    return pool, w


def test_flag_file_done_short_circuits_pane(tmp_path, monkeypatch):
    pool, w = _pool_with_worker(tmp_path)
    flag_dir = Path(w.worktree) / ".agentorchestr"
    flag_dir.mkdir()
    (flag_dir / "done.flag").write_text("done\nshipped\n")
    monkeypatch.setattr(pool, "read_output",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not read pane")))
    state, summary = pool.detect_completion("w01")
    assert state == "done" and summary == "shipped"


def test_flag_file_blocked_and_failed(tmp_path, monkeypatch):
    pool, w = _pool_with_worker(tmp_path)
    flag_dir = Path(w.worktree) / ".agentorchestr"
    flag_dir.mkdir()
    monkeypatch.setattr(pool, "read_output", lambda *a, **kw: "")
    for state_str in ("blocked", "failed"):
        (flag_dir / "done.flag").write_text(f"{state_str}\nreason\n")
        state, summary = pool.detect_completion("w01")
        assert state == state_str and summary == "reason"


def test_garbled_flag_falls_through_to_pane_regex(tmp_path, monkeypatch):
    pool, w = _pool_with_worker(tmp_path)
    (Path(w.worktree) / ".agentorchestr").mkdir()
    (Path(w.worktree) / ".agentorchestr" / "done.flag").write_text("nonsense\n")
    monkeypatch.setattr(pool, "read_output",
                        lambda *a, **kw: "WORKER_DONE: from pane")
    state, _ = pool.detect_completion("w01")
    assert state == "done"


def test_no_flag_no_sentinel_returns_running(tmp_path, monkeypatch):
    pool, _ = _pool_with_worker(tmp_path)
    monkeypatch.setattr(pool, "read_output", lambda *a, **kw: "still thinking")
    assert pool.detect_completion("w01") == ("running", "")


def test_quoted_sentinel_does_not_trigger(tmp_path, monkeypatch):
    pool, _ = _pool_with_worker(tmp_path)
    text = "I'll write `WORKER_DONE: foo` when done\nstill working"
    monkeypatch.setattr(pool, "read_output", lambda *a, **kw: text)
    assert pool.detect_completion("w01")[0] == "running"


def test_real_sentinel_at_line_start(tmp_path, monkeypatch):
    pool, _ = _pool_with_worker(tmp_path)
    monkeypatch.setattr(pool, "read_output",
                        lambda *a, **kw: "doing stuff\nWORKER_DONE: ok\n")
    state, summary = pool.detect_completion("w01")
    assert state == "done" and summary == "ok"


@pytest.mark.parametrize("rx,line", [
    (_DONE_RE,    "WORKER_DONE: x"),
    (_BLOCKED_RE, "WORKER_BLOCKED: y"),
    (_FAILED_RE,  "WORKER_FAILED: z"),
])
def test_sentinel_regexes_match(rx, line):
    assert rx.search(line) is not None


@pytest.mark.parametrize("text", [
    "I'll print `WORKER_DONE: foo` later",
    "echo \"WORKER_FAILED: nope\"",
    "  prefix WORKER_DONE: mid-line",
])
def test_sentinel_regexes_reject_inline(text):
    for rx in (_DONE_RE, _BLOCKED_RE, _FAILED_RE):
        assert rx.search(text) is None, f"wrongly matched: {text!r}"
