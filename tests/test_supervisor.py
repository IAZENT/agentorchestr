"""Regression tests for Supervisor.run() crash-safety.

Origin: a real bug where Supervisor.run() only caught
asyncio.CancelledError, so any other exception inside the goal future
or run_sse_async() escaped and the trailing `return state, summary`
raised UnboundLocalError.  These tests pin the contract: run() always
returns a (state, summary) tuple, even when its internals blow up.
"""
from __future__ import annotations

import asyncio

import pytest

from supervisor import Supervisor


class _FailingMCP:
    """Stand-in MCP app whose run_sse_async() raises immediately."""
    def __init__(self, *_a, **_kw):
        pass
    async def run_sse_async(self):
        raise RuntimeError("mcp transport went boom")


class _NoopPool:
    """The minimum pool surface Supervisor.run() / Supervisor.cleanup() touch."""
    def __init__(self):
        self.tmux_session_name = "orch-fake"
        self.host = "127.0.0.1"
        self._started: list[list[str]] = []

    @property
    def attach_command(self) -> str:
        return f"tmux attach -t {self.tmux_session_name}"

    async def start_session(self, argv):
        self._started.append(list(argv))

    async def cleanup(self, *, keep_tmux: bool = True):
        pass


@pytest.mark.asyncio
async def test_supervisor_run_returns_failed_on_internal_exception(
    monkeypatch, tmp_path,
):
    """If the goal future or MCP server raises, run() returns
    ('failed', '<error>') — never UnboundLocalError."""

    # Build a minimal Supervisor and replace its build_supervisor_app +
    # write_mcp_config + pool with cheap stubs.
    sup = Supervisor.__new__(Supervisor)
    sup.project_root = tmp_path
    sup.goal = "test"
    sup.lead_agent = {"name": "kiro", "cmd": "kiro"}
    sup.worker_agents = []
    sup.store = None
    sup.session_id = "abcd1234"
    sup.host = "127.0.0.1"
    sup.port = 1
    sup.memory = None
    sup.skills = None
    sup.pool = _NoopPool()
    sup._goal_future = None
    sup._mcp_app = None
    sup._mcp_task = None

    # Patch the symbols supervisor.run() reaches for.
    import supervisor as sup_mod
    monkeypatch.setattr(sup_mod, "build_supervisor_app",
                         lambda *a, **kw: _FailingMCP())
    monkeypatch.setattr(sup_mod, "write_mcp_config", lambda *a, **kw: None)
    monkeypatch.setattr(sup_mod, "_supervisor_argv",
                         lambda lead, prompt: ["bash", "-lc", "true"])

    state, summary = await sup.run()
    assert state == "failed"
    # Either the MCP failure or the resulting goal-future never-resolved
    # state — both are acceptable, the contract is just "no UnboundLocalError".
    assert summary  # non-empty
    assert isinstance(summary, str)


@pytest.mark.asyncio
async def test_supervisor_run_handles_cancelled_error(monkeypatch, tmp_path):
    """The pre-existing CancelledError path must still report 'cancelled'."""
    class _LongMCP:
        async def run_sse_async(self):
            await asyncio.sleep(60)  # effectively forever for the test

    sup = Supervisor.__new__(Supervisor)
    sup.project_root = tmp_path
    sup.goal = "test"
    sup.lead_agent = {"name": "kiro", "cmd": "kiro"}
    sup.worker_agents = []
    sup.store = None
    sup.session_id = "abcd1234"
    sup.host = "127.0.0.1"
    sup.port = 1
    sup.memory = None
    sup.skills = None
    sup.pool = _NoopPool()
    sup._goal_future = None
    sup._mcp_app = None
    sup._mcp_task = None

    import supervisor as sup_mod
    monkeypatch.setattr(sup_mod, "build_supervisor_app",
                         lambda *a, **kw: _LongMCP())
    monkeypatch.setattr(sup_mod, "write_mcp_config", lambda *a, **kw: None)
    monkeypatch.setattr(sup_mod, "_supervisor_argv",
                         lambda lead, prompt: ["bash", "-lc", "true"])

    # Run sup.run() and cancel it from outside.
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        state, summary = await task
    except asyncio.CancelledError:
        pytest.fail("Supervisor.run() let CancelledError propagate; "
                    "it should swallow it and report 'cancelled'.")
    assert state == "cancelled"
    assert "cancelled" in summary.lower()


def test_mcp_bridge_exports_availability_flag():
    """orchestrator.main() reads mcp_bridge.MCP_AVAILABLE to fail fast.
    The flag must exist and be a bool."""
    import mcp_bridge
    assert hasattr(mcp_bridge, "MCP_AVAILABLE")
    assert isinstance(mcp_bridge.MCP_AVAILABLE, bool)
