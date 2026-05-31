"""Tests for the supervisor MCP bridge.

We don't start a real SSE server; we just verify the builder registers
the right tool surface and that mark_goal_done/_failed wire to callbacks.
"""
from __future__ import annotations

import json

import pytest

import mcp_bridge
from state_store import StateStore
from worker_pool import WorkerPool


@pytest.fixture
async def pool(tmp_path):
    db = tmp_path / "state.db"
    store = StateStore(db_path=str(db))
    await store.init()
    await store.save_session(
        "deadbeef",
        {"goal": "test", "plan": {}, "agents": [], "status": "active"},
    )
    p = WorkerPool(str(tmp_path), "deadbeef", store)
    yield p
    await store.close()


@pytest.mark.asyncio
async def test_supervisor_app_registers_expected_tools(pool):
    pytest.importorskip("mcp.server.fastmcp")
    app = mcp_bridge.build_supervisor_app(
        pool,
        available_agents=[{"name": "openclaude", "cmd": "openclaude", "type": "cli",
                            "capabilities": {}}],
        default_worker_agent={"name": "openclaude", "cmd": "openclaude", "type": "cli",
                               "capabilities": {}},
    )
    tools = await app.list_tools()
    names = {t.name for t in tools}
    expected = {
        "spawn_worker", "read_worker_output", "send_to_worker",
        "wait_for_worker", "list_workers", "kill_worker",
        "cross_review", "verify", "report_progress",
        "mark_goal_done", "mark_goal_failed", "list_perspectives",
    }
    assert expected <= names, f"missing tools: {expected - names}"


@pytest.mark.asyncio
async def test_mark_goal_done_invokes_callback(pool):
    pytest.importorskip("mcp.server.fastmcp")
    captured: dict = {}

    async def on_done(summary: str) -> None:
        captured["summary"] = summary

    app = mcp_bridge.build_supervisor_app(
        pool,
        available_agents=[{"name": "openclaude", "cmd": "openclaude",
                            "type": "cli", "capabilities": {}}],
        default_worker_agent={"name": "openclaude", "cmd": "openclaude",
                               "type": "cli", "capabilities": {}},
        on_goal_done=on_done,
    )
    # FastMCP exposes call_tool() returning a tuple (content, metadata)
    result = await app.call_tool("mark_goal_done", {"summary": "shipped it"})
    # Different FastMCP versions return different shapes; accept any iterable
    assert captured.get("summary") == "shipped it"
    # Result should at minimum mention acknowledgment
    payload = ""
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "__iter__"):
        for item in result:
            payload += getattr(item, "text", str(item))
    else:
        payload = str(result)
    assert "acknowledged" in payload


@pytest.mark.asyncio
async def test_list_perspectives_returns_known_set(pool):
    pytest.importorskip("mcp.server.fastmcp")
    app = mcp_bridge.build_supervisor_app(
        pool,
        available_agents=[{"name": "openclaude", "cmd": "openclaude",
                            "type": "cli", "capabilities": {}}],
        default_worker_agent={"name": "openclaude", "cmd": "openclaude",
                               "type": "cli", "capabilities": {}},
    )
    raw = await app.call_tool("list_perspectives", {})
    if isinstance(raw, tuple):
        raw = raw[0]
    payload = ""
    for item in raw:
        payload += getattr(item, "text", str(item))
    names = json.loads(payload)
    assert "implementer" in names
    assert "verifier" in names
