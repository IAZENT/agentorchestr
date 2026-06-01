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
        "discover_running_agents", "send_to_external_agent",
        "read_from_external_agent",
        "compact_session", "clear_tool_results",
        "web_research",
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


@pytest.mark.asyncio
async def test_compact_session_rewrites_old_summaries(pool):
    """compact_session must shorten finished workers' summaries while
    preserving the most recent keep_last_n verbatim."""
    pytest.importorskip("mcp.server.fastmcp")
    # Seed 4 workers, oldest -> newest.
    for i, st in enumerate(["done", "done", "failed", "running"]):
        await pool.store.upsert_worker(
            "deadbeef",
            {
                "worker_id": f"w{i+1:02d}", "perspective": "implementer",
                "agent_name": "kiro", "worktree": "/tmp", "branch": "",
                "pane_id": "", "task": f"task {i}", "state": st,
                "summary": f"long detailed summary for w{i+1:02d} " * 20,
                "started_at": 1000.0 + i, "finished_at": None,
            },
        )

    app = mcp_bridge.build_supervisor_app(
        pool,
        available_agents=[{"name": "openclaude", "cmd": "openclaude",
                           "type": "cli", "capabilities": {}}],
        default_worker_agent={"name": "openclaude", "cmd": "openclaude",
                              "type": "cli", "capabilities": {}},
    )
    raw = await app.call_tool("compact_session", {"keep_last_n": 1})
    if isinstance(raw, tuple):
        raw = raw[0]
    payload = ""
    for item in raw:
        payload += getattr(item, "text", str(item))
    body = json.loads(payload)
    # 3 finished, keep_last_n=1 -> 2 of them get compacted.
    assert body["compacted"] == 2
    assert body["total_workers"] == 4

    rows = {r["worker_id"]: r for r in await pool.store.get_workers("deadbeef")}
    assert rows["w01"]["summary"].startswith("[compacted]")
    assert rows["w02"]["summary"].startswith("[compacted]")
    # Most recent finished worker (w03) is kept verbatim.
    assert not rows["w03"]["summary"].startswith("[compacted]")
    # Running worker untouched.
    assert not rows["w04"]["summary"].startswith("[compacted]")


@pytest.mark.asyncio
async def test_web_research_uses_cache_on_second_call(pool, tmp_path, monkeypatch):
    """First call hits the network (mocked) and persists to memory; second
    call must short-circuit and return cached=True without calling research()."""
    pytest.importorskip("mcp.server.fastmcp")

    # Spin up a real MemoryFederation rooted in tmp_path.
    import memory.federation as fed_mod
    monkeypatch.setattr(fed_mod, "GLOBAL_DIR", tmp_path / "global")
    monkeypatch.setattr(fed_mod, "HAS_SQLITE_VEC", False)
    monkeypatch.setattr(fed_mod, "HAS_FASTEMBED", False)
    from memory import MemoryFederation
    project = tmp_path / "proj"
    project.mkdir()
    mem = MemoryFederation(project)
    await mem.init()

    # Patch research.research to count network calls and return canned data.
    import research as research_mod
    calls: list[str] = []
    async def _fake_research(query, *, k=5, fetch_top_n=2, max_chars_per_body=4000):
        calls.append(query)
        return research_mod.ResearchPayload(
            query=query, fetched_at=12345.0,
            results=[research_mod.SearchResult(title="X", snippet="snip", url="https://x")],
            fetched_bodies={"https://x": "BODY"},
        )
    monkeypatch.setattr(research_mod, "research", _fake_research)

    app = mcp_bridge.build_supervisor_app(
        pool,
        available_agents=[{"name": "openclaude", "cmd": "openclaude",
                           "type": "cli", "capabilities": {}}],
        default_worker_agent={"name": "openclaude", "cmd": "openclaude",
                              "type": "cli", "capabilities": {}},
        memory=mem,
    )

    # First call → hits the mocked network.
    raw = await app.call_tool("web_research", {"query": "fastapi 0.115"})
    if isinstance(raw, tuple): raw = raw[0]
    payload = json.loads("".join(getattr(i, "text", str(i)) for i in raw))
    assert payload["cached"] is False
    assert calls == ["fastapi 0.115"]

    # Second call → cached.
    raw = await app.call_tool("web_research", {"query": "fastapi 0.115"})
    if isinstance(raw, tuple): raw = raw[0]
    payload = json.loads("".join(getattr(i, "text", str(i)) for i in raw))
    assert payload["cached"] is True
    assert calls == ["fastapi 0.115"]  # research() NOT re-invoked

    mem.close()


@pytest.mark.asyncio
async def test_web_research_force_refresh_bypasses_cache(pool, tmp_path, monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    import memory.federation as fed_mod
    monkeypatch.setattr(fed_mod, "GLOBAL_DIR", tmp_path / "global")
    monkeypatch.setattr(fed_mod, "HAS_SQLITE_VEC", False)
    monkeypatch.setattr(fed_mod, "HAS_FASTEMBED", False)
    from memory import MemoryFederation
    project = tmp_path / "proj"; project.mkdir()
    mem = MemoryFederation(project); await mem.init()

    import research as research_mod
    calls: list[str] = []
    async def _fake_research(query, *, k=5, fetch_top_n=2, max_chars_per_body=4000):
        calls.append(query)
        return research_mod.ResearchPayload(
            query=query, fetched_at=12345.0,
            results=[research_mod.SearchResult(title="X", snippet="s", url="https://x")],
        )
    monkeypatch.setattr(research_mod, "research", _fake_research)

    app = mcp_bridge.build_supervisor_app(
        pool,
        available_agents=[{"name": "x", "cmd": "x", "type": "cli", "capabilities": {}}],
        default_worker_agent={"name": "x", "cmd": "x", "type": "cli", "capabilities": {}},
        memory=mem,
    )
    await app.call_tool("web_research", {"query": "q1"})
    await app.call_tool("web_research", {"query": "q1", "force_refresh": True})
    assert calls == ["q1", "q1"]
    mem.close()
