"""Tests for the FastAPI dashboard endpoints.

We exercise build_app(store) with httpx.ASGITransport — no real uvicorn
process, no port binding.  Confirms the live-path /workers endpoint
returns the rows StateStore.upsert_worker has stored.
"""
from __future__ import annotations

import httpx
import pytest

from agentorchestr.dashboard.server import build_app
from agentorchestr.state_store import StateStore


@pytest.fixture
async def store(tmp_path):
    db = tmp_path / "state.db"
    s = StateStore(db_path=str(db))
    await s.init()
    yield s
    await s.close()


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_healthz(store):
    async with _client(build_app(store)) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_sessions_list_includes_worker_counts(store):
    """The /api/sessions aggregator must surface live worker counts so
    supervisor-driven sessions don't appear empty."""
    await store.save_session(
        "abc12345",
        {"goal": "build it", "plan": {}, "agents": ["kiro"], "status": "active"},
    )
    # 2 done, 1 running
    for wid, st in [("w01", "done"), ("w02", "done"), ("w03", "running")]:
        await store.upsert_worker(
            "abc12345",
            {
                "worker_id": wid, "perspective": "implementer",
                "agent_name": "kiro", "worktree": "/tmp", "branch": "",
                "pane_id": "", "task": "x", "state": st, "summary": "",
                "started_at": 1.0, "finished_at": None,
            },
        )

    async with _client(build_app(store)) as c:
        r = await c.get("/api/sessions")
    assert r.status_code == 200
    sessions = r.json()["sessions"]
    assert len(sessions) == 1
    s = sessions[0]
    assert s["id"] == "abc12345"
    assert s["workers_total"] == 3
    assert s["workers_done"] == 2
    assert s["workers_running"] == 1


@pytest.mark.asyncio
async def test_session_workers_endpoint(store):
    await store.save_session(
        "deadbeef",
        {"goal": "g", "plan": {}, "agents": [], "status": "active"},
    )
    await store.upsert_worker(
        "deadbeef",
        {
            "worker_id": "w01", "perspective": "tester",
            "agent_name": "openclaude", "worktree": "/tmp", "branch": "b",
            "pane_id": "%1", "task": "run tests", "state": "done",
            "summary": "all green", "started_at": 1.0, "finished_at": 2.0,
        },
    )

    async with _client(build_app(store)) as c:
        r = await c.get("/api/sessions/deadbeef/workers")
    assert r.status_code == 200
    workers = r.json()["workers"]
    assert len(workers) == 1
    assert workers[0]["state"] == "done"
    assert workers[0]["summary"] == "all green"


@pytest.mark.asyncio
async def test_session_detail_now_includes_workers(store):
    await store.save_session(
        "abcd0001",
        {"goal": "g", "plan": {}, "agents": [], "status": "active"},
    )
    await store.upsert_worker(
        "abcd0001",
        {
            "worker_id": "w01", "perspective": "implementer",
            "agent_name": "kiro", "worktree": "/tmp", "branch": "",
            "pane_id": "", "task": "x", "state": "running", "summary": "",
            "started_at": 1.0, "finished_at": None,
        },
    )

    async with _client(build_app(store)) as c:
        r = await c.get("/api/sessions/abcd0001")
    assert r.status_code == 200
    body = r.json()
    assert "workers" in body
    assert len(body["workers"]) == 1
    assert body["workers"][0]["worker_id"] == "w01"


@pytest.mark.asyncio
async def test_session_detail_404(store):
    async with _client(build_app(store)) as c:
        r = await c.get("/api/sessions/nonexistent")
    assert r.status_code == 404
