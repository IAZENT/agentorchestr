"""Tests for the SQLite state store."""
from __future__ import annotations

import pytest

from state_store import StateStore


@pytest.fixture
async def store(tmp_path):
    db = tmp_path / "state.db"
    s = StateStore(db_path=str(db))
    await s.init()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_session_roundtrip(store):
    await store.save_session("abc12345", {
        "goal": "build it",
        "plan": {"tasks": [{"id": "t001"}]},
        "agents": ["kiro"],
        "status": "active",
    })
    row = await store.get_session("abc12345")
    assert row["goal"] == "build it"
    assert row["plan"]["tasks"][0]["id"] == "t001"
    assert row["agents"] == ["kiro"]
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_session_partial_update_keeps_existing_fields(store):
    await store.save_session("abc12345", {
        "goal": "build it", "plan": {"tasks": []}, "agents": ["kiro"], "status": "active",
    })
    await store.save_session("abc12345", {"status": "paused"})
    row = await store.get_session("abc12345")
    assert row["status"] == "paused"
    assert row["goal"] == "build it"
    assert row["agents"] == ["kiro"]


@pytest.mark.asyncio
async def test_task_lifecycle(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.assign_task("s1", "t001", "worker-0")
    assert await store.get_task_status("s1", "t001") == "running"
    await store.update_task("s1", "t001", "done", {"summary": "ok"})
    assert await store.get_task_status("s1", "t001") == "done"
    results = await store.get_results("s1")
    assert len(results) == 1
    assert results[0]["status"] == "done"
    assert results[0]["assigned_worker"] == "worker-0"


@pytest.mark.asyncio
async def test_retry_count_increments(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.update_task("s1", "t001", "running", {})
    assert await store.get_retry_count("s1", "t001") == 0
    await store.increment_retry("s1", "t001")
    await store.increment_retry("s1", "t001")
    assert await store.get_retry_count("s1", "t001") == 2


@pytest.mark.asyncio
async def test_completed_tasks_query(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.update_task("s1", "t001", "done", {})
    await store.update_task("s1", "t002", "failed", {})
    await store.update_task("s1", "t003", "done", {})
    completed = await store.get_completed_tasks("s1")
    assert sorted(completed) == ["t001", "t003"]


@pytest.mark.asyncio
async def test_task_ledger_and_stall_count(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.save_task_ledger("s1",
        facts=["python 3.13"], guesses=["maybe needs uv"],
        plan=["t001: build it"],
    )
    ledger = await store.get_task_ledger("s1")
    assert ledger["facts"] == ["python 3.13"]
    assert ledger["stall_count"] == 0

    n1 = await store.increment_stall_count("s1")
    n2 = await store.increment_stall_count("s1")
    assert n1 == 1 and n2 == 2

    await store.reset_stall_count("s1")
    ledger = await store.get_task_ledger("s1")
    assert ledger["stall_count"] == 0


@pytest.mark.asyncio
async def test_list_sessions_with_counts(store):
    await store.save_session("s1", {"goal": "first goal", "plan": {}, "agents": [], "status": "active"})
    await store.save_session("s2", {"goal": "second", "plan": {}, "agents": [], "status": "paused"})
    await store.update_task("s1", "t1", "done", {})
    await store.update_task("s1", "t2", "running", {})
    await store.update_task("s1", "t3", "pending", {})
    await store.update_task("s2", "t1", "failed", {})

    rows = await store.list_sessions(limit=10)
    by_id = {r["id"]: r for r in rows}
    assert by_id["s1"]["done"] == 1
    assert by_id["s1"]["running"] == 1
    assert by_id["s1"]["pending"] == 1
    assert by_id["s1"]["total"] == 3
    assert by_id["s2"]["failed"] == 1


@pytest.mark.asyncio
async def test_reset_running_tasks_for_crash_recovery(store):
    """After a crash, any task stuck in 'running' should be requeued so
    --resume can re-dispatch it instead of leaving it orphaned forever."""
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.update_task("s1", "t1", "done", {})
    await store.assign_task("s1", "t2", "worker-0")  # status='running'
    await store.assign_task("s1", "t3", "worker-1")  # status='running'
    await store.update_task("s1", "t4", "pending", {})

    n = await store.reset_running_tasks("s1")
    assert n == 2

    results = {r["task_id"]: r for r in await store.get_results("s1")}
    assert results["t2"]["status"] == "pending"
    assert results["t2"]["assigned_worker"] is None
    assert results["t3"]["status"] == "pending"
    assert results["t1"]["status"] == "done"
    assert results["t4"]["status"] == "pending"
    # Idempotent: running again resets nothing
    assert await store.reset_running_tasks("s1") == 0


@pytest.mark.asyncio
async def test_checkpoint_storage(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.save_checkpoint("s1", "t001", {"completed": ["t001"], "progress": {"percent": 33}})
    last = await store.get_last_checkpoint("s1")
    assert last["task_id"] == "t001"
    assert last["state"]["progress"]["percent"] == 33


@pytest.mark.asyncio
async def test_worker_upsert_and_query(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    await store.upsert_worker("s1", {
        "worker_id": "w01", "perspective": "implementer", "agent_name": "openclaude",
        "worktree": "/tmp/orch/wt", "branch": "orch/s1/wt", "pane_id": "%1",
        "task": "build it", "state": "running", "summary": "",
        "started_at": 1.0, "finished_at": None,
    })
    rows = await store.get_workers("s1")
    assert len(rows) == 1
    assert rows[0]["state"] == "running"

    # Update — same primary key, new state
    await store.upsert_worker("s1", {
        "worker_id": "w01", "perspective": "implementer", "agent_name": "openclaude",
        "worktree": "/tmp/orch/wt", "branch": "orch/s1/wt", "pane_id": "%1",
        "task": "build it", "state": "done", "summary": "shipped",
        "started_at": 1.0, "finished_at": 2.0,
    })
    rows = await store.get_workers("s1")
    assert len(rows) == 1
    assert rows[0]["state"] == "done"
    assert rows[0]["summary"] == "shipped"


@pytest.mark.asyncio
async def test_reset_running_workers_for_crash_recovery(store):
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    for wid, st in [("w01", "running"), ("w02", "starting"), ("w03", "done")]:
        await store.upsert_worker("s1", {
            "worker_id": wid, "perspective": "implementer",
            "agent_name": "x", "worktree": "/tmp", "branch": "",
            "pane_id": "", "task": "", "state": st, "summary": "",
            "started_at": 1.0, "finished_at": None,
        })
    n = await store.reset_running_workers("s1")
    assert n == 2  # w01 and w02 got moved to 'orphaned'
    rows = {r["worker_id"]: r for r in await store.get_workers("s1")}
    assert rows["w01"]["state"] == "orphaned"
    assert rows["w02"]["state"] == "orphaned"
    assert rows["w03"]["state"] == "done"
    # idempotent
    assert await store.reset_running_workers("s1") == 0
