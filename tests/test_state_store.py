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
        "worktree": "/tmp/agentorchestr/wt", "branch": "agentorchestr/s1/wt", "pane_id": "%1",
        "task": "build it", "state": "running", "summary": "",
        "started_at": 1.0, "finished_at": None,
    })
    rows = await store.get_workers("s1")
    assert len(rows) == 1
    assert rows[0]["state"] == "running"

    # Update — same primary key, new state
    await store.upsert_worker("s1", {
        "worker_id": "w01", "perspective": "implementer", "agent_name": "openclaude",
        "worktree": "/tmp/agentorchestr/wt", "branch": "agentorchestr/s1/wt", "pane_id": "%1",
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


@pytest.mark.asyncio
async def test_op_log_idempotent_replay(store):
    """begin_op + finish_op once; second begin_op with the same key returns
    the cached result without re-executing."""
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})

    op_id, prior = await store.begin_op(
        "s1", "spawn_worker", "spawn:abc123",
        {"task": "build it", "perspective": "implementer"},
    )
    assert prior is None
    assert op_id is not None

    await store.finish_op(op_id, {"worker_id": "w01", "branch": "feat/x"})

    # Second call with the SAME key MUST return the cached result.
    op_id2, prior2 = await store.begin_op(
        "s1", "spawn_worker", "spawn:abc123",
        {"task": "different args don't matter", "perspective": "tester"},
    )
    assert op_id2 is None, "completed op must NOT get a new row"
    assert prior2 == {"worker_id": "w01", "branch": "feat/x"}


@pytest.mark.asyncio
async def test_op_log_pending_ops_for_resume(store):
    """An op begun but not finished is still 'pending' — resume sees it."""
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    op_id_a, _ = await store.begin_op("s1", "spawn_worker", "spawn:a", {"x": 1})
    op_id_b, _ = await store.begin_op("s1", "verify",       "verify:b", {"y": 2})
    await store.finish_op(op_id_a, {"worker_id": "w01"})
    # b is still pending
    pending = await store.get_pending_ops("s1")
    assert len(pending) == 1
    assert pending[0]["kind"] == "verify"
    assert pending[0]["idempotency_key"] == "verify:b"
    assert pending[0]["args"] == {"y": 2}


@pytest.mark.asyncio
async def test_op_log_history_survives_failure(store):
    """A failed op stays in the log so resume can retry or skip it."""
    await store.save_session("s1", {"goal": "g", "plan": {}, "agents": [], "status": "active"})
    op_id, _ = await store.begin_op("s1", "verify", "v:1", {})
    await store.finish_op(op_id, {"err": "tests crashed"}, state="failed")
    log = await store.get_op_log("s1")
    assert len(log) == 1
    assert log[0]["state"] == "failed"
    assert log[0]["result"] == {"err": "tests crashed"}
