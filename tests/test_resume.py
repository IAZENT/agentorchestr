"""Tests for the op_log durable-execution / --resume path.

Verifies that:
  1. begin_op returns (op_id, None) on first call.
  2. A second call with the same idempotency key returns (None, prior_result).
  3. finish_op persists the result so the replay path works.
  4. reset_running_workers transitions stale 'running' rows to 'orphaned'.
"""
from __future__ import annotations

import pytest

from agentorchestr.state_store import StateStore


@pytest.fixture
async def store(tmp_path):
    s = StateStore(db_path=str(tmp_path / "state.db"))
    await s.init()
    await s.save_session("sess1", {"goal": "g", "status": "active"})
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_begin_op_first_call(store):
    op_id, prior = await store.begin_op("sess1", "spawn_worker", "key-abc",
                                        {"task": "do x"})
    assert op_id is not None
    assert prior is None


@pytest.mark.asyncio
async def test_begin_op_replay_returns_prior_result(store):
    op_id, _ = await store.begin_op("sess1", "spawn_worker", "key-abc", {})
    await store.finish_op(op_id, {"worker_id": "w01"})

    op_id2, prior = await store.begin_op("sess1", "spawn_worker", "key-abc", {})
    assert op_id2 is None
    assert prior is not None
    assert prior["worker_id"] == "w01"


@pytest.mark.asyncio
async def test_finish_op_persists_result(store):
    op_id, _ = await store.begin_op("sess1", "verify", "key-verify", {})
    await store.finish_op(op_id, {"exit_code": 0, "passed": True})

    _, prior = await store.begin_op("sess1", "verify", "key-verify", {})
    assert prior["exit_code"] == 0
    assert prior["passed"] is True


@pytest.mark.asyncio
async def test_different_keys_are_independent(store):
    id1, _ = await store.begin_op("sess1", "spawn_worker", "key-1", {})
    id2, _ = await store.begin_op("sess1", "spawn_worker", "key-2", {})
    assert id1 != id2
    await store.finish_op(id1, {"worker_id": "w01"})
    await store.finish_op(id2, {"worker_id": "w02"})

    _, p1 = await store.begin_op("sess1", "spawn_worker", "key-1", {})
    _, p2 = await store.begin_op("sess1", "spawn_worker", "key-2", {})
    assert p1["worker_id"] == "w01"
    assert p2["worker_id"] == "w02"


@pytest.mark.asyncio
async def test_reset_running_workers_marks_orphaned(store):
    await store.upsert_worker("sess1", {
        "worker_id": "w01", "perspective": "implementer", "agent_name": "kiro",
        "worktree": "/tmp/x", "branch": "", "pane_id": "", "task": "t",
        "state": "running", "summary": "", "started_at": 0.0, "finished_at": None,
    })
    await store.reset_running_workers("sess1")
    workers = await store.get_workers("sess1")
    assert workers[0]["state"] == "orphaned"


@pytest.mark.asyncio
async def test_get_pending_ops_returns_unfinished(store):
    op_id, _ = await store.begin_op("sess1", "spawn_worker", "key-pending", {})
    pending = await store.get_pending_ops("sess1")
    assert any(op["idempotency_key"] == "key-pending" for op in pending)
    await store.finish_op(op_id, {})
    pending2 = await store.get_pending_ops("sess1")
    assert not any(op["idempotency_key"] == "key-pending" for op in pending2)
