#!/usr/bin/env python3
"""
state_store.py — SQLite Persistence + Checkpointing
=====================================================
Inspired by: LangGraph checkpointing, CrewAI state persistence, OpenAI Sessions

Provides async SQLite storage for sessions, tasks, and checkpoints.
Enables resume without re-running completed tasks.
"""

import json
import time
import aiosqlite
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / ".orch" / "state.db"


class StateStore:
    """Async SQLite state store for session/task persistence and checkpointing."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self):
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                plan TEXT,
                agents TEXT,
                status TEXT DEFAULT 'active',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                result TEXT,
                retry_count INTEGER DEFAULT 0,
                assigned_worker TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, task_id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                state_snapshot TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS task_ledger (
                session_id TEXT PRIMARY KEY,
                facts TEXT,
                guesses TEXT,
                plan TEXT,
                stall_count INTEGER DEFAULT 0,
                updated_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS workers (
                session_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                perspective TEXT,
                agent_name TEXT,
                worktree TEXT,
                branch TEXT,
                pane_id TEXT,
                task TEXT,
                state TEXT DEFAULT 'starting',
                summary TEXT,
                started_at REAL,
                finished_at REAL,
                PRIMARY KEY (session_id, worker_id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(session_id, status);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id);
            CREATE INDEX IF NOT EXISTS idx_workers_session ON workers(session_id);
            CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(session_id, state);
        """)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    # ── Session CRUD ──────────────────────────────────────────────

    async def save_session(self, session_id: str, data: dict):
        """Save or update a session."""
        now = time.time()
        existing = await self.get_session(session_id)

        if existing:
            # Merge data into existing session
            updates = []
            params = []
            for key in ("goal", "plan", "agents", "status"):
                if key in data:
                    updates.append(f"{key} = ?")
                    val = data[key]
                    params.append(json.dumps(val) if isinstance(val, (dict, list)) else val)
            if updates:
                updates.append("updated_at = ?")
                params.append(now)
                params.append(session_id)
                await self._db.execute(
                    f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
                    params
                )
        else:
            await self._db.execute(
                "INSERT INTO sessions (id, goal, plan, agents, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    data.get("goal", ""),
                    json.dumps(data.get("plan", {})) if isinstance(data.get("plan"), dict) else data.get("plan"),
                    json.dumps(data.get("agents", [])) if isinstance(data.get("agents"), list) else data.get("agents"),
                    data.get("status", "active"),
                    now, now
                )
            )
        await self._db.commit()

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID."""
        async with self._db.execute(
            "SELECT id, goal, plan, agents, status, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "goal": row[1],
                "plan": json.loads(row[2]) if row[2] else None,
                "agents": json.loads(row[3]) if row[3] else [],
                "status": row[4],
                "created_at": row[5],
                "updated_at": row[6],
            }

    # ── Task State Transitions ────────────────────────────────────

    async def update_task(self, session_id: str, task_id: str, status: str, result: dict = None):
        """Update or insert a task status."""
        now = time.time()
        result_json = json.dumps(result) if result else None

        await self._db.execute("""
            INSERT INTO tasks (session_id, task_id, status, result, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, task_id) DO UPDATE SET
                status = excluded.status,
                result = COALESCE(excluded.result, tasks.result),
                updated_at = excluded.updated_at
        """, (session_id, task_id, status, result_json, now, now))
        await self._db.commit()

    async def assign_task(self, session_id: str, task_id: str, worker_id: str):
        """Record which worker is assigned to a task."""
        now = time.time()
        await self._db.execute("""
            INSERT INTO tasks (session_id, task_id, status, assigned_worker, created_at, updated_at)
            VALUES (?, ?, 'running', ?, ?, ?)
            ON CONFLICT(session_id, task_id) DO UPDATE SET
                assigned_worker = excluded.assigned_worker,
                status = 'running',
                updated_at = excluded.updated_at
        """, (session_id, task_id, worker_id, now, now))
        await self._db.commit()

    async def get_results(self, session_id: str) -> list[dict]:
        """Get all task results for a session."""
        async with self._db.execute(
            "SELECT task_id, status, result, retry_count, assigned_worker FROM tasks WHERE session_id = ?",
            (session_id,)
        ) as cursor:
            results = []
            async for row in cursor:
                results.append({
                    "task_id": row[0],
                    "status": row[1],
                    "result": json.loads(row[2]) if row[2] else None,
                    "retry_count": row[3],
                    "assigned_worker": row[4],
                })
            return results

    async def get_task_status(self, session_id: str, task_id: str) -> Optional[str]:
        """Get current status of a specific task."""
        async with self._db.execute(
            "SELECT status FROM tasks WHERE session_id = ? AND task_id = ?",
            (session_id, task_id)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    # ── Retry Tracking ────────────────────────────────────────────

    async def get_retry_count(self, session_id: str, task_id: str) -> int:
        """Get current retry count for a task."""
        async with self._db.execute(
            "SELECT retry_count FROM tasks WHERE session_id = ? AND task_id = ?",
            (session_id, task_id)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def increment_retry(self, session_id: str, task_id: str):
        """Increment retry count for a task."""
        await self._db.execute("""
            UPDATE tasks SET retry_count = retry_count + 1, updated_at = ?
            WHERE session_id = ? AND task_id = ?
        """, (time.time(), session_id, task_id))
        await self._db.commit()

    # ── Checkpointing (LangGraph/CrewAI-inspired) ────────────────

    async def save_checkpoint(self, session_id: str, task_id: str, state: dict):
        """Save a checkpoint after task completion."""
        await self._db.execute(
            "INSERT INTO checkpoints (session_id, task_id, state_snapshot, created_at) VALUES (?, ?, ?, ?)",
            (session_id, task_id, json.dumps(state), time.time())
        )
        await self._db.commit()

    async def get_last_checkpoint(self, session_id: str) -> Optional[dict]:
        """Get the most recent checkpoint for a session."""
        async with self._db.execute(
            "SELECT task_id, state_snapshot, created_at FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "task_id": row[0],
                "state": json.loads(row[1]),
                "created_at": row[2],
            }

    async def get_completed_tasks(self, session_id: str) -> list[str]:
        """Get list of completed task IDs for resume."""
        async with self._db.execute(
            "SELECT task_id FROM tasks WHERE session_id = ? AND status = 'done'",
            (session_id,)
        ) as cursor:
            return [row[0] async for row in cursor]

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions with done/total counts (for --list-sessions)."""
        rows: list[dict] = []
        async with self._db.execute(
            "SELECT id, goal, status, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ) as cursor:
            async for r in cursor:
                rows.append({
                    "id": r[0], "goal": r[1], "status": r[2],
                    "created_at": r[3], "updated_at": r[4],
                })
        # Annotate with task counts
        for row in rows:
            async with self._db.execute(
                "SELECT status, COUNT(*) FROM tasks WHERE session_id = ? GROUP BY status",
                (row["id"],)
            ) as c:
                counts = {s: n async for s, n in c}
            row["done"] = counts.get("done", 0)
            row["failed"] = counts.get("failed", 0)
            row["running"] = counts.get("running", 0)
            row["pending"] = counts.get("pending", 0)
            row["total"] = sum(counts.values())
        return rows

    async def reset_running_tasks(self, session_id: str) -> int:
        """Crash recovery: any task left in `running` after a crash is
        re-queued as `pending` so it can be reassigned on resume.

        Returns the number of tasks that were reset.
        """
        async with self._db.execute(
            "SELECT COUNT(*) FROM tasks WHERE session_id = ? AND status = 'running'",
            (session_id,)
        ) as c:
            row = await c.fetchone()
            n = row[0] if row else 0
        if n:
            await self._db.execute(
                "UPDATE tasks SET status = 'pending', assigned_worker = NULL, updated_at = ? "
                "WHERE session_id = ? AND status = 'running'",
                (time.time(), session_id),
            )
            await self._db.commit()
        return n

    # ── Task Ledger (Magentic-One-inspired) ───────────────────────

    async def save_task_ledger(self, session_id: str, facts: list, guesses: list, plan: list):
        """Save/update the task ledger (Magentic-One outer loop)."""
        now = time.time()
        await self._db.execute("""
            INSERT INTO task_ledger (session_id, facts, guesses, plan, stall_count, updated_at)
            VALUES (?, ?, ?, ?, 0, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                facts = excluded.facts,
                guesses = excluded.guesses,
                plan = excluded.plan,
                stall_count = 0,
                updated_at = excluded.updated_at
        """, (session_id, json.dumps(facts), json.dumps(guesses), json.dumps(plan), now))
        await self._db.commit()

    async def get_task_ledger(self, session_id: str) -> Optional[dict]:
        """Get the current task ledger."""
        async with self._db.execute(
            "SELECT facts, guesses, plan, stall_count FROM task_ledger WHERE session_id = ?",
            (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "facts": json.loads(row[0]) if row[0] else [],
                "guesses": json.loads(row[1]) if row[1] else [],
                "plan": json.loads(row[2]) if row[2] else [],
                "stall_count": row[3],
            }

    async def increment_stall_count(self, session_id: str) -> int:
        """Increment stall count and return new value."""
        await self._db.execute("""
            UPDATE task_ledger SET stall_count = stall_count + 1, updated_at = ?
            WHERE session_id = ?
        """, (time.time(), session_id))
        await self._db.commit()
        async with self._db.execute(
            "SELECT stall_count FROM task_ledger WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def reset_stall_count(self, session_id: str):
        """Reset stall count after progress."""
        await self._db.execute(
            "UPDATE task_ledger SET stall_count = 0, updated_at = ? WHERE session_id = ?",
            (time.time(), session_id)
        )
        await self._db.commit()

    # ── Worker rows (supervisor mode) ──────────────────────────────

    async def upsert_worker(self, session_id: str, row: dict) -> None:
        """Insert or update a worker row.

        `row` keys: worker_id, perspective, agent_name, worktree, branch,
        pane_id, task, state, summary, started_at, finished_at.
        """
        await self._db.execute("""
            INSERT INTO workers
              (session_id, worker_id, perspective, agent_name, worktree,
               branch, pane_id, task, state, summary, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, worker_id) DO UPDATE SET
              perspective  = excluded.perspective,
              agent_name   = excluded.agent_name,
              worktree     = excluded.worktree,
              branch       = excluded.branch,
              pane_id      = excluded.pane_id,
              task         = excluded.task,
              state        = excluded.state,
              summary      = excluded.summary,
              finished_at  = excluded.finished_at
        """, (
            session_id,
            row["worker_id"],
            row.get("perspective", ""),
            row.get("agent_name", ""),
            row.get("worktree", ""),
            row.get("branch", ""),
            row.get("pane_id", "") or "",
            row.get("task", ""),
            row.get("state", "starting"),
            row.get("summary", ""),
            row.get("started_at"),
            row.get("finished_at"),
        ))
        await self._db.commit()

    async def get_workers(self, session_id: str) -> list[dict]:
        async with self._db.execute(
            "SELECT worker_id, perspective, agent_name, worktree, branch, "
            "pane_id, task, state, summary, started_at, finished_at "
            "FROM workers WHERE session_id = ? ORDER BY started_at",
            (session_id,),
        ) as cur:
            rows = []
            async for r in cur:
                rows.append({
                    "worker_id": r[0], "perspective": r[1], "agent_name": r[2],
                    "worktree": r[3], "branch": r[4], "pane_id": r[5],
                    "task": r[6], "state": r[7], "summary": r[8],
                    "started_at": r[9], "finished_at": r[10],
                })
            return rows

    async def reset_running_workers(self, session_id: str) -> int:
        """Crash recovery: any worker stuck in 'starting'/'running' becomes
        'orphaned' so the supervisor can re-spawn or skip on resume."""
        async with self._db.execute(
            "SELECT COUNT(*) FROM workers WHERE session_id = ? "
            "AND state IN ('starting','running')",
            (session_id,),
        ) as c:
            row = await c.fetchone()
            n = row[0] if row else 0
        if n:
            await self._db.execute(
                "UPDATE workers SET state='orphaned', finished_at=? "
                "WHERE session_id=? AND state IN ('starting','running')",
                (time.time(), session_id),
            )
            await self._db.commit()
        return n
