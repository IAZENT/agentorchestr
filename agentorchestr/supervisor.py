"""
supervisor.py — top-level coordinator for agentorchestr "auto" mode
============================================================

Flow:
  1. Pick a free port for the MCP server.
  2. Build the WorkerPool and write MCP client config so the lead agent
     can find the bridge.
  3. Compose the supervisor system prompt (goal + tool reference).
  4. Start a tmux session; pane 0 runs the lead agent INTERACTIVELY with
     that prompt.
  5. Start the MCP server in the same Python process.
  6. Block on a future that resolves when the lead calls
     `mark_goal_done` or `mark_goal_failed`.
  7. On exit: keep tmux alive (so the user can scroll back), prune
     git worktrees, persist final state.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import socket
import time
from pathlib import Path
from typing import Optional

from agentorchestr.mcp_bridge import (
    DEFAULT_MCP_TRANSPORT,
    build_supervisor_app,
    remove_mcp_config,
    write_mcp_config,
)
from agentorchestr.state_store import StateStore
from agentorchestr.worker_pool import WorkerPool


SUPERVISOR_PROMPT_TEMPLATE = """\
You are the SUPERVISOR for an agentorchestr multi-agent coding session.

═══════════════════════════════════════════════════════
YOUR ROLE — READ THIS CAREFULLY
═══════════════════════════════════════════════════════
You are a PURE ORCHESTRATOR. You do NOT write code, edit files, or run
commands directly. Your only job is to:
  1. Decompose the goal into tasks.
  2. Spawn the right workers for each task via spawn_worker().
  3. Monitor their output via read_worker_output().
  4. Send corrections via send_to_worker() when a worker is stuck.
  5. Call verify() before finishing.
  6. Call mark_goal_done(summary) or mark_goal_failed(reason) to end.

If you find yourself writing code or editing a file — STOP. Spawn a
worker instead.

═══════════════════════════════════════════════════════
MANDATORY FIRST ACTION
═══════════════════════════════════════════════════════
You MUST call spawn_worker() as your FIRST tool call, always.
Even for a one-line fix, spawn an implementer. Never attempt the work
yourself.

═══════════════════════════════════════════════════════
AVAILABLE TOOLS (via MCP server "agentorchestr")
═══════════════════════════════════════════════════════
  spawn_worker(task, perspective, file_scope?, agent_hint?)
      perspective ∈ {{implementer, tester, reviewer, security,
                      performance, verifier, researcher}}
  read_worker_output(worker_id, tail_lines?)
  send_to_worker(worker_id, message)
  wait_for_worker(worker_id, timeout_s?)
  list_workers()
  kill_worker(worker_id)
  cross_review(worker_a, worker_b, focus)
  verify(file_scope?, success_criteria?)   ← spawns a verifier internally
  web_research(query)                      ← 24h cache, fast
  discover_running_agents()               ← find agents in other terminals
  compact_session() / clear_tool_results() ← if context fills up
  report_progress(message)
  mark_goal_done(summary)
  mark_goal_failed(reason)

═══════════════════════════════════════════════════════
WORKER SCALING RULES
═══════════════════════════════════════════════════════
  Trivial  (1 file, clear fix):  1 implementer → verify → done
  Simple   (few files, 1 concern): implementer + verify
  Moderate (new feature):         implementer + tester + reviewer + verify
  Complex  (multiple subsystems): researcher first, then 2–3 parallel
                                  implementers (max 4) + reviewer + verify
  Security-sensitive: always add a security worker alongside implementer.

Research BEFORE implementing when: external APIs, library versions,
recent spec changes, or anything you're uncertain about.

═══════════════════════════════════════════════════════
COMPLETION RULE
═══════════════════════════════════════════════════════
ALWAYS call verify() before mark_goal_done(). Never skip it.
If verify() fails: spawn a new implementer with the failure as context,
then verify() again. Max 3 fix attempts before mark_goal_failed().

If a worker reports WORKER_BLOCKED: read its tail with
read_worker_output(), diagnose the issue, send_to_worker() with a
concrete fix, or kill_worker() and respawn with a different perspective.

Keep YOUR messages concise — your own token usage compounds per turn.

════════════════════════════════════════════════════════
GOAL:
{goal}
════════════════════════════════════════════════════════

Now: plan in 2–3 sentences (internally), then immediately call
spawn_worker() — do not ask for clarification unless the goal is
completely undecidable.
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_port(host: str, port: int, *, timeout: float = 10.0, interval: float = 0.15) -> bool:
    """Poll until host:port accepts TCP connections or timeout expires.

    Returns True if the port opened in time, False otherwise.
    CancelledError propagates immediately so task cancellation is never eaten.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=interval
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except asyncio.CancelledError:
            raise  # never swallow cancellation
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            await asyncio.sleep(interval)
    return False


def _supervisor_argv(lead: dict, prompt_path: str) -> list[str]:
    """argv to run the lead agent INTERACTIVELY with `prompt_path` as
    its first user message.

    Thin delegate to agent_launcher.build_argv with role='supervisor', so
    the worker-launch path in WorkerPool and this lead-launch path stay in
    sync without a duplicated per-agent if/elif chain.
    """
    from agentorchestr.agent_launcher import build_argv
    return build_argv(lead, prompt_path, role="supervisor")


class Supervisor:
    def __init__(
        self,
        *,
        project_root: str,
        goal: str,
        lead_agent: dict,
        worker_agents: list[dict],
        store: StateStore,
        session_id: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        memory: object | None = None,
        skills: object | None = None,
    ):
        self.project_root = Path(project_root)
        self.goal = goal
        self.lead_agent = lead_agent
        self.worker_agents = worker_agents
        self.store = store
        self.session_id = session_id
        self.host = host
        self.port = port or _free_port()
        self.memory = memory
        self.skills = skills
        self.mcp_transport = DEFAULT_MCP_TRANSPORT
        self.pool = WorkerPool(
            str(self.project_root), session_id, store,
            memory=memory, skills=skills,
        )
        self._goal_future: asyncio.Future | None = None
        self._mcp_app = None
        self._mcp_task: asyncio.Task | None = None
        self._written_mcp_paths: list[str] = []  # cleaned up on exit
        self._mcp_entry_key: str | None = None

    @property
    def attach_command(self) -> str:
        return self.pool.attach_command

    async def _on_goal_done(self, summary: str) -> None:
        if self._goal_future and not self._goal_future.done():
            self._goal_future.set_result(("done", summary))

    async def _on_goal_failed(self, reason: str) -> None:
        if self._goal_future and not self._goal_future.done():
            self._goal_future.set_result(("failed", reason))

    async def run(self) -> tuple[str, str]:
        """Run the supervisor session. Returns (state, summary).

        state: 'done' | 'failed' | 'cancelled' | 'timeout'.

        Correct startup order (fixes the race condition where the lead agent
        tried to connect before the MCP server was bound):
          1. Build the MCP app object (no I/O yet).
          2. Start the MCP server as a background task.
          3. Wait until the port is actually accepting connections.
          4. Write MCP config files pointing at the live port.
          5. Compose the supervisor prompt.
          6. Launch the lead agent in tmux — it can now connect immediately.
        """
        self._goal_future = asyncio.get_event_loop().create_future()

        # 1. Build MCP server (no network I/O yet)
        default_worker_agent = (
            self.worker_agents[0] if self.worker_agents else self.lead_agent
        )
        self._mcp_app = build_supervisor_app(
            self.pool,
            available_agents=self.worker_agents or [self.lead_agent],
            default_worker_agent=default_worker_agent,
            on_goal_done=self._on_goal_done,
            on_goal_failed=self._on_goal_failed,
            host=self.host,
            port=self.port,
            memory=self.memory,
        )

        # 2. Start MCP server in the background FIRST
        mcp_transport = getattr(self, "mcp_transport", DEFAULT_MCP_TRANSPORT)
        if mcp_transport == "streamable-http" and hasattr(self._mcp_app, "run_streamable_http_async"):
            self._mcp_task = asyncio.create_task(self._mcp_app.run_streamable_http_async())
        else:
            self._mcp_task = asyncio.create_task(self._mcp_app.run_sse_async())

        def _on_mcp_done(t: asyncio.Task) -> None:
            if self._goal_future and not self._goal_future.done():
                if t.cancelled():
                    return
                exc = t.exception()
                if exc:
                    self._goal_future.set_result(
                        ("failed", f"MCP server crashed: {exc}")
                    )

        self._mcp_task.add_done_callback(_on_mcp_done)

        # 3. Wait until the MCP server is actually accepting connections.
        #    This eliminates the race where the lead agent starts and gets
        #    "connection refused" before the server has bound its port.
        ready = await _wait_for_port(self.host, self.port, timeout=12.0)
        if not ready:
            self._mcp_task.cancel()
            return "failed", f"MCP server did not bind to {self.host}:{self.port} within 12s"

        # 4. Write MCP config — server is live so the URL is valid now.
        #    Use project-local paths so we never corrupt the user's global
        #    MCP config. Only write the lead agent's config file.
        self._mcp_entry_key = f"agentorchestr-{self.session_id}"
        self._written_mcp_paths = write_mcp_config(
            self.port,
            transport=mcp_transport,
            project_root=str(self.project_root),
            lead_agent_name=self.lead_agent.get("name"),
            session_id=self.session_id,
        )

        # 5. Compose the supervisor's first user message
        memory_preamble = ""
        if self.memory is not None:
            try:
                memory_preamble = await self.memory.load_static_context()
            except Exception:
                memory_preamble = ""
        body = SUPERVISOR_PROMPT_TEMPLATE.format(goal=self.goal.strip())
        prompt = (memory_preamble + "\n\n---\n\n" + body) if memory_preamble else body
        prompt_dir = Path(f"/tmp/agentorchestr-{self.session_id}")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / "supervisor_prompt.txt"
        prompt_path.write_text(prompt)

        # 6. Launch lead agent in tmux — MCP is already live, so first
        #    tool call from the lead will succeed immediately.
        await self.pool.start_session(_supervisor_argv(self.lead_agent, str(prompt_path)))

        state: str = "failed"
        summary: str = "supervisor never reached completion"
        try:
            state, summary = await self._goal_future
        except asyncio.CancelledError:
            state, summary = "cancelled", "supervisor task cancelled"
        except Exception as e:  # noqa: BLE001
            state, summary = "failed", f"{type(e).__name__}: {e}"
        finally:
            self._mcp_task.cancel()
            try:
                await asyncio.wait_for(self._mcp_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        return state, summary

    async def cleanup(self, *, keep_tmux: bool = True) -> None:
        # Remove the agentorchestr entry from every config file we wrote.
        # This restores the user's original MCP server list exactly.
        if self._written_mcp_paths:
            remove_mcp_config(
                self._written_mcp_paths,
                entry_key=self._mcp_entry_key or "agentorchestr",
            )
        await self.pool.cleanup(keep_tmux=keep_tmux)