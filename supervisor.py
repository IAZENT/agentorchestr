"""
supervisor.py — top-level coordinator for ORCH "auto" mode
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
from pathlib import Path
from typing import Optional

from mcp_bridge import build_supervisor_app, write_mcp_config
from state_store import StateStore
from worker_pool import WorkerPool


SUPERVISOR_PROMPT_TEMPLATE = """\
You are the SUPERVISOR for an ORCH multi-agent coding session.

THE GOAL (from the user):
{goal}

YOUR TOOL SURFACE (MCP server "orch"):
  spawn_worker(task, perspective, file_scope?, agent_hint?)
      Launches a worker agent in its own tmux pane + git worktree.
      Perspectives: implementer, tester, reviewer, security,
                    performance, verifier.
  read_worker_output(worker_id, tail_lines?)   - inspect a pane
  send_to_worker(worker_id, message)            - follow-up instruction
  wait_for_worker(worker_id, timeout_s?)        - block until WORKER_DONE
  list_workers()                                - all workers + states
  kill_worker(worker_id)                        - terminate a worker
  cross_review(worker_a, worker_b, focus?)      - peer review
  verify(file_scope, success_criteria, ...)     - run FULL test suite
  report_progress(message)                      - visible in dashboard
  mark_goal_done(summary)                       - finish successfully
  mark_goal_failed(reason)                      - cannot achieve goal

OPERATING PRINCIPLES (Anthropic + AWS CAO + arXiv 2511.16708):
  1. Decompose the goal CONTEXT-CENTRICALLY, not problem-centrically.
     A worker that builds a feature should also write its tests; do
     not split planner/implementer/tester/reviewer if they share most
     of the same context. Split only when context is genuinely
     independent (different modules, different domains).
  2. Effort scaling:
       - simple change       -> 1 implementer + verify()
       - moderate change     -> 1 implementer + 1 reviewer + verify()
       - complex change      -> 2-3 parallel implementers on independent
                                modules, then a security/performance
                                reviewer, then verify()
     Cap at 4 parallel workers unless the goal genuinely cannot be
     decomposed below that.
  3. ALWAYS run verify() before mark_goal_done. The verifier MUST run
     the FULL test suite — never accept partial test runs.
  4. If a worker reports WORKER_BLOCKED, read its output and either
     send_to_worker with a corrective instruction OR spawn a different
     worker with a different perspective. Do not loop forever.
  5. Token budget: keep your own messages short. Read only the tail of
     a worker's pane unless you genuinely need more.
  6. When unsure between two implementations, run them in parallel
     workers and use cross_review() to choose.

RESULT PROTOCOL:
  When everything is satisfied AND the verifier passed, call
  mark_goal_done with a one-paragraph summary.
  If you've concluded the goal cannot be achieved (missing tools,
  ambiguous requirements, repeated failures), call mark_goal_failed
  with a clear reason.

START NOW. Begin by analyzing the goal in 2-3 sentences, then call
list_perspectives() if you want a refresher, then spawn your first
worker. Do not ask the user for clarification unless the goal is
literally undecidable; assume reasonable defaults.
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _supervisor_argv(lead: dict, prompt_path: str) -> list[str]:
    """argv to run the lead agent INTERACTIVELY with `prompt_path` as
    its first user message. Mirrors worker_pool's _agent_argv but is
    duplicated here intentionally so the supervisor's launch path can
    diverge in the future (e.g. allow stdin streaming)."""
    name = lead["name"]
    cmd = lead["cmd"]
    q = shlex.quote
    if name in ("claude", "openclaude", "amp"):
        return ["bash", "-lc", f"{q(cmd)} --dangerously-skip-permissions \"$(cat {q(prompt_path)})\""]
    if name == "kiro" or cmd in ("kiro-cli", "kirocli"):
        return ["bash", "-lc", f"{q(cmd)} chat --trust-all-tools \"$(cat {q(prompt_path)})\""]
    if name == "opencode":
        return ["bash", "-lc", f"{q(cmd)} run \"$(cat {q(prompt_path)})\""]
    if name == "aider":
        return ["bash", "-lc", f"{q(cmd)} --yes-always --message-file {q(prompt_path)}"]
    if name == "codex":
        return ["bash", "-lc", f"{q(cmd)} exec --full-auto \"$(cat {q(prompt_path)})\""]
    if name == "gemini":
        return ["bash", "-lc", f"{q(cmd)} --prompt \"$(cat {q(prompt_path)})\""]
    return ["bash", "-lc", f"{q(cmd)} \"$(cat {q(prompt_path)})\""]


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
    ):
        self.project_root = Path(project_root)
        self.goal = goal
        self.lead_agent = lead_agent
        self.worker_agents = worker_agents
        self.store = store
        self.session_id = session_id
        self.host = host
        self.port = port or _free_port()
        self.pool = WorkerPool(str(self.project_root), session_id, store)
        self._goal_future: asyncio.Future | None = None
        self._mcp_app = None
        self._mcp_task: asyncio.Task | None = None

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
        """
        self._goal_future = asyncio.get_event_loop().create_future()

        # 1. Build MCP server bound to our pool
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
        )

        # 2. Write MCP client config so the lead agent finds the bridge
        write_mcp_config(self.port, transport="sse")

        # 3. Compose the supervisor's first user message
        prompt = SUPERVISOR_PROMPT_TEMPLATE.format(goal=self.goal.strip())
        prompt_dir = Path(f"/tmp/orch-{self.session_id}")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / "supervisor_prompt.txt"
        prompt_path.write_text(prompt)

        # 4. Start tmux session + lead agent in pane 0
        await self.pool.start_session(_supervisor_argv(self.lead_agent, str(prompt_path)))

        # 5. Run the MCP server in this loop until the goal future fires
        self._mcp_task = asyncio.create_task(self._mcp_app.run_sse_async())

        try:
            state, summary = await self._goal_future
        except asyncio.CancelledError:
            state, summary = "cancelled", "supervisor task cancelled"
        finally:
            self._mcp_task.cancel()
            try:
                await self._mcp_task
            except (asyncio.CancelledError, Exception):
                pass

        return state, summary

    async def cleanup(self, *, keep_tmux: bool = True) -> None:
        await self.pool.cleanup(keep_tmux=keep_tmux)
