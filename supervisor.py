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

Your tools are exposed via the MCP server "orch" — list them with
list_tools() if you forget. Key capabilities you should know about:
  • spawn_worker(perspective=…) where perspective is one of
    {{implementer, tester, reviewer, security, performance,
      verifier, researcher}}
  • verify(...) MUST run the FULL test suite before mark_goal_done
  • web_research(query) hits a 24h cache; spawn a 'researcher' worker
    for deeper multi-source synthesis
  • discover_running_agents() finds agents the user opened in other
    terminals so you can dispatch to them without cold-start
  • compact_session() / clear_tool_results() if your context fills up
  • mark_goal_done(summary) | mark_goal_failed(reason) to finish

OPERATING PRINCIPLES:
  1. Decompose CONTEXT-centrically, not problem-centrically. One worker
     that builds + tests beats four workers that hand off.
  2. Effort scaling: simple → 1 impl + verify. Moderate → +1 reviewer.
     Complex → 2–3 parallel implementers (cap 4) + reviewer + verify.
  3. RESEARCH BEFORE BUILDING for anything time-sensitive (library
     versions, recent specs, external APIs). Cached results carry into
     the implementer worker for free.
  4. ALWAYS verify() with the full suite before mark_goal_done.
  5. If a worker reports WORKER_BLOCKED, read the tail and either
     send_to_worker(fix) or spawn a different perspective. Don't loop.
  6. Keep YOUR messages short — your own tokens compound across turns.

START: read the goal below, plan in 2–3 sentences, then act. Don't ask
for clarification unless the goal is undecidable; assume sensible
defaults.

────────────────────────────────────────────────────────────
GOAL:
{goal}
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
        self.pool = WorkerPool(
            str(self.project_root), session_id, store,
            memory=memory, skills=skills,
        )
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
            memory=self.memory,
        )

        # 2. Write MCP client config so the lead agent finds the bridge
        write_mcp_config(self.port, transport="sse")

        # 3. Compose the supervisor's first user message
        memory_preamble = ""
        if self.memory is not None:
            try:
                memory_preamble = await self.memory.load_static_context()
            except Exception:
                memory_preamble = ""
        body = SUPERVISOR_PROMPT_TEMPLATE.format(goal=self.goal.strip())
        if memory_preamble:
            prompt = memory_preamble + "\n\n---\n\n" + body
        else:
            prompt = body
        prompt_dir = Path(f"/tmp/orch-{self.session_id}")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / "supervisor_prompt.txt"
        prompt_path.write_text(prompt)

        # 4. Start tmux session + lead agent in pane 0
        await self.pool.start_session(_supervisor_argv(self.lead_agent, str(prompt_path)))

        # 5. Run the MCP server in this loop until the goal future fires
        self._mcp_task = asyncio.create_task(self._mcp_app.run_sse_async())

        # Defaults so a failure inside the future / finally never leaks an
        # UnboundLocalError out to orchestrator.main().
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
                await self._mcp_task
            except (asyncio.CancelledError, Exception):
                pass

        return state, summary

    async def cleanup(self, *, keep_tmux: bool = True) -> None:
        await self.pool.cleanup(keep_tmux=keep_tmux)
