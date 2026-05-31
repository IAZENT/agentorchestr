"""
mcp_bridge.py — supervisor's tool surface
==========================================

This is the MCP server the lead/supervisor agent connects to. It exposes
the WorkerPool as a set of tools so the lead can:

    spawn_worker(task, perspective, file_scope, agent_hint?)
    read_worker_output(worker_id, tail_lines?)
    send_to_worker(worker_id, message)
    wait_for_worker(worker_id, timeout_s?)
    list_workers()
    kill_worker(worker_id)
    cross_review(worker_a, worker_b, focus)
    verify(file_scope, success_criteria)
    report_progress(message)
    mark_goal_done(summary)
    mark_goal_failed(reason)

The lead is itself a CLI agent (claude / kiro-cli / openclaude / etc.)
that supports MCP. We start the server in-process; the orchestrator
writes MCP config files pointing at it.

Two execution modes:
  1. Server-mode (spawn from CLI):
       python mcp_bridge.py --transport sse --port 8765
     Useful when running the bridge separately for ad-hoc testing.

  2. In-process (used by orchestrator.py):
       from mcp_bridge import build_supervisor_app
       app = build_supervisor_app(worker_pool, ...)
       await app.run_sse_async()
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment]

import perspectives
from worker_pool import WorkerPool, worker_to_row


def build_supervisor_app(
    pool: WorkerPool,
    *,
    available_agents: list[dict],
    default_worker_agent: dict,
    on_goal_done=None,
    on_goal_failed=None,
    host: str = "127.0.0.1",
    port: int = 8765,
):
    """Construct a FastMCP server bound to a live WorkerPool.

    `available_agents`: list of detected agents the supervisor may pick
        from. The supervisor names one in `agent_hint`.
    `default_worker_agent`: fallback when the supervisor doesn't specify.
    `on_goal_done`/`on_goal_failed`: async callbacks the orchestrator
        installs to be notified when the supervisor finishes.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "mcp library not installed. Run: pip install 'mcp>=1.20'"
        )

    mcp = FastMCP("orch-supervisor", host=host, port=port)
    agents_by_name = {a["name"]: a for a in available_agents}

    def _resolve_agent(hint: str | None) -> dict:
        if hint and hint in agents_by_name:
            return agents_by_name[hint]
        return default_worker_agent

    @mcp.tool()
    async def spawn_worker(
        task: str,
        perspective: str = "implementer",
        file_scope: list[str] | None = None,
        pass_criteria: str = "",
        extra_context: str = "",
        agent_hint: str | None = None,
    ) -> str:
        """Launch a worker in a new tmux pane + git worktree.

        perspective: one of {implementer, tester, reviewer, security,
                              performance, verifier}.
        file_scope:  paths the worker should restrict edits to.
        agent_hint:  preferred CLI agent name; falls back to default.
        Returns a JSON string with worker_id and the user-facing tmux
        attach command.
        """
        if not perspectives.is_valid(perspective):
            return json.dumps({
                "error": f"unknown perspective {perspective!r}",
                "valid": perspectives.perspective_names(),
            })
        agent = _resolve_agent(agent_hint)
        worker = await pool.spawn_worker(
            task=task,
            perspective=perspective,
            agent=agent,
            file_scope=list(file_scope or []),
            pass_criteria=pass_criteria,
            extra_context=extra_context,
        )
        return json.dumps({
            "worker_id": worker.id,
            "perspective": worker.perspective,
            "agent": agent["name"],
            "worktree": worker.worktree,
            "branch": worker.branch,
            "tmux_pane": worker.pane_id,
        })

    @mcp.tool()
    def read_worker_output(worker_id: str, tail_lines: int = 80) -> str:
        """Capture the last N lines from a worker's tmux pane."""
        try:
            return pool.read_output(worker_id, tail_lines=tail_lines)
        except KeyError as e:
            return f"[error] {e}"

    @mcp.tool()
    def send_to_worker(worker_id: str, message: str) -> str:
        """Type a follow-up instruction into a running worker's pane.
        Useful for 'fix this bug' / 'try a different approach' loops."""
        try:
            pool.send_message(worker_id, message)
            return json.dumps({"sent": True, "worker_id": worker_id})
        except KeyError as e:
            return json.dumps({"sent": False, "error": str(e)})

    @mcp.tool()
    async def wait_for_worker(worker_id: str, timeout_s: int = 300) -> str:
        """Block until the worker writes WORKER_DONE / WORKER_BLOCKED /
        WORKER_FAILED on its last line, or until timeout."""
        try:
            state, summary = await pool.wait_until_done(
                worker_id, timeout=float(timeout_s)
            )
            return json.dumps({
                "worker_id": worker_id, "state": state, "summary": summary,
            })
        except KeyError as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def list_workers() -> str:
        """List all workers in this session with their current state."""
        out = [
            {
                "worker_id": w.id,
                "perspective": w.perspective,
                "agent": w.agent.get("name"),
                "state": w.state,
                "summary": w.summary,
                "task": w.task[:120],
            }
            for w in pool.list()
        ]
        return json.dumps(out)

    @mcp.tool()
    def kill_worker(worker_id: str) -> str:
        try:
            pool.kill_worker(worker_id)
            return json.dumps({"killed": True, "worker_id": worker_id})
        except KeyError as e:
            return json.dumps({"killed": False, "error": str(e)})

    @mcp.tool()
    async def cross_review(
        worker_a: str,
        worker_b: str,
        focus: str = "correctness",
    ) -> str:
        """Have worker A review worker B's diff and vice-versa.

        Implementation: send a focused review prompt to each worker's
        pane referring to the other's worktree. Returns a JSON object
        with both reviews (their tail output after the review).
        """
        try:
            wa, wb = pool.get(worker_a), pool.get(worker_b)
        except KeyError as e:
            return json.dumps({"error": str(e)})

        prompt_a = (
            f"Cross-review request: read the diff in {wb.worktree} "
            f"(branch {wb.branch}) against this project's main branch. "
            f"Focus: {focus}. Write up to 8 numbered concerns, then end "
            f"with WORKER_DONE."
        )
        prompt_b = prompt_a.replace(wb.worktree, wa.worktree).replace(
            wb.branch, wa.branch
        )
        pool.send_message(worker_a, prompt_a)
        pool.send_message(worker_b, prompt_b)
        # Don't block on completion — supervisor can read outputs later.
        return json.dumps({
            "review_started": True,
            "tip": "call read_worker_output for each worker after a few seconds",
        })

    @mcp.tool()
    async def verify(
        file_scope: list[str] | None = None,
        success_criteria: str = "all tests pass, no lint errors",
        agent_hint: str | None = None,
    ) -> str:
        """Spawn a verifier subagent that runs the FULL test suite.

        Implements Anthropic's 'verification subagent' pattern with the
        explicit 'no shortcuts' instruction. Blocks until the verifier
        produces WORKER_DONE / WORKER_FAILED.
        """
        agent = _resolve_agent(agent_hint)
        verifier = await pool.spawn_worker(
            task=(
                "Verify the current changes meet the success criteria. "
                "Run the project's FULL test suite and any linters / "
                "type-checkers. Do NOT stop after a subset of tests."
            ),
            perspective="verifier",
            agent=agent,
            file_scope=list(file_scope or []),
            pass_criteria=success_criteria,
        )
        state, summary = await pool.wait_until_done(verifier.id, timeout=600.0)
        return json.dumps({
            "verifier_id": verifier.id,
            "state": state,
            "summary": summary,
        })

    @mcp.tool()
    def report_progress(message: str) -> str:
        """Append a progress note (visible in the dashboard)."""
        log = Path(f"/tmp/orch-{pool.orch_session_id}/progress.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(message.rstrip() + "\n")
        return json.dumps({"logged": True})

    @mcp.tool()
    async def mark_goal_done(summary: str) -> str:
        """Tell the orchestrator the supervisor is finished successfully.
        Triggers run shutdown."""
        if on_goal_done is not None:
            await on_goal_done(summary)
        return json.dumps({"acknowledged": True, "state": "done"})

    @mcp.tool()
    async def mark_goal_failed(reason: str) -> str:
        """Tell the orchestrator the supervisor cannot achieve the goal."""
        if on_goal_failed is not None:
            await on_goal_failed(reason)
        return json.dumps({"acknowledged": True, "state": "failed"})

    @mcp.tool()
    def list_perspectives() -> str:
        """Names of valid worker perspectives."""
        return json.dumps(perspectives.perspective_names())

    return mcp


# ── standalone CLI (kept for ad-hoc use + config writing) ─────────────

def write_mcp_config(port: int = 8765, transport: str = "sse") -> None:
    """Write MCP client config files for all supported agents."""
    url = f"http://localhost:{port}/sse"
    config = {
        "mcpServers": {
            "orch": {
                "url": url if transport == "sse" else None,
                "command": "python" if transport == "stdio" else None,
                "args": [str(Path(__file__).resolve()), "--transport", "stdio"]
                        if transport == "stdio" else None,
                "description": "ORCH supervisor bridge",
            }
        }
    }
    config["mcpServers"]["orch"] = {
        k: v for k, v in config["mcpServers"]["orch"].items() if v is not None
    }
    locations = {
        "Kiro":        os.path.expanduser("~/.kiro/settings/mcp.json"),
        "Claude Code": os.path.expanduser("~/.claude/mcp.json"),
        "Cursor":      os.path.expanduser("~/.cursor/mcp.json"),
        "Windsurf":    os.path.expanduser("~/.windsurf/mcp.json"),
    }
    for agent, path in locations.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Wrote MCP config for {agent}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ORCH supervisor MCP bridge")
    parser.add_argument("--transport", choices=["sse", "stdio", "streamable-http"], default="sse")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--write-configs", action="store_true",
                        help="Write MCP client config files for known agents")
    args = parser.parse_args()

    if args.write_configs:
        write_mcp_config(args.port, transport=args.transport)
        return 0

    if not MCP_AVAILABLE:
        print("ERROR: mcp not installed.  pip install 'mcp>=1.20'", file=sys.stderr)
        return 1

    print(
        "This module is a library used by the orchestrator. "
        "To run a standalone tool-stub server (no WorkerPool), use "
        "`python mcp_bridge.py --write-configs` to write client configs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
