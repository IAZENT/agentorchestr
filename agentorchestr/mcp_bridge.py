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

from agentorchestr.agent_detector import AGENT_REGISTRY

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment]

DEFAULT_MCP_TRANSPORT = "streamable-http" if MCP_AVAILABLE and hasattr(FastMCP, "run_streamable_http_async") else "sse"


def build_supervisor_app(
    pool: WorkerPool,
    *,
    available_agents: list[dict],
    default_worker_agent: dict,
    on_goal_done=None,
    on_goal_failed=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    memory: object | None = None,
):
    """Construct a FastMCP server bound to a live WorkerPool.

    `available_agents`: list of detected agents the supervisor may pick
        from. The supervisor names one in `agent_hint`.
    `default_worker_agent`: fallback when the supervisor doesn't specify.
    `on_goal_done`/`on_goal_failed`: async callbacks the orchestrator
        installs to be notified when the supervisor finishes.
    `memory`: optional MemoryFederation; enables cache-aware web_research.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "mcp library not installed. Run: pip install 'mcp>=1.20'"
        )

    mcp = FastMCP("agentorchestr-supervisor", host=host, port=port)
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
        """Launch a worker (tmux pane + git worktree). perspective ∈ {implementer, tester, reviewer, security, performance, verifier, researcher}."""
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
        """Send a follow-up instruction into a running worker's pane."""
        try:
            pool.send_message(worker_id, message)
            return json.dumps({"sent": True, "worker_id": worker_id})
        except KeyError as e:
            return json.dumps({"sent": False, "error": str(e)})

    @mcp.tool()
    async def wait_for_worker(worker_id: str, timeout_s: int = 300) -> str:
        """Block until the worker writes WORKER_DONE/BLOCKED/FAILED, or timeout."""
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
    async def kill_worker(worker_id: str) -> str:
        """Terminate a worker (SIGTERM, then SIGKILL fallback)."""
        try:
            await pool.kill_worker(worker_id)
            return json.dumps({"killed": True, "worker_id": worker_id})
        except KeyError as e:
            return json.dumps({"killed": False, "error": str(e)})

    @mcp.tool()
    async def cross_review(
        worker_a: str,
        worker_b: str,
        focus: str = "correctness",
    ) -> str:
        """Have worker A review B's diff and vice-versa. Non-blocking."""
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
        """Spawn a verifier subagent that runs the FULL test suite. Blocking, 600s timeout."""
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
        log = Path(f"/tmp/agentorchestr-{pool.agentorchestr_session_id}/progress.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(message.rstrip() + "\n")
        return json.dumps({"logged": True})

    @mcp.tool()
    async def mark_goal_done(summary: str) -> str:
        """Signal goal completion to the orchestrator. Triggers run shutdown."""
        if on_goal_done is not None:
            await on_goal_done(summary)
        return json.dumps({"acknowledged": True, "state": "done"})

    @mcp.tool()
    async def mark_goal_failed(reason: str) -> str:
        """Signal that the goal cannot be achieved. Triggers run shutdown."""
        if on_goal_failed is not None:
            await on_goal_failed(reason)
        return json.dumps({"acknowledged": True, "state": "failed"})

    @mcp.tool()
    def list_perspectives() -> str:
        """Names of valid worker perspectives."""
        return json.dumps(perspectives.perspective_names())

    @mcp.tool()
    async def discover_running_agents(include_tmux: bool = True,
                                       mdns_timeout: float = 1.0) -> str:
        """Find CLI agents already running on this host (mDNS + fs cards + tmux scan)."""
        from agentorchestr.discovery import discover  # local import keeps mcp_bridge cheap
        agents = await discover(mdns_timeout=mdns_timeout, include_tmux=include_tmux)
        return json.dumps([a.to_dict() for a in agents])

    @mcp.tool()
    async def send_to_external_agent(agent_id: str, text: str) -> str:
        """Send input to a shim-wrapped discovered agent (transport='shim_socket' only)."""
        from agentorchestr.discovery import discover, ShimClient
        agents = await discover(include_tmux=False)
        match = next((a for a in agents if a.id == agent_id), None)
        if not match or not match.socket:
            return json.dumps({"sent": False, "error": f"agent {agent_id!r} not found or no socket"})
        client = ShimClient(match.socket)
        result = await client.send(text)
        return json.dumps({"agent_id": agent_id, **result})

    @mcp.tool()
    async def read_from_external_agent(agent_id: str, max_bytes: int = 4096) -> str:
        """Capture recent output from a shim-wrapped discovered agent."""
        from agentorchestr.discovery import discover, ShimClient
        agents = await discover(include_tmux=False)
        match = next((a for a in agents if a.id == agent_id), None)
        if not match or not match.socket:
            return json.dumps({"output": "", "error": f"agent {agent_id!r} not found or no socket"})
        client = ShimClient(match.socket)
        result = await client.read(max_bytes=max_bytes)
        return json.dumps({"agent_id": agent_id, **result})

    @mcp.tool()
    async def web_research(query: str, max_results: int = 5,
                            fetch_top_n: int = 2,
                            force_refresh: bool = False,
                            ttl_hours: float = 24.0) -> str:
        """DuckDuckGo search + body fetch. 24h cache; auto-persists into project memory."""
        from agentorchestr.research import research as _do_research
        if not query or not query.strip():
            return json.dumps({"error": "empty query"})

        # 1. Cache hit?
        if memory is not None and not force_refresh:
            try:
                cached = await memory.get_cached_research(query)
            except Exception:
                cached = None
            if cached is not None:
                return json.dumps({**cached, "cached": True})

        # 2. Live search.
        payload = await _do_research(query, k=max_results, fetch_top_n=fetch_top_n)
        payload_dict = payload.to_dict()

        # 3. Persist for next time.
        if memory is not None and payload.results:
            try:
                await memory.cache_research(query, payload_dict, ttl_hours=ttl_hours)
            except Exception:
                pass

        return json.dumps({**payload_dict, "cached": False})

    @mcp.tool()
    async def compact_session(keep_last_n: int = 3,
                               include_running: bool = False) -> str:
        """Rewrite older finished workers' summaries as short stubs to keep the context window below 70%."""
        rows = await pool.store.get_workers(pool.agentorchestr_session_id)
        # Order: oldest first.
        rows.sort(key=lambda r: r.get("started_at") or 0.0)
        finished = [
            r for r in rows
            if include_running or r.get("state") in ("done", "failed", "killed", "blocked", "timeout")
        ]
        # Keep the most recent keep_last_n out of compaction.
        if keep_last_n > 0:
            finished = finished[:-keep_last_n] if len(finished) > keep_last_n else []

        compacted_count = 0
        for r in finished:
            old_summary = (r.get("summary") or "").strip()
            if old_summary.startswith("[compacted]"):
                continue  # already compacted
            new_summary = f"[compacted] {r.get('perspective','?')}/{r.get('state','?')} — " + (
                (old_summary[:120] + "…") if len(old_summary) > 120 else old_summary
            )
            r2 = dict(r)
            r2["summary"] = new_summary
            await pool.store.upsert_worker(pool.agentorchestr_session_id, r2)
            compacted_count += 1

        return json.dumps({
            "compacted": compacted_count,
            "kept_recent": min(keep_last_n, len(rows)),
            "total_workers": len(rows),
        })

    @mcp.tool()
    def clear_tool_results(progress_log: bool = True) -> str:
        """Truncate /tmp/agentorchestr-<session>/progress.log to drop noise from the next iteration."""
        cleared: list[str] = []
        if progress_log:
            log = Path(f"/tmp/agentorchestr-{pool.agentorchestr_session_id}/progress.log")
            if log.exists():
                try:
                    log.write_text("")  # truncate
                    cleared.append(str(log))
                except OSError as e:
                    return json.dumps({"cleared": cleared, "error": str(e)})
        return json.dumps({"cleared": cleared})

    return mcp


# ── standalone CLI (kept for ad-hoc use + config writing) ─────────────

def _mcp_entry(port: int, transport: str, session_id: str | None = None) -> tuple[str, dict]:
    """Build a named agentorchestr mcpServers entry."""
    if transport == "sse":
        entry = {
            "url": f"http://127.0.0.1:{port}/sse",
            "description": "agentorchestr supervisor bridge (session-scoped, auto-removed on exit)",
        }
    elif transport == "streamable-http":
        entry = {
            "url": f"http://127.0.0.1:{port}/mcp",
            "description": "agentorchestr supervisor bridge (session-scoped, auto-removed on exit)",
        }
    else:
        entry = {
            "command": sys.executable,
            "args": [str(Path(__file__).resolve()), "--transport", "stdio"],
            "description": "agentorchestr supervisor bridge (session-scoped, auto-removed on exit)",
        }

    key = "agentorchestr" if session_id is None else f"agentorchestr-{session_id}"
    entry["_managed_by"] = "agentorchestr"
    if session_id is not None:
        entry["_session_id"] = session_id
    return key, entry


def _agent_config_locations(project_root: str | None = None) -> dict[str, tuple[list[str], list[str]]]:
    """
    Return candidate MCP config paths for every known agent.

    Search order per agent (first found wins at write time):
      1. Project-local config  — project-specific config paths the client
         may respect for this agent.
      2. Global config         — user config paths in the home directory.

    We prefer project-local paths so we never accidentally corrupt the
    user's global MCP server list when a project root is supplied.
    """
    proj = Path(project_root).resolve() if project_root else None

    def local(subpath: str) -> str | None:
        if proj is None:
            return None
        return str(proj / subpath)

    def expand_user(path: str) -> str:
        return os.path.expanduser(path)

    def agent_project_paths(agent_name: str) -> list[str]:
        paths: list[str | None] = [
            local(f".agentorchestr/mcp_configs/{agent_name}/mcp.json"),
            local(f".agentorchestr/mcp_configs/{agent_name}/settings.json"),
            local(f".{agent_name}/settings.json"),
            local(f".{agent_name}/mcp.json"),
        ]
        if agent_name == "claude":
            paths.insert(0, local(".claude/settings.json"))
            paths.insert(1, local(".claude/mcp.json"))
        return [p for p in paths if p]

    def agent_global_paths(agent_name: str) -> list[str]:
        paths: list[str] = [
            expand_user(f"~/.{agent_name}/settings.json"),
            expand_user(f"~/.{agent_name}/mcp.json"),
            expand_user(f"~/.{agent_name}/config/mcp.json"),
        ]
        if agent_name == "claude":
            paths = [
                expand_user("~/.claude/settings.json"),
                expand_user("~/.claude/mcp.json"),
                expand_user("~/.claude/config/mcp.json"),
            ]
        elif agent_name == "kiro":
            paths.insert(0, expand_user("~/.kiro/settings/mcp.json"))
        return paths

    locations: dict[str, tuple[list[str], list[str]]] = {}
    for agent_name in AGENT_REGISTRY:
        locations[agent_name] = (
            agent_project_paths(agent_name),
            agent_global_paths(agent_name),
        )
    return locations


def _agent_config_candidates(agent_name: str, project_root: str | None = None) -> tuple[list[str], list[str]]:
    """Return candidate local/global config paths for any agent name."""
    locations = _agent_config_locations(project_root)
    if agent_name in locations:
        return locations[agent_name]

    def local(subpath: str) -> str | None:
        if project_root is None:
            return None
        return str(Path(project_root).resolve() / subpath)

    def expand_user(path: str) -> str:
        return os.path.expanduser(path)

    local_paths: list[str] = [
        local(f".agentorchestr/mcp_configs/{agent_name}/mcp.json"),
        local(f".agentorchestr/mcp_configs/{agent_name}/settings.json"),
        local(f".{agent_name}/settings.json"),
        local(f".{agent_name}/mcp.json"),
    ]
    global_paths = [
        expand_user(f"~/.{agent_name}/settings.json"),
        expand_user(f"~/.{agent_name}/mcp.json"),
        expand_user(f"~/.{agent_name}/config/mcp.json"),
    ]
    return [p for p in local_paths if p], global_paths


def _read_existing_config(path: str) -> dict | None:
    """Read an existing MCP config file.

    Returns the decoded JSON object if readable, {} if the file is absent,
    or None if the file is present but invalid. We avoid clobbering malformed
    user config files.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return None


def _write_merged_config(path: str, entry: dict, entry_key: str = "agentorchestr") -> bool:
    """
    Merge our entry into the existing config at `path`.

    - Reads the existing file (if any) to preserve all other mcpServers.
    - Adds/replaces only the named entry key.
    - Writes atomically via a temp file + rename so a crash never leaves
      a half-written config.
    - Returns True on success, False on error.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_config(path)
    if existing is None:
        print(f"[warn] skipping {path}: invalid JSON", file=sys.stderr)
        return False

    servers = existing.get("mcpServers", {})
    if not isinstance(servers, dict):
        servers = {}

    servers[entry_key] = entry
    existing["mcpServers"] = servers

    tmp = p.with_suffix(".agentorchestr_tmp")
    try:
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception as exc:
        print(f"[warn] could not write {path}: {exc}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def write_mcp_config(
    port: int = 8765,
    transport: str = "streamable-http",
    project_root: str | None = None,
    lead_agent_name: str | None = None,
    session_id: str | None = None,
) -> list[str]:
    """
    Inject the agentorchestr MCP entry into agent config files.

    Strategy (safe-by-default):
      • Prefer project-local paths  (.agentorchestr/mcp_configs/<agent>/mcp.json)
        so we never touch global configs when a project root is known.
      • If no project-local path exists for an agent, fall back to the
        global path — but MERGE (not overwrite) so existing entries survive.
      • If only one agent is the lead, only write its config (no need to
        pollute every agent's config for a single-lead session).
      • Each session uses its own config key when possible so concurrent
        sessions do not collide.

    Returns list of paths that were successfully written.
    """
    entry_key, entry = _mcp_entry(port, transport, session_id=session_id)
    written: list[str] = []

    agent_names = [lead_agent_name] if lead_agent_name else list(_agent_config_locations(project_root).keys())
    for agent_name in agent_names:
        if agent_name is None:
            continue
        local_paths, global_paths = _agent_config_candidates(agent_name, project_root)
        candidates = [p for p in local_paths if p] + global_paths
        for target in candidates:
            if _write_merged_config(target, entry, entry_key=entry_key):
                written.append(target)
                print(f"✓ MCP config ({agent_name}): {target}")
                break

    return written


def remove_mcp_config(
    written_paths: list[str],
    entry_key: str = "agentorchestr",
) -> None:
    """
    Remove the agentorchestr entry from every config file that was written
    during this session.  Called by Supervisor.cleanup().

    - If the only remaining key under mcpServers is the managed entry,
      the entire mcpServers block is left as {} (don't delete the file —
      the user may have other top-level keys we don't know about).
    - If the file no longer exists, skip silently.
    """
    for path in written_paths:
        p = Path(path)
        if not p.exists():
            continue
        data = _read_existing_config(path)
        if data is None:
            continue
        servers = data.get("mcpServers", {})
        if isinstance(servers, dict) and entry_key in servers:
            del servers[entry_key]
            data["mcpServers"] = servers
            tmp = p.with_suffix(".agentorchestr_tmp")
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(p)
            except Exception as exc:
                print(f"[warn] could not clean up {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="agentorchestr supervisor MCP bridge")
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