#!/usr/bin/env python3
"""
ORCH — supervisor-worker agent orchestrator
============================================

Two modes:

  AUTO (default):
      python orchestrator.py --goal "Build a JWT auth module"
    A lead/supervisor CLI agent runs in tmux pane 0 with MCP tools
    (spawn_worker, verify, cross_review, ...). It decides how many
    workers to spawn, what perspectives, and how to verify. Workers
    run interactively in their own panes + git worktrees.

  MANUAL:
      python orchestrator.py --manual --task "fix the login bug"
    Claude-squad-style direct dispatch — one tmux session, one agent,
    one task. No supervisor, no MCP. Use this when you just want a
    pane in tmux running an agent on a worktree.

Other:
  --detect           list installed agents + LLM keys
  --list-sessions    show recent sessions (resumable ones flagged)
  --resume <id>      pick up a crashed session
  --dashboard        launch the FastAPI dashboard (http://localhost:3000)
  --use-ollama       include local Ollama in LLM router (off by default)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from agent_detector import AgentDetector
from agents.agent_registry import AgentRegistry
from hooks import HookManager
from router import LLMRouter
from state_store import StateStore
from supervisor import Supervisor

console = Console()

BANNER = """
[bold cyan]
  ██████╗ ██████╗  ██████╗██╗  ██╗
 ██╔═══██╗██╔══██╗██╔════╝██║  ██║
 ██║   ██║██████╔╝██║     ███████║
 ██║   ██║██╔══██╗██║     ██╔══██║
 ╚██████╔╝██║  ██║╚██████╗██║  ██║
  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
[/bold cyan]
[dim]supervisor-worker · tmux · MCP[/dim]
"""


# ── interactive helpers ────────────────────────────────────────────────

def _interactive_pick(available: list[dict]) -> tuple[list[dict], dict | None]:
    """Pick worker agents + designate the lead. Returns (workers, lead).

    Used when neither --agents nor --lead are given.
    """
    console.print("\n[bold cyan]═══ Available agents ═══[/bold cyan]")
    for i, a in enumerate(available, 1):
        caps = a.get("capabilities", {})
        flags = " ".join(filter(None, [
            "parallel" if caps.get("parallel") else "",
            "mcp" if caps.get("mcp") else "",
        ]))
        console.print(
            f"  [cyan]{i}.[/cyan] {a['name']:<14} {a['type']:<5} "
            f"orch={caps.get('orchestrate', 0)} impl={caps.get('implement', 0)} "
            f"[dim]{flags}[/dim]"
        )

    console.print(
        "\nPick worker agents [dim](comma-separated, 'all', or Enter for "
        "auto-select):[/dim] ",
        end="",
    )
    raw = input().strip().lower()
    if not raw or raw == "all":
        selected = available[:] if raw == "all" else available[:3]
    else:
        selected = []
        for part in raw.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(available):
                    selected.append(available[idx])
            except ValueError:
                continue
        if not selected:
            console.print("[yellow]No valid pick — using top-ranked agents.[/yellow]")
            selected = available[:3]

    if not selected:
        return [], None

    if len(selected) == 1:
        return selected, selected[0]

    cli_subset = [a for a in selected if a["type"] in ("sdk", "cli")] or selected
    default_lead = max(cli_subset, key=lambda a: a["capabilities"].get("orchestrate", 0))
    default_idx = selected.index(default_lead) + 1

    console.print("\n[bold cyan]Selected:[/bold cyan]")
    for i, a in enumerate(selected, 1):
        marker = " [yellow](default lead)[/yellow]" if a is default_lead else ""
        console.print(f"  [cyan]{i}.[/cyan] {a['name']}{marker}")

    console.print(
        f"\nWhich is the LEAD/supervisor? "
        f"[dim](1-{len(selected)}, Enter for #{default_idx} = "
        f"{default_lead['name']}):[/dim] ",
        end="",
    )
    raw = input().strip()
    lead = default_lead
    if raw:
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(selected):
                lead = selected[idx]
        except ValueError:
            pass
    return selected, lead


def _resolve_agents(
    args, available: list[dict], registry: AgentRegistry
) -> tuple[list[dict], dict | None]:
    """Map CLI args to (workers, lead). Falls back to interactive picker."""
    if args.agents:
        forced = [a for a in available if a["name"] in args.agents]
        if not forced:
            console.print(
                f"[red]None of {args.agents} are available. "
                f"Found: {[a['name'] for a in available]}[/red]"
            )
            return [], None
        if args.lead:
            match = [a for a in forced if a["name"] == args.lead]
            if not match:
                console.print(
                    f"[red]--lead {args.lead} is not in --agents {args.agents}[/red]"
                )
                return [], None
            lead = match[0]
        else:
            cli_subset = [a for a in forced if a["type"] in ("sdk", "cli")] or forced
            lead = max(cli_subset, key=lambda a: a["capabilities"].get("orchestrate", 0))
        return forced, lead

    if not args.interactive:
        # Best-effort default: top 3 ranked CLI agents, lead = #1
        cli = [a for a in available if a["type"] in ("sdk", "cli")][:3]
        if not cli:
            return [], None
        return cli, cli[0]

    return _interactive_pick(available)


def _read_goal(args) -> str:
    if args.goal:
        return args.goal.strip()
    if args.interactive:
        console.print(
            "[cyan]Enter your goal (multi-line; press Enter twice to submit):[/cyan]"
        )
        lines: list[str] = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        return "\n".join(lines).strip()
    return ""


def _print_detection(available: list[dict], llm: LLMRouter) -> None:
    console.print("\n[bold cyan]═══ Agents on PATH ═══[/bold cyan]")
    if not available:
        console.print("  [red]none detected.[/red] Install at least one (kiro-cli, "
                      "claude-code, openclaude, opencode, aider, codex, gemini-cli)")
    for a in available:
        caps = a.get("capabilities", {})
        ver = a.get("version", "")
        flags = " ".join(filter(None, [
            "parallel" if caps.get("parallel") else "",
            "mcp" if caps.get("mcp") else "",
        ]))
        console.print(
            f"  [green]✓[/green] {a['name']:<12} {a['type']:<4} {ver:<22} [dim]{flags}[/dim]"
        )

    providers = {n: s for n, s in llm.provider_status().items() if s["available"]}
    if providers:
        console.print("\n[bold cyan]═══ LLM backends ═══[/bold cyan]")
        for n, s in providers.items():
            cd = f"  [yellow](cooldown {s['cooldown_s']}s)[/yellow]" if s.get("cooldown_s") else ""
            console.print(f"  [green]✓[/green] {n:<12} {s.get('limit', '')}{cd}")
    else:
        console.print(
            "\n[dim]No external LLM keys configured. "
            "ORCH's free-LLM router is used only for legacy quality-gate evaluation; "
            "auto mode runs entirely on the agents' own auth.[/dim]"
        )


async def _print_session_list(store: StateStore) -> None:
    from datetime import datetime
    rows = await store.list_sessions(limit=20)
    if not rows:
        console.print("[dim]No sessions in ~/.orch/state.db yet.[/dim]")
        return
    console.print("\n[bold cyan]═══ Recent sessions ═══[/bold cyan]")
    console.print(f"  [dim]{'id':<10} {'status':<10} {'tasks':<10} {'updated':<19} goal[/dim]")
    for r in rows:
        unfinished = r.get("pending", 0) + r.get("running", 0)
        resumable = r["status"] in ("active", "paused") and unfinished > 0
        marker = "[yellow]*[/yellow]" if resumable else " "
        ts = datetime.fromtimestamp(r["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
        tasks = f"{r.get('done', 0)}/{r.get('total', 0)}"
        if r.get("failed"):
            tasks += f" ({r['failed']}!)"
        goal = (r["goal"] or "").replace("\n", " ")[:60]
        console.print(
            f" {marker}[cyan]{r['id']:<10}[/cyan] {r['status']:<10} {tasks:<10} {ts}  {goal}"
        )
    console.print(
        "\n[dim]Resume an unfinished session: "
        "[cyan]python orchestrator.py --resume <id>[/cyan][/dim]"
    )


def _attach_to_tmux(session_name: str) -> None:
    """Open a new terminal attached to the supervisor's tmux session.
    No-op when stdin isn't a TTY."""
    import shutil as _sh
    import subprocess as _sp
    if os.environ.get("TMUX"):
        _sp.Popen(["tmux", "switch-client", "-t", session_name],
                  stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        console.print(f"[dim]→ switched tmux client to {session_name}[/dim]")
        return
    for term, args in (
        ("kitty", ["kitty", "-e"]),
        ("alacritty", ["alacritty", "-e"]),
        ("wezterm", ["wezterm", "start", "--"]),
        ("gnome-terminal", ["gnome-terminal", "--"]),
        ("konsole", ["konsole", "-e"]),
        ("xterm", ["xterm", "-e"]),
    ):
        if _sh.which(term):
            _sp.Popen(args + ["tmux", "attach", "-t", session_name],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            console.print(f"[dim]→ opened {term} attached to {session_name}[/dim]")
            return
    console.print(
        f"[yellow]No terminal emulator found.[/yellow] Run: "
        f"[cyan]tmux attach -t {session_name}[/cyan]"
    )


# ── main ───────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="ORCH — supervisor-worker agent orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--goal", type=str, help="Goal for AUTO mode (supervisor + workers)")
    parser.add_argument("--task", type=str, help="Task for MANUAL mode (single agent)")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: one agent on one task, no supervisor")
    parser.add_argument("--agents", nargs="+",
                        help="Restrict to these agent names (lead picked from these)")
    parser.add_argument("--lead", type=str,
                        help="Force a specific lead agent (must be in --agents)")
    parser.add_argument("--project", default=os.getcwd(), help="Project root")
    parser.add_argument("--detect", action="store_true",
                        help="Show detected agents + LLM keys and exit")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List recent sessions and exit")
    parser.add_argument("--resume", type=str, help="Resume a session by id")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the FastAPI dashboard and exit")
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt for goal + agents at startup")
    parser.add_argument("--attach", action="store_true",
                        help="Open a new terminal and tmux-attach after startup")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Include local Ollama in the legacy LLM router")
    parser.add_argument("--no-tmux-cleanup", action="store_true",
                        help="Keep tmux session alive after run (default)")
    args = parser.parse_args()

    console.print(BANNER)

    store = StateStore()
    await store.init()

    detector = AgentDetector()
    available = detector.detect()
    llm = LLMRouter(use_ollama=args.use_ollama)
    await llm.init()

    if args.detect:
        _print_detection(available, llm)
        await llm.close(); await store.close(); return 0

    if args.list_sessions:
        await _print_session_list(store)
        await llm.close(); await store.close(); return 0

    if args.dashboard:
        from dashboard.server import start_dashboard
        try:
            await start_dashboard(store)
        finally:
            await llm.close(); await store.close()
        return 0

    if not available:
        console.print(Panel(
            "[red]No supported CLI agents detected on PATH.[/red]\n"
            "Install at least one of: kiro-cli, claude-code, openclaude, "
            "opencode, aider, codex, gemini-cli.",
            border_style="red", title="no agents"
        ))
        await llm.close(); await store.close(); return 1

    # Resume?
    session_id = args.resume
    if session_id:
        session = await store.get_session(session_id)
        if not session:
            console.print(f"[red]session {session_id} not found[/red]")
            await llm.close(); await store.close(); return 1
        n_orphaned = await store.reset_running_workers(session_id)
        if n_orphaned:
            console.print(
                f"[yellow]↻ {n_orphaned} worker(s) were 'running' from a prior crash; "
                f"marked 'orphaned'. Supervisor may re-spawn as needed.[/yellow]"
            )
        goal = session["goal"]
    else:
        goal = _read_goal(args)
        session_id = uuid.uuid4().hex[:8]

    if args.manual:
        # Direct dispatch path (claude-squad-style: one task, one agent, one pane)
        try:
            return await _run_manual(args, available, store, session_id)
        finally:
            await llm.close()
            await store.close()

    if not goal:
        console.print(
            "[yellow]No goal provided. Use --goal '...', --interactive, or --manual.[/yellow]"
        )
        await llm.close(); await store.close(); return 0

    # Pick agents + lead
    registry = AgentRegistry(available)
    workers, lead = _resolve_agents(args, available, registry)
    if not workers or lead is None:
        await llm.close(); await store.close(); return 1

    console.print(
        f"\n[green]Mode:[/green] auto (supervisor-worker) | "
        f"[green]Lead:[/green] [bold]{lead['name']}[/bold] | "
        f"[green]Worker pool:[/green] {', '.join(a['name'] for a in workers)}"
    )

    # Persist a session row up-front so the dashboard / --list-sessions sees it
    await store.save_session(session_id, {
        "goal": goal, "plan": {}, "agents": [a["name"] for a in workers],
        "status": "active",
    })

    hooks = HookManager(project_root=args.project)
    await hooks.run("pre_session", {
        "session_id": session_id, "goal": goal,
        "lead": lead["name"], "workers": [a["name"] for a in workers],
    })

    sup = Supervisor(
        project_root=args.project,
        goal=goal,
        lead_agent=lead,
        worker_agents=workers,
        store=store,
        session_id=session_id,
    )

    console.print(Panel(
        f"[bold]session id:[/bold] [cyan]{session_id}[/cyan]\n"
        f"[bold]tmux:[/bold]       [cyan]{sup.attach_command}[/cyan]\n"
        f"[bold]MCP:[/bold]        http://{sup.host}:{sup.port}/sse\n"
        f"[bold]dashboard:[/bold]  http://localhost:3000  (run --dashboard)\n"
        f"[dim]The supervisor is now running. It will spawn workers as it sees fit. "
        f"Attach to tmux to watch.[/dim]",
        title="[green]auto mode running[/green]", border_style="green",
    ))

    if args.attach:
        _attach_to_tmux(sup.pool.tmux_session_name)

    try:
        state, summary = await sup.run()
    except KeyboardInterrupt:
        state, summary = "cancelled", "interrupted by user"
    finally:
        await sup.cleanup(keep_tmux=not args.no_tmux_cleanup)
        await store.save_session(session_id, {"status": state if state != "cancelled" else "paused"})
        await hooks.run("post_session", {
            "session_id": session_id, "state": state, "summary": summary,
        })
        await llm.close()
        await store.close()

    color = {"done": "green", "failed": "red", "cancelled": "yellow", "timeout": "yellow"}.get(state, "white")
    console.print(Panel(
        f"[bold]state:[/bold]   [{color}]{state}[/{color}]\n"
        f"[bold]summary:[/bold] {summary}\n"
        f"[dim]tmux session kept open: {sup.attach_command}\n"
        f"kill manually: tmux kill-session -t {sup.pool.tmux_session_name}[/dim]",
        title=f"[{color}]session {session_id} complete[/{color}]",
        border_style=color,
    ))
    return 0 if state == "done" else 2


async def _run_manual(args, available: list[dict], store: StateStore, session_id: str) -> int:
    """Manual mode: claude-squad-style single-pane single-task."""
    if not args.task:
        console.print(
            "[yellow]Use --manual with --task 'your task description'[/yellow]"
        )
        return 1
    # Pick first matching agent
    if args.agents:
        candidates = [a for a in available if a["name"] in args.agents]
    else:
        candidates = [a for a in available if a["type"] in ("sdk", "cli")][:1]
    if not candidates:
        console.print("[red]No suitable agent for manual mode[/red]")
        return 1
    agent = candidates[0]

    from worker_pool import WorkerPool
    await store.save_session(session_id, {
        "goal": args.task, "plan": {}, "agents": [agent["name"]],
        "status": "active",
    })
    pool = WorkerPool(args.project, session_id, store)
    # Use the lead-pane only (no supervisor); workers window stays empty.
    await pool.start_session([
        "bash", "-lc",
        f"echo '=== ORCH manual: {agent['name']} ==='; "
        f"{shlex.quote(agent['cmd'])}",
    ])
    # Spawn one worker with the task as its sole prompt
    worker = await pool.spawn_worker(
        task=args.task, perspective="implementer", agent=agent,
    )
    # Give the SQLite background thread a tick to flush the upsert before
    # the event loop is torn down by the caller.
    await asyncio.sleep(0.1)
    console.print(Panel(
        f"[bold]session:[/bold] [cyan]{session_id}[/cyan]\n"
        f"[bold]worker:[/bold]  [cyan]{worker.id}[/cyan] ({agent['name']}, {worker.perspective})\n"
        f"[bold]worktree:[/bold] {worker.worktree}\n"
        f"[bold]attach:[/bold]  [cyan]{pool.attach_command}[/cyan]\n"
        f"[dim]ORCH exits now. The agent keeps running in tmux. "
        f"Attach to watch and steer; review the diff in the worktree before merging.[/dim]",
        title="[green]manual mode started[/green]", border_style="green",
    ))
    if args.attach:
        _attach_to_tmux(pool.tmux_session_name)
    return 0


def _sync_main() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(_sync_main())
