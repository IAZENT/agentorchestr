#!/usr/bin/env python3
"""
agentorchestr — supervisor-worker agent orchestrator
=====================================================

Two modes:

  AUTO (default):
      agentorchestr --goal "Build a JWT auth module"
    A lead/supervisor CLI agent runs in tmux pane 0 with MCP tools.

  MANUAL:
      agentorchestr --manual --task "fix the login bug"
    Single agent, single task, no supervisor.

Other flags:
  --detect           list installed agents + LLM keys
  --list-sessions    show recent sessions (resumable ones flagged)
  --resume <id>      pick up a crashed session
  --dashboard        launch the FastAPI dashboard (http://localhost:3000)
  --init             scaffold .agentorchestr/ for this project
  --doctor           comprehensive health check
  --skill-add/list/verify/remove   skill marketplace
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agentorchestr.agent_detector import AgentDetector
from agentorchestr.router import LLMRouter
from agentorchestr.state_store import StateStore

console = Console()

BANNER = r"""
[bold cyan]
    █████╗  ██████╗ ███████╗███╗   ██╗████████╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
   ██║  ██║╚██████╔╝███████╗██║   ╚██║   ██║
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝    ╚═╝   ╚═╝
[/bold cyan]
[dim]supervisor-worker · tmux · MCP[/dim]
"""

# ── goal helpers (imported by commands/auto.py) ────────────────────────

_GOAL_WARN_BYTES = 8 * 1024
_GOAL_HARD_CAP   = 64 * 1024


def _check_goal_size(text: str, *, source: str) -> str:
    n = len(text.encode("utf-8"))
    if n > _GOAL_HARD_CAP:
        console.print(f"[red]✗ goal from {source} is {n//1024}KB > {_GOAL_HARD_CAP//1024}KB hard cap.[/red]")
        return text[:_GOAL_HARD_CAP]
    if n > _GOAL_WARN_BYTES:
        console.print(f"[yellow]⚠ goal is {n//1024}KB > {_GOAL_WARN_BYTES//1024}KB. Consider PROJECT.md for long context.[/yellow]")
    return text


def _read_goal(args) -> str:
    if getattr(args, "goal_file", None):
        path = Path(args.goal_file).expanduser()
        if not path.exists():
            console.print(f"[red]✗ --goal-file {path} not found[/red]")
            return ""
        try:
            return _check_goal_size(path.read_text(encoding="utf-8").strip(), source=str(path))
        except UnicodeDecodeError:
            console.print(f"[red]✗ --goal-file {path} is not UTF-8[/red]")
            return ""
    if args.goal:
        return _check_goal_size(args.goal.strip(), source="--goal")
    if args.interactive:
        if not sys.stdin.isatty():
            console.print("[red]✗ --interactive requires a TTY.[/red]")
            return ""
        console.print("[cyan]Enter goal. End with /end or double-blank-line.[/cyan]")
        lines: list[str] = []
        try:
            while True:
                line = input()
                if line.strip() == "/end":
                    break
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass
        return _check_goal_size("\n".join(lines).strip(), source="paste")
    return ""


def _interactive_pick(available: list[dict]) -> tuple[list[dict], dict | None]:
    if not sys.stdin.isatty():
        cli = [a for a in available if a["type"] in ("sdk", "cli")][:3]
        return (cli, cli[0]) if cli else ([], None)
    console.print("\n[bold cyan]=== Available agents ===[/bold cyan]")
    for i, a in enumerate(available, 1):
        caps = a.get("capabilities", {})
        console.print(f"  [cyan]{i}.[/cyan] {a['name']:<14} orch={caps.get('orchestrate',0)} impl={caps.get('implement',0)}")
    console.print("\nPick workers (comma-sep, 'all', Enter=auto): ", end="")
    try:
        raw = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        return [], None
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
                pass
        if not selected:
            selected = available[:3]
    if not selected:
        return [], None
    if len(selected) == 1:
        return selected, selected[0]
    cli_sub = [a for a in selected if a["type"] in ("sdk", "cli")] or selected
    lead = max(cli_sub, key=lambda a: a["capabilities"].get("orchestrate", 0))
    return selected, lead


def _resolve_agents(args, available: list[dict]) -> tuple[list[dict], dict | None]:
    if args.agents:
        forced = [a for a in available if a["name"] in args.agents]
        if not forced:
            console.print(f"[red]None of {args.agents} found.[/red]")
            return [], None
        if args.lead:
            match = [a for a in forced if a["name"] == args.lead]
            if not match:
                console.print(f"[red]--lead {args.lead} not in --agents[/red]")
                return [], None
            return forced, match[0]
        cli_sub = [a for a in forced if a["type"] in ("sdk", "cli")] or forced
        return forced, max(cli_sub, key=lambda a: a["capabilities"].get("orchestrate", 0))
    if not args.interactive:
        cli = [a for a in available if a["type"] in ("sdk", "cli")][:3]
        return (cli, cli[0]) if cli else ([], None)
    return _interactive_pick(available)


def _show_agents_table(available: list[dict]) -> None:
    if not available:
        console.print("[yellow]No agents detected on PATH.[/yellow]")
        return
    from rich.table import Table
    t = Table(show_header=True, header_style="cyan", box=None, pad_edge=False)
    for col, kw in [("#", {"justify":"right","style":"cyan","no_wrap":True}),
                    ("name", {"style":"bold"}), ("type", {}),
                    ("orch", {"justify":"right"}), ("impl", {"justify":"right"}),
                    ("flags", {"style":"dim"})]:
        t.add_column(col, **kw)
    for i, a in enumerate(available, 1):
        caps = a.get("capabilities", {})
        flags = " ".join(filter(None, ["parallel" if caps.get("parallel") else "",
                                        "mcp" if caps.get("mcp") else ""]))
        t.add_row(str(i), a["name"], a["type"],
                  str(caps.get("orchestrate","?")), str(caps.get("implement","?")), flags)
    console.print(t)


# ── heartbeat / tty helpers (kept for backward compat imports) ─────────

def _should_auto_attach() -> bool:
    from agentorchestr.tty import should_auto_attach
    return should_auto_attach()


def _attach_to_tmux(session_name: str) -> None:
    from agentorchestr.tty import attach_to_tmux
    attach_to_tmux(session_name)


async def _heartbeat_loop(store, session_id: str, stop_event, *, interval: float = 5.0):
    from agentorchestr.tty import heartbeat_loop
    await heartbeat_loop(store, session_id, stop_event, interval=interval)


# ── CLI entry point ────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="agentorchestr — supervisor-worker agent orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    from agentorchestr import __version__ as _v
    parser.add_argument("--version", action="version", version=f"agentorchestr {_v}")
    parser.add_argument("--goal", type=str)
    parser.add_argument("--goal-file", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--agents", nargs="+")
    parser.add_argument("--lead", type=str)
    parser.add_argument("--project", default=os.getcwd())
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--attach", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-tmux-cleanup", action="store_true")
    parser.add_argument("--skill-list", action="store_true")
    parser.add_argument("--skill-add", metavar="GIT_URL")
    parser.add_argument("--skill-remove", metavar="NAME")
    parser.add_argument("--skill-verify", metavar="NAME")
    parser.add_argument("--require-signed-skills", action="store_true")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--init-force", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()

    console.print(BANNER)

    store = StateStore()
    await store.init()
    with console.status("[bold cyan]Detecting agents…[/bold cyan]"):
        available = AgentDetector().detect()
    with console.status("[bold cyan]Checking LLM backends…[/bold cyan]"):
        llm = LLMRouter()
        await llm.init()

    from agentorchestr.commands.info import (
        handle_init, handle_skill_subcommand,
        print_detection, print_session_list, show_agents_table,
    )

    if args.detect:
        print_detection(available, llm)
        await llm.close(); await store.close(); return 0

    if args.list_sessions:
        await print_session_list(store)
        await llm.close(); await store.close(); return 0

    if args.skill_list or args.skill_add or args.skill_remove or args.skill_verify:
        from agentorchestr.skills import SkillRegistry
        reg = SkillRegistry(require_signature=args.require_signed_skills)
        rc = handle_skill_subcommand(reg, args)
        await llm.close(); await store.close(); return rc

    if args.doctor:
        from agentorchestr.doctor import run_doctor
        rc = run_doctor(available, llm)
        await llm.close(); await store.close(); return rc

    if args.init:
        rc = handle_init(args)
        await llm.close(); await store.close(); return rc

    if args.dashboard:
        from agentorchestr.dashboard.server import start_dashboard
        try:
            await start_dashboard(store)
        finally:
            await llm.close(); await store.close()
        return 0

    if not available:
        console.print(Panel(
            "[red]No supported CLI agents detected on PATH.[/red]\n"
            "Install at least one: kiro-cli, claude-code, openclaude, opencode, aider, codex, gemini-cli.",
            border_style="red", title="no agents",
        ))
        await llm.close(); await store.close(); return 1

    show_agents_table(available)

    session_id = args.resume or uuid.uuid4().hex[:8]
    if args.resume:
        session = await store.get_session(session_id)
        if not session:
            console.print(f"[red]session {session_id} not found[/red]")
            await llm.close(); await store.close(); return 1
        n = await store.reset_running_workers(session_id)
        if n:
            console.print(f"[yellow]↻ {n} worker(s) marked orphaned from prior crash.[/yellow]")

    try:
        if args.manual:
            from agentorchestr.commands.manual import run_manual
            return await run_manual(args, available, store, session_id)
        from agentorchestr.commands.auto import run_auto
        return await run_auto(args, available, store, llm, session_id)
    finally:
        await llm.close()
        await store.close()


def _sync_main() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(_sync_main())
