#!/usr/bin/env python3
"""
agentorchestr — supervisor-worker agent orchestrator
=====================================================

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
from rich.prompt import Prompt

from agent_detector import AgentDetector
from hooks import HookManager
from router import LLMRouter
from state_store import StateStore
from supervisor import Supervisor

console = Console()

BANNER = r"""
[bold cyan]
    █████╗  ██████╗ ███████╗███╗   ██╗████████╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
   ██║  ██║╚██████╔╝███████╗██║   ╚██║   ██║
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝    ╚═╝   ╚═╝
   ██████╗ ██████╗  ██████╗██╗  ██╗████████╗██████╗
  ██╔═══██╗██╔══██╗██╔════╝██║  ██║╚══██╔══╝██╔══██╗
  ██║   ██║██████╔╝██║     ███████║   ██║   ██████╔╝
  ██║   ██║██╔══██╗██║     ██╔══██║   ██║   ██╔══██╗
  ╚██████╔╝██║  ██║╚██████╗██║  ██║   ██║   ██║  ██║
   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
[/bold cyan]
[dim]supervisor-worker · tmux · MCP[/dim]
"""


# ── interactive helpers ────────────────────────────────────────────────

def _interactive_pick(available: list[dict]) -> tuple[list[dict], dict | None]:
    """Pick worker agents + designate the lead. Returns (workers, lead).

    Used when neither --agents nor --lead are given.
    """
    if not sys.stdin.isatty():
        # No TTY (CI / piped) — fall back to top-3 heuristic silently.
        cli = [a for a in available if a["type"] in ("sdk", "cli")][:3]
        if not cli:
            return [], None
        return cli, cli[0]
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
    try:
        raw = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]cancelled[/dim]")
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
    try:
        raw = input().strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]cancelled — using default lead[/dim]")
        return selected, default_lead
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
    args, available: list[dict]
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


# Anthropic prompt-cache write surcharge starts to bite past ~8KB.
# Beyond that we still work, just nag the user so they know they're paying.
_GOAL_WARN_BYTES = 8 * 1024
# Hard cap so a runaway paste can't blow up the prompt prefix.
_GOAL_HARD_CAP = 64 * 1024


def _read_goal(args) -> str:
    """Resolve the goal text from --goal / --goal-file / interactive paste.

    Order of precedence:
      1. --goal-file <path>   reads the file as-is (UTF-8).
      2. --goal "..."         single-line / quick goal.
      3. --interactive        multi-line paste mode; terminator: a line
                              containing only "/end" OR a blank line
                              after another blank line (Enter Enter).
                              The "/end" sentinel is preferred for
                              file-pastes that may contain blank lines.
    """
    if getattr(args, "goal_file", None):
        path = Path(args.goal_file).expanduser()
        if not path.exists():
            console.print(f"[red]✗ --goal-file {path} not found[/red]")
            return ""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            console.print(f"[red]✗ --goal-file {path} is not UTF-8[/red]")
            return ""
        return _check_goal_size(text.strip(), source=str(path))

    if args.goal:
        return _check_goal_size(args.goal.strip(), source="--goal")

    if args.interactive:
        if not sys.stdin.isatty():
            console.print(
                "[red]✗ --interactive requires a TTY. "
                "Pipe stdin? Use --goal-file PATH or --goal '...' instead.[/red]"
            )
            return ""
        console.print(
            "[cyan]Enter your goal — paste freely. End with [bold]/end[/bold] on its "
            "own line, or press Enter twice on a blank line.[/cyan]"
        )
        lines: list[str] = []
        try:
            while True:
                line = input()
                stripped = line.strip()
                if stripped == "/end":
                    break
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass  # Ctrl-D also ends paste cleanly.
        except KeyboardInterrupt:
            console.print("\n[dim]cancelled[/dim]")
            return ""
        return _check_goal_size("\n".join(lines).strip(), source="paste")

    return ""


def _check_goal_size(text: str, *, source: str) -> str:
    n = len(text.encode("utf-8"))
    if n > _GOAL_HARD_CAP:
        console.print(
            f"[red]✗ goal from {source} is {n//1024}KB > "
            f"{_GOAL_HARD_CAP//1024}KB hard cap. Trim it or break into "
            f"phases.[/red]"
        )
        return text[: _GOAL_HARD_CAP]
    if n > _GOAL_WARN_BYTES:
        console.print(
            f"[yellow]⚠ goal is {n//1024}KB > {_GOAL_WARN_BYTES//1024}KB. "
            f"Past this size you pay the prompt-cache write surcharge on "
            f"every call. Consider moving long context into PROJECT.md "
            f"so it lands in the cacheable per-project tier instead.[/yellow]"
        )
    return text


def _show_agents_table(available: list[dict]) -> None:
    if not available:
        console.print("[yellow]No agents detected on PATH.[/yellow]")
        return
    from rich.table import Table
    table = Table(show_header=True, header_style="cyan", box=None, pad_edge=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("name", style="bold")
    table.add_column("type")
    table.add_column("orch", justify="right")
    table.add_column("impl", justify="right")
    table.add_column("flags", style="dim")
    for i, a in enumerate(available, 1):
        caps = a.get("capabilities", {})
        flags = " ".join(filter(None, [
            "parallel" if caps.get("parallel") else "",
            "mcp" if caps.get("mcp") else "",
        ]))
        table.add_row(
            str(i), a["name"], a["type"],
            str(caps.get("orchestrate", "?")),
            str(caps.get("implement", "?")),
            flags,
        )
    console.print(table)


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
            "agentorchestr's free-LLM router is used only for legacy quality-gate evaluation; "
            "auto mode runs entirely on the agents' own auth.[/dim]"
        )

    # Surface the optional-but-required-for-auto-mode dependencies.
    console.print("\n[bold cyan]═══ agentorchestr dependencies ═══[/bold cyan]")
    from mcp_bridge import MCP_AVAILABLE
    try:
        import zeroconf  # noqa: F401
        zc_ok = True
    except ImportError:
        zc_ok = False
    try:
        import ddgs  # noqa: F401
        ddgs_ok = True
    except ImportError:
        ddgs_ok = False
    rows = [
        ("mcp", MCP_AVAILABLE, "REQUIRED for auto mode (lead<->agentorchestr bridge)",
         "pip install 'mcp>=1.20'"),
        ("zeroconf", zc_ok, "optional: cross-terminal agent discovery",
         "pip install 'zeroconf>=0.140'"),
        ("ddgs", ddgs_ok, "optional: web research via DuckDuckGo",
         "pip install 'ddgs>=9.0'"),
    ]
    for name, ok, why, how in rows:
        mark = "[green]✓[/green]" if ok else "[yellow]✗[/yellow]"
        console.print(f"  {mark} {name:<10} [dim]{why}[/dim]")
        if not ok:
            console.print(f"      [dim]→ {how}[/dim]")


async def _print_session_list(store: StateStore) -> None:
    from datetime import datetime
    rows = await store.list_sessions(limit=20)
    if not rows:
        console.print("[dim]No sessions in ~/.agentorchestr/state.db yet.[/dim]")
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


def _handle_skill_subcommand(reg, args) -> int:
    """Dispatch the --skill-* flags. Returns the process exit code."""
    reg.reload()
    if args.skill_list:
        skills = reg.all()
        if not skills:
            console.print("[dim]No skills installed.  Try --skill-add <git-url>[/dim]")
            return 0
        console.print("\n[bold cyan]═══ Installed skills ═══[/bold cyan]")
        for s in skills:
            sig = "[green]signed[/green]" if s.manifest.has_signature else "[yellow]unsigned[/yellow]"
            triggers = ", ".join(s.manifest.keywords[:5]) or "(no keywords)"
            console.print(
                f"  [cyan]{s.name:<30}[/cyan] v{s.manifest.version:<10} {sig}  "
                f"[dim]{triggers}[/dim]"
            )
        return 0

    if args.skill_add:
        try:
            skill = reg.add_from_git(args.skill_add)
        except (FileExistsError, RuntimeError, ValueError) as e:
            console.print(f"[red]✗ {e}[/red]")
            return 1
        console.print(
            f"[green]✓[/green] installed [cyan]{skill.name}[/cyan] "
            f"v{skill.manifest.version} from {args.skill_add}"
        )
        return 0

    if args.skill_remove:
        if reg.remove(args.skill_remove):
            console.print(f"[green]✓[/green] removed [cyan]{args.skill_remove}[/cyan]")
            return 0
        console.print(f"[red]✗ skill {args.skill_remove!r} not installed[/red]")
        return 1

    if args.skill_verify:
        info = reg.verify(args.skill_verify)
        if info["ok"]:
            kind = "signed" if info["signed"] else "unsigned"
            console.print(
                f"[green]✓[/green] {args.skill_verify} verified ({kind})"
            )
            return 0
        console.print(f"[red]✗ {args.skill_verify}: {info['error']}[/red]")
        return 1
    return 0


async def _heartbeat_loop(store: "StateStore", session_id: str,
                          stop_event: asyncio.Event,
                          *, interval: float = 5.0) -> None:
    """Print a one-line worker status summary every `interval` seconds.

    Runs alongside Supervisor.run() so the user can see progress even
    when watching the parent terminal (rather than tmux).  Stops when
    `stop_event` is set or the loop is cancelled.  Only emits when the
    counts or last-summary change, so an idle supervisor stays quiet.
    """
    last_signature: tuple = ()
    while not stop_event.is_set():
        try:
            workers = await store.get_workers(session_id)
        except Exception:
            workers = []
        counts: dict[str, int] = {}
        last_summary = ""
        last_id = ""
        for w in workers:
            st = w.get("state") or "unknown"
            counts[st] = counts.get(st, 0) + 1
            if w.get("summary"):
                last_summary = w["summary"]
                last_id = w.get("worker_id") or ""
        signature = (tuple(sorted(counts.items())), last_summary[:80])
        if workers and signature != last_signature:
            last_signature = signature
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            counts_str = "  ".join(
                f"{k}={v}" for k, v in sorted(counts.items())
            ) or "(no workers yet)"
            tail = f"  | last: {last_id} {last_summary[:80]}" if last_summary else ""
            console.print(
                f"[dim]\\[{ts}] agentorchestr {session_id}  • {counts_str}{tail}[/dim]"
            )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


def _should_auto_attach() -> bool:
    """True when agentorchestr should spawn / switch a terminal to view tmux by default.

    Mirrors the logic of `_attach_to_tmux` so we don't try to spawn a
    terminal agentorchestr can't actually use:
      * Already inside tmux  → True (we'll just `tmux switch-client`).
      * stdin isn't a TTY    → False (probably scripted run).
      * No terminal emulator → False (would silently fail).
    """
    import shutil as _sh
    if os.environ.get("TMUX"):
        return True
    if not sys.stdin.isatty():
        return False
    for term in ("kitty", "alacritty", "wezterm",
                 "gnome-terminal", "konsole", "xterm"):
        if _sh.which(term):
            return True
    return False


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
        description="agentorchestr — supervisor-worker agent orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--goal", type=str, help="Goal for AUTO mode (supervisor + workers)")
    parser.add_argument("--goal-file", type=str,
                        help="Read the goal from a file (UTF-8). Useful for "
                             "long multi-paragraph goals or design docs.")
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
    parser.add_argument("--attach", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Open a new terminal and tmux-attach after startup. "
                             "Default: auto (attach when stdin is a TTY and a "
                             "terminal emulator is available). Use --no-attach "
                             "to keep the current terminal.")
    parser.add_argument("--no-tmux-cleanup", action="store_true",
                        help="Keep tmux session alive after run (default)")

    # Skill marketplace subcommands.  All optional; if --skill-* isn't
    # passed we just behave as before.  Kept as flags (not subparsers)
    # so combining with --detect / --list-sessions stays simple.
    parser.add_argument("--skill-list", action="store_true",
                        help="List installed skills and exit")
    parser.add_argument("--skill-add", metavar="GIT_URL",
                        help="git-clone a skill into ~/.agentorchestr/skills and exit")
    parser.add_argument("--skill-remove", metavar="NAME",
                        help="Uninstall a skill by name and exit")
    parser.add_argument("--skill-verify", metavar="NAME",
                        help="Verify a skill's signature and exit")
    parser.add_argument("--require-signed-skills", action="store_true",
                        help="Refuse to load any skill without a valid ed25519 signature")
    parser.add_argument("--init", action="store_true",
                        help="Scaffold .agentorchestr/ files in the current project "
                             "(PROJECT.md, CONVENTIONS.md, hooks.json, "
                             "memory/{topics,episodes,research}/, skills/) "
                             "and append .gitignore patterns. Idempotent.")
    parser.add_argument("--init-force", action="store_true",
                        help="With --init: overwrite existing scaffolded files.")
    parser.add_argument("--setup", action="store_true",
                        help="Run the first-run wizard now, even if it has "
                             "already been completed. Detects agents, lets "
                             "you pick lead + worker layout, persists the "
                             "choice for next time.")
    parser.add_argument("--no-setup", action="store_true",
                        help="Skip the first-run wizard even if it has "
                             "never been completed (useful for CI / "
                             "non-interactive runs).")
    parser.add_argument("--doctor", action="store_true",
                        help="Run a comprehensive health check (Python, "
                             "tmux, git, mcp, optional deps, agents, LLM "
                             "keys, write access to XDG dirs) and exit. "
                             "Exit code 0 = healthy, 1 = degraded.")
    args = parser.parse_args()

    console.print(BANNER)

    store = StateStore()
    await store.init()

    with console.status("[bold cyan]Detecting agents on PATH…[/bold cyan]"):
        detector = AgentDetector()
        available = detector.detect()
    with console.status("[bold cyan]Checking LLM backends…[/bold cyan]"):
        llm = LLMRouter()
        await llm.init()

    if not (args.detect or args.list_sessions or args.skill_list
            or args.skill_add or args.skill_remove or args.skill_verify
            or args.doctor or args.init or args.dashboard):
        _show_agents_table(available)

    if args.detect:
        _print_detection(available, llm)
        await llm.close(); await store.close(); return 0

    if args.list_sessions:
        await _print_session_list(store)
        await llm.close(); await store.close(); return 0

    # Skill subcommands — exit after handling.
    if args.skill_list or args.skill_add or args.skill_remove or args.skill_verify:
        from skills import SkillRegistry
        reg = SkillRegistry(require_signature=args.require_signed_skills)
        rc = _handle_skill_subcommand(reg, args)
        await llm.close(); await store.close()
        return rc

    # `agentorchestr --doctor` runs a comprehensive health check then exits.
    if args.doctor:
        from doctor import run_doctor
        rc = run_doctor(available, llm)
        await llm.close(); await store.close()
        return rc

    # `agentorchestr --init` scaffolds the per-project layout, then exits.
    if args.init:
        from paths import init_project
        actions = init_project(args.project, force=args.init_force)
        console.print(f"\n[bold cyan]agentorchestr project init at {args.project}[/bold cyan]")
        for path, action in actions.items():
            color = {
                "created": "green", "appended": "green",
                "kept": "dim", "overwrote": "yellow",
            }.get(action, "white")
            console.print(f"  [{color}]{action:<10}[/{color}] {path}")
        console.print(
            "\n[dim]Edit PROJECT.md and CONVENTIONS.md to teach agentorchestr about "
            "this project — they're auto-loaded into the cacheable preamble "
            "of every session.[/dim]"
        )
        await llm.close(); await store.close()
        return 0

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

    # First-run wizard / --setup.  Skipped for non-interactive CLI usage
    # (--manual --task, explicit --agents/--lead, or --no-setup).
    wizard_layout = None
    try:
        import wizard
        skip_wizard = (
            args.no_setup
            or bool(args.agents)
            or bool(args.lead)
            or args.manual
            or not sys.stdin.isatty()
        )
        if not skip_wizard and wizard.needs_wizard(args, available):
            wizard_layout = wizard.run(available, args.project)
    except Exception as e:  # pragma: no cover — wizard must never break startup
        console.print(f"[yellow]wizard skipped: {e}[/yellow]")
        wizard_layout = None

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

    if not goal and not args.manual and sys.stdin.isatty():
        console.print(
            "[dim]Tip: pass [cyan]--goal '...'[/cyan] next time, or "
            "[cyan]--interactive[/cyan] for multi-line paste.[/dim]"
        )
        try:
            goal = Prompt.ask("[bold cyan]Enter your goal[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            goal = ""

    if args.manual:
        # Direct dispatch path (claude-squad-style: one task, one agent, one pane)
        try:
            return await _run_manual(args, available, store, session_id)
        finally:
            await llm.close()
            await store.close()

    if not goal:
        console.print(
            "[yellow]No goal provided. Use --goal '...', --goal-file <path>, --interactive, or --manual.[/yellow]"
        )
        await llm.close(); await store.close(); return 0

    # Pick agents + lead.  Precedence:
    #   1. Wizard layout from this session (if just run).
    #   2. Saved preferences (from a prior --setup).
    #   3. The heuristic in _resolve_agents.
    #
    # CLI overrides (--agents / --lead / --manual) skip 1 + 2 and go
    # straight to _resolve_agents which already handles them.
    workers: list[dict] = []
    lead: dict | None = None
    if wizard_layout is not None:
        workers, lead = wizard_layout.workers, wizard_layout.lead
    elif not args.agents and not args.lead and not args.manual:
        try:
            import preferences as _prefs
            saved = _prefs.load()
        except Exception:
            saved = {}
        agents_pref = saved.get("agents", {}) if saved else {}
        if agents_pref.get("lead") and agents_pref.get("workers"):
            by_name = {a["name"]: a for a in available}
            saved_lead = by_name.get(agents_pref["lead"])
            saved_workers = [by_name[n] for n in agents_pref["workers"] if n in by_name]
            missing = [n for n in agents_pref["workers"] if n not in by_name]
            if missing:
                console.print(
                    f"[yellow]⚠ saved worker(s) {missing} no longer installed — "
                    f"falling back to defaults.[/yellow]"
                )
            if saved_lead and saved_workers:
                lead = saved_lead
                count = int(agents_pref.get("worker_count", len(saved_workers)))
                count = max(1, min(count, 8))
                workers = [saved_workers[i % len(saved_workers)] for i in range(count)]
                console.print(
                    f"[dim]using saved preferences "
                    f"(lead={lead['name']}, {count} worker(s)) — "
                    f"override with --setup or --agents/--lead.[/dim]"
                )
            elif not saved_lead and agents_pref.get("lead"):
                console.print(
                    f"[yellow]⚠ saved lead {agents_pref['lead']!r} no longer "
                    f"installed — re-run with --setup to pick a new one.[/yellow]"
                )
        elif not args.no_setup and sys.stdin.isatty():
            # Fresh install, no preferences yet, interactive — hint at --setup.
            console.print(
                "[dim]No saved layout. Using top-3 heuristic. "
                "Run [cyan]agentorchestr --setup[/cyan] to choose your own.[/dim]"
            )
    if not workers or lead is None:
        workers, lead = _resolve_agents(args, available)
    if not workers or lead is None:
        # Nothing usable.  Diagnose why and exit cleanly.
        ide_only = available and all(a["type"] in ("ide", "gh") for a in available)
        if ide_only:
            console.print(Panel(
                "[red]Only IDE-style agents detected.[/red]\n"
                "agentorchestr's auto mode needs at least one CLI/SDK agent\n"
                "(claude, openclaude, kiro, opencode, aider, codex, gemini, "
                "goose, amp).\n\n"
                "Cursor / Windsurf / Copilot can't be driven from the\n"
                "supervisor's MCP surface.\n\n"
                "Install one CLI agent, or use --manual --task '...' to drive\n"
                "an agent directly without a supervisor.",
                border_style="red", title="no CLI agent available",
            ))
        else:
            console.print(Panel(
                "[red]Could not pick a lead + worker layout.[/red]\n"
                "Try [cyan]agentorchestr --setup[/cyan] to walk through it manually,\n"
                "or pass [cyan]--agents <name> [...]  --lead <name>[/cyan].",
                border_style="red", title="agent selection failed",
            ))
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

    # Memory federation: optional, no hard dep.  If sqlite-vec/fastembed are
    # missing it falls back to FTS5 + recency only — still a meaningful win.
    memory = None
    try:
        from memory import MemoryFederation
        memory = MemoryFederation(args.project)
        await memory.init()
    except Exception as e:  # pragma: no cover - tolerate missing optional deps
        console.print(f"[dim]memory federation unavailable: {e}[/dim]")
        memory = None

    # Skills: optional, never blocks the run.  --require-signed-skills
    # makes us refuse to load anything unsigned.
    skills_reg = None
    try:
        from skills import SkillRegistry
        skills_reg = SkillRegistry(require_signature=args.require_signed_skills)
        skills_reg.reload()
        if skills_reg.all():
            console.print(
                f"[dim]loaded {len(skills_reg.all())} skill(s) from "
                f"{skills_reg.skills_dir}[/dim]"
            )
    except Exception as e:  # pragma: no cover
        console.print(f"[dim]skills registry unavailable: {e}[/dim]")
        skills_reg = None

    # Auto mode hard-requires the `mcp` library because the lead agent
    # talks to agentorchestr over MCP/SSE.  Fail BEFORE we spin up tmux + worktrees.
    from mcp_bridge import MCP_AVAILABLE
    if not MCP_AVAILABLE:
        console.print(Panel(
            "[red]The `mcp` Python library is not installed.[/red]\n"
            "agentorchestr's auto mode needs it so the lead agent can call back via\n"
            "MCP/SSE. Install it into your venv:\n\n"
            "  [cyan].env/bin/pip install 'mcp>=1.20'[/cyan]\n"
            "  (or:  [cyan]python bootstrap.py --with-mcp[/cyan])\n\n"
            "If you only want to drive a single agent without MCP, use\n"
            "  [cyan]python orchestrator.py --manual --task '...'[/cyan]",
            title="[red]missing dependency[/red]", border_style="red",
        ))
        await llm.close(); await store.close(); return 1

    sup = Supervisor(
        project_root=args.project,
        goal=goal,
        lead_agent=lead,
        worker_agents=workers,
        store=store,
        session_id=session_id,
        memory=memory,
        skills=skills_reg,
    )

    console.print(Panel(
        f"[bold]session id:[/bold] [cyan]{session_id}[/cyan]\n"
        f"[bold]tmux:[/bold]       [cyan]{sup.attach_command}[/cyan]\n"
        f"[bold]MCP:[/bold]        http://{sup.host}:{sup.port}/sse\n"
        f"[bold]dashboard:[/bold]  http://localhost:3000  (run --dashboard)\n"
        f"\n"
        f"[bold]Next:[/bold]\n"
        f"  • Lead agent at the top; worker panes tile below — all in one view.\n"
        f"  • [cyan]Ctrl-B then D[/cyan] inside tmux detaches you without killing it.\n"
        f"  • [cyan]Ctrl-C[/cyan] in [italic]this[/italic] terminal cancels the whole session.\n"
        f"  • Run [cyan]python orchestrator.py --resume {session_id}[/cyan] to pick up "
        f"if you stop now.\n"
        f"[dim]Heartbeat below; full status panel when the goal completes.[/dim]",
        title="[green]auto mode running[/green]", border_style="green",
    ))

    # --attach defaults to None ("auto"); --no-attach forces False.
    want_attach = (
        _should_auto_attach() if args.attach is None else args.attach
    )
    if want_attach:
        _attach_to_tmux(sup.pool.tmux_session_name)
        # Inside tmux we just switch-client; outside tmux a new terminal
        # was opened. Either way the parent terminal stays free for the
        # heartbeat ticker / final summary panel.

    # Defaults so the `finally` clause and the closing Panel always have
    # valid values, even if sup.run() raises before its first assignment.
    state: str = "failed"
    summary: str = "supervisor never started"
    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(store, session_id, heartbeat_stop)
    )
    try:
        state, summary = await sup.run()
    except KeyboardInterrupt:
        state, summary = "cancelled", "interrupted by user"
    except Exception as e:  # noqa: BLE001 — we want the message, not the type leak
        state, summary = "failed", f"{type(e).__name__}: {e}"
        console.print(f"[red]✗ supervisor crashed:[/red] {summary}")
    finally:
        heartbeat_stop.set()
        try:
            await asyncio.wait_for(heartbeat_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            heartbeat_task.cancel()
        try:
            await sup.cleanup(keep_tmux=not args.no_tmux_cleanup)
        except Exception as e:
            console.print(f"[yellow]cleanup warning:[/yellow] {e}")
        await store.save_session(
            session_id, {"status": state if state != "cancelled" else "paused"},
        )
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
    # Use the lead-pane only (no supervisor); single-window layout.
    await pool.start_session([
        "bash", "-lc",
        f"echo '=== agentorchestr manual: {agent['name']} ==='; "
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
        f"[dim]agentorchestr exits now. The agent keeps running in tmux. "
        f"Attach to watch and steer; review the diff in the worktree before merging.[/dim]",
        title="[green]manual mode started[/green]", border_style="green",
    ))
    want_attach = (
        _should_auto_attach() if args.attach is None else args.attach
    )
    if want_attach:
        _attach_to_tmux(pool.tmux_session_name)
    return 0


def _sync_main() -> int:
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(_sync_main())
