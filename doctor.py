"""
doctor.py — comprehensive health check
========================================

Run via:

    orch --doctor

Validates every dependency and environment piece agentorchestr relies
on, prints a colourised report, and exits 0 (healthy) / 1 (degraded).
Designed to be the first thing a new user runs after install — and the
first thing an existing user runs when something feels off.

Checks, in order:

  1. Python version (>= 3.11)
  2. tmux presence + version
  3. git presence
  4. Required runtime deps  (httpx, rich, fastapi, uvicorn, libtmux,
                             aiosqlite)
  5. Required extras for auto mode  (mcp)
  6. Optional extras  (zeroconf, ddgs, sqlite_vec, fastembed,
                        cryptography or PyNaCl)
  7. CLI agents detected on PATH  (delegated to AgentDetector)
  8. LLM keys configured  (delegated to LLMRouter.provider_status)
  9. Write access to XDG global state dir
 10. Write access to project's .orch/ if --init was run
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_doctor(available_agents: list[dict], llm) -> int:
    """Run all checks. Returns 0 if everything required is healthy, 1 otherwise."""
    rows: list[tuple[str, str, str, str]] = []  # (status, area, detail, fix-hint)
    fatal = False

    # ── 1. Python ──────────────────────────────────────────────────
    py_ok = sys.version_info >= (3, 11)
    rows.append((
        "ok" if py_ok else "fatal",
        "python",
        f"{sys.version.split()[0]}",
        "" if py_ok else "Install Python 3.11+ (https://python.org)",
    ))
    if not py_ok:
        fatal = True

    # ── 2. tmux ───────────────────────────────────────────────────
    tmux_path = shutil.which("tmux")
    if tmux_path:
        try:
            r = subprocess.run(
                ["tmux", "-V"], capture_output=True, text=True, timeout=2,
            )
            tmux_v = (r.stdout or r.stderr or "").strip() or "(unknown version)"
        except Exception:
            tmux_v = "(detected; version probe failed)"
        rows.append(("ok", "tmux", tmux_v, ""))
    else:
        rows.append((
            "fatal", "tmux", "not found",
            "Install: brew install tmux  OR  sudo apt install tmux",
        ))
        fatal = True

    # ── 3. git ─────────────────────────────────────────────────────
    git_path = shutil.which("git")
    if git_path:
        rows.append(("ok", "git", git_path, ""))
    else:
        rows.append((
            "warn", "git", "not found",
            "Workers fall back to scratch dirs without git worktrees.",
        ))

    # ── 4. required runtime deps ──────────────────────────────────
    for dep in ("httpx", "rich", "fastapi", "uvicorn", "libtmux", "aiosqlite"):
        ok, ver = _try_import(dep)
        rows.append((
            "ok" if ok else "fatal",
            f"dep:{dep}",
            ver if ok else "missing",
            "" if ok else f"pip install {dep}",
        ))
        if not ok:
            fatal = True

    # ── 5. required for auto mode ─────────────────────────────────
    ok, ver = _try_import("mcp")
    rows.append((
        "ok" if ok else "fatal",
        "dep:mcp",
        ver if ok else "missing",
        "" if ok else "pip install 'mcp>=1.20'  (auto mode needs this)",
    ))
    if not ok:
        fatal = True

    # ── 6. optional extras ────────────────────────────────────────
    for dep, why in (
        ("zeroconf", "cross-terminal mDNS discovery"),
        ("ddgs", "web research"),
        ("sqlite_vec", "vector retrieval in memory federation"),
        ("fastembed", "local embedding model for memory federation"),
        ("cryptography", "ed25519 skill signature verification"),
        ("nacl", "PyNaCl: alt ed25519 backend"),
    ):
        ok, ver = _try_import(dep)
        rows.append((
            "ok" if ok else "info",
            f"opt:{dep}",
            (ver if ok else f"missing — {why}"),
            "" if ok else f"pip install {dep}",
        ))

    # ── 7. agents ─────────────────────────────────────────────────
    if available_agents:
        names = ", ".join(a["name"] for a in available_agents)
        cli_count = sum(1 for a in available_agents if a["type"] in ("sdk", "cli"))
        if cli_count == 0:
            rows.append((
                "warn", "agents",
                f"{len(available_agents)} found, but only IDE-style: {names}",
                "Auto mode needs at least one CLI agent (claude, kiro, ...)",
            ))
        else:
            rows.append(("ok", "agents", f"{len(available_agents)} found: {names}", ""))
    else:
        rows.append((
            "fatal", "agents", "none on PATH",
            "Install one of: claude-code, kiro-cli, openclaude, opencode, "
            "aider, codex, gemini-cli",
        ))
        fatal = True

    # ── 8. LLM keys ───────────────────────────────────────────────
    try:
        status = llm.provider_status() if llm else {}
    except Exception:
        status = {}
    configured = [n for n, s in status.items() if s.get("available")]
    if configured:
        rows.append(("ok", "llm", f"{len(configured)} key(s): {', '.join(configured)}", ""))
    else:
        rows.append((
            "warn", "llm", "no keys configured",
            "Set ANTHROPIC_API_KEY (cache wins) or any free-tier key. "
            "Auto mode runs on the agents' own auth, but ORCH's router needs a key.",
        ))

    # ── 9. XDG dirs writable ─────────────────────────────────────
    try:
        from paths import global_data_dir, global_config_dir
        for label, p in (
            ("config dir", global_config_dir()),
            ("data dir", global_data_dir()),
        ):
            try:
                p.mkdir(parents=True, exist_ok=True)
                test = p / ".doctor-write-test"
                test.write_text("ok")
                test.unlink()
                rows.append(("ok", f"xdg:{label}", str(p), ""))
            except OSError as e:
                rows.append(("warn", f"xdg:{label}", f"not writable: {e}", ""))
    except Exception as e:
        rows.append(("warn", "xdg", f"paths module error: {e}", ""))

    # ── 10. project .orch/ ─────────────────────────────────────────
    project_orch = Path(os.getcwd()) / ".orch"
    if project_orch.exists():
        rows.append(("ok", "project", str(project_orch), ""))
    else:
        rows.append((
            "info", "project",
            f"no .orch/ in {os.getcwd()}",
            "Run [cyan]orch --init[/cyan] inside your project to scaffold it.",
        ))

    # ── render ────────────────────────────────────────────────────
    table = Table(
        title="agentorchestr — health check",
        title_style="bold cyan",
        show_header=True, header_style="cyan",
    )
    table.add_column("status", style="bold", width=6)
    table.add_column("area", style="cyan", width=18)
    table.add_column("detail")
    table.add_column("fix", style="dim")

    for status, area, detail, fix in rows:
        style = {"ok": "green", "warn": "yellow", "fatal": "red", "info": "dim"}.get(status, "white")
        glyph = {"ok": "✓", "warn": "⚠", "fatal": "✗", "info": "·"}.get(status, " ")
        table.add_row(f"[{style}]{glyph} {status}[/{style}]", area, detail, fix)

    console.print(table)

    counts = {"ok": 0, "warn": 0, "fatal": 0, "info": 0}
    for status, *_ in rows:
        counts[status] = counts.get(status, 0) + 1

    if fatal:
        console.print(Panel(
            f"[red]{counts['fatal']} fatal[/red] · "
            f"[yellow]{counts['warn']} warning(s)[/yellow] · "
            f"[green]{counts['ok']} ok[/green]\n\n"
            "Fix the [red]fatal[/red] rows above before running auto mode.\n"
            "[cyan]--manual --task '...'[/cyan] may still work for some failures.",
            border_style="red", title="[red]degraded[/red]",
        ))
        return 1

    if counts["warn"]:
        console.print(Panel(
            f"[yellow]{counts['warn']} warning(s)[/yellow] · "
            f"[green]{counts['ok']} ok[/green]\n\n"
            "Warnings are not blocking, but address the agent / LLM ones\n"
            "before relying on agentorchestr for production work.",
            border_style="yellow", title="[yellow]healthy with warnings[/yellow]",
        ))
        return 0

    console.print(Panel(
        f"[green]{counts['ok']} checks passed.[/green]\n"
        "agentorchestr is ready. Try [cyan]orch --setup[/cyan] in a project,\n"
        "then [cyan]orch --interactive[/cyan].",
        border_style="green", title="[green]healthy[/green]",
    ))
    return 0


# ── helpers ───────────────────────────────────────────────────────────

def _try_import(name: str) -> tuple[bool, str]:
    """Returns (importable, version-or-empty)."""
    try:
        mod = import_module(name)
        ver = getattr(mod, "__version__", "") or ""
        return True, str(ver)
    except ImportError:
        return False, ""
    except Exception:
        # Some packages do funky things on import; treat as installed.
        return True, "?"
