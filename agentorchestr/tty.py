"""tty.py — terminal-attach helpers for agentorchestr."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys

from rich.console import Console

console = Console()


def should_auto_attach() -> bool:
    import shutil as _sh
    if os.environ.get("TMUX"):
        return True
    if not sys.stdin.isatty():
        return False
    for term in ("kitty", "alacritty", "wezterm", "gnome-terminal", "konsole", "xterm"):
        if _sh.which(term):
            return True
    return False


def attach_to_tmux(session_name: str) -> None:
    if not sys.stdin.isatty():
        return
    if os.environ.get("TMUX"):
        subprocess.Popen(["tmux", "switch-client", "-t", session_name],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"[dim]→ switched tmux client to {session_name}[/dim]")
        return
    for term, args in (
        ("kitty",        ["kitty", "-e"]),
        ("alacritty",    ["alacritty", "-e"]),
        ("wezterm",      ["wezterm", "start", "--"]),
        ("gnome-terminal", ["gnome-terminal", "--"]),
        ("konsole",      ["konsole", "-e"]),
        ("xterm",        ["xterm", "-e"]),
    ):
        if shutil.which(term):
            subprocess.Popen(args + ["tmux", "attach", "-t", session_name],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            console.print(f"[dim]→ opened {term} attached to {session_name}[/dim]")
            return
    console.print(f"[yellow]No terminal emulator found.[/yellow] Run: "
                  f"[cyan]tmux attach -t {session_name}[/cyan]")


async def heartbeat_loop(store, session_id: str, stop_event: asyncio.Event,
                         *, interval: float = 5.0) -> None:
    from datetime import datetime
    last_sig: tuple = ()
    while not stop_event.is_set():
        try:
            workers = await store.get_workers(session_id)
        except Exception:
            workers = []
        counts: dict[str, int] = {}
        last_summary = last_id = ""
        for w in workers:
            st = w.get("state") or "unknown"
            counts[st] = counts.get(st, 0) + 1
            if w.get("summary"):
                last_summary = w["summary"]
                last_id = w.get("worker_id") or ""
        sig = (tuple(sorted(counts.items())), last_summary[:80])
        if workers and sig != last_sig:
            last_sig = sig
            ts = datetime.now().strftime("%H:%M:%S")
            counts_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(no workers yet)"
            tail = f"  | last: {last_id} {last_summary[:80]}" if last_summary else ""
            console.print(f"[dim]\\[{ts}] agentorchestr {session_id}  • {counts_str}{tail}[/dim]")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
