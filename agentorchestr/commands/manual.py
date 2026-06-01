"""commands/manual.py — single-agent single-task mode."""
from __future__ import annotations

import shlex

from rich.console import Console
from rich.panel import Panel

from agentorchestr.state_store import StateStore
from agentorchestr.tty import attach_to_tmux, should_auto_attach
from agentorchestr.worker_pool import WorkerPool

console = Console()


async def run_manual(args, available: list[dict], store: StateStore,
                     session_id: str) -> int:
    if not args.task:
        console.print("[yellow]Use --manual with --task 'your task description'[/yellow]")
        return 1

    candidates = ([a for a in available if a["name"] in args.agents]
                  if args.agents
                  else [a for a in available if a["type"] in ("sdk", "cli")][:1])
    if not candidates:
        console.print("[red]No suitable agent for manual mode[/red]")
        return 1
    agent = candidates[0]

    await store.save_session(session_id, {
        "goal": args.task, "plan": {}, "agents": [agent["name"]], "status": "active",
    })
    pool = WorkerPool(args.project, session_id, store)
    await pool.start_session([
        "bash", "-lc",
        f"echo '=== agentorchestr manual: {agent['name']} ==='; {shlex.quote(agent['cmd'])}",
    ])
    worker = await pool.spawn_worker(task=args.task, perspective="implementer", agent=agent)

    import asyncio
    await asyncio.sleep(0.1)

    console.print(Panel(
        f"[bold]session:[/bold]  [cyan]{session_id}[/cyan]\n"
        f"[bold]worker:[/bold]   [cyan]{worker.id}[/cyan] ({agent['name']})\n"
        f"[bold]worktree:[/bold] {worker.worktree}\n"
        f"[bold]attach:[/bold]   [cyan]{pool.attach_command}[/cyan]",
        title="[green]manual mode started[/green]", border_style="green",
    ))
    want_attach = should_auto_attach() if args.attach is None else args.attach
    if want_attach:
        attach_to_tmux(pool.tmux_session_name)
    return 0
