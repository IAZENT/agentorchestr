"""commands/auto.py — supervisor-worker auto mode."""
from __future__ import annotations

import asyncio
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agentorchestr.hooks import HookManager
from agentorchestr.state_store import StateStore
from agentorchestr.supervisor import Supervisor
from agentorchestr.tty import attach_to_tmux, heartbeat_loop, should_auto_attach

console = Console()


async def run_auto(args, available: list[dict], store: StateStore,
                   llm, session_id: str | None = None) -> int:
    from agentorchestr.mcp_bridge import MCP_AVAILABLE
    if not MCP_AVAILABLE:
        console.print(Panel(
            "[red]The `mcp` Python library is not installed.[/red]\n"
            "Install: [cyan]pip install 'agentorchestr[mcp]'[/cyan]",
            title="[red]missing dependency[/red]", border_style="red",
        ))
        return 1

    from agentorchestr.orchestrator import (
        _read_goal, _resolve_agents, _show_agents_table,
    )

    if not session_id:
        session_id = uuid.uuid4().hex[:8]

    goal = _read_goal(args)
    if not goal and not args.resume:
        if args.interactive or (not args.goal and not args.goal_file):
            try:
                goal = Prompt.ask("[bold cyan]Enter your goal[/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                goal = ""
    if not goal:
        console.print("[yellow]No goal provided.[/yellow]")
        return 0

    workers, lead = _resolve_agents(args, available)
    if not workers or lead is None:
        console.print("[red]Could not pick a lead + worker layout.[/red]")
        return 1

    _show_agents_table(available)
    console.print(
        f"\n[green]Lead:[/green] [bold]{lead['name']}[/bold] | "
        f"[green]Workers:[/green] {', '.join(a['name'] for a in workers)}"
    )

    await store.save_session(session_id, {
        "goal": goal, "plan": {}, "agents": [a["name"] for a in workers],
        "status": "active",
    })

    hooks = HookManager(project_root=args.project)
    await hooks.run("pre_session", {"session_id": session_id, "goal": goal,
                                    "lead": lead["name"],
                                    "workers": [a["name"] for a in workers]})

    memory = None
    try:
        from agentorchestr.memory import MemoryFederation
        memory = MemoryFederation(args.project)
        await memory.init()
    except Exception as e:
        console.print(f"[dim]memory unavailable: {e}[/dim]")

    skills_reg = None
    try:
        from agentorchestr.skills import SkillRegistry
        skills_reg = SkillRegistry(require_signature=args.require_signed_skills)
        skills_reg.reload()
    except Exception:
        pass

    sup = Supervisor(project_root=args.project, goal=goal, lead_agent=lead,
                     worker_agents=workers, store=store, session_id=session_id,
                     memory=memory, skills=skills_reg)

    console.print(Panel(
        f"[bold]session:[/bold] [cyan]{session_id}[/cyan]\n"
        f"[bold]tmux:[/bold]    [cyan]{sup.attach_command}[/cyan]\n"
        f"[bold]MCP:[/bold]     http://{sup.host}:{sup.port}/sse\n"
        f"[dim]Ctrl-B D to detach · Ctrl-C here to cancel[/dim]",
        title="[green]auto mode running[/green]", border_style="green",
    ))

    want_attach = should_auto_attach() if args.attach is None else args.attach
    if want_attach:
        attach_to_tmux(sup.pool.tmux_session_name)

    state, summary = "failed", "supervisor never started"
    stop = asyncio.Event()
    hb = asyncio.create_task(heartbeat_loop(store, session_id, stop))
    try:
        state, summary = await sup.run()
    except KeyboardInterrupt:
        state, summary = "cancelled", "interrupted by user"
    except Exception as e:
        state, summary = "failed", f"{type(e).__name__}: {e}"
        console.print(f"[red]✗ supervisor crashed:[/red] {summary}")
    finally:
        stop.set()
        try:
            await asyncio.wait_for(hb, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            hb.cancel()
        await sup.cleanup(keep_tmux=not args.no_tmux_cleanup)
        await store.save_session(session_id,
                                 {"status": state if state != "cancelled" else "paused"})
        await hooks.run("post_session", {"session_id": session_id,
                                         "state": state, "summary": summary})

    color = {"done": "green", "failed": "red",
             "cancelled": "yellow", "timeout": "yellow"}.get(state, "white")
    console.print(Panel(
        f"[bold]state:[/bold]   [{color}]{state}[/{color}]\n"
        f"[bold]summary:[/bold] {summary}\n"
        f"[dim]tmux: {sup.attach_command}[/dim]",
        title=f"[{color}]session {session_id} complete[/{color}]",
        border_style=color,
    ))
    return 0 if state == "done" else 2
