"""commands/info.py — read-only display commands (detect, sessions, skills, init, doctor)."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def print_detection(available: list[dict], llm) -> None:
    console.print("\n[bold cyan]═══ Agents on PATH ═══[/bold cyan]")
    if not available:
        console.print("  [red]none detected.[/red]")
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

    console.print("\n[bold cyan]═══ Optional deps ═══[/bold cyan]")
    from agentorchestr.mcp_bridge import MCP_AVAILABLE
    try:
        import zeroconf; zc_ok = True  # noqa: E401
    except ImportError:
        zc_ok = False
    try:
        import ddgs; ddgs_ok = True  # noqa: E401
    except ImportError:
        ddgs_ok = False
    for name, ok, why in [
        ("mcp",      MCP_AVAILABLE, "REQUIRED for auto mode"),
        ("zeroconf", zc_ok,         "optional: cross-terminal discovery"),
        ("ddgs",     ddgs_ok,       "optional: web research"),
    ]:
        mark = "[green]✓[/green]" if ok else "[yellow]✗[/yellow]"
        console.print(f"  {mark} {name:<10} [dim]{why}[/dim]")


def show_agents_table(available: list[dict]) -> None:
    if not available:
        console.print("[yellow]No agents detected on PATH.[/yellow]")
        return
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
        table.add_row(str(i), a["name"], a["type"],
                      str(caps.get("orchestrate", "?")),
                      str(caps.get("implement", "?")), flags)
    console.print(table)


async def print_session_list(store) -> None:
    from datetime import datetime
    rows = await store.list_sessions(limit=20)
    if not rows:
        console.print("[dim]No sessions yet.[/dim]")
        return
    console.print("\n[bold cyan]═══ Recent sessions ═══[/bold cyan]")
    console.print(f"  [dim]{'id':<10} {'status':<10} {'tasks':<10} {'updated':<19} goal[/dim]")
    for r in rows:
        unfinished = r.get("pending", 0) + r.get("running", 0)
        resumable = r["status"] in ("active", "paused") and unfinished > 0
        marker = "[yellow]*[/yellow]" if resumable else " "
        ts = datetime.fromtimestamp(r["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
        tasks = f"{r.get('done', 0)}/{r.get('total', 0)}"
        goal = (r["goal"] or "").replace("\n", " ")[:60]
        console.print(f" {marker}[cyan]{r['id']:<10}[/cyan] {r['status']:<10} {tasks:<10} {ts}  {goal}")


def handle_skill_subcommand(reg, args) -> int:
    reg.reload()
    if args.skill_list:
        skills = reg.all()
        if not skills:
            console.print("[dim]No skills installed.[/dim]")
            return 0
        for s in skills:
            sig = "[green]signed[/green]" if s.manifest.has_signature else "[yellow]unsigned[/yellow]"
            console.print(f"  [cyan]{s.name:<30}[/cyan] v{s.manifest.version:<10} {sig}")
        return 0
    if args.skill_add:
        try:
            skill = reg.add_from_git(args.skill_add)
            console.print(f"[green]✓[/green] installed [cyan]{skill.name}[/cyan]")
            return 0
        except (FileExistsError, RuntimeError, ValueError) as e:
            console.print(f"[red]✗ {e}[/red]")
            return 1
    if args.skill_remove:
        if reg.remove(args.skill_remove):
            console.print(f"[green]✓[/green] removed [cyan]{args.skill_remove}[/cyan]")
            return 0
        console.print(f"[red]✗ not installed[/red]")
        return 1
    if args.skill_verify:
        info = reg.verify(args.skill_verify)
        if info["ok"]:
            console.print(f"[green]✓[/green] {args.skill_verify} verified")
            return 0
        console.print(f"[red]✗ {info['error']}[/red]")
        return 1
    return 0


def handle_init(args) -> int:
    from agentorchestr.paths import init_project
    actions = init_project(args.project, force=args.init_force)
    console.print(f"\n[bold cyan]agentorchestr init at {args.project}[/bold cyan]")
    for path, action in actions.items():
        color = {"created": "green", "appended": "green",
                 "kept": "dim", "overwrote": "yellow"}.get(action, "white")
        console.print(f"  [{color}]{action:<10}[/{color}] {path}")
    return 0
