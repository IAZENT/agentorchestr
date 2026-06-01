"""
wizard.py — first-run setup + agent layout chooser
====================================================

Two flows:

    Manual (interactive)
        1. Show every detected agent with capability stats.
        2. Ask which is the LEAD (default: highest orchestrate score).
        3. Ask which agents go in the WORKER POOL (multi-select).
        4. Ask the worker count.  N > pool size is allowed: agentorchestr cycles
           through the pool so e.g. 1 agent + count=3 spawns 3 sessions
           of the same agent in parallel — exactly what the user asked
           for ("can use a single agent with multiple session as multiple
           workers").

    Automatic
        1. Run complexity.assess() against the project.
        2. Pick the agent with the best orchestrate score as lead.
        3. Pick the top-K agents (deduped) as workers, where K matches
           the recommended_workers from the complexity tier.
        4. Confirm with the user (one keypress) before starting.

The chosen layout is persisted via preferences.save() so subsequent
runs skip the wizard unless --setup is passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from agentorchestr import complexity
from agentorchestr import preferences

console = Console()


@dataclass
class Layout:
    lead: dict                  # full agent dict for the lead
    workers: list[dict]         # ordered list of agent dicts; len() == worker_count
    mode: str                   # "manual" | "automatic"
    rationale: str = ""

    def to_prefs(self) -> dict:
        """Project this Layout into the preferences-file shape."""
        return {
            "setup": {"mode": self.mode},
            "agents": {
                "lead": self.lead.get("name", ""),
                "workers": [w.get("name", "") for w in self.workers],
                "worker_count": len(self.workers),
            },
        }


# ── public entry ──────────────────────────────────────────────────────

def needs_wizard(args, available: list[dict]) -> bool:
    """True if we should run the wizard now."""
    if getattr(args, "setup", False):
        return True
    if not available:
        return False                   # nothing to choose from
    return not preferences.is_setup_complete()


def run(available: list[dict], project_root: str) -> Optional[Layout]:
    """Drive the wizard.  Returns the chosen Layout, or None on abort.

    Ctrl-C and EOF (Ctrl-D) are handled cleanly: the wizard prints a
    one-line dim message and returns None.  The caller's existing fall-
    back path (saved prefs → heuristic) takes over from there.
    """
    if not available:
        console.print(
            "[red]No CLI agents detected on PATH. Install at least one — "
            "see `agentorchestr --detect` for the list.[/red]"
        )
        return None

    try:
        console.print("\n[bold]Select operating mode:[/bold]")
        console.print("  [cyan]1.[/cyan] Automatic — detect project complexity, pick agents for you")
        console.print("  [cyan]2.[/cyan] Manual    — choose lead and workers yourself")
        console.print("  [cyan]3.[/cyan] Skip      — use defaults, no customisation")
        choice = IntPrompt.ask(
            "\n[bold]Mode[/bold]",
            choices=["1", "2", "3"],
            default=1,
        )
        if choice == 3:
            console.print("[dim]Skipping wizard. Re-run with [cyan]agentorchestr --setup[/cyan] any time.[/dim]")
            return None

        layout = _automatic_flow(available, project_root) if choice == 1 \
            else _manual_flow(available)

        if layout is None:
            return None

        _summarize(layout)
        if Confirm.ask("\nSave this as your default layout?", default=True):
            try:
                preferences.save(layout.to_prefs())
                console.print(f"[green]✓[/green] saved to {preferences.prefs_path()}")
            except OSError as e:
                console.print(f"[yellow]⚠ could not save preferences: {e}[/yellow]")
        return layout
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]wizard cancelled — falling back to defaults. "
                      "Re-run with [cyan]agentorchestr --setup[/cyan] any time.[/dim]")
        return None


# ── flows ─────────────────────────────────────────────────────────────

def _automatic_flow(available: list[dict], project_root: str) -> Optional[Layout]:
    console.print("\n[bold cyan]Probing project complexity…[/bold cyan]")
    c = complexity.assess(project_root)
    console.print(Panel(
        f"[bold]tier:[/bold]      [cyan]{c.tier}[/cyan]\n"
        f"[bold]workers:[/bold]   {c.recommended_workers}\n"
        f"[bold]rationale:[/bold] [dim]{c.rationale}[/dim]\n"
        f"[bold]languages:[/bold] {', '.join(sorted(c.languages)) or '(none detected)'}",
        title=f"[cyan]complexity: {c.tier}[/cyan]", border_style="cyan",
    ))

    cli_agents = [a for a in available if a["type"] in ("sdk", "cli")]
    if not cli_agents:
        console.print("[red]Automatic mode needs at least one CLI/SDK agent.[/red]")
        return None

    # Pick lead = highest orchestrate score.
    lead = max(cli_agents, key=lambda a: a["capabilities"].get("orchestrate", 0))

    # Pick workers: top-K by (orchestrate + implement) / 2, deduped by name.
    by_score = sorted(
        cli_agents,
        key=lambda a: (a["capabilities"].get("orchestrate", 0)
                       + a["capabilities"].get("implement", 0)) / 2,
        reverse=True,
    )
    seen: set[str] = set()
    workers: list[dict] = []
    for a in by_score:
        if a["name"] in seen:
            continue
        workers.append(a)
        seen.add(a["name"])
        if len(workers) >= c.recommended_workers:
            break
    # If pool < recommended count, cycle through the pool.
    while len(workers) < c.recommended_workers:
        workers.append(workers[len(workers) % max(len(seen), 1)])

    return Layout(
        lead=lead, workers=workers, mode="automatic",
        rationale=f"complexity={c.tier} → {c.recommended_workers} worker(s)",
    )


def _manual_flow(available: list[dict]) -> Optional[Layout]:
    cli_agents = [a for a in available if a["type"] in ("sdk", "cli")]
    if not cli_agents:
        console.print("[red]Manual mode needs at least one CLI/SDK agent.[/red]")
        return None

    # Step 1 — pick the lead.
    default_lead = max(cli_agents, key=lambda a: a["capabilities"].get("orchestrate", 0))
    default_idx = cli_agents.index(default_lead) + 1
    raw = Prompt.ask(
        f"\nWhich agent is the [bold]LEAD[/bold]? "
        f"(1-{len(cli_agents)}, default {default_idx}={default_lead['name']})",
        default=str(default_idx),
    )
    try:
        lead_idx = int(raw) - 1
    except ValueError:
        lead_idx = default_idx - 1
    lead = cli_agents[lead_idx] if 0 <= lead_idx < len(cli_agents) else default_lead

    # Step 2 — pick the worker pool.
    raw = Prompt.ask(
        "\nWhich agents go in the [bold]WORKER POOL[/bold]?\n"
        "  (comma-separated indices, 'all', or Enter to use the lead alone)",
        default=str(default_idx),
    )
    pool: list[dict] = []
    if raw.strip().lower() == "all":
        pool = cli_agents[:]
    elif raw.strip():
        for part in raw.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(cli_agents):
                    pool.append(cli_agents[idx])
            except ValueError:
                continue
    if not pool:
        pool = [lead]

    # Step 3 — worker count (allows N copies of one agent for parallel sessions).
    default_count = max(1, len(pool))
    count = IntPrompt.ask(
        f"\nHow many [bold]workers[/bold] should run in parallel? "
        f"(default {default_count})",
        default=default_count,
    )
    count = max(1, min(int(count), 8))   # cap at 8 — tmux pane overhead

    # Build the worker list by cycling through the pool.
    workers = [pool[i % len(pool)] for i in range(count)]

    return Layout(lead=lead, workers=workers, mode="manual",
                  rationale=f"manual: lead={lead['name']}, pool={[a['name'] for a in pool]}, count={count}")


# ── helpers ───────────────────────────────────────────────────────────

def _summarize(layout: Layout) -> None:
    workers_str = ", ".join(w["name"] for w in layout.workers)
    console.print(Panel(
        f"[bold]mode:[/bold]    {layout.mode}\n"
        f"[bold]lead:[/bold]    [cyan]{layout.lead['name']}[/cyan]\n"
        f"[bold]workers:[/bold] {workers_str}\n"
        f"[dim]{layout.rationale}[/dim]",
        title="[green]chosen layout[/green]", border_style="green",
    ))
