#!/usr/bin/env python3
"""
monitor.py — Rich Live Status Table
=====================================
Inspired by: Claude Code status display, AutoGen UI

Provides a live-updating terminal display showing task status,
event log, and session statistics.
"""

import time
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live


class Monitor:
    """Live terminal monitor for orchestration status."""

    def __init__(self, store, session_id: str):
        self.store = store
        self.session_id = session_id
        self.console = Console()
        self.events: list[dict] = []
        self._live: Optional[Live] = None
        self._start_time = time.time()

    def log(self, message: str):
        """Log a timestamped event."""
        self.events.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
        })
        # Keep last 50 events
        if len(self.events) > 50:
            self.events = self.events[-50:]
        # Print to console immediately
        self.console.print(f"  [dim]{self.events[-1]['time']}[/dim] {message}")

    async def get_status_table(self) -> Table:
        """Build the current status table from store."""
        table = Table(title=f"Session {self.session_id}", expand=True)
        table.add_column("Task", style="cyan", width=8)
        table.add_column("Status", width=12)
        table.add_column("Worker", style="dim", width=12)
        table.add_column("Retries", width=8)
        table.add_column("Summary", max_width=50)

        results = await self.store.get_results(self.session_id)

        status_icons = {
            "pending": "[blue]-[/blue] Ready",
            "running": "[yellow]![/yellow] Running",
            "done": "[green]✓[/green] Done",
            "failed": "[red]✗[/red] Failed",
        }

        stats = {"pending": 0, "running": 0, "done": 0, "failed": 0}

        for r in results:
            status = r.get("status", "pending")
            stats[status] = stats.get(status, 0) + 1
            icon = status_icons.get(status, status)
            summary = ""
            if r.get("result") and r["result"].get("summary"):
                summary = r["result"]["summary"][:50]
            table.add_row(
                r["task_id"],
                icon,
                r.get("assigned_worker") or "-",
                str(r.get("retry_count", 0)),
                summary,
            )

        # Add summary row
        elapsed = int(time.time() - self._start_time)
        table.add_row(
            "TOTAL",
            f"[green]{stats['done']}[/green]/[red]{stats['failed']}[/red]/[yellow]{stats['running']}[/yellow]/[blue]{stats['pending']}[/blue]",
            "",
            "",
            f"Elapsed: {elapsed}s",
        )

        return table

    def get_event_log(self) -> Panel:
        """Build event log panel."""
        if not self.events:
            return Panel("[dim]No events yet...[/dim]", title="Event Log")
        lines = []
        for evt in self.events[-15:]:  # Last 15 events
            lines.append(f"[dim]{evt['time']}[/dim] {evt['message']}")
        return Panel("\n".join(lines), title="Event Log")

    def get_stats(self) -> dict:
        """Get quick stats."""
        elapsed = int(time.time() - self._start_time)
        return {
            "session_id": self.session_id,
            "events": len(self.events),
            "elapsed_seconds": elapsed,
        }
