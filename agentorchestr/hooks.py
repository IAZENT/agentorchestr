#!/usr/bin/env python3
"""
hooks.py — Lifecycle Hooks
============================
Inspired by: Claude Code hooks, Kiro agent hooks

Runs user-defined shell commands at orchestration lifecycle points.
Hook failures never block the main orchestration loop.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

HOOKS_CONFIG = ".agentorchestr/hooks.json"

# Hook events
EVENTS = [
    "pre_session",    # Before session starts
    "post_session",   # After session ends
    "pre_task",       # Before task is assigned to worker
    "post_task",      # After task completes (success or failure)
    "pre_eval",       # Before quality gate evaluation
    "post_eval",      # After quality gate evaluation
    "on_failure",     # When task fails after all retries
    "on_pause",       # When session is paused (Ctrl+C)
    "on_resume",      # When session is resumed
]


class HookManager:
    """Manages lifecycle hooks for orchestration events."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.hooks: dict[str, list[str]] = {}
        self._load_hooks()

    def _load_hooks(self):
        """Load hook configuration from .agentorchestr/hooks.json."""
        config_path = self.project_root / HOOKS_CONFIG
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                config = json.load(f)
            for event in EVENTS:
                if event in config:
                    cmds = config[event]
                    if isinstance(cmds, str):
                        cmds = [cmds]
                    self.hooks[event] = cmds
        except (json.JSONDecodeError, KeyError):
            pass

    def has_hooks(self, event: str) -> bool:
        """Check if any hooks are registered for an event."""
        return event in self.hooks and len(self.hooks[event]) > 0

    async def run(self, event: str, context: dict = None, timeout: float = 30.0):
        """
        Run all hooks for an event.

        Args:
            event: Hook event name (from EVENTS list)
            context: Dict passed as JSON via stdin to the hook
            timeout: Max seconds per hook command (default 30)
        """
        if event not in self.hooks:
            return

        context = context or {}
        context_json = json.dumps(context)

        for cmd in self.hooks[event]:
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.project_root),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=context_json.encode()),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except Exception:
                # Hook failures never block the main loop
                pass

    def get_config_path(self) -> Path:
        """Return the path to the hooks config file."""
        return self.project_root / HOOKS_CONFIG

    @staticmethod
    def create_example_config(project_root: str = ".") -> Path:
        """Create an example hooks config file."""
        config_path = Path(project_root) / HOOKS_CONFIG
        config_path.parent.mkdir(parents=True, exist_ok=True)

        example = {
            "pre_task": "echo 'Task starting: ' $(cat) | logger -t agentorchestr",
            "post_task": "echo 'Task complete' > /tmp/agentorchestr_last_task.log",
            "on_failure": "echo 'Task failed — review needed' | mail -s 'agentorchestr Alert' you@example.com",
        }

        with open(config_path, "w") as f:
            json.dump(example, f, indent=2)

        return config_path
