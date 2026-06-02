"""Tests for orchestrator agent selection behavior."""
from __future__ import annotations

import builtins
import sys

import pytest

from agentorchestr.orchestrator import _resolve_agents


class _Args:
    def __init__(self, interactive: bool = False, agents=None, lead=None):
        self.interactive = interactive
        self.agents = agents
        self.lead = lead


AVAILABLE_AGENTS = [
    {"name": "claude", "type": "sdk", "capabilities": {"orchestrate": 10, "implement": 10}},
    {"name": "openclaude", "type": "cli", "capabilities": {"orchestrate": 9, "implement": 9}},
    {"name": "kiro", "type": "cli", "capabilities": {"orchestrate": 8, "implement": 9}},
]


def test_resolve_agents_prompts_when_tty(monkeypatch):
    args = _Args(interactive=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    called = {"picked": False}

    def fake_interactive(available):
        called["picked"] = True
        return [available[1]], available[0]

    monkeypatch.setattr("agentorchestr.orchestrator._interactive_pick", fake_interactive)
    workers, lead = _resolve_agents(args, AVAILABLE_AGENTS)

    assert called["picked"] is True
    assert lead["name"] == "claude"
    assert [w["name"] for w in workers] == ["openclaude"]


def test_resolve_agents_auto_picks_when_not_tty(monkeypatch):
    args = _Args(interactive=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    workers, lead = _resolve_agents(args, AVAILABLE_AGENTS)

    assert lead["name"] == "claude"
    assert [w["name"] for w in workers] == ["openclaude", "kiro"]
