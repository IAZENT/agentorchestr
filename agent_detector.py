"""
agent_detector.py
==================
Detects all installed CLI coding agents on the system.
Adds zero cost — pure shutil.which() checks + version probing.

Each agent declares a list of binary candidates in priority order
(e.g. kiro is shipped as `kiro-cli` on most installs; we try `kiro` first
then fall back to `kiro-cli`, `kirocli`, etc.).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Iterable, Optional


# Registry of every known CLI agent and how to probe it.
# `cmds` is a priority-ordered list of binary names — first one found wins.
AGENT_REGISTRY: dict[str, dict] = {
    "claude":     {"cmds": ["claude", "claude-code"],            "type": "sdk", "vflag": "--version"},
    "openclaude": {"cmds": ["openclaude"],                       "type": "cli", "vflag": "--version"},
    "kiro":       {"cmds": ["kiro", "kiro-cli", "kirocli"],      "type": "cli", "vflag": "--version"},
    "opencode":   {"cmds": ["opencode"],                         "type": "cli", "vflag": "--version"},
    "aider":      {"cmds": ["aider"],                            "type": "cli", "vflag": "--version"},
    "codex":      {"cmds": ["codex"],                            "type": "cli", "vflag": "--version"},
    "gemini":     {"cmds": ["gemini", "gemini-cli"],             "type": "cli", "vflag": "--version"},
    "goose":      {"cmds": ["goose"],                            "type": "cli", "vflag": "--version"},
    "amp":        {"cmds": ["amp"],                              "type": "cli", "vflag": "--version"},
    "cursor":     {"cmds": ["cursor"],                           "type": "ide", "vflag": "--version"},
    "windsurf":   {"cmds": ["windsurf"],                         "type": "ide", "vflag": "--version"},
    "copilot":    {"cmds": ["gh"],                               "type": "gh",  "vflag": "--version"},
    "kilocode":   {"cmds": ["kilocode"],                         "type": "ide", "vflag": "--version"},
}

# Capability scores for ranking agents as orchestrator vs worker
AGENT_CAPABILITIES: dict[str, dict] = {
    "claude":     {"orchestrate": 10, "implement": 10, "parallel": True,  "mcp": True},
    "openclaude": {"orchestrate": 9,  "implement": 9,  "parallel": True,  "mcp": True},
    "kiro":       {"orchestrate": 9,  "implement": 9,  "parallel": True,  "mcp": True},
    "opencode":   {"orchestrate": 8,  "implement": 9,  "parallel": True,  "mcp": False},
    "aider":      {"orchestrate": 6,  "implement": 8,  "parallel": True,  "mcp": False},
    "codex":      {"orchestrate": 7,  "implement": 8,  "parallel": True,  "mcp": False},
    "gemini":     {"orchestrate": 8,  "implement": 8,  "parallel": True,  "mcp": False},
    "goose":      {"orchestrate": 7,  "implement": 7,  "parallel": True,  "mcp": True},
    "amp":        {"orchestrate": 7,  "implement": 8,  "parallel": True,  "mcp": False},
    "cursor":     {"orchestrate": 4,  "implement": 9,  "parallel": False, "mcp": True},
    "windsurf":   {"orchestrate": 4,  "implement": 9,  "parallel": False, "mcp": True},
    "copilot":    {"orchestrate": 3,  "implement": 7,  "parallel": False, "mcp": True},
}


def _first_on_path(cmds: Iterable[str]) -> Optional[tuple[str, str]]:
    """Return (cmd_name, absolute_path) for the first candidate on PATH."""
    for c in cmds:
        path = shutil.which(c)
        if path:
            return c, path
    return None


class AgentDetector:
    """Detects installed agents with zero API cost."""

    def detect(self) -> list[dict]:
        """
        Returns a list of available agents sorted by orchestration capability.
        Each dict: {name, cmd, path, type, version, capabilities}
        """
        available: list[dict] = []
        for name, info in AGENT_REGISTRY.items():
            cmds = info.get("cmds") or [info.get("cmd")]
            hit = _first_on_path([c for c in cmds if c])
            if not hit:
                continue
            cmd_name, path = hit

            # Special-case: gh is detected as 'copilot' only when the
            # GitHub Copilot extension is installed.
            if name == "copilot" and not self._has_gh_copilot_ext(path):
                continue

            version = self._probe_version(path, info["vflag"])
            caps = AGENT_CAPABILITIES.get(name, {
                "orchestrate": 5, "implement": 5, "parallel": False, "mcp": False
            })
            available.append({
                "name": name,
                "cmd": cmd_name,
                "path": path,
                "type": info["type"],
                "version": version,
                "capabilities": caps,
            })

        # Sort: SDK > CLI > IDE, then by orchestrate score
        priority = {"sdk": 0, "cli": 1, "ide": 2, "gh": 3}
        available.sort(key=lambda a: (priority.get(a["type"], 9), -a["capabilities"]["orchestrate"]))
        return available

    @staticmethod
    def _probe_version(path: str, flag: str) -> str:
        try:
            result = subprocess.run(
                [path, flag],
                capture_output=True, text=True, timeout=3,
            )
            line = (result.stdout or result.stderr or "").strip().split("\n")[0]
            return line[:40] if line else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _has_gh_copilot_ext(gh_path: str) -> bool:
        """gh by itself isn't Copilot — the extension must be installed."""
        try:
            r = subprocess.run([gh_path, "extension", "list"],
                               capture_output=True, text=True, timeout=3)
            return "copilot" in (r.stdout or "").lower()
        except Exception:
            return False

    def get_cli_agents(self, agents: list[dict]) -> list[dict]:
        """Only agents controllable via CLI (not IDE-only)."""
        return [a for a in agents if a["type"] in ("sdk", "cli")]

    def get_mcp_agents(self, agents: list[dict]) -> list[dict]:
        """Agents that support MCP server connections."""
        return [a for a in agents if a["capabilities"].get("mcp")]

    def get_parallel_agents(self, agents: list[dict]) -> list[dict]:
        """Agents that support multiple concurrent instances."""
        return [a for a in agents if a["capabilities"].get("parallel")]
