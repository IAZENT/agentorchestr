#!/usr/bin/env python3
"""
bootstrap.py — ORCH installer + environment check
==================================================
Installs core deps (and optional extras), then prints a status
report of detected agents, LLM keys, and tmux availability.

Safety:
  - Refuses to use `pip --break-system-packages` unless explicitly
    requested with --force-system. Recommends a venv instead.
  - Idempotent: re-running just rechecks the environment.

Usage:
  python bootstrap.py                  # core deps only
  python bootstrap.py --with-mcp       # + MCP bridge support
  python bootstrap.py --with-search    # + DuckDuckGo web search
  python bootstrap.py --with-test      # + pytest harness
  python bootstrap.py --all            # everything optional
  python bootstrap.py --check          # don't install, just report
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

CORE = [
    "aiosqlite>=0.20",
    "httpx>=0.27",
    "rich>=13.7",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "libtmux>=0.55,<0.60",
]

EXTRAS = {
    "mcp":    ["mcp>=1.20"],
    "search": ["ddgs>=9.0"],
    "test":   ["pytest>=8.0", "pytest-asyncio>=0.23"],
}

ENV_VARS = {
    "ANTHROPIC_API_KEY":  ("https://console.anthropic.com",     "Paid: 90% off cached prefix tokens"),
    "GEMINI_API_KEY":     ("https://ai.google.dev",             "Free: 1500 req/day on Gemini 2.5 Flash"),
    "GROQ_API_KEY":       ("https://console.groq.com",          "Free: 30 RPM on Llama 70B"),
    "CEREBRAS_API_KEY":   ("https://inference.cerebras.ai",     "Free: 1M tokens/day on Llama 70B"),
    "OPENROUTER_API_KEY": ("https://openrouter.ai",             "Free: 28+ models including DeepSeek R1"),
}

AGENTS = {
    "openclaude": "github.com/Gitlawb/openclaude",
    "kiro":       "amazon.com/kiro",
    "opencode":   "opencode.ai",
    "claude":     "npm install -g @anthropic-ai/claude-code",
    "aider":      "pip install aider-chat",
    "codex":      "npm install -g @openai/codex",
    "gemini":     "npm install -g @google/gemini-cli",
    "goose":      "block.github.io/goose",
    "amp":        "github.com/sourcegraph/amp",
}


def check_python() -> None:
    if sys.version_info < (3, 11):
        print(f"ERROR: Python 3.11+ required. You have {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")


def in_virtualenv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or "VIRTUAL_ENV" in os.environ
    )


def pip_install(packages: list[str], force_system: bool) -> None:
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install"]
    if not in_virtualenv():
        if force_system:
            cmd.append("--break-system-packages")
            print("⚠  Installing into system Python with --break-system-packages")
        else:
            cmd.append("--user")
            print("ℹ  Not in a venv — using --user. (Pass --force-system to override.)")
    cmd.extend(packages)
    print(f"→ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def check_env_vars() -> list[str]:
    print("\n── LLM Backends ────────────────────────────────")
    print("ORCH needs at least one hosted LLM API key.\n")
    configured = []
    for key, (url, note) in ENV_VARS.items():
        val = os.environ.get(key)
        if val:
            print(f"  ✓ {key:<20} (configured)")
            configured.append(key)
        else:
            print(f"  ✗ {key:<20} — {note}")
            print(f"      → {url}")
    if not configured:
        print("\n⚠  No LLM keys found. Set at least one in your shell, e.g.:")
        print("   export ANTHROPIC_API_KEY=...     (best caching support)")
        print("   export CEREBRAS_API_KEY=...      (fastest free tier)\n")
    return configured


def check_tmux() -> None:
    if shutil.which("tmux"):
        print("✓ tmux available (multi-pane agent sessions enabled)")
    else:
        print("⚠  tmux not found. ORCH will use subprocess mode (no visual splits).")
        print("   Install: brew install tmux  OR  sudo apt install tmux")


def check_agents() -> list[str]:
    print("\n── CLI Agents ──────────────────────────────────")
    found = []
    for agent, install in AGENTS.items():
        if shutil.which(agent):
            print(f"  ✓ {agent}")
            found.append(agent)
    if not found:
        print("  No agents detected. Install at least one:")
        for agent, install in AGENTS.items():
            print(f"    {agent:<12} — {install}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-mcp", action="store_true", help="Install MCP bridge support")
    parser.add_argument("--with-search", action="store_true", help="Install DuckDuckGo web search")
    parser.add_argument("--with-test", action="store_true", help="Install pytest test harness")
    parser.add_argument("--all", action="store_true", help="Install all optional extras")
    parser.add_argument("--check", action="store_true", help="Don't install — just report status")
    parser.add_argument("--force-system", action="store_true",
                        help="Allow --break-system-packages outside a venv (NOT recommended)")
    args = parser.parse_args()

    print("═══════════════════════════════════════")
    print("  ORCH — bootstrap & environment check")
    print("═══════════════════════════════════════\n")

    check_python()

    if not args.check:
        to_install = list(CORE)
        if args.all or args.with_mcp:    to_install += EXTRAS["mcp"]
        if args.all or args.with_search: to_install += EXTRAS["search"]
        if args.all or args.with_test:   to_install += EXTRAS["test"]
        print(f"\n→ Installing {len(to_install)} package(s)...")
        try:
            pip_install(to_install, args.force_system)
            print("✓ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ pip install failed: {e}")
            return 1

    configured = check_env_vars()
    print()
    found_agents = check_agents()
    print()
    check_tmux()

    print("\n═══════════════════════════════════════")
    if found_agents and configured:
        print(f"✓ Ready. {len(found_agents)} agent(s), {len(configured)} LLM key(s).")
        print("\nQuick start:")
        print("  python orchestrator.py --detect")
        print('  python orchestrator.py --goal "Build a REST API for user auth"')
    else:
        if not found_agents:
            print("⚠  Install at least one CLI agent.")
        if not configured:
            print("⚠  Configure at least one hosted LLM API key.")
    print("═══════════════════════════════════════")
    return 0


if __name__ == "__main__":
    sys.exit(main())
