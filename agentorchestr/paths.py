"""
paths.py — single source of truth for agentorchestr filesystem layout
==============================================================

agentorchestr separates global, per-project, and per-session state.  The exact
locations follow the XDG Base Directory specification on Linux/macOS,
with a backward-compatibility layer for installations that still keep
state under ``~/.agentorchestr/``.

Layout (XDG-compliant; defaults shown):

    Global (cross-project):
        $XDG_CONFIG_HOME/agentorchestr/         (~/.config/agentorchestr)
            mcp.json                    optional global MCP overrides
            hooks.json                  optional global hooks
        $XDG_DATA_HOME/agentorchestr/           (~/.local/share/agentorchestr)
            state.db                    sqlite — sessions, workers, op_log
            skills/                     installed skill bundles
            memory/                     global markdown memory (cross-project)
        $XDG_CACHE_HOME/agentorchestr/          (~/.cache/agentorchestr)
            (research / embedding caches if we ever externalise them)

    Per-project (in your repo, git-trackable):
        <project>/.agentorchestr/
            PROJECT.md                  project context (auto-loaded as preamble)
            CONVENTIONS.md              style/lint/test rules (auto-loaded)
            MEMORY.md                   index of session learnings
            memory/
                topics/*.md             learned topic notes
                episodes/*.md           per-session journals
                research/*.md           cached web-research payloads
            hooks.json                  per-project lifecycle hooks
            skills/                     project-pinned skills (override globals)

    Per-session (ephemeral, lives in /tmp):
        /tmp/agentorchestr-<sid>/
            supervisor_prompt.txt
            progress.log

Backward compat
---------------
If ``~/.agentorchestr/`` exists (the pre-XDG layout) we keep using it for global
state.  This means installs that predate this module just keep working;
new installs use the XDG layout.  ``agentorchestr init`` scaffolds per-project
directories regardless.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ── XDG helpers ─────────────────────────────────────────────────────────

def _xdg(env_var: str, default_relative: str) -> Path:
    """Return $env_var if set, else $HOME/<default_relative>."""
    val = os.environ.get(env_var)
    if val:
        return Path(val).expanduser().resolve()
    return Path.home() / default_relative


def xdg_config_home() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config")


def xdg_data_home() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share")


def xdg_cache_home() -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache")


# ── agentorchestr global paths ───────────────────────────────────────────────────

LEGACY_HOME = Path.home() / ".agentorchestr"


def _use_legacy() -> bool:
    """Existing installs that already have ~/.agentorchestr keep using it.

    This avoids silently moving a user's state.db to a new location on
    upgrade — which would look like data loss.  New installs (no
    ~/.agentorchestr directory) get the XDG layout from the start.
    """
    return LEGACY_HOME.exists()


def global_config_dir() -> Path:
    if _use_legacy():
        return LEGACY_HOME
    p = xdg_config_home() / "agentorchestr"
    p.mkdir(parents=True, exist_ok=True)
    return p


def global_data_dir() -> Path:
    if _use_legacy():
        return LEGACY_HOME
    p = xdg_data_home() / "agentorchestr"
    p.mkdir(parents=True, exist_ok=True)
    return p


def global_cache_dir() -> Path:
    if _use_legacy():
        return LEGACY_HOME / "cache"
    p = xdg_cache_home() / "agentorchestr"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_db_path() -> Path:
    """Where sessions / workers / op_log live."""
    return global_data_dir() / "state.db"


def global_skills_dir() -> Path:
    p = global_data_dir() / "skills"
    p.mkdir(parents=True, exist_ok=True)
    return p


def global_memory_dir() -> Path:
    p = global_data_dir() / "memory"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── per-project paths ───────────────────────────────────────────────────

def project_agentorch_dir(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / ".agentorchestr"


def project_memory_dir(project_root: str | Path) -> Path:
    return project_agentorch_dir(project_root) / "memory"


# ── per-session paths ───────────────────────────────────────────────────

def session_tmp_dir(session_id: str) -> Path:
    p = Path(f"/tmp/agentorchestr-{session_id}")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── scaffolding ─────────────────────────────────────────────────────────

PROJECT_MD_TEMPLATE = """\
# PROJECT.md

> Auto-loaded by agentorchestr at the start of every session as the per-project
> cacheable preamble.  Keep this short — every byte here is re-paid on
> every supervisor turn.  Move long-form details into skills/ or
> memory/topics/ which are loaded only when relevant.

## Stack

(language, framework, key libraries, tooling)

## Architecture

(2-3 sentences on how the codebase is laid out — module boundaries,
data flow, anything a worker needs to know to make a sane edit)

## How to verify

(exact commands the verifier should run before mark_goal_done; these
are READ literally by the supervisor)

  - Run tests:  pytest -q
  - Lint:       ruff check .
  - Type check: mypy .

## Out-of-scope

(rules that prevent workers from over-reaching: don't touch
infrastructure code, don't bump dependencies, etc.)
"""

CONVENTIONS_MD_TEMPLATE = """\
# CONVENTIONS.md

> Style / lint / test rules.  Loaded into the cacheable preamble.

- Indent: 4 spaces (Python) / 2 spaces (TS, YAML, JSON)
- Naming: snake_case in Python, camelCase in TS
- Tests live next to the code they test under tests/
- Public functions need a one-line docstring at minimum
- New deps require a quick justification comment in the PR description
"""

HOOKS_JSON_TEMPLATE = """\
{
  "_doc": "agentorchestr lifecycle hooks. Each value is a shell command run with the event JSON on stdin. See hooks.py EVENTS for the full list.",
  "pre_session": null,
  "post_session": null,
  "pre_task": null,
  "post_task": null,
  "on_failure": null
}
"""

GITIGNORE_LINES = [
    "# agentorchestr per-project state — keep markdown, exclude transient artefacts",
    ".agentorchestr/cache/",
    ".agentorchestr/*.log",
    ".agentorchestr/skills/*/  # remove this line to git-track installed skills",
    "",
]


def init_project(project_root: str | Path, *, force: bool = False) -> dict:
    """Scaffold ``<project>/.agentorchestr/`` with the recommended files.

    Idempotent: existing files are left alone unless ``force=True``,
    in which case they're overwritten.  Returns a dict mapping each
    written / skipped path to its action ("created" | "kept" | "overwrote").
    """
    root = Path(project_root).resolve()
    ao = root / ".agentorchestr"
    actions: dict[str, str] = {}

    files = {
        "PROJECT.md": PROJECT_MD_TEMPLATE,
        "CONVENTIONS.md": CONVENTIONS_MD_TEMPLATE,
        "hooks.json": HOOKS_JSON_TEMPLATE,
    }
    ao.mkdir(parents=True, exist_ok=True)
    (ao / "memory").mkdir(parents=True, exist_ok=True)
    (ao / "memory" / "topics").mkdir(parents=True, exist_ok=True)
    (ao / "memory" / "episodes").mkdir(parents=True, exist_ok=True)
    (ao / "memory" / "research").mkdir(parents=True, exist_ok=True)
    (ao / "skills").mkdir(parents=True, exist_ok=True)

    for name, body in files.items():
        path = ao / name
        if path.exists() and not force:
            actions[str(path)] = "kept"
            continue
        path.write_text(body, encoding="utf-8")
        actions[str(path)] = "overwrote" if path.exists() else "created"
        actions[str(path)] = "created"

    # Append .gitignore patterns if not already there.
    gi = root / ".gitignore"
    existing = gi.read_text() if gi.exists() else ""
    needs_append = any(
        line.strip() and line not in existing
        for line in GITIGNORE_LINES
    )
    if needs_append:
        with gi.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n" + "\n".join(GITIGNORE_LINES))
        actions[str(gi)] = "appended" if existing else "created"

    return actions
