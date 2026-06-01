"""
preferences.py — user preferences persistence (XDG-friendly)
==============================================================

Stored as TOML at $XDG_CONFIG_HOME/agentorchestr/preferences.toml.
Read at startup so the wizard can be skipped on subsequent runs.

Schema (v1):

    [version]
    schema = 1

    [setup]
    completed = true
    last_run_at = 1717000000.0
    mode = "manual" | "automatic"   # default mode the wizard saved

    [agents]
    lead = "kiro"                   # preferred lead agent name
    workers = ["kiro", "openclaude"] # worker pool (may repeat for parallel sessions)
    worker_count = 3                # how many panes to spawn

    [llm]
    primary = "anthropic"           # which provider to prefer

The file is intentionally tiny and human-editable.
"""
from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import Optional

from paths import global_config_dir


PREFS_FILENAME = "preferences.toml"
SCHEMA_VERSION = 1


def prefs_path() -> Path:
    return global_config_dir() / PREFS_FILENAME


def load() -> dict:
    """Return preferences dict, or {} if no file exists / unreadable."""
    p = prefs_path()
    if not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def save(prefs: dict) -> Path:
    """Write prefs as TOML.  Sets schema version + last_run_at."""
    out = dict(prefs)
    out.setdefault("version", {})["schema"] = SCHEMA_VERSION
    out.setdefault("setup", {})["last_run_at"] = time.time()
    out["setup"]["completed"] = True

    p = prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = _to_toml(out)
    p.write_text(text, encoding="utf-8")
    return p


def is_setup_complete() -> bool:
    return bool(load().get("setup", {}).get("completed"))


# ── tiny TOML emitter (no extra dep; tomllib is read-only in stdlib) ─

def _to_toml(d: dict) -> str:
    """Serialize a flat-ish dict to TOML.  Handles strings, ints, floats,
    bools, and lists of strings/numbers — which is all the preferences
    schema needs.
    """
    lines: list[str] = []
    # Top-level scalars first (currently none, but keeps the format clean).
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            for sk, sv in v.items():
                lines.append(f"{sk} = {_toml_value(sv)}")
        else:
            lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines).strip() + "\n"


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        # Escape backslashes and double quotes.
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if v is None:
        return '""'
    return _toml_value(str(v))
