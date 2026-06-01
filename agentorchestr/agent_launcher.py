"""
agent_launcher.py — single source of truth for "how to launch agent X interactively"
=====================================================================================

Both Supervisor (lead agent) and WorkerPool (worker agents) need to spin up
a CLI agent in a tmux pane with an initial prompt loaded from a file.  Every
agent has its own conventions for accepting that prompt:

    claude / openclaude / amp:  positional argument
    kiro-cli:                   `chat` subcommand + positional
    opencode:                   `run` + positional
    aider:                      `--message-file PATH`   (cleanest)
    codex:                      `exec --full-auto` + positional
    gemini-cli:                 `--prompt` + positional
    goose:                      `run --text` + positional

Wherever an agent supports a flag-based prompt-file form (--message-file,
--prompt-file), we prefer it: it avoids the bash-cat round-trip and is
immune to ARG_MAX limits on huge prompts.  Where we have to use a positional
argument, we use `bash -lc 'cmd "$(cat $prompt_path)"'` — `$(cat …)`
substitutes the file content as a literal string and is NOT vulnerable to
re-evaluation of `$(…)` or backticks inside that content (verified).

The `role` parameter lets the supervisor diverge from worker-launch behavior
in the future (e.g. allow stdin streaming for the lead) without re-introducing
the duplicated builder we used to have.
"""
from __future__ import annotations

import shlex


def build_argv(agent: dict, prompt_path: str, *, role: str = "worker") -> list[str]:
    """Return the argv list to launch `agent` interactively with the
    contents of `prompt_path` as its initial prompt.

    Parameters
    ----------
    agent : dict
        Agent record from agent_detector.detect() — must have at least
        ``name`` and ``cmd``.
    prompt_path : str
        Filesystem path to the prompt file.  Caller has already written
        the prompt; this function does not read or modify it.
    role : str
        Currently unused but kept so the supervisor's launch path can
        diverge later (interactive streaming, custom flags) without us
        forking the function back into two copies.
    """
    if not isinstance(agent, dict):
        raise TypeError(f"agent must be a dict, got {type(agent).__name__}")
    name = agent.get("name") or ""
    cmd = agent.get("cmd") or name
    if not cmd:
        raise ValueError(f"agent {agent!r} has no 'cmd' to launch")

    q = shlex.quote
    qp = q(prompt_path)
    qc = q(cmd)

    # Agents with a native --message-file / --prompt-file flag.  These are
    # preferred: no bash-cat round-trip, no ARG_MAX limit, the agent reads
    # the file directly.
    if name == "aider":
        return ["bash", "-lc", f"{qc} --yes-always --message-file {qp}"]

    # Positional-prompt agents.  We feed the prompt via $(cat …) inside
    # double quotes — the file content becomes a single literal argument
    # and is NOT re-evaluated for shell metacharacters.
    if name in ("claude", "openclaude", "amp"):
        return ["bash", "-lc", f"{qc} --dangerously-skip-permissions \"$(cat {qp})\""]
    if name == "kiro" or cmd in ("kiro-cli", "kirocli"):
        return ["bash", "-lc", f"{qc} chat --trust-all-tools \"$(cat {qp})\""]
    if name == "opencode":
        return ["bash", "-lc", f"{qc} run \"$(cat {qp})\""]
    if name == "codex":
        return ["bash", "-lc", f"{qc} exec --full-auto \"$(cat {qp})\""]
    if name == "gemini":
        return ["bash", "-lc", f"{qc} --prompt \"$(cat {qp})\""]
    if name == "goose":
        return ["bash", "-lc", f"{qc} run --text \"$(cat {qp})\""]

    # Default: positional argument.  Works for any agent whose CLI accepts
    # a single string as its first user message.
    return ["bash", "-lc", f"{qc} \"$(cat {qp})\""]
