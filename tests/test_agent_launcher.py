"""Tests for agent_launcher.build_argv — the single source of truth for
launching every supported CLI agent.  Both Supervisor and WorkerPool
delegate here, so a regression in this builder breaks every spawn path.
"""
from __future__ import annotations

import pytest

from agentorchestr.agent_launcher import build_argv


@pytest.mark.parametrize("name,expected_substrings", [
    ("claude",     ["claude", "--dangerously-skip-permissions", "$(cat"]),
    ("openclaude", ["openclaude", "--dangerously-skip-permissions"]),
    ("amp",        ["amp", "--dangerously-skip-permissions"]),
    ("kiro",       ["kiro", "chat", "--trust-all-tools"]),
    ("opencode",   ["opencode", "run"]),
    ("aider",      ["aider", "--yes-always", "--message-file"]),
    ("codex",      ["codex", "exec", "--full-auto"]),
    ("gemini",     ["gemini", "--prompt"]),
    ("goose",      ["goose", "run", "--text"]),
    ("unknownx",   ["unknownx", "$(cat"]),  # default fallback
])
def test_build_argv_per_agent(name, expected_substrings):
    argv = build_argv({"name": name, "cmd": name}, "/tmp/p.txt")
    assert argv[:2] == ["bash", "-lc"]
    for sub in expected_substrings:
        assert sub in argv[2], f"{name} argv missing {sub!r}: {argv[2]!r}"


def test_build_argv_aider_uses_message_file_flag_not_cat():
    """aider has --message-file natively — we MUST prefer the flag form,
    not the bash-cat fallback, because $(cat prompt) breaks for huge prompts
    that exceed ARG_MAX (~128 KB on Linux)."""
    argv = build_argv({"name": "aider", "cmd": "aider"}, "/tmp/p.txt")
    assert "--message-file" in argv[2]
    assert "$(cat" not in argv[2]


def test_build_argv_kiro_via_kirocli_alias():
    """kiro detection uses cmd ∈ {kiro-cli, kirocli} as well as name=='kiro'."""
    argv = build_argv({"name": "something-else", "cmd": "kiro-cli"}, "/tmp/p.txt")
    assert "chat" in argv[2]
    assert "--trust-all-tools" in argv[2]


def test_build_argv_quotes_cmd_and_path():
    """cmd and prompt_path must be shell-quoted to handle spaces / metachars."""
    argv = build_argv({"name": "aider", "cmd": "/opt/my agent/aider"},
                      "/tmp/has space.txt")
    # shlex.quote uses single quotes around any string with special chars
    assert "'/opt/my agent/aider'" in argv[2]
    assert "'/tmp/has space.txt'" in argv[2]


def test_build_argv_rejects_non_dict():
    with pytest.raises(TypeError):
        build_argv("kiro", "/tmp/p.txt")  # type: ignore[arg-type]


def test_build_argv_rejects_missing_cmd():
    with pytest.raises(ValueError):
        build_argv({"name": ""}, "/tmp/p.txt")


def test_build_argv_role_param_is_accepted():
    """The role parameter exists so supervisor/worker paths can diverge
    later without re-introducing duplicate builders."""
    argv_w = build_argv({"name": "kiro", "cmd": "kiro"}, "/tmp/p.txt", role="worker")
    argv_s = build_argv({"name": "kiro", "cmd": "kiro"}, "/tmp/p.txt", role="supervisor")
    assert argv_w == argv_s  # currently identical; if/when they diverge, update test
