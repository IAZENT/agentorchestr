"""Tests for AgentDetector — multi-candidate binary lookup, copilot gating."""
from __future__ import annotations

import subprocess

import pytest

import agent_detector
from agent_detector import AgentDetector


def test_first_on_path_returns_first_match(monkeypatch):
    """If only the second candidate is on PATH, that one is returned."""
    seen: list[str] = []

    def fake_which(cmd: str) -> str | None:
        seen.append(cmd)
        return "/fake/bin/" + cmd if cmd == "kiro-cli" else None

    monkeypatch.setattr(agent_detector.shutil, "which", fake_which)
    hit = agent_detector._first_on_path(["kiro", "kiro-cli", "kirocli"])
    assert hit == ("kiro-cli", "/fake/bin/kiro-cli")
    # Stops at first match
    assert seen == ["kiro", "kiro-cli"]


def test_kiro_detected_when_only_kiro_cli_present(monkeypatch):
    """Regression: user had kiro-cli but the registry only looked for 'kiro'."""
    def fake_which(cmd: str) -> str | None:
        return "/fake/bin/kiro-cli" if cmd == "kiro-cli" else None

    def fake_run(*args, **kwargs):
        class R: stdout = "kiro-cli 2.3.0\n"; stderr = ""; returncode = 0
        return R()

    monkeypatch.setattr(agent_detector.shutil, "which", fake_which)
    monkeypatch.setattr(agent_detector.subprocess, "run", fake_run)
    detected = AgentDetector().detect()
    names = [a["name"] for a in detected]
    assert "kiro" in names
    kiro = next(a for a in detected if a["name"] == "kiro")
    # Detector reports the *actual* binary it found
    assert kiro["cmd"] == "kiro-cli"
    assert kiro["path"] == "/fake/bin/kiro-cli"
    assert "2.3.0" in kiro["version"]


def test_copilot_requires_extension(monkeypatch):
    """gh on PATH is not enough — must have the copilot extension installed."""
    def fake_which(cmd: str) -> str | None:
        return "/fake/bin/" + cmd if cmd == "gh" else None

    def fake_run(*args, **kwargs):
        # First call: --version probe; second: extension list (no copilot)
        cmd = list(args[0])
        class R: returncode = 0
        if "extension" in cmd:
            R.stdout = "  fake-ext  some-other-ext\n"
            R.stderr = ""
        else:
            R.stdout = "gh version 2.50.0\n"
            R.stderr = ""
        return R()

    monkeypatch.setattr(agent_detector.shutil, "which", fake_which)
    monkeypatch.setattr(agent_detector.subprocess, "run", fake_run)
    detected = AgentDetector().detect()
    assert "copilot" not in [a["name"] for a in detected]


def test_no_agents_when_path_empty(monkeypatch):
    monkeypatch.setattr(agent_detector.shutil, "which", lambda cmd: None)
    assert AgentDetector().detect() == []
