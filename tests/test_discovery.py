"""Tests for the discovery layer.

The fs + tmux layers are pure-logic and easy to fake. The mDNS layer
needs a real zeroconf advertise-then-browse cycle, which is slow and
flaky in CI containers — we cover it lightly with a no-op test that
verifies graceful behaviour when zeroconf isn't usable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentorchestr import discovery


@pytest.fixture
def fake_sock_dir(tmp_path, monkeypatch):
    d = tmp_path / "agentorchestr-agents"
    d.mkdir()
    monkeypatch.setattr(discovery, "SHIM_SOCK_DIR", d)
    return d


def _write_card(dir_: Path, *, name: str, pid: int, sock: str = "/tmp/x.sock") -> Path:
    p = dir_ / f"{name}-{pid}.json"
    p.write_text(json.dumps({
        "shim_version": "1",
        "name": name,
        "pid": pid,
        "started_at": 1.0,
        "socket": sock,
        "host": "testhost",
        "argv": [name, "chat"],
    }))
    return p


def test_filesystem_discovery_picks_up_live_card(fake_sock_dir, monkeypatch):
    """A card whose pid is alive must be returned by the fs layer."""
    _write_card(fake_sock_dir, name="kiro", pid=os.getpid())  # we're alive
    out = discovery._discover_filesystem()
    assert len(out) == 1
    assert out[0].name == "kiro"
    assert out[0].transport == "shim_socket"
    assert out[0].pid == os.getpid()


def test_filesystem_discovery_prunes_stale_cards(fake_sock_dir):
    """A card whose pid is dead must be removed and excluded."""
    # PID 2^30 is virtually guaranteed not to exist on Linux.
    stale = _write_card(fake_sock_dir, name="claude", pid=2**30)
    out = discovery._discover_filesystem()
    assert out == []
    assert not stale.exists(), "stale card should have been deleted"


def test_filesystem_skips_invalid_json(fake_sock_dir):
    bad = fake_sock_dir / "broken.json"
    bad.write_text("not-json{")
    # Should not raise; just returns empty.
    assert discovery._discover_filesystem() == []


def test_match_agent_recognises_known_binaries():
    assert discovery._match_agent("kiro", "") == "kiro"
    assert discovery._match_agent("kiro-cli", "") == "kiro"
    assert discovery._match_agent("bash", "kiro chat") == "kiro"
    assert discovery._match_agent("vim", "editing file") is None


def test_match_agent_handles_empty():
    assert discovery._match_agent("", "") is None
    assert discovery._match_agent(None, None) is None


def test_pid_alive():
    assert discovery._pid_alive(os.getpid()) is True
    assert discovery._pid_alive(2**30) is False


def test_tmux_scan_returns_empty_when_no_tmux(monkeypatch):
    """When tmux isn't on PATH, layer 3 returns []."""
    monkeypatch.setattr(discovery.shutil, "which", lambda c: None)
    assert discovery._discover_tmux() == []


def test_tmux_scan_parses_pane_output(monkeypatch):
    """Simulate `tmux list-panes -a` and check that we extract agents."""
    fake_output = "\n".join([
        "main\t%0\t1234\tbash\tEditing config",
        "agents\t%1\t5678\tkiro\torch w01",
        "agents\t%2\t9012\tbash\tkiro chat in subshell",
        "agents\t%3\t3456\taider\tworking",
        "misc\t%4\t1\tvim\tnotes.md",
    ])
    monkeypatch.setattr(discovery.shutil, "which", lambda c: "/usr/bin/tmux")
    monkeypatch.setattr(
        discovery.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=fake_output, stderr=""),
    )
    out = discovery._discover_tmux()
    names = sorted(a.name for a in out)
    assert names == ["aider", "kiro", "kiro"]
    # All have transport='tmux'
    assert all(a.transport == "tmux" for a in out)
    # IDs are stable across runs given the same session/pane
    ids = sorted(a.id for a in out)
    assert ids[0].startswith("tmux:")


def test_tmux_scan_tolerates_missing_columns(monkeypatch):
    """If tmux returns a malformed row, we just skip it."""
    monkeypatch.setattr(discovery.shutil, "which", lambda c: "/usr/bin/tmux")
    monkeypatch.setattr(
        discovery.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="incomplete\trow", stderr=""),
    )
    assert discovery._discover_tmux() == []


@pytest.mark.asyncio
async def test_discover_merges_layers(fake_sock_dir, monkeypatch):
    """fs + tmux merge cleanly; ids stay unique."""
    _write_card(fake_sock_dir, name="kiro", pid=os.getpid())
    monkeypatch.setattr(discovery.shutil, "which", lambda c: "/usr/bin/tmux")
    monkeypatch.setattr(
        discovery.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="agents\t%1\t111\taider\tx", stderr=""
        ),
    )
    # Force mDNS to no-op so the test is deterministic.
    async def _no_mdns(timeout=1.0):
        return []
    monkeypatch.setattr(discovery, "_discover_mdns", _no_mdns)

    out = await discovery.discover(mdns_timeout=0.05)
    names = sorted(a.name for a in out)
    assert names == ["aider", "kiro"]
    assert {a.transport for a in out} == {"shim_socket", "tmux"}


@pytest.mark.asyncio
async def test_discover_no_zeroconf_does_not_raise(monkeypatch):
    """With zeroconf missing, _discover_mdns returns [] cleanly."""
    # Simulate ImportError by hiding the module.
    real_modules = sys.modules.copy()
    sys.modules.pop("zeroconf", None)
    sys.modules.pop("zeroconf.asyncio", None)
    monkeypatch.setitem(sys.modules, "zeroconf", None)
    try:
        out = await discovery._discover_mdns(timeout=0.1)
        assert out == []
    finally:
        sys.modules.clear()
        sys.modules.update(real_modules)
