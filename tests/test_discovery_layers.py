"""Tests for discovery layer fallback ordering and TTL cache.

Covers:
  1. zeroconf missing → mDNS layer skipped, no exception.
  2. Filesystem layer wins when mDNS returns nothing.
  3. Stale FS card (dead pid) is removed during _discover_filesystem.
  4. tmux layer fires only when shutil.which('tmux') is truthy.
  5. TTL cache: second call within TTL returns cached result.
  6. force_refresh bypasses the cache.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import agentorchestr.discovery as disc


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level discovery cache before each test."""
    disc._discovery_cache = None
    yield
    disc._discovery_cache = None


# ── 1. zeroconf missing → mDNS skipped ────────────────────────────────

@pytest.mark.asyncio
async def test_no_zeroconf_skips_mdns(monkeypatch):
    monkeypatch.setattr(disc, "_can_skip_mdns", lambda: True)
    mdns_called = []
    async def _fake_mdns(**kw):
        mdns_called.append(1)
        return []
    monkeypatch.setattr(disc, "_discover_mdns", _fake_mdns)
    monkeypatch.setattr(disc, "_discover_filesystem", lambda: [])
    monkeypatch.setattr(disc, "_discover_tmux", lambda: [])
    await disc.discover(include_tmux=False)
    assert mdns_called == [], "mDNS must not be called when _can_skip_mdns() is True"


# ── 2. filesystem layer wins when mDNS empty ──────────────────────────

@pytest.mark.asyncio
async def test_filesystem_wins_when_mdns_empty(monkeypatch):
    monkeypatch.setattr(disc, "_can_skip_mdns", lambda: False)
    async def _empty_mdns(**kw): return []
    monkeypatch.setattr(disc, "_discover_mdns", _empty_mdns)
    fs_agent = disc.DiscoveredAgent(id="host:1234", name="kiro",
                                    transport="shim_socket", pid=1234)
    monkeypatch.setattr(disc, "_discover_filesystem", lambda: [fs_agent])
    monkeypatch.setattr(disc, "_discover_tmux", lambda: [])
    result = await disc.discover(include_tmux=False)
    assert len(result) == 1 and result[0].id == "host:1234"


# ── 3. stale FS card with dead pid is removed ─────────────────────────

def test_stale_fs_card_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(disc, "SHIM_SOCK_DIR", tmp_path)
    card = tmp_path / "99999.json"
    card.write_text(json.dumps({"pid": 99999, "name": "kiro",
                                "host": "localhost", "socket": "/tmp/x.sock"}))
    # pid 99999 almost certainly doesn't exist
    monkeypatch.setattr(disc, "_pid_alive", lambda pid: False)
    result = disc._discover_filesystem()
    assert result == []
    assert not card.exists(), "stale card must be deleted"


# ── 4. tmux layer only fires when tmux is on PATH ─────────────────────

@pytest.mark.asyncio
async def test_tmux_layer_skipped_when_no_tmux(monkeypatch):
    monkeypatch.setattr(disc, "_can_skip_mdns", lambda: True)
    async def _empty_mdns(**kw): return []
    monkeypatch.setattr(disc, "_discover_mdns", _empty_mdns)
    monkeypatch.setattr(disc, "_discover_filesystem", lambda: [])
    tmux_called = []
    def _fake_tmux():
        tmux_called.append(1)
        return []
    monkeypatch.setattr(disc, "_discover_tmux", _fake_tmux)
    await disc.discover(include_tmux=False)
    assert tmux_called == []


# ── 5. TTL cache: second call within TTL returns cached result ─────────

@pytest.mark.asyncio
async def test_ttl_cache_returns_cached_result(monkeypatch):
    monkeypatch.setattr(disc, "_can_skip_mdns", lambda: True)
    async def _empty_mdns(**kw): return []
    monkeypatch.setattr(disc, "_discover_mdns", _empty_mdns)
    call_count = [0]
    def _counting_fs():
        call_count[0] += 1
        return []
    monkeypatch.setattr(disc, "_discover_filesystem", _counting_fs)
    monkeypatch.setattr(disc, "_discover_tmux", lambda: [])

    await disc.discover(cache_ttl_s=10.0)
    await disc.discover(cache_ttl_s=10.0)
    assert call_count[0] == 1, "filesystem should only be scanned once within TTL"


# ── 6. force_refresh bypasses cache ───────────────────────────────────

@pytest.mark.asyncio
async def test_force_refresh_bypasses_cache(monkeypatch):
    monkeypatch.setattr(disc, "_can_skip_mdns", lambda: True)
    async def _empty_mdns(**kw): return []
    monkeypatch.setattr(disc, "_discover_mdns", _empty_mdns)
    call_count = [0]
    def _counting_fs():
        call_count[0] += 1
        return []
    monkeypatch.setattr(disc, "_discover_filesystem", _counting_fs)
    monkeypatch.setattr(disc, "_discover_tmux", lambda: [])

    await disc.discover(cache_ttl_s=10.0)
    await disc.discover(cache_ttl_s=10.0, force_refresh=True)
    assert call_count[0] == 2
