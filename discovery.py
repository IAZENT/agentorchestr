"""
discovery.py — find agents the supervisor didn't spawn
========================================================

Use case: the user opens `orch-shim kiro chat` in terminal 1 and
`orch-shim claude code` in terminal 2, then later runs ORCH from
terminal 3. ORCH needs to discover those existing workers without
the user explicitly listing them.

Three layers, tried in order, results merged:

  1. mDNS (zeroconf)  — preferred when the shim is running and the
     `zeroconf` extra is installed. Service type `_orch-agent._tcp.local.`
     Catches shims on this host AND across the local network.
  2. Filesystem       — every shim drops $XDG_RUNTIME_DIR/orch-agents/<pid>.json.
     Guaranteed to work on a single host even when zeroconf is unavailable.
  3. tmux scan        — `tmux list-panes -a` filtered by known agent
     binaries. Catches agents NOT wrapped by orch-shim (best-effort, you
     can only send keys + capture-pane, no structured RPC).

Everything is cheap: full discovery completes in <100 ms typical.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Match orch_shim's defaults.
SHIM_SOCK_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/run-{os.getuid()}") / "orch-agents"
MDNS_SERVICE_TYPE = "_orch-agent._tcp.local."

# Agent binaries we recognise when falling back to bare-tmux scanning.
# Keep aligned with agent_detector.AGENT_REGISTRY's primary command names.
KNOWN_AGENT_BINARIES = {
    "claude", "claude-code",
    "openclaude",
    "kiro", "kiro-cli", "kirocli",
    "opencode",
    "aider",
    "codex",
    "gemini", "gemini-cli",
    "goose",
    "amp",
}


@dataclass
class DiscoveredAgent:
    """One agent we can reach. `transport` distinguishes shim vs tmux-only."""
    id: str                       # stable ID (host:pid for shims, session:pane for tmux)
    name: str                     # binary basename, e.g. 'kiro'
    transport: str                # 'shim_socket' | 'tmux'
    pid: int = 0
    started_at: float = 0.0
    socket: Optional[str] = None  # Unix socket path (shim only)
    tmux_session: Optional[str] = None
    tmux_pane: Optional[str] = None
    host: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Layer 1: mDNS ──────────────────────────────────────────────────────

async def _discover_mdns(timeout: float = 1.0) -> list[DiscoveredAgent]:
    """Browse `_orch-agent._tcp.local.` and return discovered shims."""
    try:
        from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf  # type: ignore
        from zeroconf import ServiceStateChange  # type: ignore
    except ImportError:
        return []

    found: dict[str, DiscoveredAgent] = {}
    seen_event = asyncio.Event()

    aiozc = AsyncZeroconf()

    def _handler(zc, service_type, name, state_change):
        if state_change != ServiceStateChange.Added:
            return
        # Schedule the actual fetch in the running loop.
        asyncio.create_task(_record(name))

    async def _record(instance: str) -> None:
        info = await aiozc.async_get_service_info(MDNS_SERVICE_TYPE, instance, timeout=int(timeout * 1000))
        if info is None:
            return
        props = {
            (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
            for k, v in (info.properties or {}).items()
        }
        try:
            pid = int(props.get("pid", "0"))
        except ValueError:
            pid = 0
        agent_id = f"{info.server.rstrip('.')}:{pid}"
        found[agent_id] = DiscoveredAgent(
            id=agent_id,
            name=props.get("name", instance.split(".")[0]),
            transport="shim_socket",
            pid=pid,
            socket=props.get("socket"),
            host=info.server.rstrip("."),
            extras={"argv": props.get("argv", "")},
        )
        seen_event.set()

    browser = AsyncServiceBrowser(aiozc.zeroconf, [MDNS_SERVICE_TYPE], handlers=[_handler])
    try:
        try:
            await asyncio.wait_for(seen_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        # Brief settle so late responders also land.
        await asyncio.sleep(0.05)
    finally:
        await browser.async_cancel()
        await aiozc.async_close()
    return list(found.values())


# ── Layer 2: filesystem ────────────────────────────────────────────────

def _discover_filesystem() -> list[DiscoveredAgent]:
    """Read every *.json card the shim has dropped in SHIM_SOCK_DIR."""
    if not SHIM_SOCK_DIR.exists():
        return []
    out: list[DiscoveredAgent] = []
    for card in SHIM_SOCK_DIR.glob("*.json"):
        try:
            data = json.loads(card.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pid = int(data.get("pid", 0))
        if pid <= 0 or not _pid_alive(pid):
            # Stale card from a previous shim that died ungracefully.
            try:
                card.unlink()
            except OSError:
                pass
            continue
        host = data.get("host", socket.gethostname())
        out.append(
            DiscoveredAgent(
                id=f"{host}:{pid}",
                name=data.get("name", card.stem),
                transport="shim_socket",
                pid=pid,
                started_at=float(data.get("started_at", 0.0)),
                socket=data.get("socket"),
                host=host,
                extras={"argv": data.get("argv", [])},
            )
        )
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, but we can't signal it — treat as alive
    return True


# ── Layer 3: tmux scan ─────────────────────────────────────────────────

def _discover_tmux() -> list[DiscoveredAgent]:
    """Find bare CLI agents running in tmux panes.

    Returns one DiscoveredAgent per matching pane. Note: with no shim
    we have only `tmux send-keys` + `capture-pane`, so the supervisor
    can drive these but lacks structured request/response.
    """
    if shutil.which("tmux") is None:
        return []
    try:
        # current_command is the immediate process; if a wrapper like
        # `bash -lc "kiro chat"` is running, current_command is 'bash' —
        # so we also scan the full pane title. This intentionally errs
        # on the side of recall over precision; ORCH validates separately.
        r = subprocess.run(
            ["tmux", "list-panes", "-a", "-F",
             "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}\t#{pane_title}"],
            capture_output=True, text=True, timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if r.returncode != 0:
        return []
    out: list[DiscoveredAgent] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        session, pane_id, pid_s, cmd, title = parts
        agent_name = _match_agent(cmd, title)
        if not agent_name:
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            pid = 0
        out.append(
            DiscoveredAgent(
                id=f"tmux:{session}:{pane_id}",
                name=agent_name,
                transport="tmux",
                pid=pid,
                tmux_session=session,
                tmux_pane=pane_id,
                host=socket.gethostname(),
                extras={"current_command": cmd, "title": title},
            )
        )
    return out


def _match_agent(cmd: str, title: str) -> Optional[str]:
    cmd = (cmd or "").lower()
    title = (title or "").lower()
    for binary in KNOWN_AGENT_BINARIES:
        # Direct match on current_command (most reliable)
        if cmd == binary or cmd == binary.split("-")[0]:
            return binary.split("-")[0]  # canonical name
        # Title match (catches `bash -lc "kiro chat"`)
        if binary in title:
            return binary.split("-")[0]
    return None


# ── public API ─────────────────────────────────────────────────────────

async def discover(*, mdns_timeout: float = 1.0,
                   include_tmux: bool = True) -> list[DiscoveredAgent]:
    """Run all enabled discovery layers and merge by id (mDNS wins)."""
    fs_agents = _discover_filesystem()
    tmux_agents = _discover_tmux() if include_tmux else []
    mdns_agents = await _discover_mdns(timeout=mdns_timeout)

    by_id: dict[str, DiscoveredAgent] = {}
    # Lowest precedence first so mDNS overwrites shim-fs which overwrites tmux.
    for src in (tmux_agents, fs_agents, mdns_agents):
        for a in src:
            by_id[a.id] = a
    return list(by_id.values())


# ── lightweight client to talk to a discovered shim ────────────────────

class ShimClient:
    """Minimal JSON-RPC over Unix socket; matches orch_shim's protocol."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._counter = 0

    async def _rpc(self, method: str, **params) -> dict:
        self._counter += 1
        rid = self._counter
        try:
            r, w = await asyncio.open_unix_connection(self.socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            return {"error": str(e)}
        try:
            req = json.dumps({"id": rid, "method": method, "params": params}).encode()
            w.write(req + b"\n")
            await w.drain()
            line = await asyncio.wait_for(r.readline(), timeout=5.0)
            try:
                return json.loads(line.decode())
            except json.JSONDecodeError:
                return {"error": "bad response", "raw": line.decode(errors="replace")}
        finally:
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

    async def info(self) -> dict:
        return await self._rpc("info")

    async def send(self, text: str) -> dict:
        return await self._rpc("send", text=text)

    async def read(self, max_bytes: int = 4096) -> dict:
        return await self._rpc("read", max_bytes=max_bytes)

    async def kill(self) -> dict:
        return await self._rpc("kill")
