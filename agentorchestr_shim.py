#!/usr/bin/env python3
"""
agentorchestr_shim.py — wrap a CLI agent so agentorchestr supervisors can find and drive it
==========================================================================

Usage:
  agentorchestr-shim <agent-binary> [args...]

What it does:
  1. Spawns the underlying agent as a subprocess with a PTY so the
     interactive REPL behaves correctly.
  2. Listens on a Unix-domain socket at $XDG_RUNTIME_DIR/agentorchestr-agents/<pid>.sock
     and serves a tiny JSON-RPC tool surface:

        {"id": 1, "method": "info"}
            -> {"id": 1, "result": {"name", "pid", "started_at", "tty"}}

        {"id": 2, "method": "send",  "params": {"text": "hello\\n"}}
            -> {"id": 2, "result": {"sent": <bytes>}}

        {"id": 3, "method": "read",  "params": {"max_bytes": 4096}}
            -> {"id": 3, "result": {"output": "...buffered stdout..."}}

        {"id": 4, "method": "kill"}
            -> {"id": 4, "result": {"killed": true}}

  3. Drops an Agent Card at $XDG_RUNTIME_DIR/agentorchestr-agents/<pid>.json so any
     agentorchestr session on this machine can discover the agent without mDNS.

  4. Optionally advertises via mDNS service type `_agentorchestr-agent._tcp.local.`
     (requires the `zeroconf` extra; safe to skip if unavailable).

Why this exists:
  Most CLI coding agents (claude, kiro, opencode, aider, codex…) only
  speak stdin/stdout. The shim turns them into addressable workers a
  supervisor can dispatch tasks to from anywhere on the host — including
  *terminals the supervisor didn't open*. That's the lift you can't get
  from tmux alone.

Single-file by design: 0 deps beyond the Python stdlib + an optional
zeroconf import. Run by users like `agentorchestr-shim kiro chat` and forget it.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import pty
import shlex
import shutil
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Optional


SOCK_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/run-{os.getuid()}") / "agentorchestr-agents"
MDNS_SERVICE_TYPE = "_agentorchestr-agent._tcp.local."
SHIM_VERSION = "1"


# ── child process management ───────────────────────────────────────────

class ShimAgent:
    """Owns the underlying agent subprocess + a stdout buffer."""

    def __init__(self, name: str, argv: list[str]):
        self.name = name
        self.argv = argv
        self.pid: int = -1
        self.master_fd: int = -1
        self.started_at: float = 0.0
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._reader_task: asyncio.Task | None = None
        self._exit_code: int | None = None

    async def start(self) -> None:
        """Fork the agent under a PTY and start a background reader."""
        pid, master_fd = pty.fork()
        if pid == 0:
            # Child: replace ourselves with the agent.
            try:
                os.execvp(self.argv[0], self.argv)
            except OSError as e:
                # If exec failed, exit the child loudly; parent sees EOF.
                os.write(2, f"agentorchestr-shim exec failed: {e}\n".encode())
                os._exit(127)

        self.pid = pid
        self.master_fd = master_fd
        self.started_at = time.time()
        # Make the master_fd non-blocking so reads never stall the loop.
        os.set_blocking(master_fd, False)
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Drain the PTY into self._buffer until the child exits.

        We register the PTY master fd with the asyncio event loop's
        reader callbacks so we get woken up exactly when bytes arrive
        — no polling, no missed wakeups, no false EOF.
        """
        loop = asyncio.get_event_loop()
        # Cap the buffer so a chatty agent can't OOM the shim.
        MAX_BUF = 1 << 20  # 1 MiB

        eof_event = asyncio.Event()

        def _on_readable() -> None:
            try:
                data = os.read(self.master_fd, 4096)
            except BlockingIOError:
                return
            except OSError as e:
                if e.errno == errno.EIO:
                    # PTY hangup — child exited.
                    eof_event.set()
                    return
                eof_event.set()
                return
            if not data:
                eof_event.set()
                return
            # Append; drop oldest half if we're over the cap.
            self._buffer.extend(data)
            if len(self._buffer) > MAX_BUF:
                del self._buffer[:len(self._buffer) - MAX_BUF // 2]

        try:
            loop.add_reader(self.master_fd, _on_readable)
        except (NotImplementedError, OSError):
            # Some platforms (Windows) lack add_reader; fall back to polling.
            await self._poll_loop()
            return

        try:
            await eof_event.wait()
        finally:
            try:
                loop.remove_reader(self.master_fd)
            except Exception:
                pass
            self._exit_code = self._reap()

    async def _poll_loop(self) -> None:
        """Fallback for platforms without loop.add_reader (Windows)."""
        MAX_BUF = 1 << 20
        while True:
            try:
                data = os.read(self.master_fd, 4096)
            except BlockingIOError:
                await asyncio.sleep(0.05)
                continue
            except OSError as e:
                if e.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            self._buffer.extend(data)
            if len(self._buffer) > MAX_BUF:
                del self._buffer[:len(self._buffer) - MAX_BUF // 2]
        self._exit_code = self._reap()

    def _read_chunk(self) -> bytes:  # kept for back-compat with tests
        try:
            return os.read(self.master_fd, 4096)
        except BlockingIOError:
            return b""
        except OSError as e:
            if e.errno == errno.EIO:
                return b""
            raise

    def _reap(self) -> int | None:
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid == 0:
                return None
            if os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            if os.WIFSIGNALED(status):
                return -os.WTERMSIG(status)
        except ChildProcessError:
            return None
        return None

    @property
    def alive(self) -> bool:
        return self._exit_code is None

    async def send(self, text: str) -> int:
        if not self.alive:
            return 0
        data = text.encode()
        return os.write(self.master_fd, data)

    async def read(self, max_bytes: int = 4096) -> str:
        async with self._lock:
            chunk = bytes(self._buffer[-max_bytes:])
        # Best-effort utf-8; surrogateescape keeps binary noise reversible.
        return chunk.decode("utf-8", errors="replace")

    async def kill(self) -> None:
        if not self.alive:
            return
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Give it a moment, then SIGKILL.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if not self.alive:
                return
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ── JSON-RPC server over a Unix socket ─────────────────────────────────

class _JsonRpcServer:
    def __init__(self, agent: ShimAgent, sock_path: Path):
        self.agent = agent
        self.sock_path = sock_path
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            self.sock_path.unlink()
        self._server = await asyncio.start_unix_server(self._handle, str(self.sock_path))
        # Tighten perms — only this user.
        os.chmod(self.sock_path, 0o600)

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self.sock_path.exists():
            try:
                self.sock_path.unlink()
            except OSError:
                pass

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    req = json.loads(line.decode())
                except json.JSONDecodeError:
                    writer.write(b'{"error":"bad json"}\n'); await writer.drain(); continue
                resp = await self._dispatch(req)
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, req: dict) -> dict:
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        try:
            if method == "info":
                result = {
                    "shim_version": SHIM_VERSION,
                    "name": self.agent.name,
                    "pid": self.agent.pid,
                    "started_at": self.agent.started_at,
                    "alive": self.agent.alive,
                    "argv": self.agent.argv,
                }
            elif method == "send":
                sent = await self.agent.send(str(params.get("text", "")))
                result = {"sent": sent}
            elif method == "read":
                output = await self.agent.read(int(params.get("max_bytes", 4096)))
                result = {"output": output}
            elif method == "kill":
                await self.agent.kill()
                result = {"killed": True}
            else:
                return {"id": rid, "error": f"unknown method {method!r}"}
            return {"id": rid, "result": result}
        except Exception as e:
            return {"id": rid, "error": f"{type(e).__name__}: {e}"}


# ── discovery side-channels ────────────────────────────────────────────

def _agent_card(agent: ShimAgent, sock_path: Path, port: int = 0) -> dict:
    return {
        "shim_version": SHIM_VERSION,
        "name": agent.name,
        "pid": agent.pid,
        "started_at": agent.started_at,
        "socket": str(sock_path),
        "host": socket.gethostname(),
        "port": port,  # 0 means "Unix socket only, no TCP"
        "argv": agent.argv,
    }


def _write_card(card: dict, card_path: Path) -> None:
    card_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = card_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(card, indent=2))
    tmp.replace(card_path)
    os.chmod(card_path, 0o600)


def _try_advertise_mdns(card: dict) -> Optional[object]:
    """Best-effort mDNS advertisement; returns the ServiceInfo handle or None."""
    try:
        from zeroconf import ServiceInfo, Zeroconf, IPVersion  # type: ignore
    except ImportError:
        return None
    try:
        zc = Zeroconf(ip_version=IPVersion.V4Only)
        # mDNS service names must be <=63 chars; encode pid for uniqueness.
        instance = f"{card['name']}-{card['pid']}.{MDNS_SERVICE_TYPE}"
        info = ServiceInfo(
            MDNS_SERVICE_TYPE,
            instance,
            addresses=[socket.inet_aton("127.0.0.1")],
            port=card.get("port") or 0,
            properties={
                "version": SHIM_VERSION,
                "name": card["name"],
                "pid": str(card["pid"]),
                "socket": card["socket"],
                # Truncate argv into TXT-record-friendly form.
                "argv": " ".join(shlex.quote(a) for a in card["argv"])[:200],
            },
            server=f"{socket.gethostname()}.local.",
        )
        zc.register_service(info)
        return (zc, info)
    except Exception:
        return None


def _stop_mdns(handle) -> None:
    if handle is None:
        return
    try:
        zc, info = handle
        zc.unregister_service(info)
        zc.close()
    except Exception:
        pass


# ── main ──────────────────────────────────────────────────────────────

async def _run(argv: list[str]) -> int:
    if not argv:
        print("agentorchestr-shim: missing agent command", file=sys.stderr)
        return 2

    # Resolve the binary so the card carries an absolute path.
    binary = shutil.which(argv[0]) or argv[0]
    full_argv = [binary, *argv[1:]]
    name = os.path.basename(argv[0])

    agent = ShimAgent(name, full_argv)
    await agent.start()

    sock_path = SOCK_DIR / f"{name}-{agent.pid}.sock"
    card_path = SOCK_DIR / f"{name}-{agent.pid}.json"
    server = _JsonRpcServer(agent, sock_path)
    await server.start()

    card = _agent_card(agent, sock_path)
    _write_card(card, card_path)
    mdns_handle = _try_advertise_mdns(card)

    print(f"agentorchestr-shim: {name} pid={agent.pid} sock={sock_path}", file=sys.stderr)
    if mdns_handle is None:
        print("agentorchestr-shim: mDNS not available, using filesystem-only discovery",
              file=sys.stderr)

    serve_task = asyncio.create_task(server.serve_forever())

    # Bridge the parent's stdin/stdout to the PTY so the user can also
    # interact with the agent directly (just like running it bare).
    bridge_in = asyncio.create_task(_bridge_stdin(agent))
    bridge_out = asyncio.create_task(_bridge_stdout(agent))

    try:
        # Exit when the agent dies.
        while agent.alive:
            await asyncio.sleep(0.25)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await agent.kill()
    finally:
        for t in (bridge_in, bridge_out, serve_task):
            t.cancel()
        await server.stop()
        _stop_mdns(mdns_handle)
        try:
            card_path.unlink()
        except OSError:
            pass

    return agent._exit_code or 0


async def _bridge_stdin(agent: ShimAgent) -> None:
    """Forward parent's stdin to the agent so the user keeps a terminal."""
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            if not data:
                break
            try:
                os.write(agent.master_fd, data)
            except OSError:
                break
    except asyncio.CancelledError:
        return


async def _bridge_stdout(agent: ShimAgent) -> None:
    """Mirror the agent's PTY output to the parent's stdout."""
    last_len = 0
    try:
        while agent.alive:
            await asyncio.sleep(0.05)
            async with agent._lock:
                chunk = bytes(agent._buffer[last_len:])
                last_len = len(agent._buffer)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
    except asyncio.CancelledError:
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wrap a CLI agent so agentorchestr can discover + drive it.",
        usage="agentorchestr-shim AGENT [ARGS...]",
    )
    # Use parse_known_args so anything after the agent passes through unchanged.
    parser.add_argument("agent", help="Agent binary, e.g. kiro / claude / opencode")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded to the agent")
    ns = parser.parse_args()
    full_argv = [ns.agent, *ns.args]
    try:
        return asyncio.run(_run(full_argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
