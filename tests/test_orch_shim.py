"""End-to-end test for orch_shim.

We use `cat` as a stand-in agent — it's universally available, takes
stdin and echoes to stdout, and exits cleanly on EOF/SIGTERM.  This
proves the shim's PTY + JSON-RPC + card-drop machinery without
requiring an actual coding agent.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket as socket_mod
import sys
import time
from pathlib import Path

import pytest

import orch_shim
from discovery import ShimClient


pytestmark = pytest.mark.skipif(
    shutil.which("cat") is None or sys.platform == "win32",
    reason="needs a Unix-y system with /bin/cat",
)


@pytest.fixture
def shim_dir(tmp_path, monkeypatch):
    d = tmp_path / "agentorchestr-agents"
    d.mkdir()
    monkeypatch.setattr(orch_shim, "SOCK_DIR", d)
    return d


async def _wait_for(predicate, *, timeout: float = 3.0, interval: float = 0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_shim_lifecycle_with_cat(shim_dir, monkeypatch):
    """Wrap `cat`, send 'hello\\n', read it back, then kill cleanly."""
    # Run the shim in a background task — we still use the same loop.
    # Disable mDNS to keep the test hermetic.
    monkeypatch.setattr(orch_shim, "_try_advertise_mdns", lambda card: None)

    # Replace the stdin/stdout bridges with no-ops so the test process
    # doesn't hang reading our own stdin.
    async def _noop(_agent):
        await asyncio.sleep(0)
    monkeypatch.setattr(orch_shim, "_bridge_stdin", _noop)
    monkeypatch.setattr(orch_shim, "_bridge_stdout", _noop)

    shim_task = asyncio.create_task(orch_shim._run(["cat"]))
    try:
        # Wait for the card + sock to materialise.
        ok = await _wait_for(lambda: any(p.suffix == ".json" for p in shim_dir.iterdir()))
        assert ok, f"shim never wrote a card to {shim_dir}"

        card_path = next(p for p in shim_dir.iterdir() if p.suffix == ".json")
        sock_path = next(p for p in shim_dir.iterdir() if p.suffix == ".sock")
        card = json.loads(card_path.read_text())
        assert card["name"] == "cat"
        assert card["pid"] > 0
        assert card["socket"] == str(sock_path)

        # Connect and ping.
        client = ShimClient(str(sock_path))
        info = await client.info()
        assert info["result"]["name"] == "cat"
        assert info["result"]["alive"] is True

        # Send a line and read the echo back.
        send_resp = await client.send("hello-shim\n")
        assert send_resp["result"]["sent"] > 0

        # The PTY echoes our writes (canonical line discipline) and then
        # cat copies them to stdout, so we expect "hello-shim" to appear
        # at least once. Allow a generous window — PTY scheduling is bursty.
        async def _saw_echo() -> bool:
            r = await client.read(max_bytes=8192)
            output = r["result"].get("output", "")
            return "hello-shim" in output

        saw = await _wait_for(_saw_echo, timeout=4.0, interval=0.1)
        if not saw:
            # Surface the buffer contents to make debug tractable.
            r = await client.read(max_bytes=8192)
            pytest.fail(
                f"never saw echo of 'hello-shim' in PTY buffer; "
                f"got: {r['result'].get('output', '')!r}"
            )

        # Kill cleanly.
        kill_resp = await client.kill()
        assert kill_resp["result"]["killed"] is True

        # The shim_task should now wind down on its own.
        await asyncio.wait_for(shim_task, timeout=3.0)

        # Card should have been cleaned up.
        assert not card_path.exists(), "shim must remove its card on shutdown"
        assert not sock_path.exists(), "shim must remove its socket on shutdown"
    finally:
        if not shim_task.done():
            shim_task.cancel()
            try:
                await shim_task
            except (asyncio.CancelledError, BaseException):
                pass
