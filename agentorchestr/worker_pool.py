"""
worker_pool.py — supervisor-worker tmux pane manager
======================================================

Single source of truth for "how do I spawn / read / send-to / kill a worker".
Every worker is one tmux pane in the orchestration's tmux session, running
its CLI agent **interactively** in an isolated git worktree (or scratch
dir if the project isn't a git repo).

Why interactive (not headless):
  * Headless flags differ across agents (claude --print, codex exec,
    aider --message, kiro-cli chat --no-interactive...) and many of them
    block on auth / tool-permission prompts when run non-interactively.
  * The supervisor agent reads the pane via `tmux capture-pane`, so the
    worker's full reasoning is visible. No fragile JSON file protocol.
  * The user can attach with `tmux attach` and watch / steer at any time.

Lifecycle of a worker:
  1. supervisor calls spawn_worker(task, perspective, file_scope, agent)
  2. WorkerPool creates a git worktree on a fresh branch
  3. WorkerPool creates a new tmux pane in the agentorchestr session
  4. WorkerPool writes the system-prompt + task as one text blob to a
     .agentorchestr/worker_<id>_prompt.txt file inside the worktree
  5. WorkerPool launches the agent CLI with that prompt as positional
     input where the agent supports it (claude/openclaude/kiro-cli all do)
  6. supervisor reads the pane with read_output(), monitors for the
     WORKER_DONE / WORKER_BLOCKED / WORKER_FAILED sentinel
  7. supervisor calls send_message() to add follow-up instructions, or
     kill_worker() to stop, or wait_until_done() to block
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentorchestr import perspectives
from agentorchestr.state_store import StateStore


# Sentinels the worker is asked to print on its last line.
#
# We anchor BOTH ends of the line and require the sentinel to start the
# line so a worker that quotes the sentinel inside an explanation
# (e.g. `I'll write "WORKER_DONE: foo" when finished`) does NOT trip
# completion.  re.MULTILINE makes ^/$ match line boundaries inside a
# multi-line buffer.
_DONE_RE     = re.compile(r"^WORKER_DONE:\s*(.*?)\s*$",     re.M)
_BLOCKED_RE  = re.compile(r"^WORKER_BLOCKED:\s*(.*?)\s*$",  re.M)
_FAILED_RE   = re.compile(r"^WORKER_FAILED:\s*(.*?)\s*$",   re.M)

# How many tmux scrollback lines to inspect when polling for completion.
# 200 is too tight: a chatty worker can push the sentinel out of view
# between polls.  2000 lines @ ~80 cols is ~160 KB to scan, still cheap.
_CAPTURE_TAIL_LINES = 2000

# When the worker writes a structured completion file (.agentorchestr/done.flag
# inside its worktree) we trust it over the pane regex.  The flag file is
# atomic: workers should write `state\nsummary` to a temp file and rename.
_DONE_FLAG_NAME = "done.flag"

# Cap the number of panes per tmux window so the lead pane (always at the
# top) stays readable.  Past this many workers, we open a second, third,
# … 'agentorchestr-N' window.  5 = 1 lead + 4 workers, matching the
# supervisor prompt's "cap 4" parallel-implementer guidance.
MAX_PANES_PER_WINDOW = 5


@dataclass
class Worker:
    id: str
    session_id: str
    perspective: str
    agent: dict           # full agent dict (name, cmd, path, type, ...)
    worktree: str
    branch: str
    pane: object | None   # libtmux Pane (None until tmux pane created)
    pane_id: str | None
    task: str
    file_scope: list[str] = field(default_factory=list)
    state: str = "starting"     # starting|running|done|blocked|failed|killed
    summary: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class WorkerPool:
    """Manages the tmux session that holds the supervisor + workers."""

    def __init__(
        self,
        project_root: str,
        agentorchestr_session_id: str,
        store: StateStore,
        *,
        tmux_session_name: Optional[str] = None,
        memory: object | None = None,
        skills: object | None = None,
    ):
        self.project_root = Path(project_root)
        self.agentorchestr_session_id = agentorchestr_session_id
        self.store = store
        self.tmux_session_name = tmux_session_name or f"agentorchestr-{agentorchestr_session_id}"
        self._tmux_session = None
        self._workers: dict[str, Worker] = {}
        self._counter = 0
        self._tmp_root = Path(f"/tmp/agentorchestr-{agentorchestr_session_id}")
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        # Optional memory federation.  When provided, spawn_worker enriches
        # extra_context with the most relevant retrieved memories so they
        # end up inside the cacheable system-prompt prefix.
        self.memory = memory
        # Optional skills registry.  Skills whose triggers match the
        # spawned worker's task get their body injected into extra_context.
        self.skills = skills

    # ── tmux session lifecycle ────────────────────────────────────────

    async def start_session(self, lead_pane_cmd: list[str]) -> object:
        """Create the tmux session and launch the lead agent in the top pane.

        Layout: ONE window with lead at the top. Worker panes split
        horizontally below it as they spawn, so the lead can monitor
        all workers without switching windows.

        `lead_pane_cmd` is the argv used to start the lead (e.g.
        ['kiro-cli', 'chat', '--trust-all-tools', '--agent', '...']).
        """
        import libtmux
        from libtmux._internal.query_list import ObjectDoesNotExist

        server = libtmux.Server()
        # If a stale session exists from a previous crash, kill it.
        try:
            old = server.sessions.get(session_name=self.tmux_session_name)
            if old:
                old.kill()
        except ObjectDoesNotExist:
            pass
        except Exception:
            pass

        self._tmux_session = server.new_session(
            session_name=self.tmux_session_name,
            window_name="agentorchestr",
            start_directory=str(self.project_root),
        )

        lead_pane = self._tmux_session.active_window.active_pane
        # Quote properly so multi-word args / prompts survive
        lead_pane.send_keys(" ".join(shlex.quote(p) for p in lead_pane_cmd), enter=True)
        self._lead_pane_id = getattr(lead_pane, "pane_id", None)

        return self._tmux_session

    @property
    def attach_command(self) -> str:
        return f"tmux attach -t {self.tmux_session_name}"

    # ── worker spawn / monitor / control ──────────────────────────────

    def _pick_or_create_window(self):
        """Return the tmux window the next worker pane should land in.

        Strategy:
          * Walk through 'agentorchestr', 'agentorchestr-2', 'agentorchestr-3', ...
            and return the first one whose pane count is below
            MAX_PANES_PER_WINDOW.
          * If every existing window is full, create the next overflow
            window and return it.  The new window's initial pane will
            host the next worker (no extra split needed).
        """
        if self._tmux_session is None:
            raise RuntimeError("worker pool not started")
        windows = list(self._tmux_session.windows or [])
        # Look for the lead window first, then numbered overflows.
        candidates: list[tuple[int, object]] = []
        for w in windows:
            name = w.window_name or ""
            if name == "agentorchestr":
                candidates.append((1, w))
            elif name.startswith("agentorchestr-"):
                try:
                    n = int(name.rsplit("-", 1)[1])
                    candidates.append((n, w))
                except ValueError:
                    continue
        candidates.sort(key=lambda x: x[0])
        for _, w in candidates:
            if len(list(w.panes or [])) < MAX_PANES_PER_WINDOW:
                return w
        # All full — create the next one.
        next_n = (candidates[-1][0] + 1) if candidates else 2
        new_name = f"agentorchestr-{next_n}"
        return self._tmux_session.new_window(
            window_name=new_name,
            attach=False,
            start_directory=str(self.project_root),
        )

    def _next_worker_id(self) -> str:
        self._counter += 1
        return f"w{self._counter:02d}"

    def _is_git(self) -> bool:
        return (self.project_root / ".git").exists()

    def _create_worktree(self, name: str) -> tuple[str, str]:
        """Returns (worktree_path, branch_name). Falls back to a scratch
        dir if the project isn't a git repo."""
        worktree_path = str(self._tmp_root / name)
        os.makedirs(worktree_path, exist_ok=True)

        if not self._is_git():
            return worktree_path, ""

        branch = f"agentorchestr/{self.agentorchestr_session_id}/{name}"
        # Fresh branch first; if branch already exists from a stale run,
        # try checking it out into a new worktree.
        r = subprocess.run(
            ["git", "worktree", "add", "-b", branch, worktree_path],
            cwd=str(self.project_root),
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            subprocess.run(
                ["git", "worktree", "add", worktree_path, branch],
                cwd=str(self.project_root),
                capture_output=True, timeout=30,
            )
        return worktree_path, branch

    def _agent_argv(self, agent: dict, prompt_path: str) -> list[str]:
        """Build the argv to launch an agent in INTERACTIVE mode.

        Thin delegate to agent_launcher.build_argv so the Supervisor's
        lead-launch path and the worker-launch path here use the same
        per-agent CLI knowledge.
        """
        from agentorchestr.agent_launcher import build_argv
        return build_argv(agent, prompt_path, role="worker")

    async def spawn_worker(
        self,
        *,
        task: str,
        perspective: str,
        agent: dict,
        file_scope: list[str] | None = None,
        pass_criteria: str = "",
        extra_context: str = "",
    ) -> Worker:
        """Create a tmux pane + git worktree, launch the worker agent.

        Idempotent on (session_id, task, perspective, agent_name): a
        re-spawn for the same triple within a session returns the
        previously-created worker instead of doubling up.  Powers
        --resume.
        """
        if not perspectives.is_valid(perspective):
            raise ValueError(f"unknown perspective {perspective!r}")
        if self._tmux_session is None:
            raise RuntimeError("worker pool not started — call start_session() first")

        # Idempotency key: a hash of the inputs the supervisor controls.
        key_blob = "\x00".join([
            self.agentorchestr_session_id, perspective, agent.get("name", ""), task,
            "\x01".join(file_scope or []),
        ])
        idem_key = "spawn:" + hashlib.sha256(key_blob.encode()).hexdigest()[:16]
        op_id, prior = await self.store.begin_op(
            self.agentorchestr_session_id, "spawn_worker", idem_key,
            # Persist the full task: SQLite handles long strings fine and
            # debuggability matters more than a few KB.  The idempotency
            # key already hashes the full task, so truncating here used to
            # create stale args mismatched with the key.
            {"task": task, "perspective": perspective,
             "agent": agent.get("name"), "file_scope": file_scope or []},
        )
        if prior is not None and prior.get("worker_id") in self._workers:
            # Already spawned in this process AND we still hold the
            # in-memory Worker — return it as-is.
            return self._workers[prior["worker_id"]]

        wid = self._next_worker_id()
        wt_name = f"{perspective}-{wid}"
        worktree, branch = self._create_worktree(wt_name)

        # Optionally enrich extra_context with retrieved memories + matching skills.
        memory_context = await self._fetch_memory_context(task)
        skills_context = self._fetch_skills_context(task, file_scope or [])
        merged_extra = "\n\n".join(
            c for c in (skills_context, memory_context, extra_context) if c
        )

        # Build the system prompt + task in one blob, write to file
        prompt = perspectives.system_prompt(
            perspective=perspective,
            task=task,
            file_scope=file_scope or [],
            pass_criteria=pass_criteria,
            extra_context=merged_extra,
        )
        prompt_dir = Path(worktree) / ".agentorchestr"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = str(prompt_dir / f"prompt_{wid}.txt")
        Path(prompt_path).write_text(prompt)

        # All panes live in tmux windows named 'agentorchestr', 'agentorchestr-2',
        # 'agentorchestr-3', ...  The lead occupies pane 0 of the first
        # window; workers tile below it.  Once a window has hit
        # MAX_PANES_PER_WINDOW (lead + N workers), new workers spill into
        # a fresh window so individual panes stay readable.
        agentorchestr_window = self._pick_or_create_window()
        existing_panes = list(agentorchestr_window.panes or [])
        is_overflow_window = agentorchestr_window.window_name != "agentorchestr"

        if (not is_overflow_window) and len(self._workers) == 0 and len(existing_panes) <= 1:
            # First worker in the lead window: split below the lead.
            pane = agentorchestr_window.split(direction="down", attach=False)
        elif is_overflow_window and len(existing_panes) == 0:
            # Brand-new overflow window: its initial pane IS our worker pane.
            pane = agentorchestr_window.active_pane
        else:
            # Subsequent workers: split the bottom-most pane.
            try:
                target = existing_panes[-1] if existing_panes else None
                if target is not None and hasattr(target, "split"):
                    pane = target.split(direction="down", attach=False)
                else:
                    pane = agentorchestr_window.split(direction="down", attach=False)
            except Exception:
                pane = agentorchestr_window.split(attach=False)
        try:
            # "main-horizontal" keeps the first pane (lead) tall on top
            # and tiles workers below — better than "even-vertical" which
            # shrinks the lead as workers grow.
            agentorchestr_window.select_layout("main-horizontal")
        except Exception:
            try:
                agentorchestr_window.select_layout("tiled")
            except Exception:
                pass

        # cd to worktree, banner, then launch agent
        pane.send_keys(f"cd {shlex.quote(worktree)}", enter=True)
        pane.send_keys(
            f"echo '=== {wid} | {perspective} | {agent['name']} | branch={branch or '(no-git)'} ==='",
            enter=True,
        )
        argv = self._agent_argv(agent, prompt_path)
        pane.send_keys(" ".join(shlex.quote(p) for p in argv), enter=True)

        worker = Worker(
            id=wid,
            session_id=self.agentorchestr_session_id,
            perspective=perspective,
            agent=agent,
            worktree=worktree,
            branch=branch,
            pane=pane,
            pane_id=getattr(pane, "pane_id", None),
            task=task,
            file_scope=file_scope or [],
            state="running",
        )
        self._workers[wid] = worker
        await self.store.upsert_worker(self.agentorchestr_session_id, worker_to_row(worker))
        if op_id is not None:
            await self.store.finish_op(
                op_id,
                {"worker_id": wid, "worktree": worktree, "branch": branch},
            )
        return worker

    def get(self, worker_id: str) -> Worker:
        if worker_id not in self._workers:
            raise KeyError(f"unknown worker {worker_id}")
        return self._workers[worker_id]

    async def _fetch_memory_context(self, task: str) -> str:
        """Retrieve the top relevant memories for `task` and format them
        as a stable, deterministic block for the cacheable prefix.

        Returns "" when no memory federation is wired up or when nothing
        relevant comes back — in which case the worker's prompt is
        unchanged.

        Research notes (kind='research') get special-cased: their stored
        body is a JSON envelope; we re-render it as compact markdown so
        the worker sees the actual sources and snippets, not raw JSON.
        """
        if self.memory is None:
            return ""
        try:
            hits = await self.memory.retrieve(task, k=3)
        except Exception:
            return ""
        if not hits:
            return ""
        parts = ["## Retrieved memories"]
        for h in hits:
            parts.append(h.header())
            body = h.body
            if h.kind == "research":
                rendered = _render_research_hit(body)
                if rendered:
                    body = rendered
            # Cap each note so the prompt budget stays bounded.
            parts.append(body[:800] if h.kind == "research" else body[:600])
        return "\n\n".join(parts)

    def _fetch_skills_context(self, task: str, file_scope: list[str]) -> str:
        """Render the bodies of all skills whose triggers match this task.

        Skills land in the cacheable prefix alongside memories so any
        repeat call within the cache TTL costs only 10% on input tokens.
        """
        if self.skills is None:
            return ""
        try:
            matched = self.skills.matching(task, file_scope)
            if not matched:
                return ""
            return self.skills.render(matched)
        except Exception:
            return ""

    def list(self) -> list[Worker]:
        return list(self._workers.values())

    def read_output(self, worker_id: str, tail_lines: int = 80) -> str:
        """Capture the worker's tmux pane content (last `tail_lines` lines).

        Uses `tmux capture-pane -p -S -<n>` which is portable across
        libtmux versions. Falls back to `pane.capture_pane()` if available.
        """
        worker = self.get(worker_id)
        pane = worker.pane
        if pane is None:
            return ""
        # Prefer the tmux CLI for predictable scrollback handling
        try:
            r = subprocess.run(
                ["tmux", "capture-pane", "-p", "-J", "-S", f"-{int(tail_lines)}",
                 "-t", str(worker.pane_id)],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return r.stdout
        except Exception:
            pass
        try:
            return "\n".join(pane.capture_pane())
        except Exception:
            return ""

    def send_message(self, worker_id: str, message: str) -> None:
        """Send a follow-up instruction to a running worker by typing it
        into the pane (most CLI agents accept follow-up prompts at their
        REPL prompt)."""
        worker = self.get(worker_id)
        if worker.pane is None:
            return
        # Best-effort: many agents read multi-line input until Enter.
        worker.pane.send_keys(message, enter=True)

    def detect_completion(self, worker_id: str) -> tuple[str, str]:
        """Inspect the worker and return (state, summary).

        Detection order (most-trusted first):
          1. Structured flag file at <worktree>/.agentorchestr/done.flag.
             Format: first line is one of done|blocked|failed, remaining
             lines are the summary.  Workers that respect the protocol
             write this file atomically (write+rename) so we never read
             a half-flushed state.
          2. Sentinel regex over the last _CAPTURE_TAIL_LINES of the pane.
             Anchored at both ends of the line so a sentinel quoted
             inside an explanation does not trigger.

        state is one of: 'running', 'done', 'blocked', 'failed'.
        """
        worker = self.get(worker_id)

        # 1. Flag file (preferred).
        flag = Path(worker.worktree) / ".agentorchestr" / _DONE_FLAG_NAME
        if flag.exists():
            try:
                content = flag.read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""
            head, _, body = content.partition("\n")
            head = head.strip().lower()
            if head in ("done", "blocked", "failed"):
                return head, body.strip()[:300]

        # 2. Pane regex fallback.
        text = self.read_output(worker_id, tail_lines=_CAPTURE_TAIL_LINES)
        for rx, st in ((_DONE_RE, "done"), (_BLOCKED_RE, "blocked"), (_FAILED_RE, "failed")):
            m = rx.search(text)
            if m:
                return st, m.group(1).strip()[:300]
        return "running", ""

    async def wait_until_done(
        self,
        worker_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.5,
    ) -> tuple[str, str]:
        """Block until the worker writes a sentinel or `timeout` elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            state, summary = self.detect_completion(worker_id)
            if state != "running":
                worker = self.get(worker_id)
                worker.state = state
                worker.summary = summary
                worker.finished_at = time.time()
                await self.store.upsert_worker(self.agentorchestr_session_id, worker_to_row(worker))
                return state, summary
            await asyncio.sleep(poll_interval)
        # Timeout: leave worker running but record the timeout against the row
        worker = self.get(worker_id)
        worker.state = "timeout"
        worker.summary = "no completion sentinel within timeout"
        worker.finished_at = time.time()
        await self.store.upsert_worker(self.agentorchestr_session_id, worker_to_row(worker))
        return "timeout", worker.summary

    async def kill_worker(self, worker_id: str) -> None:
        worker = self.get(worker_id)
        if worker.pane is not None:
            try:
                worker.pane.send_keys("C-c", enter=False, suppress_history=False)
                # If the agent ignores C-c, kill the pane outright after a beat.
                worker.pane.send_keys("exit", enter=True)
            except Exception:
                pass
        worker.state = "killed"
        worker.summary = worker.summary or "killed by supervisor"
        worker.finished_at = time.time()
        await self.store.upsert_worker(self.agentorchestr_session_id, worker_to_row(worker))

    # ── teardown ──────────────────────────────────────────────────────

    async def cleanup(self, *, keep_tmux: bool = True) -> None:
        """End the run. By default we keep tmux alive so the user can
        inspect pane output / scroll back. Worktrees are pruned either way."""
        if self._tmux_session is not None and not keep_tmux:
            try:
                self._tmux_session.kill()
            except Exception:
                pass

        # Clean up git worktrees
        if self._is_git():
            for w in self._workers.values():
                if w.worktree and Path(w.worktree).exists():
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", w.worktree],
                        cwd=str(self.project_root),
                        capture_output=True, timeout=10,
                    )
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=str(self.project_root),
                capture_output=True, timeout=10,
            )

        if not keep_tmux and self._tmp_root.exists():
            shutil.rmtree(str(self._tmp_root), ignore_errors=True)


def worker_to_row(w: Worker) -> dict:
    """Project a Worker into the dict shape state_store expects."""
    return {
        "worker_id": w.id,
        "perspective": w.perspective,
        "agent_name": (w.agent or {}).get("name", ""),
        "worktree": w.worktree,
        "branch": w.branch,
        "pane_id": w.pane_id or "",
        "task": w.task,
        "state": w.state,
        "summary": w.summary,
        "started_at": w.started_at,
        "finished_at": w.finished_at,
    }


def _render_research_hit(body: str) -> str:
    """Convert a research-note body (cache header + JSON) back to the
    compact markdown form ResearchPayload.render_markdown() produces.

    Returns "" if the body doesn't look like a research envelope.
    """
    import json as _json
    import re as _re

    m = _re.search(r"```json\s*\n(.*?)\n```", body, _re.DOTALL)
    if not m:
        return ""
    try:
        payload_dict = _json.loads(m.group(1))
    except _json.JSONDecodeError:
        return ""
    try:
        from agentorchestr.research import ResearchPayload, SearchResult
    except ImportError:
        return ""
    try:
        results = [SearchResult(**r) for r in payload_dict.get("results", [])]
        payload = ResearchPayload(
            query=payload_dict.get("query", ""),
            fetched_at=float(payload_dict.get("fetched_at") or 0.0),
            results=results,
            fetched_bodies=dict(payload_dict.get("fetched_bodies") or {}),
            error=payload_dict.get("error"),
        )
    except (TypeError, ValueError):
        return ""
    # Tighten body sizes for the prompt budget.
    return payload.render_markdown(max_chars_per_body=600)
