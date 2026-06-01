# Changelog

All notable changes to agentorchestr are documented here.  Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.7.0] – 2026-06-01

### Fixed (critical)
- **Wheel was empty** — `pyproject.toml` listed 19 top-level `py-modules` that no longer existed at the repo root after the package was moved into `agentorchestr/`. The wheel shipped as 14 KB containing only `LICENSE`. Now 303 KB with all modules.
- **6 bare imports** inside the package (`from paths import …`, `from mcp_bridge import …`, `from discovery import …`) only resolved via `sys.path` manipulation in tests. Converted to absolute `agentorchestr.X` imports so the package works after a real `pip install`.

### Added
- `agentorchestr/agent_launcher.py` — single source of truth for per-agent CLI argv. Both `Supervisor` and `WorkerPool` delegate here; `aider` now uses its native `--message-file` flag.
- `agentorchestr/tty.py` — terminal-attach helpers extracted from orchestrator.
- `agentorchestr/commands/` — `auto.py`, `manual.py`, `info.py` extracted from the 1126-line `orchestrator.py` (now 315 lines).
- `agentorchestr/__init__.py` — `__version__` via `importlib.metadata` + public re-exports (`Supervisor`, `WorkerPool`, `LLMRouter`, `StateStore`).
- `--version` flag on the CLI.
- `Makefile` with `test / build / install / clean / smoke` targets.
- `RESEARCH.md` — 3003-word consolidated design note covering MCP lifecycle, prompt-cache breakpoints, tmux pane lifecycle, and multi-agent verification protocols (4 parallel research tracks, source-cited).
- Release workflow: wheel sanity check (fails CI if wheel < 50 KB or missing `orchestrator.py`).
- 40 new tests: `test_completion.py`, `test_resume.py`, `test_discovery_layers.py`, `test_agent_launcher.py`. Total: 165 tests.

### Changed
- Worker completion sentinel regexes anchored at end-of-line — a worker quoting `WORKER_DONE:` inside an explanation no longer triggers completion.
- Added `.agentorchestr/done.flag` file-based completion fallback (atomic write+rename); `tmux capture-pane` window bumped 200 → 2000 lines.
- `MAX_PANES_PER_WINDOW = 5`; past the cap, workers spill into `agentorchestr-2`, `agentorchestr-3` … windows automatically.
- `spawn_worker` now stores the full task string in `op_log` args_json (was truncated to 500 chars, mismatching the SHA-256 idempotency key).
- `LLMRouter._call_with_fallback` logs every provider failure at `DEBUG` level instead of silently swallowing it.
- Discovery: 2s process-level TTL cache + `_can_skip_mdns()` short-circuit when `zeroconf` is not installed.
- `.gitignore` rewritten — stops ignoring committed docs (`RESEARCH.md`, `CHANGELOG.md`).

### Removed
- `bootstrap.py` — `--doctor` and `--detect` cover its scope.
- `context_manager.py` — superseded by `memory/federation.py` (XDG-aware, FTS5 + optional sqlite-vec).
- Empty `agents/` directory at repo root.
- Stale `agentorchestr.egg-info/` and `orch.egg-info/` directories.


## [0.3.0] – 2026-06-01

### Added
- **Cross-terminal agent discovery** via mDNS + filesystem cards + tmux scan
  (`agentorchestr-shim` wraps any CLI agent and advertises it).
- **Three-tier memory federation** (markdown + SQLite-FTS5 + optional sqlite-vec)
  with `~/.agentorchestr/memory/` (global) and `<project>/.agentorchestr/memory/` (per-project).
- **Skills marketplace** — git-clone-able, optionally ed25519-signed bundles
  injected into matching worker prompts. CLI: `agentorchestr --skill-add/list/verify/remove`.
- **Web research** as an MCP tool + dedicated `researcher` perspective. 24h
  cache stored in project memory so repeat queries cost nothing.
- **Durable execution** — `op_log` table + idempotency keys.  Re-spawning
  the same task in the same session reuses the worktree instead of
  duplicating.
- **Anthropic prompt-caching** — `cache_control={"type":"ephemeral"}` is now
  set on the system prompt for Anthropic-direct and OpenRouter providers,
  cutting cache-hit input cost by 90 %.
- `agentorchestr --init` scaffolds `<project>/.agentorchestr/{PROJECT,CONVENTIONS}.md`,
  `hooks.json`, `memory/{topics,episodes,research}/`, `skills/`.
- `agentorchestr --goal-file PATH` for long multi-paragraph goals.
- Heartbeat ticker prints worker counts + last summary every 5 s.
- Auto-attach to tmux by default; `--no-attach` to opt out.
- XDG Base Directory layout for global state (with backward compat for
  legacy `~/.agentorchestr/`).

### Changed
- `SUPERVISOR_PROMPT_TEMPLATE` cut from ~110 → ~32 lines (~50 % token win
  on every supervisor turn).  Goal moved to the END of the prompt so the
  prefix is byte-stable for cache hits.
- Every MCP tool docstring trimmed to one sentence — matches Anthropic's
  "MCP capability disclosure tax" guidance.
- Single-window tmux layout (`agentorchestr` window) with `main-horizontal` so the
  lead stays full-width at the top and workers tile below.
- LLM router prioritises Anthropic → Cerebras → Groq → Gemini → OpenRouter.

### Removed
- Local-LLM (Ollama) support.  Latency was incompatible with the
  supervisor's per-turn budget.  Hosted providers only.
- Dead modules: `web_search.py`, `agents/agent_registry.py`.

### Fixed
- `Supervisor.run()` and `orchestrator.main()` no longer raise
  `UnboundLocalError` when an exception fires before `state, summary`
  are assigned.
- `WorkerPool.kill_worker` and `wait_until_done` timeout now persist the
  worker state to SQLite (was leaking `running` status forever).
- Dashboard exposes `/api/sessions/{id}/workers` so live-path sessions
  no longer appear empty.
- Auto mode fails fast with a clear panel when `mcp` is not installed,
  before any tmux state is created.

## [0.2.0] – 2026-05-31

Initial public-readable revision: supervisor-worker auto mode, manual
mode, free-LLM router (Gemini / Cerebras / Groq / OpenRouter / Ollama),
SQLite checkpointing, FastAPI dashboard, MCP bridge.
