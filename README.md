# agentorchestr

A supervisor/worker orchestrator for terminal-based coding agents.

**agentorchestr** discovers installed CLI agents (Claude Code, Kiro, OpenClaude,
OpenCode, Codex, Gemini CLI, Aider, Goose…), assigns one as a **lead** that
receives an MCP tool surface, and tiles the rest as **workers** in a single
tmux session — each in its own git worktree.

```
┌─────────────────────────────────────────────────────────┐
│  tmux window "agentorchestr"                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  lead agent (kiro / claude / …)  ← MCP tools     │  │
│  ├──────────────────┬────────────────────────────────┤  │
│  │  worker w01      │  worker w02                    │  │
│  │  (implementer)   │  (tester)                      │  │
│  └──────────────────┴────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Install

```bash
# From PyPI (once published):
pipx install agentorchestr

# From source:
git clone https://github.com/IAZENT/agentorchestr && cd agentorchestr
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

MCP support (required for auto mode):

```bash
pip install 'agentorchestr[mcp]'
```

Optional extras:

```bash
pip install 'agentorchestr[discover]'   # cross-terminal mDNS discovery
pip install 'agentorchestr[research]'   # web research via DuckDuckGo
pip install 'agentorchestr[memory]'     # vector retrieval (sqlite-vec + fastembed)
```

---

## LLM keys

agentorchestr's router is used for planning and compression only — the agents
use their own auth for coding. Configure at least one:

```bash
export ANTHROPIC_API_KEY=...   # paid; 90% off cached prefix (recommended)
export CEREBRAS_API_KEY=...    # 1M tok/day free, ~2100 tok/s
export GROQ_API_KEY=...        # 30 RPM free, ~315 tok/s
export GEMINI_API_KEY=...      # 1500 req/day free, 1M context
export OPENROUTER_API_KEY=...  # 28+ free models
```

---

## Quickstart

```bash
# Check what's installed:
agentorchestr --detect

# Scaffold per-project context files:
agentorchestr --init

# Run with a goal (auto mode — lead + workers):
agentorchestr --goal "Add JWT auth to /api/users"

# Long goal from a file:
agentorchestr --goal-file design-doc.md

# Interactive multi-line paste (end with /end):
agentorchestr --interactive

# Single agent, no supervisor (manual mode):
agentorchestr --manual --task "fix the login bug"
```

agentorchestr attaches you to the tmux session automatically. Detach with
`Ctrl-B D`. Reconnect later:

```bash
agentorchestr --list-sessions
agentorchestr --resume <session_id>
```

---

## How it works

1. **Lead agent** runs in tmux pane 0 with an MCP server on `127.0.0.1:<random-port>`.
2. The lead calls `spawn_worker(task, perspective)` → agentorchestr creates a git
   worktree on a fresh branch and launches a worker agent in a new pane.
3. Workers write `WORKER_DONE: <summary>` (and a `.agentorchestr/done.flag` file)
   when finished. The lead reads their output via `read_worker_output`.
4. The lead calls `verify()` then `mark_goal_done(summary)` to end the session.

### Worker perspectives

| Perspective   | Role                                                    |
|---------------|---------------------------------------------------------|
| `implementer` | minimal-diff code change                                |
| `tester`      | tests for success + at least one failure mode           |
| `reviewer`    | numbered concerns with `file:line`                      |
| `security`    | auth / injection / secrets / crypto only                |
| `performance` | N+1, quadratic loops, sync I/O on async paths           |
| `verifier`    | runs the FULL suite + linters, no shortcuts             |
| `researcher`  | gathers external facts; produces no patches             |

### MCP tools exposed to the lead

`spawn_worker` · `read_worker_output` · `send_to_worker` · `wait_for_worker` ·
`list_workers` · `kill_worker` · `cross_review` · `verify` · `report_progress` ·
`mark_goal_done` · `mark_goal_failed` · `list_perspectives` ·
`discover_running_agents` · `send_to_external_agent` · `read_from_external_agent` ·
`web_research` · `compact_session` · `clear_tool_results`

---

## Filesystem layout

Follows the [XDG Base Directory spec](https://specifications.freedesktop.org/basedir-spec/latest/).

| Scope        | Path                                                                    | Holds                                              |
|--------------|-------------------------------------------------------------------------|----------------------------------------------------|
| Global state | `$XDG_DATA_HOME/agentorchestr/` (`~/.local/share/agentorchestr/`)       | `state.db`, `skills/`, `memory/`                   |
| Global config| `$XDG_CONFIG_HOME/agentorchestr/` (`~/.config/agentorchestr/`)          | global hooks / overrides                           |
| Per-project  | `<project>/.agentorchestr/`                                             | `PROJECT.md`, `CONVENTIONS.md`, `memory/`, `hooks.json` |
| Per-session  | `/tmp/agentorchestr-<sid>/`                                             | prompt files, progress log                         |

Existing `~/.agentorchestr/` installations keep working — agentorchestr never
silently moves your data.

### Per-project context

`agentorchestr --init` creates:

```
.agentorchestr/
├── PROJECT.md       # stack, architecture, verify commands
├── CONVENTIONS.md   # style, naming, test rules
├── hooks.json       # lifecycle hook commands
├── memory/
│   ├── topics/      # learned topic notes (auto-indexed)
│   ├── episodes/    # per-session journals
│   └── research/    # cached web-research payloads (24h TTL)
└── skills/          # project-pinned skills (override globals)
```

`PROJECT.md` and `CONVENTIONS.md` are auto-loaded into the cacheable preamble
of every session. Keep them short — every byte is re-paid on every turn.

---

## Cross-terminal worker pool

Wrap any CLI agent so agentorchestr can find and drive it from another terminal:

```bash
# Terminal 1
agentorchestr-shim kiro chat

# Terminal 2
agentorchestr-shim claude

# Terminal 3
agentorchestr --goal "build the auth module"
```

Three discovery layers run in order:

1. **mDNS** via `_agentorchestr-agent._tcp.local.` (requires `zeroconf` extra)
2. **Filesystem cards** under `$XDG_RUNTIME_DIR/agentorchestr-agents/`
3. **tmux pane scan** as a last resort (send-keys only, no structured RPC)

---

## Skills

Skills are git-clone-able bundles (instructions + optional ed25519 signature)
that auto-inject into matching worker prompts.

```bash
agentorchestr --skill-add github.com/<author>/<skill-name>
agentorchestr --skill-list
agentorchestr --skill-verify <skill-name>
agentorchestr --skill-remove <skill-name>
agentorchestr --goal "..." --require-signed-skills   # production
```

A skill directory:

```
~/.local/share/agentorchestr/skills/<skill-name>/
├── skill.toml         # name, version, triggers (keywords + file_globs)
├── skill.md           # markdown injected into matching workers' prompts
├── signature.sig      # optional ed25519 signature
└── signing-key.pub    # optional public key
```

---

## Dashboard

```bash
agentorchestr --dashboard   # http://localhost:3000
```

Endpoints: `/api/sessions`, `/api/sessions/<id>`, `/api/sessions/<id>/workers`,
`/healthz`, `/api/docs`. Auto-refreshes every 5s.

---

## CLI reference

| Flag                              | Purpose                                           |
|-----------------------------------|---------------------------------------------------|
| `--goal "..."`                    | one-shot goal (auto mode)                         |
| `--goal-file PATH`                | read goal from a UTF-8 file                       |
| `--interactive`                   | paste a multi-line goal (`/end` to submit)        |
| `--manual --task "..."`           | single agent, single task, no supervisor          |
| `--agents NAME [...]`             | restrict to these agent names                     |
| `--lead NAME`                     | force a specific lead (must be in --agents)       |
| `--init [--init-force]`           | scaffold this project's `.agentorchestr/`         |
| `--detect`                        | show installed agents + LLM keys + deps           |
| `--list-sessions`                 | recent sessions, resumable ones flagged           |
| `--resume <id>`                   | pick up a paused or crashed session               |
| `--dashboard`                     | run the FastAPI dashboard                         |
| `--attach / --no-attach`          | force / suppress auto-attach to tmux              |
| `--skill-add/list/verify/remove`  | skill marketplace                                 |
| `--require-signed-skills`         | refuse unsigned skills                            |
| `--doctor`                        | comprehensive health check (exit 0 = healthy)     |
| `--version`                       | print version and exit                            |

---

## Security model

- **`--dangerously-skip-permissions`** is passed to Claude/OpenClaude/Amp
  automatically. This is intentional — the supervisor drives the agent
  non-interactively and needs it to accept tool calls without prompting.
  Only run agentorchestr in projects you trust.

- Worker prompts are written to files inside each git worktree and fed to
  agents via `bash -lc 'agent "$(cat prompt_file)"'`. The `$(cat …)`
  substitution is literal — shell metacharacters inside the prompt are NOT
  re-evaluated (verified).

- MCP server binds to `127.0.0.1` only. The port is random and written to
  the agent's MCP config file for the duration of the session.

- Skills with `--require-signed-skills` are verified against an ed25519
  public key bundled in the skill directory. Unsigned skills are allowed
  by default; use the flag in production.

---

## Known limitations

- **tmux required** for the full multi-pane UX. Without tmux, `--manual`
  mode still works via subprocess.
- **Max 5 panes per tmux window** (1 lead + 4 workers). Past that, workers
  spill into `agentorchestr-2`, `agentorchestr-3`, … windows automatically.
- **MCP required for auto mode.** Install with `pip install 'agentorchestr[mcp]'`.
- **Hosted LLM keys** are needed for the router (planning/compression). The
  agents themselves use their own auth.
- **Linux / macOS only.** Windows is not supported (PTY + tmux dependency).

---

## Requirements

- Python 3.11+
- tmux 3.0+ (recommended)
- git (for worktrees)
- Optional: `zeroconf`, `sqlite-vec`, `fastembed`, `ddgs`, `cryptography` or `pynacl`

---

## License

[MIT](LICENSE).
