# agentorchestr

A production-grade orchestrator for terminal-based coding agents.
**agentorchestr** supervises and coordinates installed CLI agents (Claude
Code, Kiro, OpenClaude, OpenCode, Codex, Gemini CLI, Aider, Goose…) so
they can work in parallel on a single goal — using small **and** large
agents together to ship giant projects in hours, without locking you
into a proprietary stack.

* Single-window tmux layout: lead at the top, workers tile below.
* MCP bridge: the lead agent calls `spawn_worker`, `verify`,
  `web_research`, etc. through a standard MCP/SSE endpoint.
* Three-tier memory (global / per-project / per-session) backed by
  markdown + SQLite-FTS5, with a 24 h cache for web research.
* Cross-terminal agent discovery via mDNS — the lead can dispatch to
  agents you opened in other terminals.
* Skill marketplace — git-clone-able, optionally signed bundles that
  auto-attach to matching tasks.
* Sub-second LLM router across hosted free tiers (Anthropic, Cerebras,
  Groq, Gemini, OpenRouter) with prompt caching enabled by default.
* First-run wizard: detects every agent on your system, asks you to
  pick a lead and workers, or runs in **automatic mode** that decides
  for you based on project complexity.

---

## Install

```bash
pipx install agentorchestr           # once published
# or, from source:
git clone https://github.com/IAZENT/agentorchestr && cd agentorchestr
python3 -m venv .env && source .env/bin/activate
pip install -e ".[dev]"
```

Configure at least one LLM key:

```bash
export ANTHROPIC_API_KEY=...   # paid; 90 % off cached prefix
export CEREBRAS_API_KEY=...    # 1 M tok/day free, ~2100 tok/s
export GROQ_API_KEY=...        # 30 RPM free, ~315 tok/s
export GEMINI_API_KEY=...      # 1500 req/day free, 1 M context
export OPENROUTER_API_KEY=...  # 28+ free models
```

Verify your setup:

```bash
agentorchestr --detect
```

---

## Quickstart

In any project you want agentorchestr to know about:

```bash
agentorchestr --init                              # scaffold .agentorchestr/ for this project
agentorchestr --interactive                       # paste a multi-line goal, pick agents
# or:
agentorchestr --goal "Add JWT auth to /api/users"
agentorchestr --goal-file design-doc.md           # for long goals
```

agentorchestr attaches you to the supervisor's tmux session automatically. Use
`Ctrl-B D` to detach without killing it. To reconnect later:

```bash
tmux attach -t agentorchestr-<session_id>
agentorchestr --list-sessions
agentorchestr --resume <session_id>
```

---

## Filesystem layout

agentorchestr follows the
[XDG Base Directory spec](https://specifications.freedesktop.org/basedir-spec/latest/).

| Scope        | Path                                                            | Holds                                                      |
| ------------ | --------------------------------------------------------------- | ---------------------------------------------------------- |
| Global state | `$XDG_DATA_HOME/agentorchestr/` (default `~/.local/share/agentorchestr/`)         | `state.db`, installed `skills/`, cross-project `memory/`   |
| Global config| `$XDG_CONFIG_HOME/agentorchestr/` (default `~/.config/agentorchestr/`)            | global hooks / overrides                                   |
| Per-project  | `<project>/.agentorchestr/`                                              | `PROJECT.md`, `CONVENTIONS.md`, `memory/`, `hooks.json`    |
| Per-session  | `/tmp/agentorchestr-<sid>/`                                              | ephemeral prompt + progress log                            |

Existing installations with `~/.agentorchestr/` keep using it — agentorchestr never
silently moves your data.

---

## Per-project context

`agentorchestr --init` creates `<project>/.agentorchestr/PROJECT.md` and
`CONVENTIONS.md`. They are auto-loaded into the cacheable preamble of
every supervisor turn, so each byte you write here pays back across
every task in that project. Keep them short.

```
.agentorchestr/
├── PROJECT.md       # stack, architecture, verify commands
├── CONVENTIONS.md   # style, naming, test rules
├── memory/
│   ├── topics/      # learned topic notes (auto-indexed)
│   ├── episodes/    # per-session journals
│   └── research/    # cached web-research payloads
├── skills/          # project-pinned skills (override globals)
└── hooks.json       # lifecycle hook commands
```

---

## Skills

Skills are git-clone-able bundles (instructions + optional MCP servers
+ optional ed25519 signature) that auto-inject when their triggers
match a task.

```bash
agentorchestr --skill-add github.com/<author>/<skill-name>
agentorchestr --skill-list
agentorchestr --skill-verify <skill-name>
agentorchestr --skill-remove <skill-name>
agentorchestr --goal "..." --require-signed-skills   # production
```

A skill is just a directory:

```
~/.local/share/agentorchestr/skills/<skill-name>/
├── skill.toml         # name, version, triggers (keywords + file_globs)
├── skill.md           # markdown injected into matching workers' prompts
├── signature.sig      # optional ed25519 signature
└── signing-key.pub    # optional public key
```

---

## Web research

The lead can ground itself in current docs without leaving its tool
surface:

```python
web_research("FastAPI 0.115 dependency injection")
```

Results land in `<project>/.agentorchestr/memory/research/*.md`, are git-trackable,
indexed by FTS5, and cached for 24 h — so a follow-up implementer
worker inherits the same facts at zero token cost.

For deeper synthesis, delegate to a worker:

```python
spawn_worker(perspective="researcher",
             task="find the latest PyJWT auth pattern")
```

Optional dep: `pip install ddgs`. Without it, `web_research` returns
"no results" but never crashes.

---

## Cross-terminal worker pool

Wrap any CLI agent so agentorchestr can find and drive it from other terminals:

```bash
# Terminal 1
$ agentorchestr-shim claude

# Terminal 2
$ agentorchestr-shim kiro chat

# Terminal 3
$ agentorchestr --goal "build the auth module"
```

The supervisor's MCP surface gains `discover_running_agents()`,
`send_to_external_agent()`, `read_from_external_agent()`. Three
discovery layers run in order:

1. mDNS via `_agentorchestr-agent._tcp.local.` (when `zeroconf` is installed)
2. Filesystem cards under `$XDG_RUNTIME_DIR/agentorchestr-agents/`
3. Bare tmux pane scan as a last resort

---

## Worker perspectives

`spawn_worker(perspective=…)` accepts:

| Perspective   | Role                                                    |
| ------------- | ------------------------------------------------------- |
| `implementer` | minimal-diff code change                                |
| `tester`      | tests for success + at least one failure mode           |
| `reviewer`    | numbered list of concerns with `file:line`              |
| `security`    | auth / injection / secrets / crypto only                |
| `performance` | N+1, quadratic loops, sync I/O on async paths           |
| `verifier`    | runs the FULL suite + linters, no shortcuts             |
| `researcher`  | gathers external facts; produces no patches             |

---

## Dashboard

```bash
agentorchestr --dashboard            # http://localhost:3000
```

`/api/sessions`, `/api/sessions/<id>`, `/api/sessions/<id>/workers`,
`/healthz`, `/api/docs`. The HTML view auto-refreshes every 5 s.

---

## CLI reference (selected)

| Flag                            | Purpose                                          |
| ------------------------------- | ------------------------------------------------ |
| `--goal "..."`                  | one-shot goal                                    |
| `--goal-file PATH`              | read goal from a UTF-8 file                      |
| `--interactive`                 | paste a multi-line goal (`/end` to submit)       |
| `--manual --task "..."`         | single agent, single task, no supervisor         |
| `--init [--init-force]`         | scaffold this project's `.agentorchestr/`                 |
| `--detect`                      | show installed agents + LLM keys + dependencies  |
| `--list-sessions`               | recent sessions, resumable ones flagged          |
| `--resume <id>`                 | pick up a paused or crashed session              |
| `--dashboard`                   | run the FastAPI dashboard                        |
| `--attach / --no-attach`        | force / suppress auto-attach to tmux             |
| `--skill-add/list/verify/remove`| skill marketplace                                |
| `--require-signed-skills`       | refuse unsigned skills                           |

---

## Requirements

* Python 3.11+
* tmux 3.0+ (recommended; otherwise subprocess mode)
* git (for worktrees)
* Optional: `zeroconf` (cross-terminal discovery), `sqlite-vec` +
  `fastembed` (vector retrieval), `ddgs` (web research),
  `cryptography` or `pynacl` (signed-skill verification)

---

## License

[MIT](LICENSE).
