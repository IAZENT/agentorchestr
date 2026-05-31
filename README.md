# ORCH — Zero-Cost Agent Orchestrator

**A production-grade agent orchestrator that works with ANY CLI coding agent (Kiro, OpenCode, OpenClaude, Claude Code, Aider, Codex, Gemini CLI…) using 100% free LLM backends for orchestration intelligence.**

---

## The Core Idea

You have CLI coding agents installed. ORCH is the *control layer* that sits above them — breaking down goals, routing tasks, managing multiple agent instances in parallel, evaluating results, and retrying failures. All orchestration intelligence uses free LLM APIs. No paid orchestrator subscription. No locked-in vendor.

### Advanced Techniques (from top companies)
- **Two-Ledger System** (Microsoft Magentic-One): Task Ledger + Progress Ledger with stall detection
- **Wave Execution** (Kiro): independent tasks run in parallel waves
- **Three-Term Memory** (Claude Code): session/project/global memory with MEMORY.md index
- **JIT Context Injection** (Anthropic): only files in scope, 6x context reduction
- **Output Key Pattern** (Google ADK): inter-task data flow via named outputs
- **3-Gate Quality Evaluation** (OpenAI guardrails): structural → file → LLM fail-fast
- **Lifecycle Hooks** (Claude Code/Kiro): pre/post task, session, eval, failure
- **Checkpointing** (LangGraph/CrewAI): resume without re-running completed tasks
- **Stall Detection** (Magentic-One): auto re-planning after N cycles without progress
- **Effort Scaling** (Anthropic): dynamic worker count based on task complexity

```
YOUR GOAL
    ↓
[ORCH Brain — Free LLM: Gemini/Groq/Cerebras]
    ↓ spec-driven decomposition (Kiro-style: requirements → design → tasks)
[Task DAG with dependencies + Task Ledger (Magentic-One)]
    ↓ wave execution scheduler (Kiro-style parallelism)
┌───────────────────────────────────────────────────────┐
│  Wave 1 (parallel):                                   │
│    Worker 0: kiro        task t001 (no deps)          │
│    Worker 1: opencode    task t002 (no deps)          │
│    Worker 2: openclaude  task t003 (no deps)          │
├───────────────────────────────────────────────────────┤
│  Wave 2 (after Wave 1):                               │
│    Worker 0: kiro        task t004 (deps: t001,t002)  │
└───────────────────────────────────────────────────────┘
    ↓ JIT context injection (only files in scope)
[Three-Term Memory: session/project/global]
    ↓ 3-gate quality evaluation (fail-fast)
[Quality Gate: structural → file → LLM eval]
    ↓ if failed: retry with feedback injection
[State Store: SQLite checkpointing — resume without re-running]
    ↓ Progress Ledger tracks stall detection (Magentic-One)
[Auto re-planning if stalled for N cycles]
    ↓
[Human review diffs] → merge (never auto-merge)
```

---

## What Companies Like Google, Microsoft, and OpenAI Do

After research into how the giants build agent orchestration (2025–2026):

### Microsoft's 5 Patterns (Azure Architecture Center)
Microsoft identifies five production-grade orchestration patterns:
1. **Sequential** — pipeline: Agent 1 → Agent 2 → Agent N (like a CI/CD chain)
2. **Concurrent** — parallel workers on independent subtasks (what ORCH uses for coding)
3. **Handoff** — Agent A transfers control + context to Agent B for specialty work
4. **Group Chat** — multiple agents negotiate a solution (AutoGen/Magentic style)
5. **Magentic** — dynamic team formation: orchestrator selects agents at runtime based on task

Microsoft maps these to: **Semantic Kernel** (code-first SDK), **Microsoft Agent Framework** (open source, .NET + Python), and **Foundry Agent Service** (low-code/no-code declarative).

### OpenAI Agents SDK (March 2025)
The Agents SDK replaced experimental Swarm with three production primitives:
- **Handoffs** — explicit agent-to-agent transfers with conversation context
- **Guardrails** — input/output validation hooks
- **Tracing** — end-to-end observability of agent chains

Their production example: a triage agent receives input → determines intent → hands off to specialist (billing, support, account). The specialist can return control or chain further.

### Google ADK (April 2025) + A2A Protocol
Google released the Agent Development Kit (ADK) and then open-sourced the **Agent-to-Agent (A2A) Protocol** to the Linux Foundation (June 2025, backed by 50+ partners including AWS, Microsoft, Salesforce, SAP).

A2A's key innovation: **Agent Cards** — JSON documents at `/.well-known/agent.json` describing what an agent does, how to authenticate, and how to communicate. Discovery is automatic.

### The 2026 Protocol Stack (What Production Systems Use)
```
┌─────────────────────────────────────────────────┐
│  Application Layer  (your orchestrator logic)   │
├─────────────────────────────────────────────────┤
│  A2A  (agent ↔ agent delegation/coordination)  │
├─────────────────────────────────────────────────┤
│  MCP  (agent ↔ tools/data/resources)           │
├─────────────────────────────────────────────────┤
│  HTTP / SSE / WebSocket  (transport)            │
└─────────────────────────────────────────────────┘
```
- **MCP** = vertical connection (agent → tools). 97M+ monthly SDK downloads. Native in Claude, ChatGPT, Cursor, VS Code, JetBrains. OpenAI deprecated Assistants API in favor of MCP (early 2026).
- **A2A** = horizontal connection (agent ↔ agent delegation). Stateful task lifecycle built in.
- **ORCH** uses MCP as its bridge layer (the root-level `mcp_bridge.py` server) and tmux + file exchange as its agent communication layer — zero cost, works with every agent that reads files.

### The Token Efficiency Research
Academic research (arXiv) ORCH implements:
- **22.7% token reduction** with aggressive compression prompting (context-aware agents)
- **6x reduction** in initial system prompt context with just-in-time schema loading
- **10–25x reduction** in context growth rate with JIT tool definitions

ORCH implements these as:
1. `GoalDecomposer._build_minimal_context()` — JIT per-task file injection
2. `_compress_context()` in the main loop — every 5 cycles, Haiku-tier model compresses completed tasks
3. Scrubbed result schema — 100-word max summaries, not raw agent output

---

## Installation

```bash
# 1. Clone
git clone https://github.com/you/orch
cd orch

# 2. Create a venv (recommended) and bootstrap
python3 -m venv .env && source .env/bin/activate
python bootstrap.py --all          # core + mcp + search + tests
# or just core:  python bootstrap.py
# or check only: python bootstrap.py --check

# 3. Configure at least ONE LLM backend
#    Free hosted (set any of these):
export GEMINI_API_KEY=...           # ai.google.dev — 1500 req/day, recommended
export GROQ_API_KEY=...              # console.groq.com — 30 RPM
export CEREBRAS_API_KEY=...          # inference.cerebras.ai — 1M tok/day
export OPENROUTER_API_KEY=...        # openrouter.ai — 28+ free models
#    OR run a local Ollama (no key needed):
ollama serve & ollama pull llama3.1
export OLLAMA_HOST=http://localhost:11434     # optional override

# 4. Verify
python orchestrator.py --detect
```

> **Note:** ORCH does NOT use `--break-system-packages` by default. If you
> bootstrap outside a virtualenv it falls back to `pip install --user`.
> Pass `python bootstrap.py --force-system` only if you really mean it.

---

## Free LLM Provider Guide

| Provider | Free Tier | Model | Sign-up |
|---|---|---|---|
| **Google Gemini** | 1,500 req/day, 1M context | Gemini 2.5 Flash | ai.google.dev — email only |
| **Cerebras** | 1M tokens/day | Llama 3.1 70B | inference.cerebras.ai — email only |
| **Groq** | 30 RPM | Llama 3.3 70B | console.groq.com — email only |
| **OpenRouter** | 20 RPM, 28+ free models | DeepSeek R1, Qwen3 480B | openrouter.ai — email only |
| **Ollama (local)** | unlimited (your hardware) | llama3.1 / mistral / any pulled model | ollama.com — install once |

**Strategy used by ORCH:**
- Planning/decomposition → Gemini 2.5 Flash (best quality + 1M context)
- Compression/eval → Groq or Cerebras (fastest, cheapest quota)
- Fallback chain → OpenRouter free → **Ollama local** → fail
- Rate-limit aware: a `429` response puts the provider on cooldown for the
  duration the API requests, so subsequent calls skip straight to the next.

---

## Usage

```bash
# Basic: give ORCH a goal
python orchestrator.py --goal "Build a REST API for user authentication with JWT"

# Force specific agents
python orchestrator.py --goal "Refactor auth module" --agents kiro opencode

# Single-agent multi-instance (if you only have kiro)
python orchestrator.py --goal "Add tests for all API endpoints" --agents kiro --workers 3

# Dry run (plan only, no agents run)
python orchestrator.py --goal "..." --dry-run

# Resume a paused session
python orchestrator.py --resume abc12345

# Interactive mode (type multi-line goal)
python orchestrator.py --interactive

# Launch dashboard only
python orchestrator.py --dashboard   # → http://localhost:3000

# No tmux (subprocess mode, quieter output)
python orchestrator.py --goal "..." --no-tmux

# Detect available agents + LLM status
python orchestrator.py --detect
```

---

## Orchestration Modes

ORCH auto-selects one of these modes based on what you have installed:

### `multi_agent` (Best: 2+ different CLI agents)
Different agents for different strengths:
```
Kiro    → task t001 (multi-file understanding)
OpenCode → task t002 (75+ model providers)
Aider   → task t003 (git-native, auto-commits)
Claude  → task t004 (full SDK, bash tools)
```

### `single_agent_multi` (You have only Kiro or only OpenCode)
Same agent, N tmux panes = N parallel workers:
```
tmux window "agents"
├── pane 0: kiro (orchestrator role — planner)
├── pane 1: kiro (worker A — task t001)
├── pane 2: kiro (worker B — task t002)
└── pane 3: kiro (worker C — task t003)
```
Each in its own git worktree. Orchestrator coordinates, workers execute.

### `hybrid` (CLI agents + IDE agents)
CLI agents do parallel execution. IDE agents (Cursor/Windsurf) do supplementary work via MCP:
```
CLI: kiro workers (tasks)
IDE: cursor/windsurf (via MCP bridge, context injection)
```

### `ide_only` (Only Cursor/Windsurf detected — limited)
Uses context file injection + MCP bridge. Limited parallelism.
```
ORCH → writes task files → IDE agent reads via MCP
```

---

## tmux Layout

When tmux is available, ORCH builds this layout automatically:

```
orch-<session_id>
├── window 0 "control"
│   ├── pane 0 (40%): ORCH Python process (Rich status table)
│   └── pane 1 (60%): live event log
│
├── window 1 "agents"
│   ├── pane 0: WORKER-0 (orchestrator-agent role)
│   ├── pane 1: WORKER-1
│   ├── pane 2: WORKER-2
│   └── pane 3: WORKER-3 (optional)
│
└── window 2 "git"
    ├── pane 0: git log --oneline --all --graph (live watch)
    └── pane 1: test runner
```

Watch your agents in real time: `tmux attach -t orch-<session_id>`

---

## MCP Bridge (for IDE agents)

If you use Cursor, Windsurf, Kiro, Claude Code, or any MCP-capable agent:

```bash
# Start the MCP bridge over Server-Sent Events
python mcp_bridge.py --transport sse --port 8765

# Or stdio (for clients that prefer subprocess transport)
python mcp_bridge.py --transport stdio

# Write client config files for all detected agents (idempotent)
python mcp_bridge.py --write-configs

# Tools agents can call once connected:
#   get_task(agent_id)      → pull their assigned task
#   report_result(...)      → tell ORCH they're done
#   get_context()           → fetch shared project context
#   report_progress(...)    → stream progress updates
```

The bridge is built on FastMCP, which provides full SSE / stdio / streamable-HTTP
transports out of the box. Configs written by `--write-configs` go to
`~/.kiro/settings/mcp.json`, `~/.claude/mcp.json`, `~/.cursor/mcp.json`, and
`~/.windsurf/mcp.json`.

---

## Web Dashboard

```bash
# Launch alongside an orchestration session
python orchestrator.py --dashboard

# Or standalone (useful for inspecting past sessions in ~/.orch/state.db)
python -m dashboard.server --port 3000
```

Endpoints:
- `GET /` — live HTML overview, auto-refreshes every 5s
- `GET /healthz` — liveness probe
- `GET /api/sessions` — recent sessions with done/total counts
- `GET /api/sessions/{id}` — session detail (plan + ledger + task results)
- `GET /api/sessions/{id}/tasks` — task results only
- `GET /api/docs` — OpenAPI / Swagger UI

---

## Quality Gates

Every worker result passes through three gates before being marked done:

**Gate 1 — Structural check (free)**
Result JSON has required fields. Status is not "failed". No errors.

**Gate 2 — File check (free)**
Claimed changed files actually exist on disk and are non-empty.

**Gate 3 — LLM eval (uses free quota)**
Free LLM verifies result against the task's `pass_criteria`. Uses cheapest/fastest model.

If any gate fails: **retry up to 2 times**, with failure feedback injected into the next attempt's instruction. After 2 retries: mark failed, continue with other tasks.

**Gate 4 — Human review (mandatory)**
ORCH **never auto-merges**. All diffs reviewed by you before touching main branch.

---

## Token Efficiency Techniques

1. **JIT context injection** — Workers receive only the files in their `file_scope`, not the full codebase. Typically 3-6 files vs. hundreds.

2. **Structured handoff schema** — All inter-agent communication uses a fixed JSON schema (max 200-word instructions, max 100-word summaries). No prose padding.

3. **Context compression every 5 cycles** — Completed task history replaced with a 3-bullet summary using the cheapest available model.

4. **Scrubbed output before eval** — Raw agent terminal output stripped of progress messages, tool call logs, intermediate thoughts. Only file changes and test results passed to evaluator.

5. **Haiku-tier for compression** — Evaluation and compression calls use the smallest/cheapest free model. Planning calls use the best available free model.

6. **Prompt caching** — Static system prompts (routing rules, project conventions) cached across calls.

---

## Project Structure

```
orch/
├── orchestrator.py         ← main entry point + orchestration loop
├── agent_detector.py       ← finds installed agents (shutil.which)
├── router.py               ← free LLM provider chain (Gemini→Cerebras→Groq→OR→Ollama)
├── session_manager.py      ← tmux layout + subprocess fallback
├── state_store.py          ← SQLite persistence + checkpointing
├── goal_decomposer.py      ← spec-driven goal → task DAG (Kiro-style)
├── scheduler.py            ← DAG scheduler with wave execution
├── quality_gate.py         ← 3-gate fail-fast evaluation
├── monitor.py              ← Rich live status table
├── context_manager.py      ← three-term memory + JIT context + output keys
├── hooks.py                ← lifecycle hooks (pre/post task, session, eval)
├── mcp_bridge.py           ← MCP server for IDE agent integration (FastMCP)
├── bootstrap.py            ← env check + pip installer (no --break-system)
├── pyproject.toml          ← canonical install metadata
│
├── dashboard/              ← FastAPI dashboard (HTML + JSON API)
│   ├── __init__.py
│   └── server.py
│
├── tests/                  ← pytest-asyncio (scheduler/store/decomposer/router)
│
├── agents/
│   ├── __init__.py
│   └── agent_registry.py   ← mode selection, agent capability ranking
│
└── .orch/                  ← project memory + hooks config
    ├── MEMORY.md           ← auto-learned project conventions
    └── hooks.json          ← lifecycle hook configuration
```

---

## Agent Capability Reference

| Agent | Type | Parallel | MCP | Best For |
|---|---|---|---|---|
| claude (Claude Code) | SDK | ✓ | ✓ | File-heavy work, bash, full SDK control |
| openclaude (OpenClaude) | CLI | ✓ | ✓ | Open-source agent, extensible, community |
| kiro | CLI | ✓ | ✓ | Multi-file understanding, spec-driven dev |
| opencode | CLI | ✓ | ✗ | 75+ LLM providers, session sharing, privacy |
| aider | CLI | ✓ | ✗ | Git-native, auto-commits, pair programming |
| codex | CLI | ✓ | ✗ | OpenAI model users |
| gemini | CLI | ✓ | ✗ | 1M token context window tasks |
| goose | CLI | ✓ | ✓ | Block-based task execution |
| cursor | IDE | ✗ | ✓ | Visual editing, inline refactoring |
| windsurf | IDE | ✗ | ✓ | Multi-file, large codebase navigation |

---

## Extending ORCH

### Add a new CLI agent
Edit `core/agent_detector.py`, add to `AGENT_REGISTRY` and `AGENT_CAPABILITIES`.

### Add a new LLM provider
Edit `llm/router.py`, add to `PROVIDERS` list with the appropriate `type` handler.

### Add a custom quality gate
Subclass `QualityGate` and override `evaluate()`.

### Add a custom task type
Edit `GoalDecomposer.ROUTING_RULES` to teach the planner about the new task type.

---

## Research Notes

This project implements patterns from:
- AWS CLI Agent Orchestrator (CAO) — supervisor-worker over MCP in tmux
- Anthropic Claude Agent SDK — session management, tool use, multi-turn
- Microsoft Azure AI Agent Orchestration Patterns — sequential, concurrent, handoff, magentic
- Google A2A Protocol + ADK — Agent Cards, stateful task lifecycle
- arXiv research on context compression — 22.7% token reduction with compression prompting
- arXiv research on JIT tool loading — 6x reduction in system prompt context

ORCH deliberately does NOT use LangGraph, CrewAI, AutoGen, or any framework — those add dependency weight and abstract the underlying mechanics. ORCH is ~600 lines of Python that you can read in an hour.

---

*ORCH never auto-merges. You review every diff. That's the quality gate that makes the whole system trustworthy.*
