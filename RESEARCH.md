# Agent Orchestration Deep Research
## Techniques from Google, Microsoft, OpenAI, Anthropic, AWS, and Open-Source Frameworks

---

## Table of Contents
1. [The Protocol Stack (2026)](#1-the-protocol-stack-2026)
2. [Microsoft: AutoGen + Magentic-One](#2-microsoft-autogen--magentic-one)
3. [Google: ADK + A2A Protocol](#3-google-adk--a2a-protocol)
4. [OpenAI: Agents SDK + Swarm](#4-openai-agents-sdk--swarm)
5. [Anthropic: Multi-Agent Research System](#5-anthropic-multi-agent-research-system)
6. [AWS: Bedrock Multi-Agent Collaboration](#6-aws-bedrock-multi-agent-collaboration)
7. [Open-Source: CrewAI](#7-open-source-crewai)
8. [Open-Source: LangGraph](#8-open-source-langgraph)
9. [Cross-Cutting Patterns](#9-cross-cutting-patterns)
10. [Key Takeaways for ORCH](#10-key-takeaways-for-orch)

---

## 1. The Protocol Stack (2026)

The industry has converged on a layered protocol architecture:

```
┌─────────────────────────────────────────────────┐
│  Application Layer  (your orchestrator logic)   │
├─────────────────────────────────────────────────┤
│  A2A  (agent ↔ agent delegation/coordination)  │
├─────────────────────────────────────────────────┤
│  MCP  (agent ↔ tools/data/resources)           │
├─────────────────────────────────────────────────┤
│  HTTP / SSE / gRPC / WebSocket  (transport)     │
└─────────────────────────────────────────────────┘
```

- **MCP** = vertical connection (agent → tools). 97M+ monthly SDK downloads. Native in Claude, ChatGPT, Cursor, VS Code, JetBrains.
- **A2A** = horizontal connection (agent ↔ agent delegation). Stateful task lifecycle. Backed by Google, AWS, Microsoft, Salesforce, SAP, 50+ partners. Now under Linux Foundation.
- **ORCH** should use MCP as its bridge layer and A2A-inspired task lifecycle for inter-agent coordination.

---

## 2. Microsoft: AutoGen + Magentic-One

### Architecture
Magentic-One is Microsoft's generalist multi-agent system built on AutoGen. It uses a **hierarchical orchestrator** pattern with 5 agents:

1. **Orchestrator** — lead agent for task decomposition, planning, delegation, progress tracking, corrective actions
2. **WebSurfer** — Chromium browser control (navigation, interaction, reading)
3. **FileSurfer** — local file reading (PDF, PPTX, WAV, markdown)
4. **Coder** — code writing, analysis, artifact creation
5. **ComputerTerminal** — shell execution, library installation

### The Two-Loop Architecture (Critical Design)

The Orchestrator operates through **two nested loops**:

#### Outer Loop: Task Ledger Management
Maintains three categories:
- **Facts** — established knowledge gathered during execution
- **Guesses** — educated hypotheses about unknowns
- **Plan** — current strategy for accomplishing the task

When sustained lack of progress is detected, returns to outer loop to update Task Ledger and generate a fresh plan. This is **strategic course-correction**.

#### Inner Loop: Progress Ledger Management
Tracks:
- Current task progress
- Task assignments to specific agents

At each step, the Orchestrator:
1. Creates/updates Progress Ledger
2. Self-reflects on task progress
3. Checks if task is completed
4. If not done, delegates subtask to appropriate agent
5. After agent finishes, updates Progress Ledger
6. Repeats

#### Stall Detection & Re-planning
The Orchestrator continuously makes one of three decisions:
1. **Task complete** → terminate
2. **Progress being made** → delegate next subtask
3. **Progress stalled** (stall count > 2) → return to outer loop and re-plan

### AutoGen Patterns (5 Patterns)

| Pattern | Mechanism |
|---------|-----------|
| **Sequential** | Pipeline: Agent 1 → Agent 2 → Agent N |
| **Concurrent** | Parallel workers on independent subtasks |
| **Handoff** | Agent A transfers control + context to Agent B |
| **Selector Group Chat** | Centralized selector chooses who speaks next in shared context |
| **Swarm** | Localized, tool-based selector in shared context |
| **GraphFlow** | DAG-based execution with directed graph of agents |

### Key Design Decisions
- Model-agnostic (can mix GPT-4o, o1-preview, etc.)
- Plug-and-play agent design (add/remove without restructuring)
- Agents can post on social media, email people, draft FOI requests (safety concern!)
- Recommends sandboxed Docker containers + human-in-the-loop

---

## 3. Google: ADK + A2A Protocol

### Google ADK Orchestration Patterns

ADK provides two generations:
1. **Template workflows** (original) — deterministic, predefined execution
2. **Graph-based + Dynamic workflows** (ADK 2.0) — more flexible

#### Sequential Agent
- Runs sub-agents in strict, predefined order
- Data flows through **shared session state** via `output_key` mechanism
- Each agent declares `output_key` (e.g., `output_key="generated_code"`)
- Downstream agents reference prior outputs with `{generated_code}` in instructions
- All sub-agents share single `InvocationContext` within one turn
- Includes temporary namespace (`temp:`) for ephemeral data

Example pipeline:
```
CodeWriterAgent (output_key="generated_code")
  → CodeReviewerAgent (output_key="review_comments", reads {generated_code})
  → CodeRefactorerAgent (reads {generated_code} + {review_comments})
```

#### Parallel Agent
- Executes sub-agents **concurrently**
- **No automatic sharing** of conversation history or state between branches
- Each sub-agent operates in its own execution branch
- Results collected after all complete (order may be non-deterministic)
- State sharing approaches:
  - Shared `InvocationContext` (with locks for concurrency)
  - External database/message queue
  - Post-processing collection
- Typical pattern: Parallel → Sequential merger
  ```
  SequentialAgent([
      ParallelAgent([researcher1, researcher2, researcher3]),
      MergerAgent  # reads all output_keys and synthesizes
  ])
  ```

#### Loop Agent
- Repeatedly executes sub-agents in sequence
- **Does NOT inherently know when to stop** — must implement termination
- Two termination strategies:
  1. `max_iterations` parameter (safety bound)
  2. Sub-agent calls `exit_loop` tool (sets `actions.escalate = True`)
- Typical pattern: Writer → Loop(Critic → Refiner)
  - Critic reviews, outputs feedback or completion phrase
  - Refiner reads critique; if completion phrase → calls `exit_loop`

### A2A Protocol (Agent-to-Agent)

#### Agent Cards
JSON metadata documents at `/.well-known/agent.json` describing:
- Agent identity and capabilities
- Endpoint URL
- Skills list
- Authentication requirements
- Custom protocol extensions

Discovery is automatic — clients fetch Agent Cards to learn about available agents.

#### Task Lifecycle
Tasks have defined states:
- **Interrupted states**: `input-required`, `auth-required`
- **Terminal states**: `completed`, `canceled`, `rejected`, `failed`

Key rules:
- Once terminal → cannot restart (must create new task in same context)
- Once task created → agent only returns Task objects
- Once complete → no more messages can be sent

#### Key Data Structures
| Structure | Purpose |
|-----------|---------|
| **Task** | Stateful unit of work with unique ID and lifecycle |
| **Message** | Single turn of communication (role: "user" or "agent") |
| **Part** | Content container: text, raw binary, URL, or structured JSON data |
| **Artifact** | Concrete output (document, image, data) with artifactId |
| **contextId** | Groups related tasks/messages together |

#### Communication Patterns
1. **Request/Response** — polling for long-running tasks
2. **Streaming via SSE** — real-time incremental results
3. **Push Notifications** — async webhook for disconnected scenarios

#### Protocol
- HTTP(S) transport
- JSON-RPC 2.0 payload format
- Also supports gRPC binding
- Auth via OAuth, API keys, mTLS (declared in Agent Card)

---

## 4. OpenAI: Agents SDK + Swarm

### Swarm (Experimental, Educational)
The predecessor — extremely minimal:

**Two primitives only:**
1. **Agents** — encapsulate `instructions` + `tools`
2. **Handoffs** — function returns another Agent to transfer control

**Stateless design** — no server-side memory. All state in messages + context variables the caller manages.

**Agent Loop** (`client.run()`):
1. Get completion from current Agent
2. Execute tool calls, append results
3. Switch Agent if handoff triggered
4. Update context variables
5. If no new function calls → return to caller

**Handoffs** — when a function returns an Agent, execution transfers. System prompt swaps but chat history persists.

**Context Variables** — shared state dict threaded through execution. Functions can read/modify them.

### OpenAI Agents SDK (Production Replacement)

**Three core primitives:**
1. **Agents** — LLMs with instructions and tools
2. **Handoffs** — agent-to-agent delegation
3. **Guardrails** — input/output validation (run in parallel with agent, fail-fast)

#### Handoffs (Deep Detail)

Handoffs are **represented as tools to the LLM**. When "Refund Agent" has a handoff, the LLM sees `transfer_to_refund_agent` as a tool. Model decides to invoke it like any function call.

**Configuration options:**
| Parameter | Purpose |
|-----------|---------|
| `agent` | Target agent receiving control |
| `tool_name_override` | Custom tool name |
| `tool_description_override` | Custom description |
| `on_handoff` | Callback fired on invocation |
| `input_type` | Pydantic schema for tool-call arguments |
| `input_filter` | Transform conversation history for next agent |
| `is_enabled` | Dynamic enable/disable at runtime |

**Context passing:**
- Default: full conversation history transfers
- `input_filter` function transforms `HandoffInputData` (input_history, pre_handoff_items, new_items)
- Pre-built filters like `remove_all_tools` strip tool calls from history
- `nest_handoff_history` (beta) collapses prior transcripts into summary

#### Agents-as-Tools
Manager keeps control, calls specialists via `Agent.as_tool()`:
- Manager owns the user-facing conversation
- Specialists handle bounded subtasks
- Manager combines outputs from multiple specialists
- Shared guardrails enforced in one place

#### Runner
Entry point for execution. Handles:
- Per-run orchestration controls
- Conversation state
- Tool invocation loop
- Streaming

#### Sessions
Persistent memory across runs:
- SQLAlchemy, SQLite, Redis, MongoDB, Encrypted, Dapr backends

#### Code-Based Orchestration Patterns
| Pattern | Description |
|---------|-------------|
| Structured outputs | Classify task → route programmatically |
| Chaining | research → outline → draft → critique → improve |
| Evaluator loops | while loop with evaluator until criteria met |
| Parallel execution | `asyncio.gather` for concurrent agents |

---

## 5. Anthropic: Multi-Agent Research System

### Orchestrator-Worker Architecture
Lead agent (LeadResearcher) coordinates specialized subagents in parallel.

#### Lead Agent Responsibilities
1. **Query analysis and strategy** — analyzes query, develops strategy, spawns subagents
2. **Planning via extended thinking** — uses thinking to plan approach, assess tools, determine complexity
3. **Delegation with detailed instructions** — each subagent gets objective, output format, tool guidance, task boundaries
4. **Synthesis and iteration** — synthesizes results, decides if more research needed
5. **Effort scaling** — explicit rules:
   - Simple: 1 agent, 3-10 tool calls
   - Comparison: 2-4 subagents, 10-15 calls each
   - Complex: 10+ subagents with clearly divided responsibilities

#### Parallel Exploration (Two Levels)
- **Subagent-level**: 3-5 subagents in parallel
- **Tool-level**: 3+ tools in parallel per subagent
- Cuts research time by up to 90%

#### Context Management Strategy
1. **Memory persistence** — LeadResearcher saves plan to Memory (context window > 200K tokens = truncation)
2. **Summarization and handoffs** — agents summarize completed work, store in external memory before new tasks
3. **Retrieval of stored context** — agents retrieve stored context from memory
4. **Filesystem outputs** — subagents store work externally, pass lightweight references back

#### Key Design Patterns

**Dynamic vs. static retrieval**: Multi-step search that dynamically finds, adapts, analyzes (not static RAG)

**Breadth-first search**: Start with short, broad queries, evaluate what's available, then narrow

**Interleaved thinking**: Subagents evaluate quality after tool results, identify gaps, refine queries

**Error resilience**: Resume from where agent was when errors occurred. Let agent know when tool fails, let it adapt.

**Tool self-improvement**: Tool-testing agent rewrites tool descriptions → 40% decrease in task completion time for future agents

**Token budget as primary lever**: Token usage explains 80% of variance in evaluation scores

**CitationAgent**: Dedicated agent for final citation processing

---

## 6. AWS: Bedrock Multi-Agent Collaboration

### Supervisor-Collaborator Model
- Designate one agent as **Supervisor**
- Associate one or more **Collaborator** agents
- Supervisor uses instructions to understand structure and role of each collaborator
- Hierarchical collaboration model for synchronous real-time responses

### Key Design Rules
- Minimize overlapping responsibilities between agents
- Each agent optimized for specific use case
- All agents have full Bedrock capabilities (tools, action groups, knowledge bases, guardrails)
- Supervisor automatically creates and executes plan across collaborators
- Routes relevant requests to appropriate collaborator

---

## 7. Open-Source: CrewAI

### Core Structure
- **Crew** — top-level orchestrator, defines workflow strategy
- **Agent** — individual workers with roles, goals, backstories
- **Task** — discrete work units assigned to agents

### Orchestration Patterns

#### Sequential Process
Tasks run in order. Output from earlier tasks feeds into later ones.

#### Hierarchical Process
Manager agent coordinates, "delegating tasks and validating outcomes before proceeding." Requires `manager_llm` or `manager_agent`.

#### Planning Mode
All crew data sent to AgentPlanner before each iteration. Plan injected into task descriptions.

### Memory System
Three types:
- **Short-term** — current execution context
- **Long-term** — persisted across runs
- **Entity memory** — tracking specific entities

Shared at crew level. Embedder config powers operations.

### Execution Modes
| Mode | Method |
|------|--------|
| Sync | `kickoff()` |
| Batch sync | `kickoff_for_each()` |
| Native async | `akickoff()` |
| Native async batch | `akickoff_for_each()` |

### Checkpointing
State saved after configurable events (default: task completion). Stored as JSON or SQLite. Restore via `Crew.from_checkpoint()`.

### Tool Caching
`cache=True` stores tool results so identical calls don't re-execute.

---

## 8. Open-Source: LangGraph

### Graph-Based Execution
Inspired by Google's Pregel system. Discrete **super-steps**:
- Nodes running in parallel share same super-step
- Sequential nodes occupy separate ones
- Nodes start `inactive`, become `active` when receiving new state
- Terminates when all nodes `inactive` and no messages in transit

### State Management
Each state key has independent reducer:
- **Default (overwrite)**: updates replace existing value
- **Custom via `Annotated`**: e.g., `Annotated[list[str], add]` concatenates
- **`add_messages` reducer**: tracks message IDs, overwrites existing messages
- **`Overwrite` type**: bypass reducer, directly overwrite

Multiple schemas: `InputState`, `OutputState`, `OverallState`, `PrivateState`

### Multi-Agent Patterns
- **Supervisor**: `Command` with `goto` and `graph=Command.PARENT`
- **Swarm**: Conditional edges and `Send` objects for parallel fan-out
- **Hierarchical**: Subgraphs with parent navigation

### Checkpointing
- Saves state at super-step boundaries
- Task results inside nodes also checkpointed
- Resumption skips completed work
- State keys can be added/removed safely

### Human-in-the-Loop
`interrupt()` pauses execution. Caller resumes with `Command(resume="yes")`. Value passed to `resume` becomes return value of `interrupt()`.

---

## 9. Cross-Cutting Patterns

### Pattern 1: Orchestrator-Worker (Most Common)
Used by: Anthropic, Microsoft, AWS, OpenAI (agents-as-tools)
- Lead agent decomposes, plans, delegates
- Workers execute specialized subtasks
- Lead synthesizes results

### Pattern 2: Handoff Chain
Used by: OpenAI, AutoGen Swarm
- Agent A → Agent B → Agent C
- Each agent owns the conversation when active
- Full context transfers

### Pattern 3: Parallel Fan-Out + Merge
Used by: Google ADK, Anthropic, LangGraph
- Multiple agents run concurrently on independent subtasks
- Results merged by downstream agent or code

### Pattern 4: Evaluator Loop
Used by: OpenAI, Google ADK Loop, CrewAI
- Task agent runs in loop with evaluator
- Until output meets criteria or max iterations

### Pattern 5: Two-Ledger System (Magentic-One)
Used by: Microsoft only
- Task Ledger (facts, guesses, plan) — outer loop
- Progress Ledger (current state, assignments) — inner loop
- Stall detection triggers re-planning

### Context Management Techniques

| Technique | Used By | Effect |
|-----------|---------|--------|
| JIT context injection | ORCH, Anthropic | Only files in scope, not full codebase |
| Summarization | Anthropic, OpenAI | Compress completed work phases |
| External memory/filesystem | Anthropic, CrewAI | Store work externally, pass references |
| Session state keys | Google ADK, LangGraph | Shared state with output_key pattern |
| Context compression | ORCH, Anthropic | Every N cycles, compress completed tasks |
| Nested handoff history | OpenAI SDK | Collapse prior transcripts into summary |
| Separate context windows | Anthropic | Each subagent has own window, parallel exploration |

### Failure Handling Techniques

| Technique | Used By | Description |
|-----------|---------|-------------|
| Stall detection + re-plan | Microsoft | Count consecutive no-progress steps, trigger outer loop |
| Retry with feedback | ORCH, CrewAI | Inject failure reason into next attempt |
| Error resilience | Anthropic | Resume from where error occurred, let model adapt |
| Guardrails (fail-fast) | OpenAI | Validate inputs/outputs in parallel, terminate early |
| Escalation tools | Google ADK | Sub-agent calls `exit_loop` tool to signal completion |
| Checkpointing | LangGraph, CrewAI | Save state, resume without re-running completed work |

---

## 10. Key Takeaways for ORCH

### What ORCH Already Does Well
- Free LLM provider chain with fallback
- Agent detection across 12+ CLI agents
- tmux-based visual orchestration
- File-based task/result exchange
- JIT context injection
- Quality gate evaluation
- Session persistence (planned)

### What ORCH Should Adopt

#### From Magentic-One: Two-Ledger System
```
Task Ledger: facts, guesses, plan
Progress Ledger: current state, assignments
Stall detection → re-planning after N failures
```

#### From Anthropic: Effort Scaling
```
Simple task → 1 worker, few tool calls
Medium → 2-3 workers, parallel
Complex → 10+ workers with clearly divided responsibilities
```

#### From OpenAI SDK: Handoff Filters
```
When handing off task results between workers,
filter conversation history to remove noise.
Only pass: task instruction, changed files, test results.
```

#### From Google ADK: Output Key Pattern
```
Each worker declares output_key.
Downstream workers reference {output_key} in instructions.
Shared session state for inter-worker data flow.
```

#### From LangGraph: Checkpointing at Super-Steps
```
Save state after each task completion.
Resume without re-running completed tasks.
Safe for crashes and pauses.
```

#### From Anthropic: Tool Self-Improvement
```
When a tool fails, rewrite its description.
40% decrease in future task completion time.
```

#### From CrewAI: Planning Mode
```
Before execution, send all task data to planner.
Plan injected into each task description.
```

#### From A2A Protocol: Agent Cards
```
Each worker agent exposes a card describing:
- capabilities, skills, auth requirements
- how to communicate with it
Enable automatic agent discovery and routing.
```

#### From All: Parallel Fan-Out + Sequential Merge
```
Phase 1: Parallel workers on independent subtasks
Phase 2: Sequential merger agent synthesizes results
Wrap in SequentialAgent([ParallelAgent([...]), MergerAgent])
```

### Architecture Recommendation for ORCH

```
User Goal
    ↓
ORCH Brain (free LLM: Gemini → Cerebras → Groq → OpenRouter → Ollama)
    ↓
GoalDecomposer (uses Task Ledger pattern from Magentic-One)
    ↓
DependencyScheduler (DAG with parallel fan-out)
    ↓
┌─────────────────────────────────────────────┐
│  Parallel Phase:                            │
│  Worker 0: kiro        task t001           │
│  Worker 1: opencode    task t002           │  ← independent tasks
│  Worker 2: aider       task t003           │
├─────────────────────────────────────────────┤
│  Sequential Phase:                          │
│  Merger: any agent     synthesize results   │  ← after all parallel done
└─────────────────────────────────────────────┘
    ↓
QualityGate (structural + file + LLM eval, inspired by OpenAI guardrails)
    ↓
ProgressLedger (track state, detect stalls, trigger re-plan)
    ↓
StateStore (checkpointing, resume without re-running)
    ↓
Human Review (never auto-merge)
```

---

## Sources

- Microsoft Magentic-One: https://www.microsoft.com/en-us/research/blog/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/
- Google ADK: https://adk.dev/agents/workflow-agents/
- A2A Protocol: https://a2a-protocol.org/latest/specification/
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- OpenAI Swarm: https://github.com/openai/swarm
- Anthropic Multi-Agent: https://www.anthropic.com/engineering/built-multi-agent-research-system
- CrewAI: https://docs.crewai.com/concepts/crews
- LangGraph: https://docs.langchain.com/oss/python/langgraph/graph-api
- AWS Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-multi-agent-collaboration.html
