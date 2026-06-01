"""
memory — three-tier memory federation for ORCH
================================================

Three tiers, each backed by markdown that's git-friendly and human-editable:

    Global      ~/.orch/memory/
        global.md            preferences across all projects
        skills/              imported skill definitions (Phase-4)

    Project     <project>/.orch/memory/
        PROJECT.md           architecture, conventions (auto-loaded)
        CONVENTIONS.md       style/lint/test rules
        topics/*.md          one file per learned topic
        episodes/*.md        per-session journal entries

    Session     <worktree>/.orch/
        ledger.jsonl         append-only tool-call log (Phase-2 op_log)

Retrieval is hybrid:

  • FTS5 always-on.  SQLite ships with FTS5 in every install since 3.20,
    so this works everywhere with zero new dependencies.
  • sqlite-vec optional.  When the extension is loadable AND fastembed
    is installed we maintain a vector index alongside FTS5 and combine
    the two scores with reciprocal-rank fusion (cheap, parameter-free).
  • Recency boost.  A note from yesterday outranks a stale note from
    last month at equal relevance — matches how humans recall.

Public surface:

    MemoryFederation(project_root) - main entry point
        .load_static_context()      → str (CLAUDE-md style preamble)
        .index_topic(name, body)    → write + index a topic
        .index_episode(sid, body)   → record a session journal
        .retrieve(query, k=5)       → list[MemoryHit]

The federation is designed to feed into the *cacheable* part of a
worker's prompt (Phase-4 cache_control).  Stable bytes first, then
the dynamic task at the end.
"""
from .federation import MemoryFederation, MemoryHit

__all__ = ["MemoryFederation", "MemoryHit"]
