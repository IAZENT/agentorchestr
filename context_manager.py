"""
core/context_manager.py
========================
Three-tier memory system inspired by Claude Code's CLAUDE.md + auto memory.

Tier 1: Session memory (in-process, ephemeral, compressed every N cycles)
Tier 2: Project memory (.orch/MEMORY.md + topic files, persistent)
Tier 3: Global memory (~/.orch/global_memory.md, persistent across projects)

Also handles JIT context injection and output key substitution.
"""

import json
import os
from pathlib import Path
from typing import Optional

from router import LLMRouter

MEMORY_DIR = ".orch/memory"
MEMORY_INDEX = ".orch/MEMORY.md"
CONVENTIONS_FILE = ".orch/conventions.md"
GLOBAL_MEMORY = os.path.expanduser("~/.orch/global_memory.md")

# Token budgets
MAX_FILE_CHARS = 3000
MAX_CONTEXT_TOKENS = 4000  # ~4000 tokens for task context


class ContextManager:
    """Three-tier memory and context management."""

    def __init__(self, project_root: str = ".", llm_router: Optional[LLMRouter] = None):
        self.project_root = Path(project_root)
        self.llm = llm_router
        self._session_memory: list[str] = []
        self._project_memory: str = ""
        self._global_memory: str = ""
        self._loaded = False
        self._output_store: dict[str, str] = {}
        self._token_usage: dict[str, int] = {}

    async def load_context(self):
        """Load all memory tiers. Call once at session start."""
        if self._loaded:
            return

        # Tier 2: Project memory
        self._project_memory = self._read_project_memory()

        # Tier 3: Global memory
        self._global_memory = self._read_global_memory()

        self._loaded = True

    async def get_task_context(self, task: dict) -> str:
        """
        Build JIT context for a task.
        Includes: file contents, project conventions, relevant memories,
        compressed completed-task summary, output store.
        """
        parts = []

        # Global memory (cross-project learnings)
        if self._global_memory:
            parts.append(f"## Global Knowledge\n{self._global_memory[:500]}")

        # Project conventions
        conventions = self._read_conventions()
        if conventions:
            parts.append(f"## Project Conventions\n{conventions[:500]}")

        # Relevant topic memories
        relevant_memories = self._find_relevant_memories(task)
        if relevant_memories:
            parts.append(f"## Relevant Memories\n{relevant_memories[:500]}")

        # Completed task summary
        output_store = task.get("_output_store", {})
        if output_store:
            output_lines = [f"- {k}: {v}" for k, v in output_store.items()]
            parts.append(f"## Previous Task Outputs\n" + "\n".join(output_lines))

        # JIT file injection (only files in scope)
        file_scope = task.get("file_scope", [])
        if file_scope:
            file_context = self._build_jit_context(file_scope)
            if file_context:
                parts.append(f"## Relevant Files\n{file_context}")

        # Output key substitution in instruction
        instruction = task.get("instruction", "")
        if output_store:
            for key, value in output_store.items():
                placeholder = "{" + key + "}"
                if placeholder in instruction:
                    instruction = instruction.replace(placeholder, value)

        if instruction:
            parts.append(f"## Task Instruction\n{instruction}")

        return "\n\n".join(parts)

    async def compress_completed(self, completed_tasks: list[dict]) -> str:
        """
        Compress completed task history into a brief summary.
        Uses the cheapest available LLM.
        """
        if not completed_tasks:
            return ""

        if not self.llm:
            # Fallback: simple text summary
            lines = []
            for t in completed_tasks:
                lines.append(f"- {t.get('task_id', '?')}: {t.get('summary', 'completed')[:80]}")
            return "\n".join(lines)

        # Build compression prompt
        task_text = "\n".join(
            f"- {t.get('task_id', '?')} ({t.get('type', '?')}): {t.get('summary', '')[:100]}"
            for t in completed_tasks
        )

        prompt = f"""Summarize these completed tasks in 3 bullet points MAX.
Focus only on: what changed, what tests passed, what remains blocked.
Return plain text, no markdown, under 150 words.

Completed tasks:
{task_text}"""

        try:
            return await self.llm.compress(prompt)
        except Exception:
            # Fallback
            return f"Completed {len(completed_tasks)} tasks"

    async def save_session_memory(self, session_id: str, learnings: str):
        """Save key learnings from a session to project memory."""
        if not learnings:
            return

        memory_dir = self.project_root / MEMORY_DIR
        memory_dir.mkdir(parents=True, exist_ok=True)

        # Write session learnings
        session_file = memory_dir / f"session_{session_id}.md"
        with open(session_file, "w") as f:
            f.write(f"# Session {session_id} Learnings\n\n{learnings}\n")

        # Update MEMORY.md index
        index_path = self.project_root / MEMORY_INDEX
        index_path.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if index_path.exists():
            existing = index_path.read_text()

        # Add entry if not already present
        entry = f"- [Session {session_id}](memory/session_{session_id}.md)\n"
        if entry not in existing:
            with open(index_path, "a") as f:
                f.write(entry)

    def _read_project_memory(self) -> str:
        """Read project-level memory from .orch/MEMORY.md."""
        index_path = self.project_root / MEMORY_INDEX
        if not index_path.exists():
            return ""

        try:
            content = index_path.read_text()
            # Only load first 200 lines (like Claude Code)
            lines = content.split("\n")[:200]
            return "\n".join(lines)
        except OSError:
            return ""

    def _read_conventions(self) -> str:
        """Read project conventions from .orch/conventions.md."""
        conv_path = self.project_root / CONVENTIONS_FILE
        if not conv_path.exists():
            return ""

        try:
            return conv_path.read_text()[:1000]
        except OSError:
            return ""

    def _read_global_memory(self) -> str:
        """Read global cross-project memory."""
        if not os.path.exists(GLOBAL_MEMORY):
            return ""

        try:
            with open(GLOBAL_MEMORY) as f:
                lines = f.readlines()[:100]
            return "".join(lines)
        except OSError:
            return ""

    def _find_relevant_memories(self, task: dict) -> str:
        """Find memory topic files relevant to the task's file scope."""
        memory_dir = self.project_root / MEMORY_DIR
        if not memory_dir.exists():
            return ""

        relevant = []
        file_scope = task.get("file_scope", [])
        task_type = task.get("type", "")

        for mem_file in memory_dir.glob("*.md"):
            if mem_file.name.startswith("session_"):
                continue  # Skip session files

            try:
                content = mem_file.read_text()[:300]
                # Simple relevance check: if any file scope keywords appear
                if any(f.split("/")[-1].split(".")[0] in content.lower() for f in file_scope):
                    relevant.append(content)
                elif task_type and task_type in content.lower():
                    relevant.append(content)
            except OSError:
                continue

        return "\n---\n".join(relevant[:3])  # Max 3 relevant memories

    def build_jit_context(self, task: dict) -> str:
        """
        Build JIT context for a task (synchronous version for orchestrator).
        Returns file contents + conventions + memories + research for the task's scope.
        """
        parts = []

        # Project conventions
        conventions = self._read_conventions()
        if conventions:
            parts.append(f"## Project Conventions\n{conventions[:500]}")

        # Relevant topic memories
        relevant_memories = self._find_relevant_memories(task)
        if relevant_memories:
            parts.append(f"## Relevant Memories\n{relevant_memories[:500]}")

        # Web research context (if enriched by web_search module)
        research = task.get("_research_context", "")
        if research:
            parts.append(research)

        # JIT file injection
        file_scope = task.get("file_scope", [])
        if file_scope:
            file_context = self._build_jit_context(file_scope)
            if file_context:
                parts.append(f"## Relevant Files\n{file_context}")

        return "\n\n".join(parts)

    def resolve_output_refs(self, text: str) -> str:
        """Resolve {output_key} references in text using stored outputs."""
        for key, value in self._output_store.items():
            placeholder = "{" + key + "}"
            if placeholder in text:
                text = text.replace(placeholder, value)
        return text

    def set_output_key(self, task_id: str, value: str):
        """Store an output value for a completed task."""
        self._output_store[task_id] = value

    def compress_session(self, summary: str):
        """Replace session memory with a compressed summary."""
        self._session_memory = [summary]

    def track_tokens(self, category: str, count: int):
        """Track token usage by category."""
        self._token_usage[category] = self._token_usage.get(category, 0) + count

    def get_token_usage(self) -> dict:
        """Get token usage breakdown."""
        total = sum(self._token_usage.values())
        return {**self._token_usage, "total": total}

    def _build_jit_context(self, file_scope: list[str]) -> str:
        """
        Build JIT context by reading only the files in scope.
        Each file truncated to MAX_FILE_CHARS.
        """
        parts = []
        total_chars = 0

        for rel_path in file_scope:
            if total_chars >= MAX_CONTEXT_TOKENS * 3:  # rough char-to-token ratio
                break

            full_path = self.project_root / rel_path
            if not full_path.exists():
                parts.append(f"### {rel_path}\n[File not found]")
                continue

            try:
                content = full_path.read_text()[:MAX_FILE_CHARS]
                total_chars += len(content)
                parts.append(f"### {rel_path}\n```\n{content}\n```")
            except OSError:
                parts.append(f"### {rel_path}\n[Could not read file]")

        return "\n\n".join(parts)
