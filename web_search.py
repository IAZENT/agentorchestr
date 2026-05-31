"""
web_search.py — Free Web Search for Agents
=============================================
Provides web search capability to coding agents using DuckDuckGo (no API key).
Agents can search for latest docs, Stack Overflow answers, npm packages, etc.

Usage from agent task:
  The task instruction can mention "search the web for X" and the orchestrator
  will inject relevant search results into the task context.
"""

from typing import Optional

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    Returns list of {title, snippet, url} results.
    No API key required.
    """
    if not HAS_DDGS:
        return []

    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "")[:100],
                    "snippet": r.get("body", "")[:300],
                    "url": r.get("href", ""),
                    "source": "DuckDuckGo",
                })
            return results
    except Exception:
        return []


async def search_and_format(query: str, max_results: int = 5) -> str:
    """Search and return formatted results for injection into agent context."""
    results = await web_search(query, max_results)
    if not results:
        return f"No web results found for: {query}"

    parts = [f"## Web Search Results for: {query}\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"{i}. **{r['title']}**")
        parts.append(f"   {r['snippet']}")
        if r['url']:
            parts.append(f"   Source: {r['url']}")
        parts.append("")

    return "\n".join(parts)


async def extract_search_queries_from_task(task_instruction: str) -> list[str]:
    """
    Extract search queries from a task instruction using heuristics.
    Returns list of queries that would help the agent complete the task.
    """
    research_keywords = [
        "latest", "recent", "new version", "documentation", "docs",
        "how to", "best practice", "example", "tutorial",
        "npm package", "pip package", "library", "framework", "API reference",
        "migration guide", "changelog", "release notes", "stack overflow",
        "search", "research", "look up", "find out",
    ]

    queries = []
    lower = task_instruction.lower()

    for keyword in research_keywords:
        if keyword in lower:
            idx = lower.index(keyword)
            start = max(0, idx - 30)
            end = min(len(task_instruction), idx + len(keyword) + 60)
            context = task_instruction[start:end].strip()
            queries.append(context)
            break

    return queries


async def enrich_task_with_research(task: dict) -> dict:
    """
    If a task mentions research-related keywords, search the web
    and inject results into the task context.
    """
    instruction = task.get("instruction", "")
    queries = await extract_search_queries_from_task(instruction)

    if not queries:
        return task

    research_parts = []
    for query in queries[:2]:
        results = await search_and_format(query, max_results=3)
        research_parts.append(results)

    if research_parts:
        research_context = "\n\n".join(research_parts)
        task["_research_context"] = research_context

    return task
