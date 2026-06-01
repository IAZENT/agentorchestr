"""
research.py — web research for ORCH workers
=============================================

Two layers:

  1. SEARCH (DuckDuckGo, no API key required, free):
       results = await search(query, k=5)
     Returns [{title, snippet, url, source}].

  2. FETCH (httpx, follow redirects, strip HTML):
       text = await fetch_url(url, max_chars=4000)
     Returns plain-text content suitable for prompt injection.

  3. RESEARCH (orchestrated):
       payload = await research(query, k=5, fetch_top_n=2)
     Runs search, fetches the top N results' bodies, returns a
     structured dict the supervisor / workers can drop into a prompt.

Designed to be cheap and skippable:
  - DuckDuckGo via the optional `ddgs` package; if missing, search()
    returns []  and the caller still gets a usable empty payload.
  - All HTTP errors are swallowed.  Research must never break a
    supervisor run.

Used by:
  - mcp_bridge.web_research()  → tool the lead/researcher worker calls.
  - memory.federation.cache_research() persists the payload as a note
    so future workers retrieve it from memory (90%-cache-hit territory).
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

try:
    from ddgs import DDGS  # type: ignore
    HAS_DDGS = True
except ImportError:  # pragma: no cover - optional dep
    HAS_DDGS = False
    DDGS = None  # type: ignore[assignment]


DEFAULT_TIMEOUT = 8.0
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; ORCH-research/1.0)"


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str = "duckduckgo"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchPayload:
    query: str
    fetched_at: float
    results: list[SearchResult] = field(default_factory=list)
    fetched_bodies: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "fetched_at": self.fetched_at,
            "results": [r.to_dict() for r in self.results],
            "fetched_bodies": dict(self.fetched_bodies),
            "error": self.error,
        }

    def render_markdown(self, max_chars_per_body: int = 1500) -> str:
        """Stable, deterministic markdown — designed to land in the
        cacheable prefix of a worker prompt."""
        if not self.results and not self.error:
            return f"## Web research: {self.query}\n\n(no results)"
        parts = [f"## Web research: {self.query}"]
        if self.error:
            parts.append(f"_error: {self.error}_")
        for i, r in enumerate(self.results, 1):
            parts.append(f"\n### {i}. {r.title}")
            if r.snippet:
                parts.append(r.snippet)
            if r.url:
                parts.append(f"<{r.url}>")
            body = self.fetched_bodies.get(r.url)
            if body:
                parts.append("```")
                parts.append(body[:max_chars_per_body])
                parts.append("```")
        return "\n".join(parts)


# ── search layer ──────────────────────────────────────────────────────

async def search(query: str, k: int = 5) -> list[SearchResult]:
    """DuckDuckGo text search.  Returns [] when ddgs isn't installed
    or the search fails for any reason."""
    if not HAS_DDGS or not query.strip():
        return []
    # ddgs is sync — push to a thread so we don't block the event loop.
    return await asyncio.to_thread(_search_sync, query, k)


def _search_sync(query: str, k: int) -> list[SearchResult]:
    out: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                out.append(SearchResult(
                    title=(r.get("title") or "")[:200],
                    snippet=(r.get("body") or "")[:400],
                    url=(r.get("href") or "").strip(),
                ))
    except Exception:
        return []
    return out


# ── fetch layer ───────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


async def fetch_url(url: str, *, max_chars: int = 4000,
                    timeout: float = DEFAULT_TIMEOUT,
                    client: Optional[httpx.AsyncClient] = None) -> str:
    """GET the URL, follow redirects, strip HTML, return plain text.

    Returns "" on any error.  Total budget is `timeout` seconds.
    """
    if not url:
        return ""
    own = client is None
    if own:
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
    try:
        try:
            r = await client.get(url)
        except (httpx.HTTPError, OSError):
            return ""
        if r.status_code >= 400:
            return ""
        ct = r.headers.get("content-type", "").lower()
        text = r.text
        if "html" in ct or "<html" in text[:500].lower():
            text = _TAG_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text[:max_chars]
    finally:
        if own:
            await client.aclose()


# ── high-level research ───────────────────────────────────────────────

async def research(query: str, *, k: int = 5, fetch_top_n: int = 2,
                   max_chars_per_body: int = 4000) -> ResearchPayload:
    """Search + fetch top-N bodies in one shot.

    The combination is what makes prompts useful — snippets alone are
    too thin, full pages are too noisy.  fetch_top_n=2 is a reasonable
    default: the lead gets ~6-8KB of grounded context for ~2s of latency.
    """
    payload = ResearchPayload(query=query, fetched_at=time.time())
    if not query.strip():
        payload.error = "empty query"
        return payload

    payload.results = await search(query, k=k)
    if not payload.results:
        payload.error = payload.error or "no results (or ddgs not installed)"
        return payload

    # Fetch the top N bodies in parallel.
    targets = [r.url for r in payload.results[:fetch_top_n] if r.url]
    if not targets:
        return payload

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    ) as client:
        bodies = await asyncio.gather(
            *(fetch_url(u, max_chars=max_chars_per_body, client=client) for u in targets),
            return_exceptions=True,
        )
    for url, body in zip(targets, bodies):
        if isinstance(body, str) and body:
            payload.fetched_bodies[url] = body

    return payload
