"""
router.py
==========
LLM router for agentorchestr orchestration intelligence.

Provider priority (the first one with credentials wins; rest are fallbacks):
  1. Anthropic direct         — paid, but cache reads cost 10% of input price
                                (set ANTHROPIC_API_KEY to enable)
  2. Cerebras                 — 1M tokens/day free on Llama 3.1 70B (~2100 tok/s)
  3. Groq                     — 30 RPM free on Llama 70B (~315 tok/s)
  4. Google Gemini AI Studio  — 1500 req/day free, 1M-token context
  5. OpenRouter               — 28+ free models including DeepSeek R1
                                (forwards cache_control to Anthropic-shaped models)

Token efficiency:
  Providers that support Anthropic's prompt-caching protocol
  (anthropic, openrouter -> claude-* models) get cache_control={type:ephemeral}
  on the system prompt automatically. This costs 1.25x base price on first
  write and 10% (90% off) on every subsequent read within the 5-min TTL.
  Anthropic engineering blog: ~80% of multi-agent perf variance is explained
  by token usage, so this is the single highest-leverage optimization.

Resilience:
  - Sequential fallback on any error.
  - Rate-limit awareness: HTTP 429 / quota errors put a provider on
    cooldown (default 60s) so we stop hammering it.
  - Each call records `last_provider`, `last_latency_ms`.

Note: agentorchestr no longer ships an Ollama / local-LLM fallback.  Local models
were too slow for the orchestration latency budget — every supervisor
turn pays the cost.  Stick to the hosted free tiers above; they're all
sub-second to first token.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import httpx

log = logging.getLogger(__name__)


PROVIDERS = [
    {
        "name": "anthropic",
        "env": "ANTHROPIC_API_KEY",
        "limit": "paid — cache hits 90% off (5m TTL) or 90% off + 2x write (1h TTL)",
        "model_plan": os.environ.get("ANTHROPIC_PLAN_MODEL", "claude-sonnet-4-5"),
        "model_compress": os.environ.get("ANTHROPIC_COMPRESS_MODEL", "claude-haiku-4-5"),
        "url": "https://api.anthropic.com/v1/messages",
        "type": "anthropic",
        "cache_control": True,
    },
    {
        "name": "cerebras",
        "env": "CEREBRAS_API_KEY",
        "limit": "1M tokens/day free (~2100 tok/s)",
        "model_plan": "llama3.1-70b",
        "model_compress": "llama3.1-8b",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "type": "openai_compat",
        "cache_control": False,
    },
    {
        "name": "groq",
        "env": "GROQ_API_KEY",
        "limit": "30 RPM free (~315 tok/s)",
        "model_plan": "llama-3.3-70b-versatile",
        "model_compress": "llama-3.1-8b-instant",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "type": "openai_compat",
        "cache_control": False,
    },
    {
        "name": "gemini",
        "env": "GEMINI_API_KEY",
        "limit": "1500 req/day free, 1M-token context",
        "model_plan": "gemini-2.5-flash",
        "model_compress": "gemini-2.5-flash",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "type": "gemini",
        "cache_control": False,
    },
    {
        "name": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "limit": "20 RPM free (28+ models)",
        "model_plan": "deepseek/deepseek-r1:free",
        "model_compress": "meta-llama/llama-3.1-8b-instruct:free",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "type": "openai_compat",
        # OpenRouter forwards cache_control to Anthropic-shaped models, but
        # silently ignores it for others — safe to always include.
        "cache_control": True,
    },
]

PLAN_SYSTEM_PROMPT = """You are an expert software project orchestrator.
Given a goal and project context, decompose it into concrete coding tasks.
Each task must be self-contained, scoped to specific files, and achievable by
a single coding agent in one session.

Return ONLY valid JSON, no markdown, no preamble:
{
  "tasks": [
    {
      "id": "t001",
      "type": "implement|test|refactor|review",
      "instruction": "...(max 200 words, imperative voice)...",
      "file_scope": ["relative/path/to/file.py"],
      "depends_on": [],
      "agent_hint": "any|claude|kiro|aider",
      "pass_criteria": "Tests pass, no lint errors, feature works as described",
      "budget_turns": 20
    }
  ],
  "summary": "One sentence overview"
}"""

COMPRESS_SYSTEM_PROMPT = """Summarize completed tasks in 3 bullet points MAX.
Focus only on: what changed, what tests passed, what remains blocked.
Return plain text, no markdown, under 150 words."""

EVAL_SYSTEM_PROMPT = """You are a code review evaluator.
Given a task specification and worker result, determine if the task passed.
Return ONLY valid JSON:
{"passed": true|false, "feedback": "brief reason if failed, empty if passed"}"""

# Default cooldown when a provider returns a quota / rate-limit error.
DEFAULT_COOLDOWN_S = 60.0


class _RateLimited(Exception):
    """Raised internally when a provider responds with 429 / quota exceeded."""
    def __init__(self, provider: str, retry_after: float):
        super().__init__(f"{provider} rate-limited (retry after {retry_after:.1f}s)")
        self.provider = provider
        self.retry_after = retry_after


class LLMRouter:
    """Routes LLM calls across hosted free providers with automatic fallback
    and rate-limit awareness.  No local-LLM fallback — see module docstring."""

    def __init__(self, cooldown_s: float = DEFAULT_COOLDOWN_S):
        self.providers: list[dict] = []
        self.active_provider: Optional[str] = None
        self.last_latency_ms: int = 0
        self._client = httpx.AsyncClient(timeout=60.0)
        self._cooldown_s = cooldown_s
        self._cooldowns: dict[str, float] = {}  # provider name -> available_at epoch

    async def init(self) -> None:
        """Probe all providers and build priority list from available ones."""
        for p in PROVIDERS:
            key = os.environ.get(p["env"]) if p["env"] else None
            if key:
                self.providers.append({**p, "_key": key})
        if self.providers:
            self.active_provider = self.providers[0]["name"]

    def has_any_free(self) -> bool:
        return len(self.providers) > 0

    def provider_status(self) -> dict:
        """Returns status of all known providers for display."""
        status = {}
        configured_names = {p["name"] for p in self.providers}
        now = time.time()
        for p in PROVIDERS:
            available = p["name"] in configured_names
            cooldown_remaining = max(0.0, self._cooldowns.get(p["name"], 0) - now)
            status[p["name"]] = {
                "available": available,
                "limit": p["limit"],
                "cooldown_s": round(cooldown_remaining, 1) if cooldown_remaining else 0,
            }
        return status

    async def plan(self, prompt: str, system: str = PLAN_SYSTEM_PROMPT) -> str:
        return await self._call_with_fallback(prompt, system, mode="plan")

    async def compress(self, prompt: str) -> str:
        return await self._call_with_fallback(prompt, COMPRESS_SYSTEM_PROMPT, mode="compress")

    async def evaluate(self, prompt: str) -> str:
        return await self._call_with_fallback(prompt, EVAL_SYSTEM_PROMPT, mode="compress")

    async def _call_with_fallback(self, prompt: str, system: str, mode: str) -> str:
        last_error: Optional[Exception] = None
        now = time.time()
        for provider in self.providers:
            cd_until = self._cooldowns.get(provider["name"], 0)
            if cd_until > now:
                last_error = _RateLimited(provider["name"], cd_until - now)
                log.debug("provider %s on cooldown for %.1fs",
                          provider["name"], cd_until - now)
                continue
            t0 = time.time()
            try:
                result = await self._call_provider(provider, prompt, system, mode)
                self.active_provider = provider["name"]
                self.last_latency_ms = int((time.time() - t0) * 1000)
                return result
            except _RateLimited as e:
                self._cooldowns[provider["name"]] = time.time() + e.retry_after
                last_error = e
                log.debug("provider %s rate-limited: retry_after=%.1fs",
                          provider["name"], e.retry_after)
                continue
            except httpx.HTTPStatusError as e:
                code = e.response.status_code if e.response is not None else "?"
                if e.response is not None and e.response.status_code in (401, 403, 500, 502, 503):
                    self._cooldowns[provider["name"]] = time.time() + 30
                last_error = e
                log.debug("provider %s HTTP %s: %s",
                          provider["name"], code, e, exc_info=True)
                continue
            except Exception as e:
                last_error = e
                # Most likely a JSON parse failure or a missing response
                # field — easy to mistake for "all providers failed" if
                # we don't surface the original error somewhere.
                log.debug("provider %s raised %s: %r",
                          provider["name"], type(e).__name__, e, exc_info=True)
                continue
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error!r}")

    async def _call_provider(self, provider: dict, prompt: str, system: str, mode: str) -> str:
        model_key = "model_plan" if mode == "plan" else "model_compress"
        model = provider[model_key]
        ptype = provider["type"]

        if ptype == "anthropic":
            return await self._call_anthropic(provider, prompt, system, model)
        if ptype == "gemini":
            return await self._call_gemini(provider, prompt, system, model)
        if ptype == "openai_compat":
            return await self._call_openai_compat(provider, prompt, system, model)
        raise ValueError(f"Unknown provider type: {ptype}")

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
        # Some APIs return seconds in the body
        body = (resp.text or "").lower()
        m = re.search(r"retry[- ]after[: ]+(\d+)", body)
        if m:
            return float(m.group(1))
        return DEFAULT_COOLDOWN_S

    async def _call_anthropic(self, provider: dict, prompt: str, system: str, model: str) -> str:
        """Anthropic Messages API with prompt-cache breakpoint on the system prompt.

        cache_control={"type":"ephemeral"} flags the system block as cacheable —
        on subsequent calls within the TTL (5 min default, 1h with header)
        Anthropic charges 10% of input price for the cached prefix tokens.
        See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        """
        headers = {
            "x-api-key": provider["_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 2048,
            "temperature": 0.1,
            # System is a list of blocks so we can mark the whole thing cacheable.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }
        r = await self._client.post(provider["url"], json=payload, headers=headers)
        if r.status_code == 429:
            raise _RateLimited(provider["name"], self._retry_after(r))
        r.raise_for_status()
        data = r.json()
        try:
            # content is a list of {type: "text", text: "..."} blocks
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Unexpected Anthropic response shape: {data}") from e

    async def _call_gemini(self, provider: dict, prompt: str, system: str, model: str) -> str:
        url = provider["url"].format(model=model, key=provider["_key"])
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.1},
        }
        r = await self._client.post(url, json=payload)
        if r.status_code == 429:
            raise _RateLimited(provider["name"], self._retry_after(r))
        r.raise_for_status()
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response shape: {data}") from e

    async def _call_openai_compat(self, provider: dict, prompt: str, system: str, model: str) -> str:
        headers = {"Authorization": f"Bearer {provider['_key']}", "Content-Type": "application/json"}
        # When the provider supports cache_control (e.g. OpenRouter -> Anthropic
        # models), put it on the system prompt as an array of content blocks.
        # Providers that don't understand the field will silently drop it.
        if provider.get("cache_control"):
            system_content: object = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        else:
            system_content = system
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
        }
        r = await self._client.post(provider["url"], json=payload, headers=headers)
        if r.status_code == 429:
            raise _RateLimited(provider["name"], self._retry_after(r))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()
