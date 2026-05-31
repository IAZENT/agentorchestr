"""
router.py
==========
Zero-cost LLM router for orchestration intelligence.

Provider priority (all free, no credit card required):
  1. Google Gemini AI Studio  — 1500 req/day, Gemini 2.5 Flash
  2. Cerebras                 — 1M tokens/day on Llama 3.1 70B
  3. Groq                     — 30 RPM on Llama 70B (315 tokens/sec)
  4. OpenRouter               — 28+ free models including DeepSeek R1
  5. Ollama (local)           — fully offline fallback (no quota at all)

Resilience:
  - Sequential fallback on any error.
  - Rate-limit awareness: HTTP 429 / quota errors put a provider on
    cooldown (default 60s) so we stop hammering it.
  - Each call records `last_provider`, `last_latency_ms`.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import httpx


PROVIDERS = [
    {
        "name": "gemini",
        "env": "GEMINI_API_KEY",
        "limit": "1500 req/day free",
        "model_plan": "gemini-2.5-flash",
        "model_compress": "gemini-2.5-flash",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "type": "gemini",
    },
    {
        "name": "cerebras",
        "env": "CEREBRAS_API_KEY",
        "limit": "1M tokens/day free",
        "model_plan": "llama3.1-70b",
        "model_compress": "llama3.1-8b",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "type": "openai_compat",
    },
    {
        "name": "groq",
        "env": "GROQ_API_KEY",
        "limit": "30 RPM free",
        "model_plan": "llama-3.3-70b-versatile",
        "model_compress": "llama-3.1-8b-instant",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "type": "openai_compat",
    },
    {
        "name": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "limit": "20 RPM free (28+ models)",
        "model_plan": "deepseek/deepseek-r1:free",
        "model_compress": "meta-llama/llama-3.1-8b-instruct:free",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "type": "openai_compat",
    },
    {
        "name": "ollama",
        "env": None,  # Detected via HTTP probe, not API key
        "limit": "local — unlimited",
        "model_plan": os.environ.get("OLLAMA_PLAN_MODEL", "llama3.1:latest"),
        "model_compress": os.environ.get("OLLAMA_COMPRESS_MODEL", "llama3.1:latest"),
        "url": (os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/") + "/api/chat",
        "type": "ollama",
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
    """Routes LLM calls across free providers with automatic fallback
    and rate-limit awareness.

    Ollama is OFF by default. Pass `use_ollama=True` (typically from the
    orchestrator's `--use-ollama` CLI flag) to include a running local
    Ollama server in the fallback chain. Environment variables alone do
    NOT enable it — explicit user intent is required.
    """

    def __init__(self, cooldown_s: float = DEFAULT_COOLDOWN_S, *, use_ollama: bool = False):
        self.providers: list[dict] = []
        self.active_provider: Optional[str] = None
        self.last_latency_ms: int = 0
        self._client = httpx.AsyncClient(timeout=60.0)
        self._cooldown_s = cooldown_s
        self._cooldowns: dict[str, float] = {}  # provider name -> available_at epoch
        self._use_ollama = bool(use_ollama)

    async def init(self) -> None:
        """Probe all providers and build priority list from available ones."""
        for p in PROVIDERS:
            if p["type"] == "ollama":
                if not self._use_ollama:
                    continue  # opt-in only
                if await self._ollama_alive(p["url"]):
                    self.providers.append({**p, "_key": ""})
                continue
            key = os.environ.get(p["env"]) if p["env"] else None
            if key:
                self.providers.append({**p, "_key": key})

        if self.providers:
            self.active_provider = self.providers[0]["name"]

    async def _ollama_alive(self, chat_url: str) -> bool:
        """Quick probe: hit /api/tags to confirm the local server responds."""
        base = chat_url.replace("/api/chat", "")
        try:
            r = await self._client.get(f"{base}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

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
                continue
            except httpx.HTTPStatusError as e:
                # Fall through to next provider; cooldown briefly on auth/server errors
                if e.response is not None and e.response.status_code in (401, 403, 500, 502, 503):
                    self._cooldowns[provider["name"]] = time.time() + 30
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error!r}")

    async def _call_provider(self, provider: dict, prompt: str, system: str, mode: str) -> str:
        model_key = "model_plan" if mode == "plan" else "model_compress"
        model = provider[model_key]
        ptype = provider["type"]

        if ptype == "gemini":
            return await self._call_gemini(provider, prompt, system, model)
        if ptype == "openai_compat":
            return await self._call_openai_compat(provider, prompt, system, model)
        if ptype == "ollama":
            return await self._call_ollama(provider, prompt, system, model)
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
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
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

    async def _call_ollama(self, provider: dict, prompt: str, system: str, model: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
        }
        # Ollama can take a while on first model load — give it a bigger budget.
        r = await self._client.post(provider["url"], json=payload, timeout=180.0)
        r.raise_for_status()
        data = r.json()
        # /api/chat returns {"message": {"role":"assistant","content":"..."}}
        try:
            return data["message"]["content"]
        except KeyError:
            # Some older builds use /api/generate shape
            return data.get("response", "")

    async def close(self) -> None:
        await self._client.aclose()
