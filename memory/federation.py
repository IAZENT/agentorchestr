"""
memory.federation — markdown + FTS5 (+ optional sqlite-vec) hybrid store
========================================================================
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ── optional deps; the federation degrades gracefully without them ────
try:
    import sqlite_vec  # type: ignore
    HAS_SQLITE_VEC = True
except ImportError:  # pragma: no cover - exercised in tests by monkeypatching
    HAS_SQLITE_VEC = False

try:
    from fastembed import TextEmbedding  # type: ignore
    HAS_FASTEMBED = True
except ImportError:  # pragma: no cover
    HAS_FASTEMBED = False


def _default_global_dir() -> Path:
    """Lazy default — checks paths.global_memory_dir() at call time so
    tests can monkeypatch GLOBAL_DIR without import-order pain."""
    try:
        from paths import global_memory_dir
        return global_memory_dir()
    except Exception:
        return Path(os.path.expanduser("~/.agentorchestr/memory"))


GLOBAL_DIR = _default_global_dir()
PROJECT_DIR_NAME = ".agentorchestr/memory"
INDEX_DB = ".agentorchestr/memory/index.sqlite"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-d, ~30 ms/sentence on CPU
EMBED_DIM = 384

# Default recency half-life: a memory loses half its boost per 30 days.
RECENCY_HALFLIFE_DAYS = 30.0
# Reciprocal-rank-fusion constant (Cormack et al.); 60 is a defensible default.
RRF_K = 60.0


@dataclass
class MemoryHit:
    id: str
    tier: str             # 'global' | 'project' | 'session'
    kind: str             # 'static' | 'topic' | 'episode'
    title: str
    body: str
    path: Optional[str] = None
    score: float = 0.0
    sources: list[str] = field(default_factory=list)  # which retrievers fired

    def header(self) -> str:
        flavour = ", ".join(self.sources) if self.sources else self.kind
        return f"### {self.title}  ({self.tier} · {flavour})"


def _hash_id(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def _read_text(path: Path, *, max_chars: int = 8192) -> str:
    try:
        return path.read_text()[:max_chars]
    except OSError:
        return ""


def _recency_score(updated_at: float, *, now: float | None = None,
                   halflife_days: float = RECENCY_HALFLIFE_DAYS) -> float:
    """Exponential decay -> 1.0 means now, 0.5 means halflife_days ago, …"""
    now = now or time.time()
    age_days = max(0.0, (now - updated_at) / 86400.0)
    return math.pow(0.5, age_days / max(halflife_days, 1e-6))


# ── the federation itself ─────────────────────────────────────────────

class MemoryFederation:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.project_dir = self.project_root / PROJECT_DIR_NAME
        self.global_dir = GLOBAL_DIR
        self.index_path = self.project_root / INDEX_DB
        self._db: Optional[sqlite3.Connection] = None
        self._embedder = None  # type: ignore[var-annotated]
        self._has_vec = HAS_SQLITE_VEC and HAS_FASTEMBED

    # public API --------------------------------------------------------

    async def init(self) -> None:
        """Create dirs + open the SQLite index. Safe to call repeatedly."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "topics").mkdir(exist_ok=True)
        (self.project_dir / "episodes").mkdir(exist_ok=True)
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # SQLite is fast enough on the foreground thread for our scale
        # (<10k notes) — keep it sync for simplicity; await dispatches it
        # off the event loop where needed.
        await asyncio.to_thread(self._open_db)
        await asyncio.to_thread(self._reindex_static_files)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    async def load_static_context(self) -> str:
        """Return the always-on preamble (global + project static files).

        This is the prefix we want Anthropic prompt caching to hit.
        Order: global -> project static -> conventions.  Stable across
        runs as long as the source markdown doesn't change.
        """
        return await asyncio.to_thread(self._load_static_sync)

    async def index_topic(self, name: str, body: str) -> None:
        """Write a topic markdown file and update the index."""
        path = self.project_dir / "topics" / f"{_safe_filename(name)}.md"
        path.write_text(body)
        await asyncio.to_thread(
            self._upsert_note,
            kind="topic",
            tier="project",
            title=name,
            body=body,
            path=str(path),
        )

    async def index_episode(self, session_id: str, body: str) -> None:
        path = self.project_dir / "episodes" / f"session_{session_id}.md"
        path.write_text(body)
        await asyncio.to_thread(
            self._upsert_note,
            kind="episode",
            tier="project",
            title=f"session {session_id}",
            body=body,
            path=str(path),
        )

    async def cache_research(self, query: str, payload_dict: dict,
                              *, ttl_hours: float = 24.0) -> None:
        """Persist a research payload as a 'research' note keyed by query.

        Stored under <project>/.agentorchestr/memory/research/<safe>.md as plain
        markdown (so a human can read / git-track it) AND mirrored into
        the FTS / vec index so future workers retrieve it by topic.
        TTL is recorded in the note header; get_cached_research enforces it.
        """
        if not query.strip():
            return
        body = self._render_research_body(query, payload_dict, ttl_hours)
        path = self.project_dir / "research" / f"{_safe_filename(query)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        await asyncio.to_thread(
            self._upsert_note,
            kind="research",
            tier="project",
            title=f"research: {query[:80]}",
            body=body,
            path=str(path),
        )

    async def get_cached_research(self, query: str) -> Optional[dict]:
        """Return a fresh cached research payload for `query`, or None.

        'Fresh' means within the TTL recorded in the note header at write
        time.  Returns the payload as a dict (matching ResearchPayload.to_dict).
        """
        if not query.strip():
            return None
        return await asyncio.to_thread(self._get_cached_research_sync, query)

    async def retrieve(self, query: str, *, k: int = 5,
                       tiers: Iterable[str] = ("global", "project")
                       ) -> list[MemoryHit]:
        """Hybrid retrieval: FTS5 + (optional) vector + recency boost."""
        if not query.strip():
            return []
        return await asyncio.to_thread(self._retrieve_sync, query, k, tuple(tiers))

    # ── implementation (sync helpers run via asyncio.to_thread) ──────

    def _open_db(self) -> None:
        # check_same_thread=False so the same connection can be reused
        # across asyncio.to_thread executor threads.  We serialise all
        # writes via the GIL + the federation's call sites, which never
        # interleave concurrent writes.
        db = sqlite3.connect(str(self.index_path), check_same_thread=False)
        db.row_factory = sqlite3.Row
        if self._has_vec:
            try:
                db.enable_load_extension(True)
                sqlite_vec.load(db)
                db.enable_load_extension(False)
            except (AttributeError, sqlite3.OperationalError):
                self._has_vec = False
        self._db = db
        # Standalone FTS5 (no content= mode) — we mirror manually on upsert.
        # Simpler than triggers and avoids "database disk image is malformed"
        # when rows are deleted out of order.
        db.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                tier TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                path TEXT,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind);
            CREATE INDEX IF NOT EXISTS idx_notes_tier ON notes(tier);
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                id UNINDEXED,
                title,
                body,
                tokenize='unicode61'
            );
        """)
        if self._has_vec:
            db.executescript(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding FLOAT[{EMBED_DIM}]
                );
            """)
        db.commit()

    def _reindex_static_files(self) -> None:
        """Crawl markdown under global + project dirs and upsert each one."""
        roots = [
            (self.global_dir, "global"),
            (self.project_dir, "project"),
        ]
        for root, tier in roots:
            if not root.exists():
                continue
            for md in root.rglob("*.md"):
                # Skip the auto-generated MEMORY.md index that just lists files.
                if md.name == "MEMORY.md":
                    continue
                kind = (
                    "topic" if "topics" in md.parts else
                    "episode" if "episodes" in md.parts else
                    "static"
                )
                title = md.stem
                body = _read_text(md)
                if not body.strip():
                    continue
                self._upsert_note(kind=kind, tier=tier, title=title,
                                  body=body, path=str(md))

    def _load_static_sync(self) -> str:
        parts: list[str] = []
        # Global is read-only header info.
        for name in ("global.md",):
            p = self.global_dir / name
            if p.exists():
                parts.append(f"# Global memory\n\n{_read_text(p, max_chars=4000)}")
        # Project: PROJECT.md + CONVENTIONS.md if present (live in project root, not memory/)
        for name in ("PROJECT.md", "CONVENTIONS.md"):
            for candidate in (self.project_root / name,
                              self.project_root / ".agentorchestr" / name):
                if candidate.exists():
                    parts.append(
                        f"# {name}\n\n{_read_text(candidate, max_chars=4000)}"
                    )
                    break
        return "\n\n".join(parts).strip()

    def _upsert_note(self, *, kind: str, tier: str, title: str,
                     body: str, path: Optional[str]) -> str:
        assert self._db is not None
        nid = _hash_id(tier, kind, path or title)
        now = time.time()
        self._db.execute("""
            INSERT INTO notes (id, kind, tier, title, body, path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                body  = excluded.body,
                path  = excluded.path,
                updated_at = excluded.updated_at
        """, (nid, kind, tier, title, body, path, now))
        # Mirror into the standalone FTS table by id.
        self._db.execute("DELETE FROM notes_fts WHERE id = ?", (nid,))
        self._db.execute(
            "INSERT INTO notes_fts(id, title, body) VALUES (?, ?, ?)",
            (nid, title, body),
        )
        if self._has_vec:
            try:
                vec = self._embed(body)
                if vec is not None:
                    blob = _vec_to_blob(vec)
                    self._db.execute(
                        "INSERT OR REPLACE INTO notes_vec(id, embedding) VALUES (?, ?)",
                        (nid, blob),
                    )
            except Exception:
                # Embedding failure must never break indexing.
                pass
        self._db.commit()
        return nid

    def _retrieve_sync(self, query: str, k: int, tiers: tuple[str, ...]) -> list[MemoryHit]:
        assert self._db is not None
        # ── FTS5 path ─────────────────────────────────────────────────
        fts_results: list[tuple[str, float]] = []
        try:
            fts_query = _sanitize_fts_query(query)
            if fts_query:
                rows = self._db.execute(f"""
                    SELECT f.id, bm25(notes_fts) AS rank
                    FROM notes_fts f JOIN notes n ON n.id = f.id
                    WHERE notes_fts MATCH ?
                      AND n.tier IN ({",".join("?" * len(tiers))})
                    ORDER BY rank LIMIT ?
                """, (fts_query, *tiers, k * 4)).fetchall()
                fts_results = [(r["id"], r["rank"]) for r in rows]
        except sqlite3.OperationalError:
            pass

        # ── vector path (optional) ────────────────────────────────────
        vec_results: list[tuple[str, float]] = []
        if self._has_vec:
            qvec = self._embed(query)
            if qvec is not None:
                try:
                    blob = _vec_to_blob(qvec)
                    rows = self._db.execute(f"""
                        SELECT v.id, distance
                        FROM notes_vec v JOIN notes n ON n.id = v.id
                        WHERE n.tier IN ({",".join("?" * len(tiers))})
                          AND embedding MATCH ?
                          AND k = ?
                        ORDER BY distance
                    """, (*tiers, blob, k * 4)).fetchall()
                    vec_results = [(r["id"], r["distance"]) for r in rows]
                except sqlite3.OperationalError:
                    pass

        # ── reciprocal rank fusion + recency boost ────────────────────
        scores: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        for rank, (nid, _score) in enumerate(fts_results):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(nid, []).append("fts")
        for rank, (nid, _score) in enumerate(vec_results):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(nid, []).append("vector")

        if not scores:
            return []

        # Apply recency boost.
        ids_csv = ",".join("?" * len(scores))
        rows = self._db.execute(f"""
            SELECT id, kind, tier, title, body, path, updated_at
            FROM notes WHERE id IN ({ids_csv})
        """, tuple(scores.keys())).fetchall()
        hits: list[MemoryHit] = []
        for row in rows:
            base = scores[row["id"]]
            recency = _recency_score(row["updated_at"])
            final = base * (1.0 + 0.5 * recency)  # +50% at zero age
            hits.append(MemoryHit(
                id=row["id"], tier=row["tier"], kind=row["kind"],
                title=row["title"], body=row["body"], path=row["path"],
                score=final, sources=sources.get(row["id"], []),
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def _embed(self, text: str) -> Optional[list[float]]:
        if not HAS_FASTEMBED:
            return None
        if self._embedder is None:
            try:
                self._embedder = TextEmbedding(model_name=EMBED_MODEL)
            except Exception:
                return None
        try:
            vecs = list(self._embedder.embed([text[:2000]]))
            return list(vecs[0]) if vecs else None
        except Exception:
            return None

    # ── research-cache helpers ────────────────────────────────────────

    @staticmethod
    def _render_research_body(query: str, payload: dict, ttl_hours: float) -> str:
        """Render a research payload as markdown with a cache header.

        The header is parsed back by _get_cached_research_sync to
        enforce TTL and reconstruct the original payload dict.

        TTL is measured from the moment of caching, not from the
        payload's `fetched_at` (which represents when the upstream
        data was retrieved — could be hours/days earlier).
        """
        ttl_s = ttl_hours * 3600.0
        cached_at = time.time()
        header = (
            f"<!--agentorchestr-research v1\n"
            f"query={query}\n"
            f"cached_at={cached_at}\n"
            f"ttl_s={ttl_s}\n"
            f"-->\n"
        )
        body_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return (
            f"{header}# Research: {query}\n\n"
            f"```json\n{body_json}\n```\n"
        )

    _RESEARCH_HEADER_RE = re.compile(
        r"<!--agentorchestr-research v1\s*\n"
        r"query=(?P<query>.*?)\s*\n"
        r"cached_at=(?P<cached_at>[\d.]+)\s*\n"
        r"ttl_s=(?P<ttl>[\d.]+)\s*\n-->",
        re.DOTALL,
    )

    def _get_cached_research_sync(self, query: str) -> Optional[dict]:
        """Look up the research note for `query`, return its payload if fresh."""
        path = self.project_dir / "research" / f"{_safe_filename(query)}.md"
        if not path.exists():
            return None
        try:
            text = path.read_text()
        except OSError:
            return None
        m = self._RESEARCH_HEADER_RE.search(text)
        if not m:
            return None
        try:
            cached_at = float(m.group("cached_at"))
            ttl_s = float(m.group("ttl"))
        except ValueError:
            return None
        if time.time() - cached_at > ttl_s:
            return None  # stale — force refresh
        # Pull the JSON block out.
        json_match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if not json_match:
            return None
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return None


# ── helpers ───────────────────────────────────────────────────────────

_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(name: str) -> str:
    return _SAFE_RE.sub("-", name).strip("-").lower() or "untitled"


_FTS_DROP = re.compile(r'[^\w\s\.\-]')


def _sanitize_fts_query(q: str) -> str:
    """FTS5 query syntax is liberal but breaks on unbalanced quotes /
    parens / colons. Strip everything but words + a few punct chars,
    then add prefix-match wildcards so partial words still hit."""
    cleaned = _FTS_DROP.sub(" ", q).split()
    if not cleaned:
        return ""
    return " OR ".join(f'{w}*' for w in cleaned[:8])


def _vec_to_blob(vec: list[float]) -> bytes:
    """sqlite-vec accepts a packed float32 blob OR the JSON form; the
    blob form sidesteps JSON-parser overhead."""
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)
