"""
skills.registry — load, match, and render skills
==================================================
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .signature import SignatureError, verify_skill_signature

def _default_skills_dir() -> Path:
    try:
        from paths import global_skills_dir
        return global_skills_dir()
    except Exception:
        return Path(os.path.expanduser("~/.agentorchestr/skills"))


DEFAULT_SKILLS_DIR = _default_skills_dir()


@dataclass
class SkillManifest:
    name: str
    version: str = "0.0.0"
    author: str = ""
    description: str = ""
    license: str = ""
    keywords: list[str] = field(default_factory=list)
    file_globs: list[str] = field(default_factory=list)
    mcp_server_module: Optional[str] = None
    has_signature: bool = False

    @classmethod
    def from_toml(cls, path: Path) -> "SkillManifest":
        data = tomllib.loads(path.read_text())
        sk = data.get("skill", {})
        triggers = data.get("triggers", {})
        mcp = data.get("mcp", {})
        sec = data.get("security", {})
        return cls(
            name=sk.get("name", path.parent.name),
            version=sk.get("version", "0.0.0"),
            author=sk.get("author", ""),
            description=sk.get("description", ""),
            license=sk.get("license", ""),
            keywords=[k.lower() for k in (triggers.get("keywords") or [])],
            file_globs=list(triggers.get("file_globs") or []),
            mcp_server_module=mcp.get("server_module"),
            has_signature=bool(sec.get("public_key")) or
                          (path.parent / "signature.sig").exists(),
        )


@dataclass
class Skill:
    manifest: SkillManifest
    path: Path
    body: str  # contents of skill.md

    @property
    def name(self) -> str:
        return self.manifest.name

    def matches(self, task: str, file_scope: Iterable[str]) -> bool:
        """Triggered if any keyword appears in the task OR any file glob
        matches an entry in file_scope.  Empty triggers = never match."""
        if not self.manifest.keywords and not self.manifest.file_globs:
            return False
        task_lower = task.lower() if task else ""
        for kw in self.manifest.keywords:
            if kw and kw in task_lower:
                return True
        for glob in self.manifest.file_globs:
            for f in file_scope:
                if fnmatch.fnmatch(f, glob):
                    return True
        return False


# ── registry ──────────────────────────────────────────────────────────

class SkillRegistry:
    def __init__(self, skills_dir: Path | str = DEFAULT_SKILLS_DIR,
                 *, require_signature: bool = False):
        self.skills_dir = Path(skills_dir)
        self.require_signature = require_signature
        self._skills: list[Skill] = []

    def reload(self) -> None:
        self._skills.clear()
        if not self.skills_dir.exists():
            return
        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                self._skills.append(self._load_one(child))
            except (FileNotFoundError, tomllib.TOMLDecodeError, SignatureError):
                # Skip malformed/un-trusted skills silently — agentorchestr skill list
                # surfaces them via verify().
                continue

    def all(self) -> list[Skill]:
        return list(self._skills)

    def matching(self, task: str, file_scope: Iterable[str]) -> list[Skill]:
        return [s for s in self._skills if s.matches(task, file_scope)]

    def render(self, skills: Iterable[Skill], *, max_chars: int = 4000) -> str:
        """Format a list of skills as a single prompt block.

        Designed to slot into the cacheable prefix of a worker prompt.
        Truncated per-skill so a single huge skill can't dominate the budget.
        """
        skills = list(skills)
        if not skills:
            return ""
        per_skill = max_chars // max(len(skills), 1)
        parts = ["## Active skills"]
        for s in skills:
            parts.append(f"### {s.name} (v{s.manifest.version})")
            if s.manifest.description:
                parts.append(f"_{s.manifest.description}_")
            parts.append(s.body[:per_skill])
        return "\n\n".join(parts)

    # ── package-management primitives (used by agentorchestr skill add/remove) ──

    def add_from_git(self, url: str, *, name: Optional[str] = None) -> Skill:
        """Clone `url` into skills_dir/<name>. Returns the loaded Skill."""
        if shutil.which("git") is None:
            raise RuntimeError("git not found on PATH")
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        target_name = name or _name_from_git_url(url)
        target = self.skills_dir / target_name
        if target.exists():
            raise FileExistsError(f"skill already installed at {target}")
        # --depth 1 to keep the install lean.
        r = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git clone failed: {r.stderr.strip()}")
        try:
            skill = self._load_one(target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        self._skills.append(skill)
        return skill

    def remove(self, name: str) -> bool:
        target = self.skills_dir / name
        if not target.exists():
            return False
        shutil.rmtree(target, ignore_errors=True)
        self._skills = [s for s in self._skills if s.name != name]
        return True

    def verify(self, name: str) -> dict:
        """Run signature verification + manifest sanity for one skill."""
        target = self.skills_dir / name
        info = {"name": name, "path": str(target), "signed": False, "ok": False, "error": None}
        if not target.exists():
            info["error"] = "not installed"
            return info
        try:
            verify_skill_signature(target)
            sig_present = (target / "signature.sig").exists() and \
                          (target / "signing-key.pub").exists()
            info["signed"] = sig_present
            info["ok"] = True
        except SignatureError as e:
            info["error"] = str(e)
        return info

    # ── internals ──────────────────────────────────────────────────────

    def _load_one(self, path: Path) -> Skill:
        manifest_path = path / "skill.toml"
        body_path = path / "skill.md"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing skill.toml in {path}")

        manifest = SkillManifest.from_toml(manifest_path)
        if self.require_signature:
            verify_skill_signature(path)
        body = body_path.read_text() if body_path.exists() else ""
        return Skill(manifest=manifest, path=path, body=body)


_GIT_URL_RE = re.compile(r"[/:]([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def _name_from_git_url(url: str) -> str:
    m = _GIT_URL_RE.search(url.rstrip("/"))
    if not m:
        raise ValueError(f"could not derive a name from {url!r}")
    return m.group(1)
