"""
complexity.py — heuristic project-complexity probe
====================================================

Used by the automatic wizard to decide how many workers to spawn.
Pure-stdlib, fast (<100 ms on a 10k-file repo), no LLM required.

The heuristic looks at:

    LOC                # via line counting on tracked source files
    file_count         # number of source files
    languages          # set of detected languages
    manifests          # presence of package.json, pyproject.toml, Cargo.toml,
                       # go.mod, build.gradle, etc.
    git_history        # commit count + active branches (proxy for project age)

It returns a Complexity dataclass + a recommended worker layout the
wizard can use as the default for automatic mode.

This is deliberately conservative: when in doubt we under-allocate
workers because token cost compounds (Anthropic: 15× tokens for
multi-agent vs single-agent, 80 % of perf variance is token usage).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Source-code file extensions we count toward LOC + complexity.
_SOURCE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".kt",
    ".rb", ".php", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp", ".swift",
    ".m", ".mm", ".scala", ".clj", ".ex", ".exs", ".erl", ".hs", ".lua",
    ".sh", ".bash", ".zsh", ".sql", ".vue", ".svelte",
}

# Build-system manifests that signal a "real" project.
_MANIFESTS = {
    "pyproject.toml": "python", "setup.py": "python", "requirements.txt": "python",
    "package.json": "javascript", "tsconfig.json": "typescript",
    "Cargo.toml": "rust", "go.mod": "go",
    "pom.xml": "java", "build.gradle": "java", "build.gradle.kts": "java",
    "Gemfile": "ruby", "composer.json": "php",
    "*.csproj": "csharp", "*.fsproj": "fsharp",
    "Makefile": "c-or-cpp", "CMakeLists.txt": "c-or-cpp",
    "Package.swift": "swift",
}

# Directories we always skip when walking the tree.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv",
    ".env", "venv", "env", "dist", "build", "out", "target",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache", "coverage",
    ".next", ".nuxt", ".cache", ".agentorchestr",
}


@dataclass
class Complexity:
    loc: int = 0
    file_count: int = 0
    languages: set[str] = field(default_factory=set)
    manifests: list[str] = field(default_factory=list)
    git_commits: int = 0
    is_git: bool = False

    # Derived
    tier: str = "trivial"   # trivial | simple | moderate | complex | giant
    recommended_workers: int = 1
    rationale: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["languages"] = sorted(self.languages)
        return d


def assess(project_root: str | Path, *, max_files: int = 5000) -> Complexity:
    """Walk the project tree once, derive a Complexity profile.

    `max_files` caps the walk so we don't pathologically scan a
    monorepo with 1M files.  Once the cap hits we still return a sane
    profile — the heuristic flips to "complex" / "giant" automatically
    based on the truncated count, which is the right answer anyway.
    """
    root = Path(project_root).resolve()
    c = Complexity(is_git=(root / ".git").exists())

    seen_files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Mutate dirnames in place so os.walk skips them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            seen_files += 1
            if seen_files >= max_files:
                break
            ext = Path(fn).suffix.lower()
            if ext in _SOURCE_EXTS:
                c.file_count += 1
                c.languages.add(_lang_for_ext(ext))
                # Cheap LOC: read file size, divide by ~50 chars/line. Avoids
                # opening every file.
                try:
                    size = (Path(dirpath) / fn).stat().st_size
                    c.loc += max(1, size // 50)
                except OSError:
                    pass
        if seen_files >= max_files:
            break

    # Manifest scan (root only — we don't go deep for build files).
    for entry in root.iterdir():
        name = entry.name
        # Direct match
        if name in _MANIFESTS and entry.is_file():
            c.manifests.append(_MANIFESTS[name])
            continue
        # Glob match (e.g. *.csproj)
        for pat, lang in _MANIFESTS.items():
            if pat.startswith("*.") and name.endswith(pat[1:]):
                c.manifests.append(lang)

    # Git commit count
    if c.is_git:
        try:
            r = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(root), capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                c.git_commits = int((r.stdout or "0").strip() or 0)
        except (subprocess.SubprocessError, ValueError, OSError):
            pass

    _classify(c, hit_cap=seen_files >= max_files)
    return c


# ── classification ────────────────────────────────────────────────────

def _classify(c: Complexity, *, hit_cap: bool) -> None:
    """Map raw counts -> tier + worker recommendation.

    Tier thresholds were picked from informal sampling of a few dozen
    public repos.  They're conservative on purpose: we under-allocate
    workers because token cost dominates perf variance.
    """
    score = 0
    score += min(c.file_count // 10, 30)           # up to +30 for file count
    score += min(c.loc // 1000, 30)                # up to +30 for LOC
    score += min(len(c.languages) * 5, 15)         # up to +15 for polyglot
    score += min(len(c.manifests) * 5, 15)         # up to +15 for manifests
    score += 5 if c.is_git and c.git_commits > 50 else 0
    if hit_cap:
        # File walker hit max_files — definitely a giant repo.
        score += 30

    if score >= 80:
        c.tier = "giant"
        c.recommended_workers = 4
    elif score >= 50:
        c.tier = "complex"
        c.recommended_workers = 3
    elif score >= 25:
        c.tier = "moderate"
        c.recommended_workers = 2
    elif score >= 10:
        c.tier = "simple"
        c.recommended_workers = 1
    else:
        c.tier = "trivial"
        c.recommended_workers = 1

    c.rationale = (
        f"score={score} "
        f"(files={c.file_count}, ~loc={c.loc}, langs={len(c.languages)}, "
        f"manifests={len(c.manifests)}, commits={c.git_commits}"
        f"{', cap-hit' if hit_cap else ''})"
    )


_EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java", ".kt": "kotlin",
    ".rb": "ruby", ".php": "php",
    ".cs": "csharp",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".swift": "swift", ".m": "objc", ".mm": "objc",
    ".scala": "scala", ".clj": "clojure",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    ".hs": "haskell", ".lua": "lua",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".sql": "sql", ".vue": "vue", ".svelte": "svelte",
}


def _lang_for_ext(ext: str) -> str:
    return _EXT_TO_LANG.get(ext, ext.lstrip("."))
