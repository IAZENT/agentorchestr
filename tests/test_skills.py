"""Tests for the skill marketplace.

We exercise loading, trigger matching, and signature verification using
fixture skills written into a tmp_path.  add_from_git is tested via a
fake `git` binary in PATH so we don't hit the network.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from agentorchestr.skills import (
    Skill, SkillManifest, SkillRegistry, SignatureError,
    verify_skill_signature,
)
from agentorchestr.skills import registry as registry_mod
from agentorchestr.skills import signature as sig_mod


# ── helpers ─────────────────────────────────────────────────────────────

def _make_skill(parent: Path, name: str, *, keywords=None,
                file_globs=None, version="0.1.0",
                signed: bool = False) -> Path:
    sk = parent / name
    sk.mkdir()
    triggers = []
    if keywords:
        triggers.append(f"keywords = {list(keywords)!r}")
    if file_globs:
        triggers.append(f"file_globs = {list(file_globs)!r}")
    triggers_block = "[triggers]\n" + "\n".join(triggers) if triggers else ""
    (sk / "skill.toml").write_text(
        f'[skill]\n'
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        f'description = "test skill"\n\n'
        f'{triggers_block}\n'
    )
    (sk / "skill.md").write_text(f"# {name}\n\nUse {name} for stuff.")
    if signed:
        # Drop placeholder signature files; we don't actually verify here
        # unless the test explicitly asks for it.
        (sk / "signing-key.pub").write_text("AAAAAA")
        (sk / "signature.sig").write_text("AAAAAA")
    return sk


# ── manifest parsing ────────────────────────────────────────────────────

def test_manifest_from_toml(tmp_path):
    sk = _make_skill(tmp_path, "auth-skill",
                     keywords=["jwt", "auth"], file_globs=["**/auth.py"])
    m = SkillManifest.from_toml(sk / "skill.toml")
    assert m.name == "auth-skill"
    assert m.version == "0.1.0"
    assert m.keywords == ["jwt", "auth"]
    assert m.file_globs == ["**/auth.py"]
    assert m.has_signature is False


def test_manifest_marks_signed_when_files_present(tmp_path):
    sk = _make_skill(tmp_path, "signed-skill",
                     keywords=["x"], signed=True)
    m = SkillManifest.from_toml(sk / "skill.toml")
    assert m.has_signature is True


# ── registry: load, match, render ──────────────────────────────────────

def test_registry_loads_skills(tmp_path):
    _make_skill(tmp_path, "auth", keywords=["jwt"])
    _make_skill(tmp_path, "perf", keywords=["latency"])
    reg = SkillRegistry(skills_dir=tmp_path)
    reg.reload()
    names = sorted(s.name for s in reg.all())
    assert names == ["auth", "perf"]


def test_registry_skips_malformed(tmp_path):
    """A directory without a skill.toml must be ignored, not crash reload."""
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "README.md").write_text("not a skill")
    _make_skill(tmp_path, "good", keywords=["x"])
    reg = SkillRegistry(skills_dir=tmp_path)
    reg.reload()
    assert [s.name for s in reg.all()] == ["good"]


def test_skill_matches_keyword(tmp_path):
    _make_skill(tmp_path, "auth", keywords=["jwt", "authentication"])
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    out = reg.matching("set up JWT authentication for the API", file_scope=[])
    assert [s.name for s in out] == ["auth"]


def test_skill_matches_file_glob(tmp_path):
    _make_skill(tmp_path, "auth", file_globs=["**/auth.py"])
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    out = reg.matching(
        "fix bug",
        file_scope=["src/users/auth.py", "src/views.py"],
    )
    assert [s.name for s in out] == ["auth"]


def test_skill_no_triggers_never_matches(tmp_path):
    _make_skill(tmp_path, "passive")  # no keywords, no globs
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    assert reg.matching("anything goes", file_scope=["x.py"]) == []


def test_render_caps_per_skill_size(tmp_path):
    sk = _make_skill(tmp_path, "verbose", keywords=["x"])
    (sk / "skill.md").write_text("A" * 10000)
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    out = reg.render(reg.all(), max_chars=200)
    # Cap is total budget split per-skill — at 1 skill, body should be <=200.
    assert "AAAA" in out
    assert len(out) < 1000  # nowhere near the 10000-char input


def test_render_empty_returns_empty(tmp_path):
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    assert reg.render([]) == ""


# ── git add / remove ────────────────────────────────────────────────────

def test_name_from_git_url():
    from agentorchestr.skills.registry import _name_from_git_url
    assert _name_from_git_url("https://github.com/cosmic/python-fastapi-jwt") == "python-fastapi-jwt"
    assert _name_from_git_url("https://github.com/cosmic/foo.git") == "foo"
    assert _name_from_git_url("git@github.com:cosmic/bar.git") == "bar"


def test_add_from_git_uses_fake_git(tmp_path, monkeypatch):
    """Stub `git clone` to copy a prebuilt skill so the test stays offline."""
    skills_dir = tmp_path / "skills"
    src = tmp_path / "src"; src.mkdir()
    _make_skill(src, "fakeskill", keywords=["x"])

    def fake_run(cmd, **kwargs):
        # cmd is ["git", "clone", "--depth", "1", url, target]
        url, target = cmd[-2], cmd[-1]
        shutil.copytree(src / "fakeskill", target)
        class R: returncode = 0; stderr = ""; stdout = ""
        return R()

    monkeypatch.setattr(registry_mod.shutil, "which", lambda c: "/usr/bin/git" if c == "git" else None)
    monkeypatch.setattr(registry_mod.subprocess, "run", fake_run)

    reg = SkillRegistry(skills_dir=skills_dir)
    skill = reg.add_from_git("https://example.com/cosmic/fakeskill.git")
    assert skill.name == "fakeskill"
    assert (skills_dir / "fakeskill").is_dir()


def test_add_from_git_refuses_duplicate(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "dupe").mkdir()
    monkeypatch.setattr(registry_mod.shutil, "which", lambda c: "/usr/bin/git")
    reg = SkillRegistry(skills_dir=skills_dir)
    with pytest.raises(FileExistsError):
        reg.add_from_git("git@example.com:x/dupe.git")


def test_remove_skill(tmp_path):
    _make_skill(tmp_path, "doomed", keywords=["x"])
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    assert reg.remove("doomed") is True
    assert reg.remove("doomed") is False  # idempotent
    reg.reload()
    assert reg.all() == []


# ── signature verification ──────────────────────────────────────────────

def test_unsigned_skill_passes_verify(tmp_path):
    _make_skill(tmp_path, "noauth", keywords=["x"])
    # No signature.sig + no signing-key.pub → caller decides; default is OK.
    verify_skill_signature(tmp_path / "noauth")  # no exception


def test_signed_skill_with_no_backend_raises(tmp_path, monkeypatch):
    """If signature.sig exists but no ed25519 backend is loadable,
    verification must FAIL closed."""
    _make_skill(tmp_path, "fakesigned", keywords=["x"], signed=True)
    monkeypatch.setattr(sig_mod, "_pick_backend", lambda: None)
    with pytest.raises(SignatureError, match="backend unavailable"):
        verify_skill_signature(tmp_path / "fakesigned")


def test_canonical_hash_deterministic(tmp_path):
    sk = _make_skill(tmp_path, "stable", keywords=["x"])
    h1 = sig_mod.canonical_hash(sk)
    h2 = sig_mod.canonical_hash(sk)
    assert h1 == h2
    # Mutate the body — hash must change.
    (sk / "skill.md").write_text("different content")
    h3 = sig_mod.canonical_hash(sk)
    assert h1 != h3


def test_registry_verify_returns_structured_info(tmp_path):
    _make_skill(tmp_path, "ok", keywords=["x"])
    reg = SkillRegistry(skills_dir=tmp_path); reg.reload()
    info = reg.verify("ok")
    assert info["ok"] is True
    assert info["signed"] is False  # unsigned
    info2 = reg.verify("does-not-exist")
    assert info2["ok"] is False
    assert info2["error"] == "not installed"
