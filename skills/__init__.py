"""
skills — reusable, sharable agent capability bundles
=====================================================

A skill is a directory containing:

    skill.toml         manifest (name, version, triggers, signature info)
    skill.md           markdown injected into worker prompts when triggered
    signature.sig      optional ed25519 signature over canonical_hash(skill)
    signing-key.pub    optional ed25519 public key (raw 32 bytes, base64)
    examples/          optional reference files (read-only at runtime)

Triggers decide when the skill's prompt gets injected:

  [triggers]
  keywords   = ["jwt", "auth"]      # any keyword present in task → match
  file_globs = ["**/auth.py"]       # any file in scope matches → match

Distribution is git-clone-based:

    orch skill add github.com/cosmic/python-fastapi-jwt
    orch skill list
    orch skill verify python-fastapi-jwt
    orch skill remove python-fastapi-jwt

Public API:

    SkillRegistry(skills_dir=~/.orch/skills)
        .reload()
        .all() -> list[Skill]
        .matching(task, file_scope) -> list[Skill]
        .render(skills) -> str   # the prompt block to prepend

Stays single-file friendly: optional ed25519 via PyNaCl OR cryptography,
gracefully no-ops if neither is installed.
"""
from .registry import Skill, SkillRegistry, SkillManifest
from .signature import SignatureError, verify_skill_signature

__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillManifest",
    "SignatureError",
    "verify_skill_signature",
]
