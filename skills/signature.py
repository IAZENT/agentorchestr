"""
skills.signature — optional ed25519 verification for skills
=============================================================

Two backends, tried in order:
  1. cryptography (already a transitive dep of many tools)
  2. PyNaCl

If neither is installed, verification raises SignatureError("backend unavailable")
and callers may choose to allow unsigned skills via --allow-unsigned.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Iterable

# Files that participate in the canonical hash. We deliberately exclude
# signature.sig and any *.pub files so signing is reproducible.
_HASHED_FILES_DEFAULT = ("skill.toml", "skill.md")


class SignatureError(Exception):
    """Raised when a skill fails to verify."""


def canonical_hash(skill_dir: Path,
                   files: Iterable[str] = _HASHED_FILES_DEFAULT) -> bytes:
    """Stable SHA-256 over the byte content of `files` in deterministic order.

    Used as the message we sign / verify.  Whitespace and line endings
    are preserved as-is — the user's checkout is the source of truth.
    """
    h = hashlib.sha256()
    for name in sorted(files):
        p = skill_dir / name
        h.update(name.encode())
        h.update(b"\x00")
        if p.exists():
            h.update(p.read_bytes())
        h.update(b"\x01")
    return h.digest()


def verify_skill_signature(skill_dir: Path) -> None:
    """Raise SignatureError if the skill is signed and the signature
    doesn't match.  No-op for unsigned skills (caller decides policy)."""
    sig_path = skill_dir / "signature.sig"
    key_path = skill_dir / "signing-key.pub"
    if not sig_path.exists() or not key_path.exists():
        # Unsigned — caller decides.
        return
    # Pick the backend FIRST so the absence of cryptography/PyNaCl always
    # produces a clean "backend unavailable" error, even when the signature
    # bytes themselves are malformed.
    backend = _pick_backend()
    if backend is None:
        raise SignatureError(
            "ed25519 backend unavailable — install cryptography or pynacl, "
            "or run with --allow-unsigned"
        )
    try:
        signature = base64.b64decode(sig_path.read_text().strip())
        pubkey_bytes = base64.b64decode(key_path.read_text().strip())
    except (OSError, ValueError) as e:
        raise SignatureError(f"could not read signature/pubkey: {e}") from e

    msg = canonical_hash(skill_dir)
    backend(pubkey_bytes, signature, msg)


def _pick_backend():
    """Return a callable verify(pubkey_bytes, signature, msg) or None."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.exceptions import InvalidSignature

        def _crypto(pubkey_bytes, signature, msg):
            try:
                key = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
                key.verify(signature, msg)
            except (InvalidSignature, ValueError) as e:
                raise SignatureError(f"signature mismatch: {e}") from e

        return _crypto
    except ImportError:
        pass

    try:
        import nacl.signing
        import nacl.exceptions

        def _nacl(pubkey_bytes, signature, msg):
            try:
                vk = nacl.signing.VerifyKey(pubkey_bytes)
                vk.verify(msg, signature)
            except nacl.exceptions.BadSignatureError as e:
                raise SignatureError("signature mismatch") from e

        return _nacl
    except ImportError:
        pass

    return None
