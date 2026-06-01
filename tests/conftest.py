"""Pytest configuration.

The package is now properly laid out under agentorchestr/ and is expected to
be on sys.path either via `pip install -e .` (editable) or because the
project root contains the package directory.  We still add the project root
to sys.path defensively so `pytest` works in a fresh checkout without
requiring an install step.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
