"""Pytest configuration — add the project root to sys.path so tests can
import the top-level modules (orchestrator, scheduler, ...) directly."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
