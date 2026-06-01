"""agentorchestr — supervisor / worker orchestrator for terminal coding agents.

Public re-exports kept deliberately small so the import surface stays
stable across refactors.  Library consumers should ``from agentorchestr
import …`` rather than reach into submodules directly.
"""
from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("agentorchestr")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
except ImportError:  # pragma: no cover - importlib.metadata always present on py3.11+
    __version__ = "0.0.0+local"

# Stable re-exports.  Add cautiously; every symbol here is a public
# contract.
from agentorchestr.router import LLMRouter  # noqa: E402
from agentorchestr.state_store import StateStore  # noqa: E402
from agentorchestr.supervisor import Supervisor  # noqa: E402
from agentorchestr.worker_pool import Worker, WorkerPool  # noqa: E402

__all__ = [
    "__version__",
    "LLMRouter",
    "StateStore",
    "Supervisor",
    "Worker",
    "WorkerPool",
]
