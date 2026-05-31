"""ORCH dashboard — minimal FastAPI app for live session inspection."""
from .server import start_dashboard, build_app

__all__ = ["start_dashboard", "build_app"]
