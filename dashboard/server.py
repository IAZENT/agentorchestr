"""
dashboard/server.py
====================
Minimal FastAPI dashboard for ORCH.

Endpoints:
  GET  /                     → HTML overview (sessions list)
  GET  /api/sessions         → JSON list of all sessions
  GET  /api/sessions/{id}    → session detail (plan + tasks + ledger)
  GET  /api/sessions/{id}/tasks → task results
  GET  /healthz              → liveness probe

Launch:
  python orchestrator.py --dashboard
  python -m dashboard.server                 # standalone
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from state_store import StateStore


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ORCH Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0e1116; color: #c9d1d9; margin: 0; padding: 24px; }
  h1 { color: #58a6ff; margin: 0 0 16px; font-size: 22px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 24px; }
  table { border-collapse: collapse; width: 100%; max-width: 1100px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; font-size: 13px; }
  th { color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 11px; }
  tr:hover { background: #161b22; }
  .status-active { color: #3fb950; }
  .status-paused { color: #d29922; }
  .status-done   { color: #8b949e; }
  .status-failed { color: #f85149; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  pre { background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
          background: #21262d; color: #c9d1d9; margin-right: 4px; }
</style>
</head>
<body>
  <h1>ORCH — sessions</h1>
  <div class="sub">live-refreshing every 5s · <a href="/api/sessions">json</a></div>
  <table id="t">
    <thead>
      <tr><th>id</th><th>status</th><th>goal</th><th>tasks done/total</th><th>updated</th></tr>
    </thead>
    <tbody></tbody>
  </table>
<script>
async function refresh() {
  const r = await fetch('/api/sessions');
  const data = await r.json();
  const tbody = document.querySelector('#t tbody');
  tbody.innerHTML = '';
  for (const s of data.sessions) {
    const tr = document.createElement('tr');
    const updated = s.updated_at ? new Date(s.updated_at * 1000).toLocaleString() : '';
    tr.innerHTML = `
      <td><a href="/api/sessions/${s.id}">${s.id}</a></td>
      <td><span class="status-${s.status}">${s.status}</span></td>
      <td>${(s.goal || '').slice(0, 90)}</td>
      <td>${s.done}/${s.total}</td>
      <td>${updated}</td>`;
    tbody.appendChild(tr);
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def build_app(store: StateStore) -> FastAPI:
    app = FastAPI(title="ORCH Dashboard", docs_url="/api/docs", redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_HTML)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/api/sessions")
    async def list_sessions() -> JSONResponse:
        # Aggregate: pull all session rows + per-session task counts.
        if store._db is None:
            return JSONResponse({"sessions": []})
        sessions: list[dict] = []
        async with store._db.execute(
            "SELECT id, goal, status, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 50"
        ) as cur:
            async for row in cur:
                sid = row[0]
                results = await store.get_results(sid)
                done = sum(1 for r in results if r["status"] == "done")
                sessions.append({
                    "id": sid,
                    "goal": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "total": len(results),
                    "done": done,
                })
        return JSONResponse({"sessions": sessions})

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str) -> JSONResponse:
        session = await store.get_session(session_id)
        if not session:
            raise HTTPException(404, f"session {session_id} not found")
        ledger = await store.get_task_ledger(session_id)
        results = await store.get_results(session_id)
        return JSONResponse({
            "session": session,
            "task_ledger": ledger,
            "tasks": results,
        })

    @app.get("/api/sessions/{session_id}/tasks")
    async def session_tasks(session_id: str) -> JSONResponse:
        return JSONResponse({"tasks": await store.get_results(session_id)})

    return app


async def start_dashboard(store: StateStore, host: str = "127.0.0.1", port: int = 3000) -> None:
    """Run the dashboard until cancelled.

    Returns when the user hits Ctrl-C; uvicorn handles signal teardown.
    """
    import uvicorn

    app = build_app(store)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    print(f"ORCH dashboard → http://{host}:{port}")
    await server.serve()


def _cli() -> int:
    """Standalone entry point: `python -m dashboard.server`."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    async def _run() -> None:
        store = StateStore()
        await store.init()
        try:
            await start_dashboard(store, host=args.host, port=args.port)
        finally:
            await store.close()

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
