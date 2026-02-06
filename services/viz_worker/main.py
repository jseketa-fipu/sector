#!/usr/bin/env python3
"""
Visualization service serving the frontend and streaming snapshots/events.

- Serves the static index.html bundled with this service.
- Proxies the Redis-backed snapshot and events (WebSocket) for the UI.
"""
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
import urllib.request
import urllib.parse

from sector.config import SIM_CONFIG
from sector.infra.redis_streams import RedisStreams
from sector.state_utils import snapshot_from_world  # type: ignore
from sector.world import create_sector

from models import VisualizationWorkerSettings


config = VisualizationWorkerSettings()  # type: ignore[call-arg]

BASE_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = BASE_DIR / "index.html"
TAIL_COUNT_DEFAULT = 50
DEFAULT_TICK_DELAY = float(SIM_CONFIG.simulation_modifiers.tick_delay)

streams = RedisStreams(url=config.redis_url)
app = FastAPI(title="Sector Viz", version="0.1.0")
API_URL = config.api_url


@app.get("/")
async def serve_index():
    return FileResponse(INDEX_PATH)


@app.get("/snapshot")
async def snapshot():
    snap = await streams.load_snapshot()
    if not snap:
        return JSONResponse({"error": "no snapshot yet"}, status_code=404)
    return snap


@app.get("/tail")
async def tail(count: int = TAIL_COUNT_DEFAULT):
    events = await streams.tail_events(count=count)
    return events


@app.get("/preview")
async def preview(seed: str):
    """
    Generate a deterministic preview of the sector without touching the live sim.
    """
    world = create_sector(seed=seed)
    snap = snapshot_from_world(
        world, tick_delay=DEFAULT_TICK_DELAY, include_ai_state=False
    )
    return snap


@app.api_route("/api/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def proxy_api(path: str, request: Request):
    """
    Proxy API calls to the internal API service so the browser can stay same-origin.
    """
    base = API_URL.rstrip("/")
    tail = path.lstrip("/")
    url = f"{base}/{tail}" if tail else base
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()
    headers = {}
    if "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]
    if "content-type" in request.headers:
        headers["Content-Type"] = request.headers["content-type"]

    if request.method == "OPTIONS":
        return Response(status_code=204)

    req = urllib.request.Request(
        url, data=body or None, headers=headers, method=request.method
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = resp.read()
            content_type = resp.headers.get("Content-Type", "application/json")
            return Response(
                content=payload, status_code=resp.status, media_type=content_type
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        content_type = exc.headers.get("Content-Type", "application/json")
        return Response(content=payload, status_code=exc.code, media_type=content_type)


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    snap = await streams.load_snapshot()
    if snap:
        await ws.send_json(snap)
    last_id = "$"
    try:
        while True:
            entries = await streams.read_events(
                last_id=last_id, count=200, block_ms=500
            )
            if not entries:
                # If no new events, send a keepalive snapshot to keep UI in sync.
                snap = await streams.load_snapshot()
                if snap:
                    await ws.send_json(snap)
                continue
            for entry_id, payload in entries:
                last_id = entry_id
                await ws.send_json(payload)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run(
        "services.viz_worker.main:app",
        host="0.0.0.0",
        port=config.port,
        reload=False,
    )
