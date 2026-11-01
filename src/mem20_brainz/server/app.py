"""FastAPI app: orchestration and serialisation for the live simulation.

This module is a thin shell over :class:`~mem20_brainz.server.runner.RunRunner`.
It contains no simulation logic and no arithmetic beyond JSON shaping. Every
voltage update happens in the backend; every derived number comes from
:mod:`mem20_brainz.readout`.

Endpoints
---------
``GET  /api/health``            liveness plus a one-line status
``GET  /api/state``             static snapshot: config, regions, layout, provenance
``POST /api/run/start``         begin streaming
``POST /api/run/pause``         stop advancing
``POST /api/run/reset``         rebuild from the configured seed
``POST /api/run/step``          advance exactly one frame, return it
``POST /api/config``            set run parameters
``POST /api/stimulus``          inject current into a region
``GET  /api/frame``             one frame as JSON (no WebSocket needed)
``WS   /ws``                    stream of frames

The WebSocket sends a ``hello`` message (the snapshot) on connect, then a
``frame`` message per step, and accepts ``{"type": "ping"}`` keepalives.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .runner import DEFAULT_STIMULUS_STRENGTH, RunConfig, RunRunner

#: The single process-wide runner. One run at a time is a deliberate limit:
#: this is an observability tool, not a multi-tenant service (ADR-0007).
_runner: RunRunner | None = None


def get_runner() -> RunRunner:
    """Return the process-wide runner, creating it on first use."""
    global _runner
    if _runner is None:
        _runner = RunRunner()
    return _runner


def set_runner(runner: RunRunner | None) -> None:
    """Replace the process-wide runner. Used by tests to isolate state."""
    global _runner
    _runner = runner


class ConfigUpdate(BaseModel):
    """Partial update of run parameters. Omitted fields are left alone."""

    dt_ms: float | None = Field(default=None, gt=0.0, le=10.0)
    steps_per_frame: int | None = Field(default=None, ge=1, le=200)
    target_fps: float | None = Field(default=None, gt=0.0, le=240.0)
    baseline_rate_hz: float | None = Field(default=None, gt=0.0, le=1000.0)
    threat_region: str | None = None
    seed: int | None = None


class StimulusRequest(BaseModel):
    """Inject an external current into one region."""

    region: str
    strength: float = Field(default=DEFAULT_STIMULUS_STRENGTH, ge=-100.0, le=100.0)
    duration_ms: float = Field(default=200.0, gt=0.0, le=60_000.0)


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="mem20-brainz live simulation",
        version="0.0.1",
        description=(
            "Streams simulation state from the mem20-brainz engine. "
            "Affective readouts are derived proxies over simulation state, "
            "not measurements of feeling."
        ),
    )

    # The Vite dev server runs on a different port, so the browser needs CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        runner = get_runner()
        with runner.locked():
            return {
                "ok": True,
                "backend": runner.engine.backend.name,
                "n_neurons": runner.network.n_neurons,
                "n_synapses": runner.network.n_synapses,
                "running": runner.running,
                "data_kind": runner.network.provenance.get("kind", "unknown"),
            }

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return get_runner().snapshot()

    @app.post("/api/run/start")
    def start() -> dict[str, Any]:
        runner = get_runner()
        runner.start()
        return {"ok": True, "running": True}

    @app.post("/api/run/pause")
    def pause() -> dict[str, Any]:
        runner = get_runner()
        runner.pause()
        return {"ok": True, "running": False}

    @app.post("/api/run/reset")
    def reset() -> dict[str, Any]:
        runner = get_runner()
        runner.reset()
        return {"ok": True, "state": runner.snapshot()}

    @app.post("/api/run/step")
    def step() -> dict[str, Any]:
        return get_runner().frame().to_dict()

    @app.get("/api/frame")
    def frame() -> dict[str, Any]:
        return get_runner().frame().to_dict()

    @app.post("/api/config")
    def config(update: ConfigUpdate) -> dict[str, Any]:
        runner = get_runner()
        try:
            cfg = runner.set_config(**update.model_dump(exclude_none=True))
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "config": cfg.to_dict()}

    @app.post("/api/stimulus")
    def stimulus(req: StimulusRequest) -> dict[str, Any]:
        runner = get_runner()
        try:
            stim = runner.inject(req.region, req.strength, req.duration_ms)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "stimulus": stim.to_dict()}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        runner = get_runner()
        await websocket.send_json({"type": "hello", "state": runner.snapshot()})

        async def pump() -> None:
            """Send frames while the run is started, at the target rate."""
            while True:
                with runner.locked():
                    running = runner.running
                    fps = runner.config.target_fps
                if not running:
                    await asyncio.sleep(0.05)
                    continue
                payload = await asyncio.to_thread(lambda: runner.frame().to_dict())
                await websocket.send_json({"type": "frame", "frame": payload})
                await asyncio.sleep(1.0 / max(fps, 1.0))

        async def listen() -> None:
            """Consume client messages so pings and disconnects are noticed."""
            while True:
                msg = await websocket.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        pump_task = asyncio.create_task(pump())
        listen_task = asyncio.create_task(listen())
        try:
            done, pending = await asyncio.wait(
                {pump_task, listen_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                # Surface a real error; a disconnect is normal and is swallowed.
                with contextlib.suppress(WebSocketDisconnect, asyncio.CancelledError):
                    task.result()
        except WebSocketDisconnect:
            pass
        finally:
            for task in (pump_task, listen_task):
                task.cancel()
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


app = create_app()


def main() -> None:
    """Entry point for ``python -m mem20_brainz.server``."""
    import uvicorn

    uvicorn.run("mem20_brainz.server.app:app", host="127.0.0.1", port=8000, reload=False)


__all__ = ["RunConfig", "app", "create_app", "get_runner", "main", "set_runner"]
