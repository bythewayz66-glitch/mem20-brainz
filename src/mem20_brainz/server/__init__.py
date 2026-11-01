"""Live observability layer: a streaming server over the existing engine.

The server orchestrates and serialises; it does not simulate. All arithmetic
lives in :mod:`mem20_brainz.simulation` (the engine and its backends) and
:mod:`mem20_brainz.readout` (the derived proxies). See ADR-0007.

Run it::

    python -m mem20_brainz.server          # uvicorn on 127.0.0.1:8000
    python -m mem20_brainz.server.headless # text summary, no browser
"""

from .frames import FRAME_SCHEMA_VERSION, Frame, RegionActivity, build_frame
from .runner import RunConfig, RunRunner, RunStats, Stimulus

__all__ = [
    "FRAME_SCHEMA_VERSION",
    "Frame",
    "RegionActivity",
    "RunConfig",
    "RunRunner",
    "RunStats",
    "Stimulus",
    "build_frame",
]
