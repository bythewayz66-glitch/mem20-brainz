"""Simulation engine.

A single-process, single-threaded reference stepping loop. It exists to pin down
interfaces (state layout, spike delivery, timestep semantics) before any
performance work. It cannot run brain-scale workloads.
"""

from .engine import SimulationEngine

__all__ = ["SimulationEngine"]
