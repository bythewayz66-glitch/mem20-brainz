"""Simulation engine.

Two engines live here, and the difference between them is the whole point:

* :class:`SimulationEngine` — the reference. A per-neuron Python loop. Slow,
  correct, and frozen as ground truth (ADR-0003). It is never optimised.
* :class:`VectorizedEngine` — the same semantics with the per-neuron loop
  replaced by a :class:`Backend`. Currently backed by :class:`NumPyBackend`,
  which is bit-identical to the reference.

Neither is brain-scale. Both are single-process and O(synapses) per step.
"""

from .backend import Backend, LIFState, NumPyBackend, VectorizedEngine
from .engine import SimulationEngine
from .event_driven import EventDrivenEngine

__all__ = [
    "Backend",
    "EventDrivenEngine",
    "LIFState",
    "NumPyBackend",
    "SimulationEngine",
    "VectorizedEngine",
]


def __getattr__(name: str):
    """Expose NumbaBackend lazily so numba stays an optional dependency."""
    if name == "NumbaBackend":
        from .numba_backend import NumbaBackend

        return NumbaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
