"""Vectorised simulation backends.

This module is the seam described in ADR-0003. The reference engine in
:mod:`mem20_brainz.simulation.engine` is the ground truth: it is a per-neuron
Python loop, it is slow, and it is never optimised. Everything fast lives here,
behind the :class:`Backend` protocol, and must be provably equivalent to it.

What is implemented
-------------------
* :class:`LIFState` — struct-of-arrays state for a LIF population.
* :class:`Backend` — the protocol every backend satisfies.
* :class:`NumPyBackend` — elementwise array kernel, **bit-identical** to the
  reference loop.
* :class:`VectorizedEngine` — the reference engine with only the per-neuron loop
  replaced. Delivery, delay handling, spike history and clock semantics are
  unchanged.

Why bit-identity is achievable here
-----------------------------------
The reference update is a sequence of scalar IEEE-754 operations. Applying the
same operations elementwise over an array produces the same bits per element:
floating-point addition and multiplication are deterministic per element and do
not depend on whether a Python loop or a NumPy ufunc drives them. The only way to
lose bit-identity is to *reassociate* the expression, which this backend
deliberately does not do.

What is deliberately NOT here
-----------------------------
CSR event-driven delivery. It is the next optimisation and it is **not**
bit-identical, because summing incoming weights in CSR order rather than edge
order is a different order of float additions. It will need a tolerance-based
equivalence test with a stated bound, and an ADR that says so. Claiming
bit-identity for it would be false.

Precision
---------
State defaults to ``float64``, which is the only precision that is
reference-equivalent. ``float32`` is available as an explicit opt-in
(``precision="float32"``) because it halves state memory, but it is **not**
reference-equivalent: measured on a heterogeneous population it drifts spike
count by ~0.14% and agrees on only ~14% of timesteps. See ADR-0007 for the
experiment and the decision. A backend refuses a state whose precision does not
match its own, rather than silently up- or down-casting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..connectome.sparse import SparseConnectome
from ..neurons.base import LIFNeuron

#: The only tier this backend implements. See ADR-0002.
TIER0 = "tier0_point_lif"

_FLOAT_FIELDS = (
    "v",
    "v_rest",
    "v_reset",
    "v_threshold",
    "tau_m",
    "r_m",
    "refractory_ms",
    "refractory_until",
)


#: Supported state precisions. ``float64`` is the reference-equivalent default;
#: ``float32`` is an opt-in memory trade documented in ADR-0007.
_DTYPES = {"float64": np.float64, "float32": np.float32}


def _full(value: float, n: int, dtype: type = np.float64) -> np.ndarray:
    """Broadcast a scalar to an array of length ``n``."""
    return np.full(n, float(value), dtype=dtype)


@dataclass
class LIFState:
    """Struct-of-arrays state for a homogeneous LIF population.

    Every field is a parallel ``float64`` array of length ``n_neurons``, except
    ``tier`` which is a fixed-width string array. This is the layout that makes
    a model class a kernel over arrays instead of a list of objects.
    """

    v: np.ndarray
    v_rest: np.ndarray
    v_reset: np.ndarray
    v_threshold: np.ndarray
    tau_m: np.ndarray
    r_m: np.ndarray
    refractory_ms: np.ndarray
    refractory_until: np.ndarray
    tier: np.ndarray
    precision: str = "float64"

    def __post_init__(self) -> None:
        if self.precision not in _DTYPES:
            raise ValueError(
                f"unknown precision {self.precision!r}; expected one of {sorted(_DTYPES)}"
            )
        dtype = _DTYPES[self.precision]
        n = len(self.v)
        for name in _FLOAT_FIELDS:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"LIFState field {name!r} has length {len(arr)}, expected {n}")
            if arr.dtype != dtype:
                raise TypeError(
                    f"LIFState field {name!r} must be {self.precision} (got {arr.dtype}); "
                    "a mixed-precision state is not reference-equivalent (ADR-0007)"
                )
        if len(self.tier) != n:
            raise ValueError(f"LIFState field 'tier' has length {len(self.tier)}, expected {n}")

    @property
    def n_neurons(self) -> int:
        return len(self.v)

    @classmethod
    def from_neurons(cls, neurons: list[LIFNeuron], precision: str = "float64") -> LIFState:
        """Build a state array from a list of reference ``LIFNeuron`` objects.

        This is the bridge used by the equivalence tests: the same parameters
        that drive the reference loop drive the array kernel.
        """
        if not neurons:
            raise ValueError("cannot build LIFState from an empty population")
        for neuron in neurons:
            if not isinstance(neuron, LIFNeuron):
                raise TypeError(
                    f"LIFState.from_neurons expects LIFNeuron, got {type(neuron).__name__}"
                )
        if precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(_DTYPES)}")
        dtype = _DTYPES[precision]
        return cls(
            v=np.array([float(x.v) for x in neurons], dtype=dtype),
            v_rest=np.array([float(x.v_rest) for x in neurons], dtype=dtype),
            v_reset=np.array([float(x.v_reset) for x in neurons], dtype=dtype),
            v_threshold=np.array([float(x.v_threshold) for x in neurons], dtype=dtype),
            tau_m=np.array([float(x.tau_m) for x in neurons], dtype=dtype),
            r_m=np.array([float(x.r_m) for x in neurons], dtype=dtype),
            refractory_ms=np.array([float(x.refractory_ms) for x in neurons], dtype=dtype),
            refractory_until=np.array([float(x._refractory_until) for x in neurons], dtype=dtype),
            tier=np.array([getattr(x, "tier", TIER0) for x in neurons], dtype="<U32"),
            precision=precision,
        )

    @classmethod
    def uniform(cls, n_neurons: int, precision: str = "float64", **overrides: float) -> LIFState:
        """Build a homogeneous population using ``LIFNeuron``'s defaults."""
        if precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(_DTYPES)}")
        dtype = _DTYPES[precision]
        defaults = LIFNeuron(id=0)
        params = {
            "v": defaults.v,
            "v_rest": defaults.v_rest,
            "v_reset": defaults.v_reset,
            "v_threshold": defaults.v_threshold,
            "tau_m": defaults.tau_m,
            "r_m": defaults.r_m,
            "refractory_ms": defaults.refractory_ms,
            "refractory_until": 0.0,
        }
        unknown = set(overrides) - set(params)
        if unknown:
            raise TypeError(f"unknown LIF parameter(s): {sorted(unknown)}")
        params.update(overrides)
        return cls(
            **{name: _full(value, n_neurons, dtype) for name, value in params.items()},
            tier=np.full(n_neurons, TIER0, dtype="<U32"),
            precision=precision,
        )


@runtime_checkable
class Backend(Protocol):
    """The only seam between the engine and the arithmetic (ADR-0003).

    Contract:

    1. ``step`` is pure with respect to everything except ``state``. It reads
       ``i_inj``, ``dt`` and ``t``; it mutates only arrays inside ``state``; it
       returns a boolean spike mask. It must not read the connectome, the clock,
       or any global.
    2. ``state`` is a struct-of-arrays, never a list of objects.
    3. Operation order is part of the contract. A backend may not reassociate
       the update expression.
    4. A backend that does not implement a tier must raise, never substitute.
    """

    name: str

    def step(
        self,
        state: LIFState,
        i_inj: np.ndarray,
        dt: float,
        t: float,
    ) -> np.ndarray:
        """Advance one timestep in place; return a boolean spike mask."""
        ...


class NumPyBackend:
    """Elementwise NumPy kernel, bit-identical to the reference LIF loop.

    The reference update, verbatim::

        if t < refractory_until: return False
        dv = (-(v - v_rest) + r_m * i_inj) / tau_m
        v += dv * dt
        if v >= v_threshold:
            v = v_reset
            refractory_until = t + refractory_ms
            return True
        return False

    The array form below applies exactly these operations, in exactly this
    order, to every element. It does not fuse, factor or reassociate anything.
    """

    name = "numpy"

    def __init__(self, precision: str = "float64") -> None:
        if precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(_DTYPES)}")
        self.precision = precision

    def step(
        self,
        state: LIFState,
        i_inj: np.ndarray,
        dt: float,
        t: float,
    ) -> np.ndarray:
        unsupported = state.tier != TIER0
        if unsupported.any():
            bad = sorted(set(state.tier[unsupported].tolist()))
            raise NotImplementedError(
                f"NumPyBackend implements only {TIER0!r}; population contains {bad}. "
                "Refusing to substitute a cheaper model (ADR-0002)."
            )
        if state.precision != self.precision:
            raise ValueError(
                f"NumPyBackend(precision={self.precision!r}) cannot step a "
                f"{state.precision!r} state; construct the backend with the same "
                "precision as the state (ADR-0007)"
            )

        i_inj = np.broadcast_to(np.asarray(i_inj, dtype=_DTYPES[self.precision]), state.v.shape)

        # A refractory neuron neither integrates nor spikes; its v is untouched.
        active = t >= state.refractory_until

        dv = (-(state.v - state.v_rest) + state.r_m * i_inj) / state.tau_m
        v_next = state.v + dv * dt

        state.v = np.where(active, v_next, state.v)

        spiked = active & (state.v >= state.v_threshold)
        state.v = np.where(spiked, state.v_reset, state.v)
        state.refractory_until = np.where(spiked, t + state.refractory_ms, state.refractory_until)
        return spiked


class VectorizedEngine:
    """The reference engine with the per-neuron loop replaced by a backend.

    Everything else is preserved exactly: the delivery step is the same code as
    :meth:`SimulationEngine._deliver`, delays are handled by the same one-step
    rule, ``spike_history[k]`` lists the ids that spiked at step ``k``, and ``t``
    advances by ``dt`` per step.

    This is a Phase 1 deliverable (ADR-0003): it removes per-neuron Python
    objects from the hot path without changing semantics. It is still O(synapses)
    per step and still single-process; it is not a brain-scale engine.
    """

    def __init__(
        self,
        state: LIFState,
        connectome: SparseConnectome,
        dt: float = 0.1,
        backend: Backend | None = None,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if connectome.n_neurons and connectome.n_neurons != state.n_neurons:
            raise ValueError(
                f"connectome has {connectome.n_neurons} neurons, population has {state.n_neurons}"
            )
        self.state = state
        self.connectome = connectome
        self.dt = dt
        self.backend: Backend = backend if backend is not None else NumPyBackend()
        self.t = 0.0
        self.spike_history: list[list[int]] = []

    @property
    def n_neurons(self) -> int:
        return self.state.n_neurons

    def _deliver(self, spikes: list[int]) -> np.ndarray:
        """Sum synaptic current arriving this step from ``spikes``.

        Byte-for-byte the reference implementation. Changing this would change
        results, so it is not a place to optimise yet.
        """
        current = np.zeros(self.n_neurons, dtype=np.float64)
        if not spikes:
            return current
        mask = np.isin(self.connectome.pre, spikes)
        if not mask.any():
            return current
        np.add.at(current, self.connectome.post[mask], self.connectome.weight[mask])
        return current

    def step(self, external_current: np.ndarray | None = None) -> list[int]:
        """Advance one timestep and return the ids of neurons that spiked."""
        prev_spikes = self.spike_history[-1] if self.spike_history else []
        current = self._deliver(prev_spikes)
        if external_current is not None:
            current = current + np.asarray(external_current, dtype=np.float64)

        spiked = self.backend.step(self.state, current, self.dt, self.t)
        spikes = np.flatnonzero(spiked).tolist()

        self.spike_history.append(spikes)
        self.t += self.dt
        return spikes

    def run(self, steps: int, external_current: np.ndarray | None = None) -> int:
        """Run ``steps`` timesteps; return total spike count."""
        total = 0
        for _ in range(steps):
            total += len(self.step(external_current))
        return total
