"""Event-driven spike delivery over a CSR connectome.

The reference engine scans the whole edge list every step::

    mask = np.isin(connectome.pre, spikes)          # O(E) per step
    np.add.at(current, connectome.post[mask], connectome.weight[mask])

That is O(E) regardless of how many neurons actually fired. At 10^14 synapses
and a 1 Hz mean rate, ~10^10 edges fire per 0.1 ms step out of 10^14 — a 10^-4
hit rate. Scanning all of them is the single largest waste in the engine.

This module replaces the scan with a lookup: for each neuron that spiked, jump
straight to its CSR row and touch only those edges. Cost becomes
O(sum of out-degrees of the neurons that fired) instead of O(E).

Why this is NOT bit-identical
-----------------------------
The reference sums contributions in **edge-list order**. This path sums them in
**CSR order**, which is grouped by presynaptic neuron. Floating-point addition
is not associative, so the two orders can differ in the last bits, and a
difference in the last bit of a membrane potential can flip a spike that sits
exactly on threshold.

That is a real, unavoidable consequence of the optimisation, not a bug. It is
why this path is gated on a tolerance-based test (see
``tests/test_event_driven.py``) and why ADR-0006 exists. Claiming bit-identity
here would be false.

One case *is* exact: if the edge list is already grouped by ``pre``, CSR order
equals edge-list order and the sums are identical. The test suite checks that
case explicitly, because it is a useful sanity check on the CSR build.
"""

from __future__ import annotations

import numpy as np

from ..connectome.csr import CSRConnectome
from .backend import Backend, LIFState, NumPyBackend


class EventDrivenEngine:
    """Vectorised engine with CSR event-driven delivery.

    Semantics are otherwise identical to :class:`VectorizedEngine`: spikes at
    step ``k`` are delivered at step ``k + 1``, ``spike_history[k]`` lists the
    ids that spiked at step ``k``, and ``t`` advances by ``dt``.

    Note on delays: like the reference engine, ``delay_ms`` is currently
    *ignored* — every spike is delivered on the next step. That is a known
    simplification shared by all three engines, not something this module
    introduces. Fixing it is a separate change that must land in the reference
    first, so the reference stays the ground truth.
    """

    def __init__(
        self,
        state: LIFState,
        connectome: CSRConnectome,
        dt: float = 0.1,
        backend: Backend | None = None,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if connectome.n_neurons != state.n_neurons:
            raise ValueError(
                f"connectome has {connectome.n_neurons} neurons, population has {state.n_neurons}"
            )
        self.state = state
        self.connectome = connectome
        self.dt = dt
        self.backend: Backend = backend if backend is not None else NumPyBackend()
        self.t = 0.0
        self.spike_history: list[list[int]] = []
        #: Number of edges actually visited, cumulative. The whole point of this
        #: engine is that this is far below ``n_synapses * steps``.
        self.edges_visited = 0

    @property
    def n_neurons(self) -> int:
        return self.state.n_neurons

    def _deliver(self, spikes: list[int]) -> np.ndarray:
        """Sum synaptic current from the CSR rows of the neurons that spiked."""
        current = np.zeros(self.n_neurons, dtype=np.float64)
        if not spikes:
            return current

        edges = self.connectome.edge_indices_for(np.asarray(spikes, dtype=np.int64))
        if edges.size == 0:
            return current

        self.edges_visited += int(edges.size)
        posts = self.connectome.indices[edges]
        weights = self.connectome.weights[edges]

        # Segmented reduction instead of an unbuffered scatter.
        #
        # ``np.add.at`` is correct but slow: it is an unbuffered scatter, so it
        # cannot vectorise and pays a per-element Python-level loop internally.
        # ``np.bincount`` with weights is a single pass that accumulates into
        # bins, which is exactly the operation we want (sum weights by
        # postsynaptic id).
        #
        # Why bincount and not ``np.add.reduceat``: reduceat needs the array
        # *sorted* by the segment key and a segment-boundary index, so it would
        # cost an argsort per step. bincount needs neither — it indexes directly
        # by the post id. The trade is that bincount always allocates a full
        # ``n_neurons`` output, which is fine because ``current`` is that size
        # anyway.
        #
        # This changes the float addition order again (bincount accumulates in
        # edge order within each bin, which is the same order as add.at, but the
        # implementation may use pairwise summation internally). It is therefore
        # still NOT bit-identical to the reference, and the tolerance test in
        # tests/test_event_driven.py remains the gate.
        current = np.bincount(posts, weights=weights, minlength=self.n_neurons)
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
