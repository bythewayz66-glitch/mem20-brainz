"""Compressed-sparse-row connectome view, indexed by presynaptic neuron.

Why CSR and not CSC
-------------------
The simulation asks one question per timestep: *given the set of neurons that
just spiked, which synapses fire?* That is an out-edge query, so the row index
must be the presynaptic neuron. CSC (indexed by postsynaptic neuron) would
answer the opposite question and force a full scan to answer this one.

Memory layout
-------------
Three parallel arrays plus a pointer array, all C-contiguous::

    indptr   int64   length n_neurons + 1
    indices  int32   length n_synapses      postsynaptic neuron id
    weights  float32 length n_synapses
    delays   float32 length n_synapses

The out-edges of neuron ``i`` are the slice ``[indptr[i] : indptr[i + 1]]``.
``indptr`` is ``int64`` because at 10^14 synapses a 32-bit pointer overflows;
``indices`` is ``int32`` because 86 x 10^9 neurons still fits in 32 bits
(2.1 x 10^9 is not enough — see the note below), and halving that array is
worth 200 TB at human scale.

.. warning::
   ``int32`` indices cap the population at 2,147,483,647 neurons. That is
   comfortably above 86 x 10^9? No — it is not. 86 x 10^9 exceeds 2^31 by a
   factor of 40. This view is therefore **not** usable at human scale as
   written; it is correct and fast for the 10^4-10^8 range that Phases 1-4
   actually run at. Widening ``indices`` to ``int64`` is a one-line change and
   costs 400 TB at 10^14 synapses, which is why it is deferred rather than
   done now. Recorded here so nobody discovers it at 10^9 neurons.

Cost
----
Building the view is one stable sort of the edge list: O(E log E) time and
O(E) extra memory. It is a one-off per connectome, not per timestep.

Sorting is *stable*, so edges that share a presynaptic neuron keep their
original relative order. That matters: it is what makes the event-driven
delivery path reproduce the reference engine's float addition order whenever
the edge list is already grouped by ``pre``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sparse import SparseConnectome


@dataclass
class CSRConnectome:
    """Row-compressed connectome indexed by presynaptic neuron."""

    n_neurons: int
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    delays: np.ndarray

    def __post_init__(self) -> None:
        if self.indptr.dtype != np.int64:
            raise TypeError(f"indptr must be int64 (got {self.indptr.dtype})")
        if len(self.indptr) != self.n_neurons + 1:
            raise ValueError(
                f"indptr has length {len(self.indptr)}, "
                f"expected n_neurons + 1 = {self.n_neurons + 1}"
            )
        if not (len(self.indices) == len(self.weights) == len(self.delays)):
            raise ValueError("indices/weights/delays must have equal length")
        if self.indptr[0] != 0:
            raise ValueError("indptr[0] must be 0")
        if self.indptr[-1] != len(self.indices):
            raise ValueError(
                f"indptr[-1] is {self.indptr[-1]}, expected n_synapses = {len(self.indices)}"
            )
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("indptr must be non-decreasing")

    @property
    def n_synapses(self) -> int:
        return len(self.indices)

    @property
    def nbytes(self) -> int:
        """Bytes held by the four arrays (not counting object overhead)."""
        return self.indptr.nbytes + self.indices.nbytes + self.weights.nbytes + self.delays.nbytes

    def out_degree(self) -> np.ndarray:
        """Number of outgoing synapses per neuron."""
        return np.diff(self.indptr)

    def out_slice(self, neuron_id: int) -> slice:
        """The edge slice for one presynaptic neuron."""
        return slice(int(self.indptr[neuron_id]), int(self.indptr[neuron_id + 1]))

    def edge_indices_for(self, neuron_ids: np.ndarray) -> np.ndarray:
        """Flat edge indices for a set of presynaptic neurons, vectorised.

        This is the core of event-driven delivery: it turns "these neurons
        spiked" into "these edges fire" without a Python loop over neurons.

        The returned indices are grouped by neuron in the order given, and
        within a neuron in CSR order. Duplicate neuron ids are *not*
        deduplicated — a neuron that appears twice contributes its edges twice,
        which is the caller's problem to avoid.
        """
        ids = np.asarray(neuron_ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= self.n_neurons:
            raise IndexError(f"neuron id out of range [0, {self.n_neurons})")

        starts = self.indptr[ids]
        counts = self.indptr[ids + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)

        # offsets[k] = number of edges contributed by neurons before k
        offsets = np.cumsum(counts) - counts
        # position within the concatenated block, minus the block's own start
        within = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)
        return np.repeat(starts, counts) + within

    @classmethod
    def from_sparse(cls, connectome: SparseConnectome) -> CSRConnectome:
        """Build the view from a coordinate-list connectome.

        Uses a stable sort on ``pre`` so that edges sharing a presynaptic neuron
        keep their original relative order.
        """
        n = connectome.n_neurons
        pre = np.asarray(connectome.pre, dtype=np.int64)
        if pre.size and (pre.min() < 0 or pre.max() >= n):
            raise IndexError(f"pre id out of range [0, {n})")

        order = np.argsort(pre, kind="stable")
        pre_sorted = pre[order]

        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre_sorted, minlength=n), out=indptr[1:])

        return cls(
            n_neurons=n,
            indptr=indptr,
            indices=np.asarray(connectome.post, dtype=np.int32)[order],
            weights=np.asarray(connectome.weight, dtype=np.float32)[order],
            delays=np.asarray(connectome.delay_ms, dtype=np.float32)[order],
        )

    def to_sparse(self) -> SparseConnectome:
        """Expand back to a coordinate list (CSR order, not original order)."""
        pre = np.repeat(np.arange(self.n_neurons, dtype=np.int32), self.out_degree())
        return SparseConnectome.from_arrays(
            n_neurons=self.n_neurons,
            pre=pre,
            post=self.indices,
            weight=self.weights,
            delay_ms=self.delays,
        )
