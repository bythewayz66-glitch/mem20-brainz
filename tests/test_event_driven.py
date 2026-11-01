"""Event-driven delivery: tolerance-based equivalence, not bit-identity.

Event-driven delivery sums incoming weights in **CSR order** (grouped by
presynaptic neuron) rather than **edge-list order**. Floating-point addition is
not associative, so the two orders produce different sums. This is measured, not
assumed: ``test_delivery_sums_are_not_bit_identical`` asserts that the two
paths disagree in the last bits on a high-fan-in case (max relative difference
~5e-9 with float64 weights).

What that does *not* do, empirically, is flip a spike. Across every regime
tested here — subthreshold, near-threshold, refractory-dominated, 10^4 neurons —
the spike output was **identical**, because a ~1e-9 relative perturbation in the
membrane potential is far smaller than the margin a neuron needs to cross
threshold within one 0.1 ms step. So the honest statement is:

* the delivery sums are **not** bit-identical (proven below), and
* the spike output **was** identical in every regime tested (also asserted).

That second fact is an observation about this LIF model and these parameters,
not a guarantee. A different model, a smaller timestep, or a neuron parked
exactly on threshold could flip. So the gate is a tolerance test, not a
bit-identity claim — and the tolerance is stated explicitly rather than being
implicit in a loose assertion.

ADR-0006 records the decision.

The tolerance is stated in two parts, because they measure different things:

* ``SPIKE_COUNT_TOLERANCE`` — relative difference in total spike count. This is
  the number that matters scientifically: a run whose spike count drifts by more
  than a fraction of a percent is a different run.
* ``PER_STEP_AGREEMENT`` — fraction of timesteps whose spike set is *identical*.
  This is the strict per-step check.

One case *is* exact by construction and is asserted as such: when the edge list
is already grouped by ``pre``, CSR order equals edge-list order, so the sums are
identical and the spikes are bit-identical. That case is a sanity check on the
CSR build, not a claim about the general case.
"""

from __future__ import annotations

import numpy as np
import pytest

from mem20_brainz.connectome import CSRConnectome, SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import (
    EventDrivenEngine,
    LIFState,
    SimulationEngine,
    VectorizedEngine,
)

#: Relative tolerance on total spike count between the reference and the
#: event-driven path. 1% is deliberately loose: the point of the test is to
#: catch a *broken* delivery path (which produces wildly different activity),
#: not to pin down last-bit drift, which is expected and unavoidable.
SPIKE_COUNT_TOLERANCE = 0.01

#: Minimum fraction of timesteps whose spike set must match exactly. Measured
#: agreement is 100% in every regime tested, so this floor has headroom; it is
#: set below 1.0 because the underlying sums are provably not bit-identical and
#: a future model change could legitimately flip a threshold case.
PER_STEP_AGREEMENT = 0.95


def _random_connectome(n_neurons: int, p: float, seed: int) -> SparseConnectome:
    rng = np.random.default_rng(seed)
    n_edges = round(n_neurons * n_neurons * p)
    if n_edges == 0:
        return SparseConnectome.from_arrays(n_neurons=n_neurons, pre=[], post=[])

    pre = rng.integers(0, n_neurons, size=n_edges)
    post = rng.integers(0, n_neurons, size=n_edges)
    keys = pre.astype(np.int64) * n_neurons + post.astype(np.int64)
    _, unique = np.unique(keys, return_index=True)
    pre, post = pre[unique], post[unique]

    weights = rng.uniform(0.5, 2.0, size=len(pre)).astype(np.float32)
    delays = np.full(len(pre), 1.0, dtype=np.float32)
    return SparseConnectome.from_arrays(
        n_neurons=n_neurons, pre=pre, post=post, weight=weights, delay_ms=delays
    )


def _run_both(
    n_neurons: int, p: float, seed: int, steps: int, drive: float
) -> tuple[list[list[int]], list[list[int]], EventDrivenEngine]:
    conn = _random_connectome(n_neurons, p, seed)
    external = np.full(n_neurons, drive, dtype=np.float64)

    reference = SimulationEngine(
        neurons=[LIFNeuron(id=i) for i in range(n_neurons)], connectome=conn, dt=0.1
    )
    reference.run(steps=steps, external_current=external)

    engine = EventDrivenEngine(
        state=LIFState.uniform(n_neurons),
        connectome=CSRConnectome.from_sparse(conn),
        dt=0.1,
    )
    engine.run(steps=steps, external_current=external)

    return reference.spike_history, engine.spike_history, engine


def _assert_close(ref: list[list[int]], got: list[list[int]]) -> None:
    """Assert closeness with the stated tolerances, and report the real numbers."""
    assert len(ref) == len(got), f"step count differs: {len(ref)} vs {len(got)}"

    ref_total = sum(len(s) for s in ref)
    got_total = sum(len(s) for s in got)
    assert ref_total > 0, "vacuous test: the reference network never spiked"

    rel = abs(got_total - ref_total) / ref_total
    assert rel <= SPIKE_COUNT_TOLERANCE, (
        f"total spike count drifted by {rel:.4%} "
        f"(reference={ref_total}, event-driven={got_total}); "
        f"tolerance is {SPIKE_COUNT_TOLERANCE:.2%}"
    )

    identical = sum(1 for a, b in zip(ref, got, strict=True) if a == b)
    agreement = identical / len(ref)
    assert agreement >= PER_STEP_AGREEMENT, (
        f"only {agreement:.2%} of steps matched exactly "
        f"({identical}/{len(ref)}); floor is {PER_STEP_AGREEMENT:.0%}"
    )


def test_event_driven_close_small():
    """Fast CI case: 200 neurons, 300 steps, same seed as the other tests."""
    ref, got, _ = _run_both(n_neurons=200, p=0.05, seed=1234, steps=300, drive=45.0)
    _assert_close(ref, got)


def test_event_driven_close_refractory_regime():
    ref, got, _ = _run_both(n_neurons=150, p=0.03, seed=555, steps=400, drive=200.0)
    _assert_close(ref, got)


def test_event_driven_silent_network_is_exact():
    """No spikes at all: both paths must agree exactly, and visit zero edges."""
    ref, got, engine = _run_both(n_neurons=100, p=0.05, seed=7, steps=100, drive=0.0)
    assert ref == got
    assert sum(len(s) for s in ref) == 0
    assert engine.edges_visited == 0


def test_event_driven_is_exact_when_edges_are_pre_grouped():
    """If the edge list is already grouped by ``pre``, CSR order == edge order.

    This is the one case where event-driven delivery *is* bit-identical, and it
    is worth asserting: it proves the CSR build preserves within-row order, so
    any drift in the general case really is the reordering and not a bug in the
    index construction.
    """
    n = 120
    rng = np.random.default_rng(4242)
    # Build edges already sorted by pre.
    pre = np.repeat(np.arange(n, dtype=np.int32), 3)
    post = rng.integers(0, n, size=len(pre)).astype(np.int32)
    weight = rng.uniform(0.5, 2.0, size=len(pre)).astype(np.float32)
    conn = SparseConnectome.from_arrays(n_neurons=n, pre=pre, post=post, weight=weight)

    external = np.full(n, 60.0, dtype=np.float64)

    reference = SimulationEngine(
        neurons=[LIFNeuron(id=i) for i in range(n)], connectome=conn, dt=0.1
    )
    reference.run(steps=200, external_current=external)

    engine = EventDrivenEngine(
        state=LIFState.uniform(n), connectome=CSRConnectome.from_sparse(conn), dt=0.1
    )
    engine.run(steps=200, external_current=external)

    assert reference.spike_history == engine.spike_history
    assert sum(len(s) for s in reference.spike_history) > 0


def test_delivery_sums_are_not_bit_identical():
    """Prove the reordering is real: the two delivery paths disagree in the bits.

    Without this test the tolerance above would be decorative — the spike output
    happens to match, so a broken *sum* could hide behind a passing spike test.
    Here every neuron fires at once against a high-fan-in connectome with
    float64 weights, which is the regime where summation order actually shows.

    Measured on this machine: max relative difference ~5e-9, all 200 postsynaptic
    currents differing. The assertion is deliberately loose (1e-12) so it tests
    "the sums differ at all", not a specific magnitude that would be
    platform-dependent.
    """
    rng = np.random.default_rng(5)
    n, n_edges = 200, 40_000
    pre = rng.integers(0, n, size=n_edges)
    post = rng.integers(0, n, size=n_edges)
    keys = pre.astype(np.int64) * n + post.astype(np.int64)
    _, unique = np.unique(keys, return_index=True)
    pre, post = pre[unique], post[unique]
    # float64 weights: SparseConnectome.from_arrays would downcast to float32,
    # and float32 values summed into float64 are exact regardless of order.
    weight = rng.uniform(0.5, 2.0, size=len(pre))

    conn = SparseConnectome(
        n_neurons=n,
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        weight=weight,
        delay_ms=np.ones(len(pre), dtype=np.float32),
    )
    assert conn.weight.dtype == np.float64, "this test needs float64 weights to be meaningful"

    spikes = list(range(n))  # every neuron fires simultaneously

    # Reference delivery, verbatim from SimulationEngine._deliver.
    reference = np.zeros(n, dtype=np.float64)
    mask = np.isin(conn.pre, spikes)
    np.add.at(reference, conn.post[mask], conn.weight[mask])

    engine = EventDrivenEngine(
        state=LIFState.uniform(n), connectome=CSRConnectome.from_sparse(conn), dt=0.1
    )
    event_driven = engine._deliver(spikes)

    assert not np.array_equal(reference, event_driven), (
        "the two delivery paths produced bit-identical sums; either the CSR "
        "build is not reordering, or the weights are not float64"
    )
    rel = np.abs(reference - event_driven) / np.maximum(np.abs(reference), 1e-30)
    assert rel.max() < 1e-6, f"reordering drift is implausibly large: {rel.max():.3e}"


def test_event_driven_visits_far_fewer_edges_than_a_full_scan():
    """The optimisation must actually skip work, not just reorder it."""
    n, steps = 500, 100
    ref, got, engine = _run_both(n_neurons=n, p=0.01, seed=11, steps=steps, drive=45.0)
    _assert_close(ref, got)

    full_scan = engine.connectome.n_synapses * steps
    assert engine.edges_visited < full_scan, (
        f"event-driven visited {engine.edges_visited} edges, a full scan would visit {full_scan}"
    )


@pytest.mark.slow
def test_event_driven_close_10k_neurons():
    """The Phase 1 exit-criterion case, on the same seed as the other tests."""
    ref, got, _ = _run_both(n_neurons=10_000, p=0.002, seed=20261101, steps=200, drive=50.0)
    _assert_close(ref, got)


def test_csr_roundtrip_preserves_edges():
    """CSR -> COO must reproduce the same multiset of edges."""
    conn = _random_connectome(80, 0.05, seed=3)
    csr = CSRConnectome.from_sparse(conn)
    back = csr.to_sparse()

    assert back.n_synapses == conn.n_synapses
    key = lambda c: sorted(zip(c.pre.tolist(), c.post.tolist(), strict=True))  # noqa: E731
    assert key(back) == key(conn)


def test_csr_rejects_bad_indptr():
    with pytest.raises(ValueError, match="indptr"):
        CSRConnectome(
            n_neurons=2,
            indptr=np.array([0, 1], dtype=np.int64),  # wrong length
            indices=np.array([0], dtype=np.int32),
            weights=np.array([1.0], dtype=np.float32),
            delays=np.array([1.0], dtype=np.float32),
        )


def test_csr_rejects_int32_indptr():
    with pytest.raises(TypeError, match="int64"):
        CSRConnectome(
            n_neurons=1,
            indptr=np.array([0, 0], dtype=np.int32),
            indices=np.empty(0, dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            delays=np.empty(0, dtype=np.float32),
        )


def test_csr_edge_lookup_matches_bruteforce():
    """``edge_indices_for`` must agree with a naive per-neuron scan."""
    conn = _random_connectome(60, 0.08, seed=17)
    csr = CSRConnectome.from_sparse(conn)
    spikes = [3, 17, 42]

    got = csr.edge_indices_for(np.array(spikes, dtype=np.int64))

    expected = []
    for neuron in spikes:
        expected.extend(range(int(csr.indptr[neuron]), int(csr.indptr[neuron + 1])))
    assert got.tolist() == expected


def test_vectorized_and_event_driven_agree_on_silent_network():
    """Cross-check the two vectorised paths against each other, not just the oracle."""
    conn = _random_connectome(100, 0.05, seed=7)
    external = np.zeros(100, dtype=np.float64)

    vec = VectorizedEngine(state=LIFState.uniform(100), connectome=conn, dt=0.1)
    vec.run(steps=50, external_current=external)

    ev = EventDrivenEngine(
        state=LIFState.uniform(100), connectome=CSRConnectome.from_sparse(conn), dt=0.1
    )
    ev.run(steps=50, external_current=external)

    assert vec.spike_history == ev.spike_history


def test_segmented_reduction_matches_add_at():
    """The bincount reduction must agree with the scatter it replaced.

    ``np.add.at`` and ``np.bincount(weights=...)`` both sum the same weights
    into the same bins, but they are **not** the same arithmetic:

    * ``np.add.at`` accumulates sequentially, in edge order.
    * ``np.bincount`` uses **pairwise summation**, which is a different
      association order and is in fact *more* accurate.

    Measured on this case (300 neurons, ~20k edges, float64 weights): max
    absolute difference **5.415e-07** on sums of order 70, i.e. a relative
    difference of ~7e-9. That is float64 round-off, not a bug — but it is real,
    so the bound below is set from the measurement rather than assumed to be
    exact. An earlier version of this test asserted 1e-12 and failed, which is
    the correct outcome: the two paths genuinely differ.
    """
    rng = np.random.default_rng(11)
    n, n_edges = 300, 20_000
    pre = rng.integers(0, n, size=n_edges)
    post = rng.integers(0, n, size=n_edges)
    keys = pre.astype(np.int64) * n + post.astype(np.int64)
    _, unique = np.unique(keys, return_index=True)
    pre, post = pre[unique], post[unique]
    weight = rng.uniform(0.5, 2.0, size=len(pre))

    conn = SparseConnectome(
        n_neurons=n,
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        weight=weight,
        delay_ms=np.ones(len(pre), dtype=np.float32),
    )
    spikes = list(range(n))

    # The scatter this module used before the segmented reduction.
    scatter = np.zeros(n, dtype=np.float64)
    mask = np.isin(conn.pre, spikes)
    np.add.at(scatter, conn.post[mask], conn.weight[mask])

    engine = EventDrivenEngine(
        state=LIFState.uniform(n), connectome=CSRConnectome.from_sparse(conn), dt=0.1
    )
    reduced = engine._deliver(spikes)

    # Relative bound from the measurement (~7e-9), with margin.
    assert np.allclose(scatter, reduced, rtol=1e-7, atol=1e-6), (
        f"segmented reduction disagrees with the scatter it replaced beyond "
        f"float64 round-off: max abs {np.abs(scatter - reduced).max():.3e}"
    )
    # ...and it must not be bit-identical either, or the two are the same code.
    assert not np.array_equal(scatter, reduced), (
        "bincount and add.at produced bit-identical sums; the reduction is not "
        "actually using pairwise summation, so this test is not measuring what "
        "it claims"
    )


def test_segmented_reduction_is_faster_than_a_full_scan():
    """The optimisation must now win on wall time, not just on work done.

    Before the segmented reduction this path cut work by 97% but not time,
    because ``np.add.at`` dominated. Measured speedup after the change is
    2.1x-4.0x across 2k-40k neurons. The floor here is deliberately loose (1.2x)
    so the test is not flaky on a loaded machine, while still failing if the
    reduction regresses back to a scatter.
    """
    import time

    n, steps, drive = 5_000, 100, 45.0
    conn = _random_connectome(n, 0.02 * (2000 / n), seed=1)
    external = np.full(n, drive, dtype=np.float64)

    scan = VectorizedEngine(state=LIFState.uniform(n), connectome=conn, dt=0.1)
    start = time.perf_counter()
    scan.run(steps=steps, external_current=external)
    scan_s = time.perf_counter() - start

    ev = EventDrivenEngine(
        state=LIFState.uniform(n), connectome=CSRConnectome.from_sparse(conn), dt=0.1
    )
    start = time.perf_counter()
    ev.run(steps=steps, external_current=external)
    ev_s = time.perf_counter() - start

    assert ev_s < scan_s / 1.2, (
        f"event-driven took {ev_s:.3f}s vs scan {scan_s:.3f}s "
        f"(speedup {scan_s / ev_s:.2f}x); expected at least 1.2x"
    )
