"""Equivalence tests: the vectorised backend must reproduce the reference engine.

The reference engine (``SimulationEngine``) is the ground truth (ADR-0003). These
tests assert that ``VectorizedEngine`` + ``NumPyBackend`` produce **bit-identical**
spike output on the same seed, same parameters, same connectome.

Bit-identity is the strong claim, and it is the one that makes every later
optimisation trustworthy: if a fast path can drift silently, no comparison
between runs means anything. If these tests ever fail, the backend is wrong, not
the test.

Two cases:

* ``test_equivalence_small`` — fast, runs in CI on every push.
* ``test_equivalence_10k_neurons`` — ~10^4 neurons, fixed seed. Slower; marked
  ``slow`` so it can be deselected with ``-m "not slow"`` but runs by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import (
    LIFState,
    NumPyBackend,
    SimulationEngine,
    VectorizedEngine,
)


def _random_connectome(
    n_neurons: int, p: float, seed: int, *, delay_ms: float = 1.0
) -> SparseConnectome:
    """Erdos-Renyi connectome with a fixed seed, as a COO edge list.

    Edges are sampled directly rather than by masking a dense N x N matrix:
    ADR-0001 forbids a dense neuron-by-neuron structure anywhere in the project,
    tests included. Sampling ``n_neurons**2 * p`` candidate pairs and rejecting
    duplicates gives the same distribution without ever materialising the matrix.
    """
    rng = np.random.default_rng(seed)
    n_edges = round(n_neurons * n_neurons * p)
    if n_edges == 0:
        return SparseConnectome.from_arrays(n_neurons=n_neurons, pre=[], post=[])

    # Sample with replacement, then drop duplicate (pre, post) pairs.
    pre = rng.integers(0, n_neurons, size=n_edges)
    post = rng.integers(0, n_neurons, size=n_edges)
    keys = pre.astype(np.int64) * n_neurons + post.astype(np.int64)
    _, unique = np.unique(keys, return_index=True)
    pre, post = pre[unique], post[unique]

    weights = rng.uniform(0.5, 2.0, size=len(pre)).astype(np.float32)
    delays = np.full(len(pre), delay_ms, dtype=np.float32)
    return SparseConnectome.from_arrays(
        n_neurons=n_neurons, pre=pre, post=post, weight=weights, delay_ms=delays
    )


def _run_both(
    n_neurons: int,
    p: float,
    seed: int,
    steps: int,
    drive: float,
    *,
    dt: float = 0.1,
) -> tuple[list[list[int]], list[list[int]]]:
    """Run the reference and the vectorised engine on identical inputs."""
    conn = _random_connectome(n_neurons, p, seed)
    external = np.full(n_neurons, drive, dtype=np.float64)

    reference = SimulationEngine(
        neurons=[LIFNeuron(id=i) for i in range(n_neurons)],
        connectome=conn,
        dt=dt,
    )
    reference.run(steps=steps, external_current=external)

    state = LIFState.uniform(n_neurons)
    vectorised = VectorizedEngine(state=state, connectome=conn, dt=dt)
    vectorised.run(steps=steps, external_current=external)

    return reference.spike_history, vectorised.spike_history


def _assert_identical(ref: list[list[int]], vec: list[list[int]]) -> None:
    assert len(ref) == len(vec), f"step count differs: {len(ref)} vs {len(vec)}"
    for k, (a, b) in enumerate(zip(ref, vec, strict=True)):
        assert a == b, (
            f"spike sets differ at step {k}: reference={a[:20]} vectorised={b[:20]} "
            f"(len {len(a)} vs {len(b)})"
        )


def test_equivalence_small():
    """Fast CI case: 200 neurons, 300 steps, dense enough to be chaotic."""
    ref, vec = _run_both(n_neurons=200, p=0.05, seed=1234, steps=300, drive=45.0)
    _assert_identical(ref, vec)
    # Guard against a vacuous pass: the network must actually be spiking.
    assert sum(len(s) for s in ref) > 0


def test_equivalence_no_input():
    """Silent network: both engines must agree that nothing spikes."""
    ref, vec = _run_both(n_neurons=100, p=0.05, seed=7, steps=100, drive=0.0)
    _assert_identical(ref, vec)
    assert sum(len(s) for s in ref) == 0


def test_equivalence_subthreshold():
    """Weak drive: integration without spikes, so v trajectories must agree.

    Steady state is ``v_rest + r_m * i_inj``, so with ``r_m = 10`` MOhm a drive of
    1.0 nA gives -60 mV, safely below the -50 mV threshold. (A drive of 5.0 nA
    would give -20 mV and spike, which is why this value is not arbitrary.)
    """
    ref, vec = _run_both(n_neurons=100, p=0.02, seed=99, steps=200, drive=1.0)
    _assert_identical(ref, vec)
    assert sum(len(s) for s in ref) == 0


def test_equivalence_refractory_regime():
    """Strong drive: neurons spend most of their time refractory."""
    ref, vec = _run_both(n_neurons=150, p=0.03, seed=555, steps=400, drive=200.0)
    _assert_identical(ref, vec)
    assert sum(len(s) for s in ref) > 0


def test_equivalence_heterogeneous_parameters():
    """Per-neuron parameter spread must be honoured, not averaged."""
    n = 120
    rng = np.random.default_rng(2024)
    neurons = [
        LIFNeuron(
            id=i,
            v_rest=float(rng.uniform(-72.0, -68.0)),
            v_threshold=float(rng.uniform(-52.0, -48.0)),
            tau_m=float(rng.uniform(15.0, 25.0)),
            r_m=float(rng.uniform(8.0, 12.0)),
            refractory_ms=float(rng.uniform(1.0, 3.0)),
        )
        for i in range(n)
    ]
    conn = _random_connectome(n, 0.05, seed=31)
    external = np.full(n, 60.0, dtype=np.float64)

    # Build the array state BEFORE running the reference: SimulationEngine
    # mutates the neuron objects in place, so reading them afterwards would
    # capture the end-of-run state instead of the initial state.
    state = LIFState.from_neurons(neurons)

    reference = SimulationEngine(neurons=neurons, connectome=conn, dt=0.1)
    reference.run(steps=300, external_current=external)

    vectorised = VectorizedEngine(state=state, connectome=conn, dt=0.1)
    vectorised.run(steps=300, external_current=external)

    _assert_identical(reference.spike_history, vectorised.spike_history)
    assert sum(len(s) for s in reference.spike_history) > 0


@pytest.mark.slow
def test_equivalence_10k_neurons():
    """~10^4 neurons, fixed seed. The Phase 1 exit-criterion test case."""
    ref, vec = _run_both(n_neurons=10_000, p=0.002, seed=20261101, steps=200, drive=50.0)
    _assert_identical(ref, vec)
    assert sum(len(s) for s in ref) > 0


def test_backend_rejects_unsupported_tier():
    """A backend must raise on a tier it does not implement (ADR-0002)."""
    state = LIFState.uniform(4)
    state.tier[:] = "tier2_multicompartment"
    backend = NumPyBackend()
    with pytest.raises(NotImplementedError, match="tier2_multicompartment"):
        backend.step(state, np.zeros(4), 0.1, 0.0)


def test_state_rejects_float32():
    """float32 state changes spike output, so it is refused (ADR-0003)."""
    state = LIFState.uniform(4)
    state.v = state.v.astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        LIFState(
            v=state.v,
            v_rest=state.v_rest,
            v_reset=state.v_reset,
            v_threshold=state.v_threshold,
            tau_m=state.tau_m,
            r_m=state.r_m,
            refractory_ms=state.refractory_ms,
            refractory_until=state.refractory_until,
            tier=state.tier,
        )


def test_vectorized_engine_rejects_population_mismatch():
    with pytest.raises(ValueError):
        VectorizedEngine(
            state=LIFState.uniform(3),
            connectome=SparseConnectome.from_arrays(n_neurons=5, pre=[], post=[]),
        )


def test_backend_protocol_is_satisfied():
    """NumPyBackend must structurally satisfy the Backend protocol."""
    from mem20_brainz.simulation import Backend

    assert isinstance(NumPyBackend(), Backend)
