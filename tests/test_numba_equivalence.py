"""Numba backend equivalence: bit-identical to the reference engine.

The claim under test is the strong one — *bit-identical* spikes, not merely
close. It is asserted on the same fixed-seed connectome used by the NumPy
equivalence test, so the two backends are compared against the same oracle.

If numba is not installed the whole module is skipped, because numba is an
optional dependency (``pip install 'mem20-brainz[numba]'``). CI runs this module
in a dedicated job that installs the extra, so the skip is not a way to hide a
failure — it is a way to keep the base install numpy-only.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba", reason="numba is an optional dependency")

from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import (
    LIFState,
    SimulationEngine,
    VectorizedEngine,
)
from mem20_brainz.simulation.numba_backend import NumbaBackend


def _random_connectome(n_neurons: int, p: float, seed: int) -> SparseConnectome:
    """Same generator as the NumPy equivalence test, so the oracle is shared."""
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


def _run_reference_and_numba(
    n_neurons: int, p: float, seed: int, steps: int, drive: float
) -> tuple[list[list[int]], list[list[int]]]:
    conn = _random_connectome(n_neurons, p, seed)
    external = np.full(n_neurons, drive, dtype=np.float64)

    reference = SimulationEngine(
        neurons=[LIFNeuron(id=i) for i in range(n_neurons)], connectome=conn, dt=0.1
    )
    reference.run(steps=steps, external_current=external)

    engine = VectorizedEngine(
        state=LIFState.uniform(n_neurons), connectome=conn, dt=0.1, backend=NumbaBackend()
    )
    engine.run(steps=steps, external_current=external)

    return reference.spike_history, engine.spike_history


def _assert_identical(ref: list[list[int]], got: list[list[int]]) -> None:
    assert len(ref) == len(got), f"step count differs: {len(ref)} vs {len(got)}"
    for k, (a, b) in enumerate(zip(ref, got, strict=True)):
        assert a == b, (
            f"spike sets differ at step {k}: reference={a[:20]} numba={b[:20]} "
            f"(len {len(a)} vs {len(b)})"
        )


def test_numba_bit_identical_small():
    """Fast CI case: 200 neurons, 300 steps, same seed as the NumPy test."""
    ref, got = _run_reference_and_numba(n_neurons=200, p=0.05, seed=1234, steps=300, drive=45.0)
    _assert_identical(ref, got)
    assert sum(len(s) for s in ref) > 0


def test_numba_bit_identical_refractory_regime():
    """Strong drive: most neurons refractory most of the time."""
    ref, got = _run_reference_and_numba(n_neurons=150, p=0.03, seed=555, steps=400, drive=200.0)
    _assert_identical(ref, got)
    assert sum(len(s) for s in ref) > 0


def test_numba_bit_identical_subthreshold():
    """No spikes: the integration trajectory must still agree exactly."""
    ref, got = _run_reference_and_numba(n_neurons=100, p=0.02, seed=99, steps=200, drive=1.0)
    _assert_identical(ref, got)
    assert sum(len(s) for s in ref) == 0


@pytest.mark.slow
def test_numba_bit_identical_10k_neurons():
    """The Phase 1 exit-criterion case, on the same seed as the NumPy test."""
    ref, got = _run_reference_and_numba(
        n_neurons=10_000, p=0.002, seed=20261101, steps=200, drive=50.0
    )
    _assert_identical(ref, got)
    assert sum(len(s) for s in ref) > 0


def test_numba_rejects_unsupported_tier():
    """A backend must raise on a tier it does not implement (ADR-0002)."""
    state = LIFState.uniform(4)
    state.tier[:] = "tier2_multicompartment"
    with pytest.raises(NotImplementedError, match="tier2_multicompartment"):
        NumbaBackend().step(state, np.zeros(4), 0.1, 0.0)


def test_numba_satisfies_backend_protocol():
    from mem20_brainz.simulation import Backend

    assert isinstance(NumbaBackend(), Backend)


def test_numba_is_not_imported_by_default():
    """The base package must not pull numba in at import time.

    Checked in a subprocess so the assertion is not polluted by this module
    having already imported numba.
    """
    import subprocess
    import sys

    code = (
        "import sys, mem20_brainz; "
        "assert 'numba' not in sys.modules, 'numba was imported eagerly'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
