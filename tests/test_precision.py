"""Numeric precision: what float32 costs, pinned as a test.

ADR-0007 decided that ``float64`` is the default and ``float32`` is an explicit
opt-in. This file pins the *measured* consequences of that choice so the ADR
cannot drift away from the code.

The headline finding, which is not the one you would guess
----------------------------------------------------------
float32 is **not** uniformly unsafe. It depends entirely on whether the
population is homogeneous:

* **Homogeneous population** (every neuron identical, driven identically): the
  network runs in lockstep, every neuron crosses threshold on the same step, and
  float32 reproduces the spike count *exactly* — 0.00% drift across every seed
  tested. The per-step spike sets still differ on ~5% of steps, but only because
  a whole-population spike is emitted one step earlier or later, not because
  activity changed.
* **Heterogeneous population** (tau_m and v_threshold jittered, which is what a
  real circuit looks like): float32 drifts the spike count by **+0.14% on
  average** (max 0.17% over 5 seeds) and agrees on only **~14% of timesteps**.

So the honest statement is: float32 is acceptable only for a homogeneous
lockstep population, which is a benchmark artefact rather than a model of
anything. For any heterogeneous circuit it changes the answer, and the 2.8 TB
it saves at 86e9 neurons is not available.

The tests below assert the *direction and rough magnitude* of that divergence,
not exact numbers, because the exact numbers are platform- and seed-dependent.
"""

from __future__ import annotations

import numpy as np
import pytest

from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.simulation import LIFState, NumPyBackend, VectorizedEngine

#: Measured mean absolute spike-count drift for a heterogeneous population over
#: 5 seeds at 1000 neurons / 500 steps. The bound is deliberately loose (2x the
#: measured value) so the test catches a *regression in kind* — float32 becoming
#: wildly wrong, or silently becoming exact — without being flaky.
HETERO_DRIFT_BOUND = 0.005  # 0.5%, measured 0.14%

#: Measured per-step agreement for a heterogeneous population. float32 must be
#: clearly worse than float64 here; the floor is far below the measured ~14%.
HETERO_PER_STEP_FLOOR = 0.05


def _connectome(n: int, p: float, seed: int) -> SparseConnectome:
    rng = np.random.default_rng(seed)
    n_edges = round(n * n * p)
    pre = rng.integers(0, n, size=n_edges)
    post = rng.integers(0, n, size=n_edges)
    keys = pre.astype(np.int64) * n + post.astype(np.int64)
    _, unique = np.unique(keys, return_index=True)
    pre, post = pre[unique], post[unique]
    weights = rng.uniform(0.5, 2.0, size=len(pre)).astype(np.float32)
    return SparseConnectome.from_arrays(
        n_neurons=n,
        pre=pre,
        post=post,
        weight=weights,
        delay_ms=np.full(len(pre), 1.0, dtype=np.float32),
    )


def _run(precision: str, n: int, p: float, seed: int, steps: int, drive: float, hetero: bool):
    conn = _connectome(n, p, seed)
    state = LIFState.uniform(n, precision=precision)
    if hetero:
        rng = np.random.default_rng(seed + 7)
        state.tau_m = (state.tau_m * rng.uniform(0.8, 1.2, n)).astype(state.tau_m.dtype)
        state.v_threshold = (state.v_threshold + rng.uniform(-3, 3, n)).astype(
            state.v_threshold.dtype
        )
    engine = VectorizedEngine(
        state=state, connectome=conn, dt=0.1, backend=NumPyBackend(precision=precision)
    )
    engine.run(steps=steps, external_current=np.full(n, drive, dtype=np.float64))
    return engine.spike_history


def test_float64_is_the_default():
    """The default must be the reference-equivalent precision."""
    assert LIFState.uniform(4).precision == "float64"
    assert LIFState.uniform(4).v.dtype == np.float64
    assert NumPyBackend().precision == "float64"


def test_float32_halves_state_memory():
    """The whole point of float32: half the bytes per neuron."""
    n = 1000
    f64 = LIFState.uniform(n)
    f32 = LIFState.uniform(n, precision="float32")
    fields = ("v", "v_rest", "v_reset", "v_threshold", "tau_m", "r_m", "refractory_ms")
    bytes64 = sum(getattr(f64, f).nbytes for f in fields)
    bytes32 = sum(getattr(f32, f).nbytes for f in fields)
    assert bytes32 * 2 == bytes64


def test_float32_is_exact_for_a_homogeneous_population():
    """Lockstep regime: float32 reproduces the spike count exactly.

    This is the one regime where the memory saving looks free. It is recorded
    because it is the trap: a benchmark built on a homogeneous population would
    conclude float32 is safe, and that conclusion does not transfer.
    """
    ref = _run("float64", 500, 0.02, seed=1, steps=300, drive=45.0, hetero=False)
    got = _run("float32", 500, 0.02, seed=1, steps=300, drive=45.0, hetero=False)
    assert sum(map(len, ref)) == sum(map(len, got))
    assert sum(map(len, ref)) > 0, "vacuous: the network never spiked"


def test_float32_drifts_on_a_heterogeneous_population():
    """The regime that matters: float32 changes the answer.

    Asserts the drift is real and bounded, and that per-step agreement collapses.
    If this test ever starts failing because float32 became exact, that is a
    signal worth investigating — not something to loosen the bound for.
    """
    drifts = []
    agreements = []
    for seed in (1, 2, 3):
        ref = _run("float64", 1000, 0.02, seed=seed, steps=500, drive=45.0, hetero=True)
        got = _run("float32", 1000, 0.02, seed=seed, steps=500, drive=45.0, hetero=True)

        ref_total = sum(map(len, ref))
        got_total = sum(map(len, got))
        assert ref_total > 0, "vacuous: the reference network never spiked"

        drifts.append(abs(got_total - ref_total) / ref_total)
        agreements.append(sum(1 for a, b in zip(ref, got, strict=True) if a == b) / len(ref))

    mean_drift = float(np.mean(drifts))
    mean_agreement = float(np.mean(agreements))

    assert mean_drift > 0, (
        "float32 produced an identical spike count on every seed; the ADR's "
        "claim that it drifts is no longer supported by this code"
    )
    assert mean_drift < HETERO_DRIFT_BOUND, (
        f"float32 drift {mean_drift:.3%} exceeds the documented bound "
        f"{HETERO_DRIFT_BOUND:.2%}; ADR-0007 needs revisiting"
    )
    assert mean_agreement < 0.5, (
        f"float32 agreed on {mean_agreement:.1%} of steps, which contradicts the "
        "documented ~14%; the ADR's evidence no longer holds"
    )
    assert mean_agreement > HETERO_PER_STEP_FLOOR, (
        f"float32 agreed on only {mean_agreement:.1%} of steps — worse than the "
        "documented floor, suggesting a real bug rather than precision drift"
    )


def test_backend_refuses_a_mismatched_precision():
    """A backend must not silently up- or down-cast a state."""
    state = LIFState.uniform(8, precision="float32")
    with pytest.raises(ValueError, match="precision"):
        NumPyBackend(precision="float64").step(state, np.zeros(8), 0.1, 0.0)


def test_state_rejects_mixed_precision_fields():
    """A half-converted state is a bug, not a configuration."""
    state = LIFState.uniform(8)
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


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        LIFState.uniform(4, precision="float16")
    with pytest.raises(ValueError, match="unknown precision"):
        NumPyBackend(precision="bfloat16")
