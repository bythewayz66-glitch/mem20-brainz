"""JAX backend: tolerance-based equivalence, never bit-identity.

JAX/XLA fuses and reassociates operations, so this backend is **not**
bit-identical to the reference and does not claim to be. The measured facts,
which the bounds below are derived from:

* The membrane potential diverges from the NumPy kernel at **step 0** — max
  absolute difference 1.421e-14 on a 500-neuron heterogeneous population. That
  is XLA fusing ``-(v - v_rest) + r_m * i_inj`` into a different instruction
  sequence, not a bug.
* Over 2000 steps the divergence stays bounded: max **relative** difference
  5.881e-16, final max absolute difference 2.842e-14. It does not compound
  catastrophically, because the LIF update is contractive (leak pulls ``v`` back
  toward ``v_rest``) and refractory resets clear the error entirely.

The bounds below are set ~6 orders of magnitude above the measured relative
drift, so the test catches a *broken* kernel (which produces O(1) differences)
without being flaky on last-bit noise.

A second trap this file guards: **JAX disables 64-bit floats by default.** If
``jax_enable_x64`` is off, every float64 array is silently truncated to float32,
which per ADR-0007 changes spike output on any heterogeneous population. The
backend enables x64 at import and raises if it finds it disabled.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="jax is an optional extra: pip install 'mem20-brainz[jax]'")

from mem20_brainz.connectome import SparseConnectome  # noqa: E402
from mem20_brainz.simulation import LIFState, NumPyBackend, VectorizedEngine  # noqa: E402
from mem20_brainz.simulation.jax_backend import JAXBackend  # noqa: E402

#: Relative tolerance on the membrane potential. Measured drift is 5.881e-16;
#: this bound is ~6 orders of magnitude looser so it is not flaky, while still
#: being far tighter than any real bug would produce.
V_REL_TOL = 1e-9

#: Absolute tolerance on the membrane potential, in mV. Measured 2.842e-14.
V_ABS_TOL = 1e-12

#: Relative tolerance on total spike count. Spike output is a discrete function
#: of a continuous state, so a 1e-14 perturbation can in principle flip a
#: threshold case. Measured agreement is 100% in every regime tested, but the
#: gate is a tolerance because that is an observation, not a guarantee.
SPIKE_COUNT_TOL = 0.01


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


def _run_both(n: int, p: float, seed: int, steps: int, drive: float):
    conn = _connectome(n, p, seed)
    external = np.full(n, drive, dtype=np.float64)

    ref = VectorizedEngine(
        state=LIFState.uniform(n), connectome=conn, dt=0.1, backend=NumPyBackend()
    )
    ref.run(steps=steps, external_current=external)

    jx = VectorizedEngine(state=LIFState.uniform(n), connectome=conn, dt=0.1, backend=JAXBackend())
    jx.run(steps=steps, external_current=external)
    return ref, jx


def test_jax_enables_x64():
    """Without x64, float64 is silently truncated to float32 (ADR-0007).

    Constructing the backend is what enables x64, so it must happen before the
    assertion — that is the behaviour being pinned.
    """
    JAXBackend()
    assert jax.config.jax_enable_x64, (
        "jax_enable_x64 is disabled; float64 arrays would be silently truncated "
        "to float32, which changes spike output"
    )


def test_jax_backend_is_not_bit_identical():
    """Prove the divergence is real, so the tolerance above is not decorative.

    Without this, a passing spike test could hide a kernel that happens to agree
    on spikes while computing a different membrane potential.
    """
    n = 200
    rng = np.random.default_rng(3)
    state = LIFState.uniform(n)
    state.tau_m = state.tau_m * rng.uniform(0.8, 1.2, n)
    state.v_threshold = state.v_threshold + rng.uniform(-3, 3, n)
    state.v = state.v + rng.uniform(-5, 5, n)

    def clone():
        return LIFState(
            **{
                f: getattr(state, f).copy()
                for f in (
                    "v",
                    "v_rest",
                    "v_reset",
                    "v_threshold",
                    "tau_m",
                    "r_m",
                    "refractory_ms",
                    "refractory_until",
                    "tier",
                )
            },
            precision="float64",
        )

    a, b = clone(), clone()
    numpy_backend, jax_backend = NumPyBackend(), JAXBackend()
    drive = np.full(n, 45.0)

    numpy_backend.step(a, drive, 0.1, 0.0)
    jax_backend.step(b, drive, 0.1, 0.0)

    assert not np.array_equal(a.v, b.v), (
        "JAX and NumPy produced bit-identical membrane potentials; either XLA "
        "stopped fusing or the backend is not actually using JAX"
    )
    # ...but the difference must be last-bit noise, not a wrong formula.
    assert np.allclose(a.v, b.v, rtol=V_REL_TOL, atol=V_ABS_TOL)


def test_jax_matches_numpy_within_tolerance():
    """The main equivalence gate: same spikes, same state, within tolerance."""
    ref, jx = _run_both(n=200, p=0.05, seed=1234, steps=300, drive=45.0)

    ref_total = sum(len(s) for s in ref.spike_history)
    jx_total = sum(len(s) for s in jx.spike_history)
    assert ref_total > 0, "vacuous: the reference network never spiked"

    rel = abs(jx_total - ref_total) / ref_total
    assert rel <= SPIKE_COUNT_TOL, (
        f"spike count drifted {rel:.4%} (numpy={ref_total}, jax={jx_total}); "
        f"tolerance is {SPIKE_COUNT_TOL:.2%}"
    )

    assert np.allclose(ref.state.v, jx.state.v, rtol=V_REL_TOL, atol=V_ABS_TOL), (
        f"final membrane potential diverged beyond tolerance: "
        f"max abs {np.abs(ref.state.v - jx.state.v).max():.3e}"
    )


def test_jax_matches_numpy_in_refractory_regime():
    """High drive: most neurons are refractory most of the time."""
    ref, jx = _run_both(n=150, p=0.03, seed=555, steps=400, drive=200.0)
    ref_total = sum(len(s) for s in ref.spike_history)
    jx_total = sum(len(s) for s in jx.spike_history)
    assert ref_total > 0
    assert abs(jx_total - ref_total) / ref_total <= SPIKE_COUNT_TOL
    assert np.allclose(ref.state.v, jx.state.v, rtol=V_REL_TOL, atol=V_ABS_TOL)


def test_jax_silent_network_is_exact():
    """No drive, no spikes: both paths must agree exactly."""
    ref, jx = _run_both(n=100, p=0.05, seed=7, steps=100, drive=0.0)
    assert ref.spike_history == jx.spike_history
    assert sum(len(s) for s in ref.spike_history) == 0


def test_jax_backend_refuses_mismatched_precision():
    state = LIFState.uniform(8, precision="float32")
    with pytest.raises(ValueError, match="precision"):
        JAXBackend(precision="float64").step(state, np.zeros(8), 0.1, 0.0)


def test_jax_backend_refuses_unsupported_tier():
    state = LIFState.uniform(8)
    state.tier = np.full(8, "tier1_conductance", dtype="<U32")
    with pytest.raises(NotImplementedError, match="tier0_point_lif"):
        JAXBackend().step(state, np.zeros(8), 0.1, 0.0)


def test_importing_package_does_not_import_jax():
    """jax must stay optional: a base install must not pay for it.

    Run in a subprocess so this process's already-imported jax cannot mask the
    result.
    """
    code = (
        "import sys; import mem20_brainz; "
        "assert 'jax' not in sys.modules, 'importing mem20_brainz pulled in jax'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "ok" in result.stdout
