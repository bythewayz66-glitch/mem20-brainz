"""JAX backend: the same LIF kernel, compiled by XLA, and differentiable.

JAX is an **optional** dependency. It is imported lazily inside
:meth:`JAXBackend.step` so that ``import mem20_brainz`` never requires it, and a
missing install produces a clear error rather than an ImportError at import time.

Why this backend exists
-----------------------
Not for speed. On a single CPU, XLA does not beat the NumPy kernel on a
population this small, and the benchmark records that honestly. This backend
exists for one reason the others cannot offer: **``jax.grad`` through a
simulation**. That is what makes parameter fitting possible — see
``examples/fit_synaptic_weight.py``.

Bit-identity: not claimed, and not achievable
---------------------------------------------
XLA fuses and reassociates operations. ``a * b + c`` may become a single FMA,
``(x - y) + z`` may be reordered, and reductions may be split across lanes. Each
of those changes the last bits. So this backend is gated on a **tolerance-based**
equivalence test with an explicitly stated bound, never on bit-identity.

There is a second, subtler trap: **JAX disables 64-bit floats by default.**
Without ``jax_enable_x64``, every ``float64`` array is silently truncated to
``float32``, which — per ADR-0007 — changes spike output on any heterogeneous
population. This module enables ``x64`` at import time and *verifies* it, because
a silent downcast would make the tolerance test meaningless.

The gradient is a surrogate, and that matters
---------------------------------------------
A spiking neuron's output is a step function of its membrane potential: it is
discontinuous at threshold, so its true derivative is zero almost everywhere and
undefined at the threshold. ``jax.grad`` through the spike test therefore returns
a gradient that is **not** the derivative of the model — it is the derivative of
whatever smooth function XLA happens to differentiate through the comparison.

This module does not pretend otherwise. :func:`spike_rate_loss` is defined over a
**smooth proxy** — the mean membrane potential, which is continuous in the
parameters — and the fitting example optimises that proxy, then *reports the
resulting spike rate as a measured outcome*, not as a differentiated quantity.
Any claim that a gradient through the spike train itself is exact would be false.
"""

from __future__ import annotations

import os

import numpy as np

from .backend import _DTYPES, TIER0, LIFState

#: Message used when jax is not installed. Kept as a constant so the test can
#: assert on it without duplicating the wording.
MISSING_JAX_HINT = (
    "JAXBackend requires the optional 'jax' dependency. "
    "Install it with:  pip install 'mem20-brainz[jax]'"
)


def _require_jax():
    """Import jax lazily, with an actionable error if it is absent.

    Also enables 64-bit floats. JAX reads ``JAX_ENABLE_X64`` at import time, so
    the environment variable is set *before* the import. If jax was already
    imported by something else with x64 disabled, the check below catches it
    rather than letting a silent float32 downcast corrupt the comparison.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised only without jax
        raise ImportError(MISSING_JAX_HINT) from exc

    if not jax.config.jax_enable_x64:
        # jax reads JAX_ENABLE_X64 at import, so if something else imported jax
        # first the env var is too late. ``config.update`` still works at
        # runtime, so use it rather than failing: a silent float32 downcast is
        # the failure mode we are guarding against, and fixing it is strictly
        # better than refusing to run.
        jax.config.update("jax_enable_x64", True)

    if not jax.config.jax_enable_x64:  # pragma: no cover - defensive
        raise RuntimeError(
            "JAXBackend requires 64-bit floats, but jax_enable_x64 could not be "
            "enabled. Without x64, float64 arrays are silently truncated to "
            "float32, which changes spike output (ADR-0007)."
        )
    return jax, jnp


def _lif_step_jax(jnp, state_arrays, i_inj, dt, t):
    """One LIF step as a pure function of arrays.

    Mirrors the reference control flow exactly: a refractory neuron neither
    integrates nor spikes, and the spike test runs on the *updated* ``v``.

    Written with ``jnp.where`` rather than boolean indexing so it is traceable
    and differentiable. ``jnp.where`` evaluates both branches, which is fine
    here because neither branch can produce a NaN or an out-of-range index.
    """
    v, v_rest, v_reset, v_threshold, tau_m, r_m, refractory_ms, refractory_until = state_arrays

    active = t >= refractory_until

    dv = (-(v - v_rest) + r_m * i_inj) / tau_m
    v_next = v + dv * dt

    v_after = jnp.where(active, v_next, v)

    spiked = active & (v_after >= v_threshold)
    v_out = jnp.where(spiked, v_reset, v_after)
    refractory_out = jnp.where(spiked, t + refractory_ms, refractory_until)

    return (v_out, v_rest, v_reset, v_threshold, tau_m, r_m, refractory_ms, refractory_out), spiked


class JAXBackend:
    """XLA-compiled LIF kernel, tolerance-equivalent to the reference.

    Not bit-identical: XLA fuses and reassociates. See the module docstring and
    ADR-0008 for the measured bound.
    """

    name = "jax"

    def __init__(self, precision: str = "float64") -> None:
        if precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(_DTYPES)}")
        self.precision = precision
        self._jax, self._jnp = _require_jax()
        self._step_fn = None

    def _get_step_fn(self):
        """JIT the step function once per process."""
        if self._step_fn is None:
            jnp = self._jnp
            self._step_fn = self._jax.jit(
                lambda arrays, i, dt, t: _lif_step_jax(jnp, arrays, i, dt, t)
            )
        return self._step_fn

    def _arrays(self, state: LIFState):
        return (
            state.v,
            state.v_rest,
            state.v_reset,
            state.v_threshold,
            state.tau_m,
            state.r_m,
            state.refractory_ms,
            state.refractory_until,
        )

    def _store(self, state: LIFState, arrays) -> None:
        (
            state.v,
            state.v_rest,
            state.v_reset,
            state.v_threshold,
            state.tau_m,
            state.r_m,
            state.refractory_ms,
            state.refractory_until,
        ) = arrays

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
                f"JAXBackend implements only {TIER0!r}; population contains {bad}. "
                "Refusing to substitute a cheaper model (ADR-0002)."
            )
        if state.precision != self.precision:
            raise ValueError(
                f"JAXBackend(precision={self.precision!r}) cannot step a "
                f"{state.precision!r} state; construct the backend with the same "
                "precision as the state (ADR-0007)"
            )

        jnp = self._jnp
        dtype = _DTYPES[self.precision]
        i_inj = np.broadcast_to(np.asarray(i_inj, dtype=dtype), state.v.shape)

        arrays = tuple(jnp.asarray(a) for a in self._arrays(state))
        new_arrays, spiked = self._get_step_fn()(
            arrays, jnp.asarray(i_inj), jnp.asarray(dt, dtype=dtype), jnp.asarray(t, dtype=dtype)
        )
        self._store(state, tuple(np.asarray(a) for a in new_arrays))
        return np.asarray(spiked)


def simulate_jax(
    state: LIFState,
    i_inj_per_step: np.ndarray,
    dt: float = 0.1,
    steps: int | None = None,
):
    """Run a whole simulation inside one JIT trace, returning spike counts.

    This is the differentiable entry point. Unlike :class:`JAXBackend`, which
    steps one timestep at a time from Python, this function traces the entire
    loop, so ``jax.grad`` can flow through it.

    Deliberately **feedforward**: there is no recurrent synaptic delivery here.
    A recurrent connectome would need a scatter inside the traced loop, which is
    possible but is a separate piece of work with its own equivalence question.
    Stating the limitation is the point — this is a parameter-fitting tool, not a
    replacement for the engine.

    Returns the per-step spike counts as a JAX array of shape ``(steps,)``.
    """
    _jax, jnp = _require_jax()
    dtype = _DTYPES[state.precision]
    steps = len(i_inj_per_step) if steps is None else steps

    arrays0 = tuple(
        jnp.asarray(a)
        for a in (
            state.v,
            state.v_rest,
            state.v_reset,
            state.v_threshold,
            state.tau_m,
            state.r_m,
            state.refractory_ms,
            state.refractory_until,
        )
    )
    drive = jnp.asarray(np.asarray(i_inj_per_step, dtype=dtype))

    # A plain Python loop over a traced function is simpler to read than
    # lax.scan here, and jax.grad handles it fine because each iteration is
    # traced with the same primitives.
    arrays = arrays0
    counts = []
    t = 0.0
    for k in range(steps):
        arrays, spiked = _lif_step_jax(jnp, arrays, drive[k], dt, t)
        counts.append(jnp.sum(spiked.astype(dtype)))
        t += dt
    return jnp.stack(counts)


def spike_rate_loss(
    state: LIFState,
    i_inj_per_step: np.ndarray,
    target_mean_v: float,
    dt: float = 0.1,
) -> object:
    """Smooth loss over the mean membrane potential — the differentiable proxy.

    **This is not a loss over spikes.** The spike train is discontinuous in the
    parameters, so a gradient through it is not the derivative of the model. The
    mean membrane potential *is* continuous, so this loss has a real gradient.

    The fitting example optimises this proxy and then reports the resulting spike
    rate as a measured outcome. See ADR-0008.
    """
    _jax, jnp = _require_jax()
    dtype = _DTYPES[state.precision]
    steps = len(i_inj_per_step)

    def run(v0):
        arrays = (
            jnp.asarray(v0),
            jnp.asarray(state.v_rest),
            jnp.asarray(state.v_reset),
            jnp.asarray(state.v_threshold),
            jnp.asarray(state.tau_m),
            jnp.asarray(state.r_m),
            jnp.asarray(state.refractory_ms),
            jnp.asarray(state.refractory_until),
        )
        drive = jnp.asarray(np.asarray(i_inj_per_step, dtype=dtype))
        total = jnp.asarray(0.0, dtype=dtype)
        t = 0.0
        for k in range(steps):
            arrays, _ = _lif_step_jax(jnp, arrays, drive[k], dt, t)
            total = total + jnp.mean(arrays[0])
            t += dt
        return total / steps

    return (run(jnp.asarray(state.v)) - target_mean_v) ** 2
