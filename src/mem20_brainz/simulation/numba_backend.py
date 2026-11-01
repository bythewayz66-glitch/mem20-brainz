"""Numba backend: the same LIF kernel, JIT-compiled.

Numba is an **optional** dependency. It is imported lazily inside
:meth:`NumbaBackend.step` so that ``import mem20_brainz`` never requires it, and
a missing install produces a clear error rather than an ImportError at import
time.

Bit-identity
------------
The claim is that this backend produces *bit-identical* spikes to the reference
loop. That is only true if the compiler does not reassociate or contract the
arithmetic, so the kernel is compiled with:

* ``fastmath=False`` (the default, stated explicitly) — no reassociation, no
  reciprocal substitution, no NaN/Inf assumptions.
* ``parallel=False`` — no thread-level reordering.
* ``nogil=True`` only, which affects locking and not arithmetic.

The one real hazard is **FMA contraction**: LLVM may fuse ``a * b + c`` into a
single fused multiply-add, which rounds once instead of twice and changes the
last bit. Numba does not enable contraction without ``fastmath``, and the
equivalence test in ``tests/test_numba_equivalence.py`` is what actually proves
it on the target platform. If that test ever fails, the honest response is to
report the failure and fall back to a tolerance test — not to weaken the
assertion.

The kernel also deliberately mirrors the reference's *control flow*, not just
its formula: a refractory neuron is skipped entirely (its ``v`` is untouched),
and the spike test happens after the integration step, on the updated ``v``.
"""

from __future__ import annotations

import numpy as np

from .backend import TIER0, LIFState

#: Message used when numba is not installed. Kept as a constant so the test can
#: assert on it without duplicating the wording.
MISSING_NUMBA_HINT = (
    "NumbaBackend requires the optional 'numba' dependency. "
    "Install it with:  pip install 'mem20-brainz[numba]'"
)


def _require_numba():
    """Import numba lazily, with an actionable error if it is absent."""
    try:
        import numba
    except ImportError as exc:  # pragma: no cover - exercised only without numba
        raise ImportError(MISSING_NUMBA_HINT) from exc
    return numba


def _build_kernel(numba):
    """Compile the LIF kernel once per process.

    Kept as a factory so the module imports cleanly without numba installed.
    """

    @numba.njit(cache=True, fastmath=False, parallel=False, nogil=True)
    def _lif_step(
        v,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        r_m,
        refractory_ms,
        refractory_until,
        i_inj,
        dt,
        t,
        spiked,
    ):
        n = v.shape[0]
        for i in range(n):
            # Refractory: no integration, no spike, v untouched.
            if t < refractory_until[i]:
                spiked[i] = False
                continue

            # Exactly the reference expression, in exactly this order.
            dv = (-(v[i] - v_rest[i]) + r_m[i] * i_inj[i]) / tau_m[i]
            v[i] = v[i] + dv * dt

            if v[i] >= v_threshold[i]:
                v[i] = v_reset[i]
                refractory_until[i] = t + refractory_ms[i]
                spiked[i] = True
            else:
                spiked[i] = False

    return _lif_step


class NumbaBackend:
    """JIT-compiled LIF kernel, bit-identical to the reference loop.

    The compiled kernel is cached on the class, so constructing many backends
    does not recompile. Compilation happens on the first :meth:`step` call.
    """

    name = "numba"

    _kernel = None

    def __init__(self) -> None:
        # Fail fast and clearly if numba is missing, rather than at first step.
        _require_numba()

    @classmethod
    def _get_kernel(cls):
        if cls._kernel is None:
            cls._kernel = _build_kernel(_require_numba())
        return cls._kernel

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
                f"NumbaBackend implements only {TIER0!r}; population contains {bad}. "
                "Refusing to substitute a cheaper model (ADR-0002)."
            )

        i_inj = np.broadcast_to(np.asarray(i_inj, dtype=np.float64), state.v.shape)
        # The kernel writes into a contiguous buffer; broadcast views are not
        # writable, so materialise when needed.
        if not i_inj.flags.writeable or not i_inj.flags.c_contiguous:
            i_inj = np.ascontiguousarray(i_inj, dtype=np.float64)

        spiked = np.zeros(state.n_neurons, dtype=np.bool_)
        kernel = self._get_kernel()
        kernel(
            state.v,
            state.v_rest,
            state.v_reset,
            state.v_threshold,
            state.tau_m,
            state.r_m,
            state.refractory_ms,
            state.refractory_until,
            i_inj,
            float(dt),
            float(t),
            spiked,
        )
        return spiked
