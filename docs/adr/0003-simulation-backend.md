# 0003 — Simulation backend stack and the `Backend` boundary

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz (lead architect)
- **Supersedes:** the provisional "Python + JIT/GPU escape hatches" and
  "reference implementation before optimisation" entries in `ARCHITECTURE.md`

## Context

The reference engine (`SimulationEngine`) is a per-neuron Python loop. It is
correct and it is slow: roughly 10⁴–10⁵ neurons·steps/s, which is ~10⁵× short of
what Phase 1 asks for. It must be kept, because it is the only thing that defines
what "correct" means.

The risk in optimising is not that the fast path is slow — it is that the fast
path is *silently different*. A vectorised update that reassociates a float
expression, or a CSR gather that sums incoming weights in a different order,
produces different spikes on a chaotic network. Once that happens, every
comparison between a fast run and a reference run is meaningless, and the
project loses the ability to say what its own results mean.

So the boundary has to be drawn before the optimisation, not after. Two
constraints follow:

1. **The reference engine is frozen as ground truth.** It is not deleted, not
   "cleaned up", and not optimised. It is the oracle.
2. **Every backend must be provably equivalent to it** on a fixed seed, and the
   proof must be a test that runs in CI.

Language and runtime choice is secondary to that. The candidates:

| Stack | Strength | Weakness |
|---|---|---|
| Pure NumPy | Zero build cost, already a dependency, elementwise ops are IEEE-754 exact | No fused loops; intermediate arrays allocate |
| Numba | JIT'd loops, no temporaries, near-C speed, easy to keep operation order | Compile time; harder to debug; per-tier specialisation |
| JAX | JIT + `vmap`/`pmap`, autodiff for parameter fitting, GPU | Functional purity required; float reassociation under `jit` can break bit-identity; heavier dependency |
| MPI / task graph | Only route past one node's memory | Phase 4; large complexity budget |

## Decision

**A `Backend` protocol is the only seam between the engine and the arithmetic.**

```python
class Backend(Protocol):
    name: str
    def step(self, state, i_inj, dt, t) -> np.ndarray:  # bool array, True = spiked
        ...
```

The contract, stated so it can be violated:

1. **`step` is pure with respect to everything except `state`.** It reads
   `i_inj`, `dt`, `t`; it mutates only the arrays inside `state`; it returns a
   boolean spike mask. It must not read the connectome, the clock, or any global.
2. **`state` is a struct-of-arrays, not a list of objects.** `LIFState` holds
   `v`, `v_rest`, `v_reset`, `v_threshold`, `tau_m`, `r_m`, `refractory_ms`,
   `refractory_until` as parallel `float64` arrays. `float64` is mandatory for
   the reference-equivalent backends: `float32` changes results.
3. **Operation order is part of the contract.** The LIF update is
   `dv = (-(v - v_rest) + r_m * i_inj) / tau_m`, then `v = v + dv * dt`. A
   backend may not reassociate this into e.g. `v + dt/tau_m * (...)`, because
   that is a different float result.
4. **A backend that does not implement a tier must raise.** No substitution.
5. **Adoption order is NumPy → Numba → JAX → MPI.** Each is added only when the
   previous one is proven equivalent and benchmarked.

**Implemented now:** `NumPyBackend`, plus `VectorizedEngine`, which keeps the
reference engine's delivery step byte-for-byte and replaces only the per-neuron
loop with an array kernel. This is the first Phase 1 bullet ("remove per-neuron
Python objects from the hot path") and it is **bit-identical** to the reference
by construction: elementwise IEEE-754 operations do not depend on whether they
are applied in a loop or over an array, and the delivery step is unchanged.

**Deliberately not yet done:** CSR event-driven delivery. This is the next
optimisation and it is *not* bit-identical, because summing incoming weights in
CSR order rather than edge order is a different float addition order. It will
therefore be gated on a tolerance-based equivalence test with a stated bound, and
the ADR that introduces it must say so explicitly. Claiming bit-identity for it
would be false.

## Consequences

**Easy:** a fast kernel can be swapped in without touching callers; the
reference engine stays available as an oracle forever; equivalence is a CI
assertion rather than a hope; JAX's autodiff can later be used for parameter
fitting behind the same seam.

**Hard:** `float64` state doubles memory versus `float32` (602 GB → 1.2 TB at
86 × 10⁹ neurons for the minimal state). That is acceptable now and will need a
 documented, tested precision decision before Phase 4 — it is an open question,
not a settled one. Backends must be written to preserve operation order, which
rules out some convenient NumPy idioms.

**Owed to the reader:** any performance number states which backend produced it,
and any correctness claim states which backend it was verified against.

## Alternatives considered

- **Optimise the reference engine in place.** Rejected: destroys the oracle. The
  moment the reference is fast, there is nothing left to check the fast path
  against.
- **Skip the protocol; write one fast engine.** Rejected: forecloses Numba/JAX/MPI
  and makes equivalence untestable.
- **JAX first.** Rejected as the *first* backend: `jit` may reassociate float
  operations, so bit-identity against the reference is not guaranteed, and the
  project would be unable to state its equivalence claim. JAX is still the
  intended GPU path, with a tolerance-based test.
- **`float32` state from the start.** Rejected for the reference-equivalent
  backends: it changes spike output. Revisit as an explicit, tested precision
  mode.
