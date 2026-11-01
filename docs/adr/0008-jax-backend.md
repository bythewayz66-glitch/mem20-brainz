# ADR-0008: JAX backend and surrogate-gradient parameter fitting

- **Status:** Accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz architecture
- **Related:** ADR-0003 (simulation backend), ADR-0007 (numeric precision)

## Context

Three backends now exist: the reference loop, NumPy, and Numba. All three answer
the same question — "run this network" — and all three are gated on equivalence
to the reference. None of them can answer a different question: **"what
parameter value would make this network behave the way I want?"**

That question is the whole of model fitting, and it is the reason to add a
fourth backend. JAX is not here for speed. On a single CPU at these population
sizes XLA does not beat the NumPy kernel, and the benchmark records that rather
than hiding it. JAX is here because `jax.grad` can differentiate through a
simulation, which no other backend can do.

Two obstacles had to be measured before this could be decided.

### Obstacle 1: XLA is not bit-identical, and cannot be made so

XLA fuses and reassociates. Measured on a 500-neuron heterogeneous population:

| quantity | measured |
|---|---|
| first divergence | **step 0** |
| max absolute difference at step 0 | 1.421e-14 |
| max **relative** difference over 2000 steps | 5.881e-16 |
| final max absolute difference | 2.842e-14 |

The divergence starts immediately — XLA compiles `-(v - v_rest) + r_m * i_inj`
into a different instruction sequence than NumPy's ufunc chain. It does **not**
compound catastrophically, because the LIF update is contractive (the leak term
pulls `v` back toward `v_rest`) and refractory resets clear accumulated error
outright.

So bit-identity is not available, and no amount of compiler-flag tuning would
produce it. The gate must be a tolerance test.

### Obstacle 2: JAX silently disables 64-bit floats

JAX reads `JAX_ENABLE_X64` at import time and defaults to **off**. With x64 off,
every `float64` array is silently truncated to `float32` — which, per ADR-0007,
drifts spike count by ~0.14% and agrees on only ~14% of timesteps for a
heterogeneous population. A tolerance test run in that state would be measuring
the wrong thing and would not know it.

The backend therefore sets `JAX_ENABLE_X64` before importing jax, and calls
`jax.config.update("jax_enable_x64", True)` afterwards in case something else
imported jax first. It then *verifies* the flag rather than assuming it.

### Obstacle 3: the gradient through a spike is not a gradient

A spiking neuron's output is a step function of its membrane potential. It is
discontinuous at threshold, so its true derivative is zero almost everywhere and
undefined at the threshold. `jax.grad` through the spike comparison returns the
derivative of whatever smooth function XLA happens to differentiate through the
comparison — **not** the derivative of the model.

This is not a technicality. Two attempts to fit a parameter failed because of it,
and both failures are worth recording because they are the reason the final
approach is what it is.

**Attempt 1 — mean membrane potential as the proxy.** The idea was to optimise a
smooth quantity (mean `v`) that is monotone in the drive. It is not monotone:
once neurons spike they reset to `v_reset`, so pushing harder can *lower* the
mean `v`. Measured, the fit diverged — loss rose from 3.569 to 106.7 while the
rate climbed from 56.75 Hz to 386.75 Hz, ending 346.75 Hz from a 40 Hz target.

**Attempt 2 — straight-through estimator with the operands swapped.** Written as
`hard + stop_gradient(soft - hard)`, which makes the forward pass hard and the
backward pass **zero**. The gradient vanished identically at every drive tested
(0.000e+00 at i = 1, 2, 2.5, 3, 3.47, 4, 5) and the fit never moved. The correct
form is `soft + stop_gradient(hard - soft)`.

## Decision

1. **Add `JAXBackend` as an optional extra**, implementing the existing
   `Backend` protocol. Lazy import, actionable `ImportError`, `pytest.importorskip`
   in tests, and a subprocess test asserting `import mem20_brainz` does not put
   `jax` in `sys.modules`.

2. **Gate it on a tolerance test, never bit-identity.** Bounds derived from the
   measurements above with ~6 orders of magnitude of margin:
   - relative tolerance on membrane potential: **1e-9** (measured 5.881e-16)
   - absolute tolerance on membrane potential: **1e-12** mV (measured 2.842e-14)
   - relative tolerance on spike count: **1%**

   A companion test asserts the two backends are **not** bit-identical, so the
   tolerance cannot silently become decorative.

3. **Require x64 and verify it.** The backend enables it and raises if it cannot.

4. **Fit with a straight-through estimator, and say so.** Forward pass is the
   true spike count; only the derivative is substituted with a sigmoid surrogate
   of width `kappa = 2.0` mV. The reported rate is therefore a real measurement
   of a real simulation, and the gradient is a heuristic.

5. **`simulate_jax` is feedforward only.** No recurrent synaptic delivery inside
   the traced loop. A recurrent scatter is possible but is separate work with its
   own equivalence question, and pretending otherwise would overstate what the
   backend does.

## Consequences

**Good**

- Parameter fitting exists. `scripts/fit_jax_example.py` fits a drive to a 40 Hz
  target: **2.0 nA → 3.4746 nA**, loss **2.560e+04 → 0.000e+00**, rate
  **0.00 → 40.00 Hz**, error **+0.00 Hz**. It converges in ~120 of 300
  iterations.
- The tolerance bounds are traceable to a measurement, not chosen for
  convenience.
- The two failed attempts are documented, so the next person does not repeat
  them.

**Bad**

- A fourth backend is a fourth thing to keep equivalent. The tolerance test is
  weaker than the bit-identity tests that guard NumPy and Numba, and that
  weakness is permanent.
- The gradient is a surrogate. It works for this 1-D monotone problem and would
  not be trustworthy for a high-dimensional fit without a proper method.
- `jax` is a heavy dependency (~100 MB with CPU wheels) for a feature that is
  currently one example script.

**Neutral**

- The learning rate is `1e-6`, not the `1e-4` that would be conventional. The
  gradient at the target drive is ~5.4e3, so a conventional rate overshoots by
  ~75 nA per step and the fit diverges to −189 nA. This is a property of the
  loss scaling (squared error over a spike *count*, not a rate), not of the
  method.

## Alternatives considered

**Use `jax.grad` directly on the spike count.** Rejected: the derivative is zero
almost everywhere, so the fit does not move. This is attempt 2 above, in effect.

**Use mean membrane potential as a smooth proxy.** Rejected on measurement: it is
non-monotone in the drive because of resets, and the fit diverged (attempt 1).

**Use a proper surrogate-gradient method (SuperSpike, etc.).** Not rejected —
deferred. It is the right answer for high-dimensional fitting and is the natural
next step once there is a multi-parameter problem to fit. For a single scalar
with a monotone response, the STE is sufficient and much simpler.

**Skip JAX and fit by bisection.** Rejected as a substitute, though it is worth
noting that bisection solves *this* problem exactly and faster. The point of the
JAX backend is not this one fit; it is that gradients generalise to problems
where bisection does not apply. The example is a demonstration of the mechanism,
not the best possible solver for a 1-D root-find.

## Open questions

- Does the STE converge for a multi-parameter fit (e.g. fitting a weight
  distribution rather than one scalar)? Untested, and the honest expectation is
  that it needs a better surrogate.
- What is the crossover where JAX's `jit` compilation cost is amortised? At 200
  neurons it is not; at 10⁶ it may be. Unmeasured.
- Can recurrent delivery be traced without a scatter that destroys the gradient?
  `jax.lax.segment_sum` or a dense matmul over a partition would work; both need
  their own equivalence test.
