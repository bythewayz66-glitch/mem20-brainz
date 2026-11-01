# ADR-0007: Numeric precision — float64 default, float32 as an opt-in

- **Status:** Accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz architecture
- **Supersedes:** the blanket "float32 is not permitted" statement in ADR-0003
- **Related:** ADR-0003 (simulation backend), ADR-0001 (sparse storage)

> **Numbering note.** The in-flight observability layer (`src/mem20_brainz/server/`)
> cites "ADR-0007" for its own decision. That reference is stale — this record
> owns 0007. The server's decision needs renumbering to 0009 when that work
> lands. Recorded here so the collision is not discovered later.

## Context

At 86 × 10⁹ neurons, per-neuron state memory is the first hard wall, and it is
much closer than the synapse wall. The `LIFState` struct-of-arrays carries eight
`float64` fields per neuron:

```
v, v_rest, v_reset, v_threshold, tau_m, r_m, refractory_ms, refractory_until
```

That is the actual field list from `simulation/backend.py`, not an estimate.
Recomputing from it:

| precision | bytes/field | fields | bytes/neuron | at 86 × 10⁹ neurons |
|---|---:|---:|---:|---:|
| float64 | 8 | 8 | 64 | **5.5 TB** |
| float32 | 4 | 8 | 32 | **2.8 TB** |

So float32 would save **2.8 TB** — a factor of two on the single largest
in-memory structure in the project. That is a large enough prize that the
question cannot be settled by preference. It has to be measured.

Note that the 1.2 TB / 602 GB figures quoted in `ARCHITECTURE.md` are for a
*minimal* state (a single membrane potential plus a refractory counter, 14 B and
7 B per neuron). They are correct for that narrower definition but they are not
the state this code actually allocates. The 5.5 TB / 2.8 TB numbers above are
the ones that apply to `LIFState` as implemented, and they are the ones this
decision is made against.

## The experiment

Same fixed-seed network, same drive, same connectome, run twice — once with
`float64` state, once with `float32` state — and the spike output compared.
Reproduce with `tests/test_precision.py`, which pins the results below.

Configuration: 1000 neurons, p = 0.02, dt = 0.1 ms, 500 steps, drive 45 nA,
5 seeds. "Heterogeneous" means `tau_m` jittered ±20% and `v_threshold` jittered
±3 mV per neuron — i.e. a circuit that is not a single repeated cell.

### Result 1 — homogeneous population: float32 is exact

| seed | float64 spikes | float32 spikes | drift |
|---|---:|---:|---:|
| 1 | 16,000 | 16,000 | 0.00% |
| 2 | 16,000 | 16,000 | 0.00% |
| 3 | 16,000 | 16,000 | 0.00% |

Every neuron is identical and identically driven, so the population runs in
lockstep: all neurons cross threshold on the same step, and the spike *count* is
reproduced exactly. Per-step spike sets still differ on ~5% of steps, but only
because the whole population's spike is emitted one step earlier or later.

**This is the trap.** A benchmark built on a homogeneous population would
conclude float32 is free. That conclusion does not transfer.

### Result 2 — heterogeneous population: float32 changes the answer

| seed | float64 spikes | float32 spikes | drift | per-step agreement |
|---|---:|---:|---:|---:|
| 1 | 16,490 | 16,514 | +0.15% | 14.6% |
| 2 | 16,469 | 16,491 | +0.13% | 14.6% |
| 3 | 16,454 | 16,475 | +0.13% | 13.4% |
| 4 | 16,507 | 16,524 | +0.10% | 14.8% |
| 5 | 16,444 | 16,472 | +0.17% | 14.6% |

Mean absolute drift **0.14%**, max **0.17%**. Per-step agreement collapses to
**~14%** — the two runs are not the same run.

### Result 3 — the drift does not settle

Spike-count drift over increasing run length (heterogeneous, seed 1):

| steps | float64 | float32 | drift |
|---:|---:|---:|---:|
| 50 | 2,000 | 2,000 | 0.00% |
| 100 | 3,257 | 3,542 | +8.75% |
| 200 | 6,719 | 6,766 | +0.70% |
| 400 | 13,145 | 13,457 | +2.37% |
| 800 | 26,331 | 26,519 | +0.71% |

The drift is not a small constant offset that averages out. It oscillates
between 0% and ~9% depending on where in the population's phase the run happens
to stop. There is no run length at which float32 becomes trustworthy.

## Decision

1. **`float64` is the default and the only reference-equivalent precision.**
   Every backend defaults to it, and the equivalence tests are defined against
   it.

2. **`float32` is available as an explicit, per-state opt-in** —
   `LIFState.uniform(n, precision="float32")` plus a matching
   `NumPyBackend(precision="float32")`. It is a *tested mode*, not a global
   switch, because the answer depends on the population, not on the machine.

3. **A backend refuses a state whose precision does not match its own.** It
   raises rather than silently casting. A half-converted state is a bug, and
   `LIFState.__post_init__` rejects mixed-precision fields for the same reason.

4. **float32 is documented as acceptable only for homogeneous lockstep
   populations** — which is a benchmark artefact, not a model of a circuit. For
   any heterogeneous population it is not acceptable, and the 2.8 TB saving is
   **not available**.

5. **The 2.8 TB is therefore not banked in the roadmap.** Phase 4 planning must
   assume float64 state (5.5 TB at 86 × 10⁹ neurons) and find the memory
   elsewhere: partitioning, out-of-core state, or a genuinely different state
   representation.

## Consequences

**Good**

- The default is the one that reproduces the reference, so no result silently
  depends on a precision choice.
- The memory question is now answered with numbers rather than intuition, and
  the answer is "no free lunch" — which is more useful than a hopeful one.
- The trap is documented and pinned by a test, so a future benchmark cannot
  rediscover the homogeneous case and conclude float32 is safe.

**Bad**

- 5.5 TB of state at human scale is a real constraint that Phase 4 must solve
  some other way. This ADR removes an option rather than providing one.
- Two precisions means two code paths to keep equivalent, and the float32 path
  is only tested for *documented divergence*, not for correctness — because
  there is no correct answer to compare it against.

**Neutral**

- `float16` and `bfloat16` are rejected outright. At 10 bits of mantissa the
  membrane potential cannot resolve the ~0.1 mV differences that decide
  threshold crossings, so they are not worth measuring.

## Alternatives considered

**Make float32 the default to bank the 2.8 TB.** Rejected: it changes spike
output on any realistic circuit (0.14% drift, 14% per-step agreement), which
would make every downstream result depend on a precision choice nobody recorded.

**Keep the blanket float32 ban from ADR-0003.** Rejected as under-evidenced. The
ban was correct in conclusion but asserted rather than measured, and it would
have blocked the homogeneous case where float32 genuinely is exact. An ADR that
says "no" without a number is not a decision, it is a preference.

**Use float32 for state but float64 for accumulation.** Rejected for now: it
does not address the dominant error source, which is the *state* rounding
compounding over thousands of steps, not the per-step sum. Worth revisiting if
Phase 4 needs the memory badly enough to measure it.

**Mixed precision by tier (float32 for tier0, float64 for tier1/2).** This is
the most promising direction and is *not* decided here. Tier1/2 carry channel
state where the same argument would need re-running; assuming the tier0 result
transfers would be exactly the kind of unmeasured claim this ADR exists to
avoid. It needs its own experiment once tier1 exists.

## Open questions

- Does the drift shrink with a smaller `dt`? A smaller step means smaller
  per-step increments, which may make float32's relative error worse, not
  better. Unmeasured.
- Does the drift depend on fan-in? The heterogeneous test used p = 0.02; a
  high-fan-in regime sums many more terms per step and may diverge faster.
- Is there a state representation that is both compact and exact — e.g. fixed
  point with a per-neuron scale? That would sidestep the trade entirely and is
  worth a spike before Phase 4 commits to 5.5 TB.
