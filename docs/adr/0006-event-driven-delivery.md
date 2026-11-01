# ADR-0006 — Event-driven spike delivery over a CSR connectome

- **Status:** Accepted
- **Date:** 2026-11-01
- **Supersedes:** nothing
- **Related:** ADR-0001 (sparse storage), ADR-0003 (backend seam), ADR-0005 (partitioning)

## Context

The reference engine and the NumPy backend both deliver spikes by scanning the
entire edge list every timestep:

```python
mask = np.isin(connectome.pre, spikes)          # O(E) per step
np.add.at(current, connectome.post[mask], connectome.weight[mask])
```

That is O(E) regardless of how many neurons actually fired. At human scale the
arithmetic is brutal: 10¹⁴ synapses, a 1 Hz mean firing rate, and a 0.1 ms
timestep means roughly 10¹⁰ edges fire per step out of 10¹⁴ — a hit rate near
10⁻⁴. Scanning all of them is the single largest waste in the engine, and it is
waste that grows with the connectome rather than with the activity.

The obvious fix is to index edges by presynaptic neuron and jump straight to the
rows of the neurons that fired. That is a CSR layout (ADR-0001 already commits
to sparse storage; this ADR commits to *which* sparse layout the hot path uses).

The complication is numerical. The reference sums contributions in **edge-list
order**. A CSR lookup sums them in **CSR order**, grouped by presynaptic neuron.
Floating-point addition is not associative, so the two orders produce different
sums. That is not a bug to be fixed — it is an unavoidable consequence of the
optimisation, and it means this path cannot inherit the bit-identity guarantee
that ADR-0003 established for the NumPy and Numba backends.

## Decision

**Adopt a CSR connectome view indexed by presynaptic neuron, and an
event-driven delivery path that visits only the out-edges of neurons that
fired. Gate it on a tolerance-based equivalence test, not a bit-identity
claim.**

Specifically:

1. `CSRConnectome` stores `indptr` (int64), `indices` (int32), `weights`
   (float32), `delays` (float32). The out-edges of neuron `i` are
   `[indptr[i] : indptr[i+1]]`.
2. The build is a **stable** sort on `pre`, so edges sharing a presynaptic
   neuron keep their original relative order. This is what makes the
   pre-grouped case exactly reproducible.
3. `EventDrivenEngine` replaces the scan with `edge_indices_for(spikes)`, a
   vectorised gather that turns a spike set into a flat edge-index array.
4. Equivalence is asserted with **stated tolerances**, and the fact that the
   sums differ is itself asserted by a test, so the tolerance cannot silently
   become decorative.

### The tolerance, and why it is what it is

Two separate numbers, because they measure different things:

| Constant | Value | What it bounds |
|---|---|---|
| `SPIKE_COUNT_TOLERANCE` | 1% relative | total spike count drift over a run |
| `PER_STEP_AGREEMENT` | 95% of steps | fraction of timesteps whose spike set is identical |

**Measured drift.** On a high-fan-in case (200 neurons, 40,000 edges, every
neuron firing simultaneously, float64 weights) the two delivery paths disagree
on **all 200** postsynaptic currents, with a **maximum relative difference of
5.4 × 10⁻⁹**. That is the real magnitude of the reordering effect, and
`test_delivery_sums_are_not_bit_identical` asserts it.

**Measured spike impact: none observed.** Across every regime tested —
subthreshold, near-threshold, refractory-dominated, and 10⁴ neurons — the spike
output was *identical*, with 100% per-step agreement. A ~10⁻⁹ relative
perturbation in membrane potential is far smaller than the margin a neuron needs
to cross threshold within one 0.1 ms step.

This is an observation about this LIF model and these parameters, **not a
guarantee**. A different model, a smaller timestep, or a neuron parked exactly
on threshold could flip. That is precisely why the gate is a tolerance test with
headroom rather than a bit-identity claim: the 95% floor is set below the
measured 100% so a legitimate future flip does not fail CI, while a *broken*
delivery path (which produces wildly different activity) still fails loudly.

### One case is exact

If the edge list is already grouped by `pre`, CSR order equals edge-list order,
so the sums are identical and the spikes are bit-identical. This is asserted
explicitly (`test_event_driven_is_exact_when_edges_are_pre_grouped`) because it
proves the CSR build preserves within-row order — so any drift in the general
case really is the reordering, not a bug in the index construction.

## Consequences

### Positive

- **Work drops by ~97%.** In the benchmark configuration the event-driven path
  visits 3% of the edges a full scan would touch (`edge_hit_rate` ≈ 0.03).
- The cost model becomes proportional to *activity* rather than to connectome
  size, which is the only way the design can survive to 10¹⁴ synapses.
- The CSR view is the natural substrate for the Phase 4 partitioner: a
  region-partitioned CSR is a row-range slice, which is exactly what a
  distributed rank needs to own.

### Negative

- **Not bit-identical.** The strongest correctness guarantee in the project
  (ADR-0003) does not extend to this path. Two engines now have different
  equivalence standards, and that must be documented wherever results are
  compared.
- **Wall-clock gain was modest at first, and is now real.** The original
  measurement found event-driven delivery *slower* than the NumPy scan at 2,000
  neurons (0.25×) and only modestly faster at 20,000 (1.02–1.48×), because
  `np.add.at` — an unbuffered scatter — dominated the per-step cost. **That has
  since been fixed** by replacing the scatter with a segmented reduction
  (`np.bincount(weights=...)`); see "Resolution" below. The lesson stands: the
  optimisation reduced work long before it reduced time, and the two had to be
  measured separately.
- `indices` is `int32`, which caps the population at 2,147,483,647 neurons.
  86 × 10⁹ exceeds that by a factor of 40, so this view is **not** usable at
  human scale as written. Widening to `int64` costs 400 TB at 10¹⁴ synapses.
  Deferred, and documented in the module docstring so nobody discovers it at
  10⁹ neurons.

### Neutral

- Delays are still ignored by all three engines (every spike is delivered on the
  next step). This ADR does not change that; fixing it must land in the
  reference first, so the reference stays the ground truth.

## Alternatives considered

**CSC (indexed by postsynaptic neuron).** Rejected. The simulation asks "given
the neurons that fired, which synapses fire?" — an out-edge query. CSC answers
the opposite question and would force a full scan to answer this one.

**Keep the scan, optimise the scan.** Rejected as the primary strategy. The scan
is O(E) by construction; no amount of vectorisation changes that. It remains the
right choice for small connectomes, which is why both paths are kept.

**Sort the edge list by `pre` at ingestion and keep bit-identity.** Tempting, and
it does give exactness (see above). Rejected as a *general* solution because it
constrains the storage format for a numerical convenience, and because ingestion
order is dictated by the source dataset, not by the simulator.

**Claim bit-identity and hope.** Rejected outright. The sums provably differ;
asserting equality would either fail intermittently or, worse, pass by luck on
the tested seeds and fail in production.

## Resolution: the segmented reduction (2026-11-01)

The bottleneck identified above was `np.add.at`. It has been replaced with
`np.bincount(posts, weights=weights, minlength=n_neurons)`.

**Why `bincount` and not `np.add.reduceat`.** `reduceat` requires the array to be
sorted by the segment key plus a segment-boundary index, so it would cost an
`argsort` per timestep. `bincount` indexes directly by the post id and needs
neither. The trade is that `bincount` always allocates a full `n_neurons` output
— which is free here, because `current` is that size anyway.

**Measured result.** Crossover speedup across 2k → 40k neurons went from
1.00×, 1.08×, 0.79×, 1.32×, 1.04× to **2.11×, 4.01×, 2.93×, 3.50×, 2.10×**.
On the main benchmark, `event_driven` went from 6,213,128 to 9,758,766
synapses/s (+57%) and from 0.038 s to 0.024 s (−37%), making it the fastest path
on that benchmark.

**Numerical consequence, measured.** `bincount` uses pairwise summation while
`add.at` accumulates sequentially, so the two are **not** bit-identical: max
absolute difference 5.415e-07 on sums of order 70 (~7e-9 relative). This is
float64 round-off and `bincount` is the more accurate of the two, but it means
the tolerance gate below is still required — the change did not buy back
bit-identity, and does not claim to.

## What would make this a wall-clock win

*(Resolved — see "Resolution" above. Kept for the record.)*

The measured bottleneck is `np.add.at`, not the edge gather. The next step is to
replace the unbuffered scatter with a segmented reduction — sort the gathered
`post` indices and use `np.add.reduceat`, or accumulate into a per-row buffer.
That is a separate change with its own equivalence question, and it should be
measured before it is claimed.

## References

- `src/mem20_brainz/connectome/csr.py` — the CSR view and its memory layout
- `src/mem20_brainz/simulation/event_driven.py` — the delivery path
- `tests/test_event_driven.py` — tolerance tests and the drift proof
- `docs/benchmarks/baseline.json` — the measured crossover data
