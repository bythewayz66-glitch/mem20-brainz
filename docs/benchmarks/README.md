# Benchmarks

Measured performance for every engine path, recorded as a committed JSON
artifact so a number in the docs can always be traced to a run.

## Read this first

**These numbers come from one machine.** The machine is recorded in
`baseline.json` under `environment` — CPU model, core count, Python, NumPy and
Numba versions. They are:

- **not** a claim about whole-brain performance,
- **not** a comparison with NEURON, NEST, Brian, Arbor or any other simulator,
- **not** a scaling projection. Nothing here extrapolates to 86 × 10⁹ neurons.

They exist for one purpose: to detect regressions and to keep the roadmap's
claims honest. If a doc says "the NumPy backend is faster than the reference",
this file is the evidence.

## Running it

```bash
pip install -e ".[dev,numba]"     # numba is optional; omit it to skip that path
python scripts/benchmark.py --markdown
```

`--markdown` prints a table to stdout in addition to writing the JSON. Without
it you just get the file.

To regenerate the committed baseline after an intentional change:

```bash
python scripts/benchmark.py
git add docs/benchmarks/baseline.json
git commit -m "bench: refresh baseline"
```

## `--check` mode

```bash
python scripts/benchmark.py --check
```

Re-runs the benchmark and **fails** if the committed baseline is stale relative
to the current code. It checks three things:

1. the benchmark configuration is unchanged,
2. the package version is unchanged,
3. **spike counts are unchanged** for every path.

The third check is the important one. Spike counts are deterministic for a fixed
seed, so if they move, the engine changed — which is a correctness signal, not a
performance one. CI runs this on every push, so a silent change to simulation
semantics cannot land without someone noticing.

`--check` deliberately does **not** compare timings. Wall-clock numbers vary
between runs on the same machine (scheduler noise, thermal state, cache
pressure), so asserting on them would produce a flaky CI job that gets disabled
within a week. Timings are recorded for humans; spike counts are asserted for
machines.

## What is measured

Four paths, on identical inputs (same seed, same connectome, same drive):

| path | engine | delivery |
|---|---|---|
| `reference` | `SimulationEngine` | per-neuron Python loop, full edge scan |
| `numpy` | `VectorizedEngine` + `NumPyBackend` | full edge scan |
| `numba` | `VectorizedEngine` + `NumbaBackend` | full edge scan |
| `event_driven` | `EventDrivenEngine` | CSR lookup, visits only fired rows |

Two metrics, because they answer different questions:

- **synapses/s** — edges processed per second. For the scan-based paths this is
  work *attempted* (`n_synapses × steps / elapsed`). For the event-driven path
  it is work *actually done* (`edges_visited / elapsed`). The JSON records both
  `synapses_attempted` and `synapses_visited` so the two are never confused.
- **neurons/s** — `n_neurons × steps / elapsed`. The unit the roadmap's Phase 1
  exit criterion is phrased in.

## The crossover measurement

`baseline.json` also carries a `crossover` array: a sweep over population size
(2k → 40k neurons) comparing scan against event-driven wall time, with `p`
scaled so mean out-degree stays roughly constant.

This exists because "is event-driven faster?" is the wrong question. The right
one is "at what size does it start to win?"

**The honest answer from the committed data: the crossover is now real, and it
came from replacing the scatter.** Measured speedup across 2k → 40k neurons is
**2.11×, 4.01×, 2.93×, 3.50×, 2.10×** — event-driven wins at every size tested.

That is a change from the previous baseline, which measured 1.00×, 1.08×, 0.79×,
1.32×, 1.04× — bouncing around 1.0 with no trend. The work reduction was always
there (~97%, a 3% edge hit rate, consistently); what was missing was a wall-clock
win, because `np.add.at` — an unbuffered scatter — dominated the per-step cost.

Replacing it with `np.bincount(weights=...)` (a segmented reduction) is what
converted the work reduction into time. Before/after on the main benchmark:

| path | before (synapses/s) | after (synapses/s) | change |
|---|---:|---:|---:|
| `event_driven` | 6,213,128 | 9,758,766 | **+57%** |
| `event_driven` elapsed | 0.038 s | 0.024 s | **−37%** |

`event_driven` is now the **fastest path on the main benchmark** (0.024 s vs
0.026 s for numba and 0.028 s for numpy), which it was not before.

Two caveats worth keeping:

- The speedup is not monotone in population size (4.01× at 5k, 2.93× at 10k,
  3.50× at 20k). These are single-run timings and the variation is within
  scheduler noise. The *direction* is stable; the exact ratio is not.
- `bincount` uses pairwise summation while `add.at` accumulates sequentially, so
  the two are not bit-identical — measured max absolute difference 5.415e-07 on
  sums of order 70 (~7e-9 relative). That is float64 round-off, and it is why
  the event-driven path is still gated on a tolerance test rather than
  bit-identity. See ADR-0006.

## Configuration

Fixed in `scripts/benchmark.py` as `CONFIG`:

| parameter | value |
|---|---|
| neurons | 2,000 |
| connection probability | 0.01 |
| synapses | ~39,800 |
| steps | 200 |
| dt | 0.1 ms |
| drive | 45.0 nA |
| seed | 20261101 |

Changing any of these invalidates the committed baseline, which is exactly what
`--check` is for.

## Files

- `baseline.json` — the committed results, with environment metadata
- `scripts/benchmark.py` — the harness (`--check`, `--markdown`, `--out`)
