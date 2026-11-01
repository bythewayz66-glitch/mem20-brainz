# Roadmap

Phases are sequential: each one's exit criterion is the next one's entry
condition. Dates are deliberately absent — this is a side project, and a
schedule that slips is worse than no schedule. What matters is the exit
criterion.

**Current phase: 1 — Vectorised core (in progress).**

---

## Phase 0 — Scaffold *(complete)*

**Goal:** a repository that is honest, installable, testable, and has frozen
interfaces.

- [x] Public repo, MIT license, Python-focused `.gitignore` with large-data rules
- [x] Installable package (`pyproject.toml`, `src/` layout)
- [x] Module skeleton: `neurons`, `synapses`, `regions`, `connectome`,
      `simulation`, `io`, `viz`
- [x] Reference `LIFNeuron`, `Synapse`, `Region`, `SparseConnectome`,
      `SimulationEngine`
- [x] Smoke test suite
- [x] JSON Schemas for fixtures, neuron records, dataset manifests
- [x] Mission README with an explicit implemented/aspirational split
- [x] ADR directory populated (`docs/adr/0001`–`0006`) with an index
- [x] CI: `pytest` + `ruff` on push and pull request (Python 3.11 and 3.12)

**Exit criterion:** `pip install -e ".[dev]"` and `pytest` pass on a clean
checkout, and the interfaces in `ARCHITECTURE.md` are stable enough that Phase 1
can be built behind them without changing callers. **Met.**

---

## Phase 1 — Vectorised core *(in progress)*

**Goal:** prove the engine can be fast, without changing semantics.

- [x] Struct-of-arrays neuron state (`LIFState`); per-neuron Python objects
      removed from the hot path
- [x] `Backend` protocol: `step(state, i_inj, dt, t) -> spike_mask`
- [x] NumPy backend (vectorised state update), **bit-identical** to the reference
- [x] Equivalence tests: bit-identical spikes on a 10⁴-neuron fixed-seed case
- [x] Numba backend (JIT inner loop), **bit-identical** to the reference
- [x] JAX backend (JIT; enables autodiff for parameter fitting) —
      tolerance-gated, not bit-identical (ADR-0008)
- [x] Gradient-based parameter fitting via a straight-through estimator
      (`scripts/fit_jax_example.py`; surrogate gradient, ADR-0008)
- [x] CSR connectome view indexed by presynaptic neuron
- [x] Event-driven delivery: only touch edges whose `pre` fired
      (tolerance-gated, not bit-identical — ADR-0006)
- [x] Segmented reduction (`np.bincount`) replacing the `np.add.at` scatter —
      turns the work reduction into a wall-clock win (2.1×–4.0×, ADR-0006)
- [x] Numeric precision decision: float64 default, float32 as a tested opt-in
      (ADR-0007)
- [x] Benchmark harness with a recorded baseline (synapses/s, neurons/s)
- [x] Live observability layer: streaming server, frame schema, React frontend,
      headless mode (ADR-0009)
- [x] Affective readout proxies with inputs, formula and `derived_proxy` tag
      ([`docs/readout.md`](docs/readout.md))

**Exit criterion:** 10⁶ LIF neurons for 1 second of biological time (10,000
steps) on a single GPU, with spike output bit-identical to the reference
implementation on a 10⁴-neuron test case.

**Note on CSR event-driven delivery:** it is *not* bit-identical to the
reference, because summing incoming weights in CSR order rather than edge order
is a different order of float additions. It is gated on a tolerance-based
equivalence test with a stated bound (ADR-0006). Measured: the delivery sums
differ by up to 5.4 × 10⁻⁹ relative, and no spike flips were observed in any
tested regime — but that is an observation, not a guarantee.

**Note on the benchmark result:** event-driven delivery reduces *work* by ~97%
(3% edge hit rate) but does **not** yet reduce wall time, because `np.add.at`
dominates the per-step cost. The win is asymptotic. See
[`docs/benchmarks/`](docs/benchmarks/README.md) and ADR-0006.

---

## Phase 2 — Real data

**Goal:** one measured dataset, end to end, with provenance.

- [ ] Dataset manifest format finalised and validated in CI
- [ ] Ingest one mesoscale tracer dataset (region-to-region projection
      strengths) into a region-level connectome
- [ ] Ingest one cell-type density atlas into region composition
- [ ] Statistical synapse generator: expand region-level strengths into
      cell-type-specific synapse counts, with every generated edge flagged
      synthetic
- [ ] Run-record writer: config + seed + manifests + code version
- [ ] First published figure, with a synthetic/measured breakdown

**Exit criterion:** a run whose connectivity can be traced, edge class by edge
class, to a cited measured source or to a documented synthetic rule — and a
figure that says which is which.

---

## Phase 3 — Region atlas and fidelity tiers

**Goal:** a whole-brain *structure* at region resolution, with per-region model
classes.

- [ ] Human parcellation loaded (region list, volumes, neuron counts)
- [ ] Fidelity map: each region assigned a neuron model class
- [ ] Mixed-fidelity simulation (cheap bulk + detailed regions)
- [ ] Validation suite: which measured statistics must a run reproduce?
- [ ] Sensitivity analysis: how much does output depend on the synthetic edges?

**Exit criterion:** a whole-brain region-level model that runs, with a published
fidelity map and a validation report stating what it does and does not
reproduce.

---

## Phase 4 — Distributed execution

**Goal:** break the single-machine memory wall.

- [ ] Anatomy-aware partitioner (minimise cut edges, respect tracts)
- [ ] Sharded on-disk connectome format with streaming reads
- [ ] MPI or task-graph runtime; boundary spike exchange per timestep
- [ ] Checkpoint/restart for long runs
- [ ] Fault tolerance: a lost partition must not silently corrupt a run

**Exit criterion:** 10⁹ neurons across multiple nodes, with a measured
scaling efficiency curve and a documented cut-edge overhead.

---

## Phase 5 — Human scale

**Goal:** 86 × 10⁹ neurons, ~10¹⁴ synapses.

- [ ] Full-brain neuron-count allocation from measured densities
- [ ] Sharded storage sized for ~1.6 TB of edge data (plus working set)
- [ ] Long-run scheduling (days of wall clock per second of biological time)
- [ ] Published fidelity map and synthetic/measured breakdown for the full run
- [ ] Reproducibility: a third party can rebuild the run from the manifest

**Exit criterion:** a full-brain run exists, is reproducible from its manifest,
and its documentation states precisely which parts are measured and which are
generated.

---

## Explicit non-goals

- **Not** a claim of consciousness, sentience, or human-level behaviour.
- **Not** a clinical or diagnostic tool.
- **Not** a claim that a synthetic connectome is a connectome. Generated edges
  are labelled as generated, always.
- **Not** a benchmark-chasing project. Speed matters only where it unblocks a
  phase.
