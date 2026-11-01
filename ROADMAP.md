# Roadmap

Phases are sequential: each one's exit criterion is the next one's entry
condition. Dates are deliberately absent — this is a side project, and a
schedule that slips is worse than no schedule. What matters is the exit
criterion.

**Current phase: 0 — Scaffold.**

---

## Phase 0 — Scaffold *(in progress)*

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
- [ ] ADR directory populated (`docs/adr/0001`–`0005`)
- [ ] CI: `pytest` + `ruff` on push

**Exit criterion:** `pip install -e ".[dev]"` and `pytest` pass on a clean
checkout, and the interfaces in `ARCHITECTURE.md` are stable enough that Phase 1
can be built behind them without changing callers.

---

## Phase 1 — Vectorised core

**Goal:** prove the engine can be fast, without changing semantics.

- [ ] Struct-of-arrays neuron state; remove per-neuron Python objects from the
      hot path
- [ ] `Backend` protocol: `step(state, incoming_spikes) -> spikes`
- [ ] NumPy backend (vectorised state update)
- [ ] Numba backend (JIT inner loop)
- [ ] JAX backend (JIT + `vmap`; enables autodiff for parameter fitting)
- [ ] CSR connectome view with precomputed per-neuron incoming slices
- [ ] Event-driven delivery: only touch edges whose `pre` fired
- [ ] Benchmark harness with a recorded baseline (synapses/s, neurons/s)
- [ ] Equivalence tests: every backend produces identical spikes to the
      reference on the same seed

**Exit criterion:** 10⁶ LIF neurons for 1 second of biological time (10,000
steps) on a single GPU, with spike output bit-identical to the reference
implementation on a 10⁴-neuron test case.

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
