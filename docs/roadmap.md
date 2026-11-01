# Technical roadmap

Phase-by-phase technical breakdown. The authoritative, higher-level version with
exit criteria is [`ROADMAP.md`](../ROADMAP.md); this file is the engineering
detail.

## Phase 0 — Scaffold

**Deliverables:** package layout, reference primitives, smoke tests, schemas,
mission docs.

**Technical debt accepted:**

- `SparseConnectome.add()` is O(n) — fine for tests, must not be used in bulk.
- `SimulationEngine._deliver` uses `np.isin` over all edges — O(synapses) per
  step regardless of sparsity.
- `io.load_config`'s YAML fallback handles flat files only.

**Exit:** clean-checkout install + green tests + frozen interfaces.

## Phase 1 — Vectorised core

**Target:** 10⁶ neurons, 1 s biological time, one GPU.

**Design:**

```python
class Backend(Protocol):
    def step(self, state: NeuronState, incoming: np.ndarray) -> np.ndarray: ...
```

- `NeuronState` is a struct-of-arrays: `v`, `v_rest`, `v_threshold`, `v_reset`,
  `tau_m`, `r_m`, `refractory_until`, all `float32` arrays of length N.
- `incoming` is the summed synaptic current per neuron for this step.
- The backend returns a boolean spike mask.

**Delivery:** precompute a CSR structure (`indptr`, `indices`, `weights`) over
`pre`. For a spike set S, iterate `indptr[pre]` slices for `pre ∈ S` and
`np.add.at` into the current vector. Cost becomes O(edges out of spiking
neurons), not O(all edges).

**Equivalence testing:** every backend must produce a bit-identical spike mask to
the reference on a fixed seed and a 10⁴-neuron random connectome. This is the
gate that lets us trust the fast path.

## Phase 2 — Real data

**Target:** one measured dataset, end to end.

**Pipeline:**

```
external storage ──► manifest (measured: true) ──► ingest ──► region matrix
                                                              │
                                          cell-type atlas ──► │
                                                              ▼
                                              synapse generator (synthetic)
                                                              │
                                                              ▼
                                                    SparseConnectome
```

**Key constraint:** the generator's output is synthetic by construction. Every
generated edge must be traceable to the rule that produced it, and the run
record must state the measured/synthetic ratio.

## Phase 3 — Region atlas and fidelity tiers

**Target:** whole-brain structure at region resolution.

- Parcellation: region list, volumes, neuron counts from measured densities.
- Fidelity map: `{region: model_class}` recorded in the run config.
- Mixed-fidelity engine: group neurons by model class, run one kernel per class
  per step.
- Validation suite: the list of measured statistics a run must reproduce. This
  is the project's biggest open scientific gap.

## Phase 4 — Distributed execution

**Target:** 10⁹ neurons across nodes.

- Partitioner: minimise cut edges, respect anatomical tracts. Likely a hybrid —
  region-local partitions plus a replicated long-range router.
- Sharded storage: on-disk CSR shards, memory-mapped, streamed per timestep.
- Runtime: MPI or a task graph; boundary spike exchange each step.
- Checkpoint/restart: long runs must survive a node loss without silent
  corruption.

## Phase 5 — Human scale

**Target:** 86 × 10⁹ neurons, ~10¹⁴ synapses.

- Neuron allocation from measured densities per region.
- ~1.6 TB of edge data, sharded, streamed.
- Wall-clock: days per second of biological time. Long-run scheduling and
  checkpointing are prerequisites, not nice-to-haves.
- Reproducibility: a third party rebuilds the run from its manifest alone.

## Cross-cutting work

| Concern | Where it lands |
|---|---|
| Provenance (measured vs synthetic) | Phase 2, enforced in CI |
| Reproducibility (config + seed + version) | Phase 2 |
| Validation | Phase 3 |
| Performance baselines | Phase 1, tracked thereafter |
| Documentation accuracy | every phase |
