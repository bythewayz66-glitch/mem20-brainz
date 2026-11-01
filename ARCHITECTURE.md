# Architecture

This document is the top-level design record for `mem20-brainz`. It states the
module boundaries, the interfaces between them, and the decisions behind them.

> **Scope note:** this describes the *intended* architecture. Only the parts
> marked **implemented** exist in code today. See the status table in
> [`README.md`](README.md#current-status).

## Design principles

1. **Interfaces before performance.** The reference implementation is slow and
   correct. Optimisation happens behind a frozen interface, so a fast kernel can
   replace a slow one without touching callers.
2. **Sparse by default.** Nothing in this project may allocate a dense
   N×N structure. Connectivity is always an edge list or a compressed sparse
   form.
3. **Provenance is part of the data.** Every dataset, every config, every run
   carries a record of where it came from and whether it is measured or
   synthetic.
4. **Fidelity is explicit.** A neuron declares its model class. A run declares
   its fidelity map. There is no implicit "default neuron".
5. **No silent fallbacks.** If a dependency or dataset is missing, raise. Do not
   substitute a plausible-looking default.

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│  scripts/            CLI entry points: ingest, build, bench  │
├──────────────────────────────────────────────────────────────┤
│  simulation/         stepping loop, scheduling, partitioning │
├──────────────────────────────────────────────────────────────┤
│  neurons/   synapses/   regions/   connectome/               │
│  (state)    (edges)     (grouping) (storage)                 │
├──────────────────────────────────────────────────────────────┤
│  io/                 config, fixtures, manifests, shards     │
├──────────────────────────────────────────────────────────────┤
│  viz/                optional, read-only over run outputs    │
└──────────────────────────────────────────────────────────────┘
```

Dependencies point **downward only**. `viz` may import anything; `io` may import
`connectome`; `simulation` may import `neurons`, `synapses`, `connectome`;
nothing below may import `simulation`. This keeps the data model usable without
the engine, which matters for analysis and for testing.

## Module boundaries

### `neurons/` — **implemented (placeholder)**

Owns per-neuron state and dynamics.

- `Neuron` — the interface: `id`, membrane state `v`, and
  `step(i_inj, dt, t) -> bool` (True if the cell spiked).
- `LIFNeuron` — leaky integrate-and-fire placeholder with textbook parameters.
  **Not calibrated against any measurement.** Carries an explicit `tier` field
  (ADR-0002).

**Contract:** `step` must be pure with respect to everything except the neuron's
own state. It must not read the connectome, the clock, or global state. This is
what makes vectorisation possible: a model class becomes a struct-of-arrays
(`LIFState`) and `step` becomes a kernel over those arrays.

**Planned:** `EIFNeuron`, `AdExNeuron`, `MultiCompartmentNeuron`, and a
`NeuronModel` registry so a config can name a model class per region.

### `synapses/` — **implemented (minimal)**

Owns the properties of a single connection: `pre`, `post`, `weight`,
`delay_ms`, `sign`.

**Contract:** `Synapse` is immutable. Plasticity, when it arrives, will not
mutate `Synapse` objects — it will update the weight arrays inside
`SparseConnectome`, because per-object mutation does not survive vectorisation.

### `regions/` — **implemented (container only)**

Owns grouping: a named set of neurons with a declared cell-type composition.

**Contract:** cell-type fractions must sum to 1.0 (validated in
`__post_init__`). A region is a *declaration*, not a data source — it does not
load an atlas.

### `connectome/` — **implemented (reference)**

Owns connectivity storage. `SparseConnectome` is a coordinate list: four
parallel arrays (`pre`, `post`, `weight`, `delay_ms`).

**Contract:** arrays are always equal length; indices are `int32`; weights and
delays are `float32`. `add()` is O(n) and exists for tests only — bulk
construction goes through `from_arrays`.

**Planned:** CSR/CSC views, sharded on-disk format, and a streaming iterator that
yields only the edges whose presynaptic neuron fired this timestep.

### `simulation/` — **implemented (reference + NumPy backend)**

Owns time. Two engines, and the difference between them is the point:

- `SimulationEngine` — the **reference**. A per-neuron Python loop. Slow,
  correct, and frozen as ground truth (ADR-0003). It is never optimised.
- `VectorizedEngine` — the same semantics with the per-neuron loop replaced by a
  `Backend`. Currently backed by `NumPyBackend`, which is **bit-identical** to
  the reference and verified as such in CI.

**Contract:** fixed `dt`; spikes emitted at step *k* are delivered at step
*k + ceil(delay/dt)*; `spike_history[k]` is the list of neuron ids that spiked at
step *k*. The engine validates that the connectome's `n_neurons` matches the
population size and raises otherwise.

**`Backend` contract** (ADR-0003): `step(state, i_inj, dt, t) -> bool mask`. It
is pure with respect to everything except `state`, it mutates only arrays inside
`state`, and it must not reassociate the update expression — operation order is
part of the contract. A backend that does not implement a tier must raise, never
substitute (ADR-0002). State is `float64`; `float32` changes spike output and is
refused.

**Planned:** Numba and JAX backends, a CSR view with event-driven delivery, and a
partitioner. Note that CSR event-driven delivery is *not* bit-identical to the
reference (different float addition order), so it will be gated on a
tolerance-based test with a stated bound rather than a bit-identity claim.

### `io/` — **implemented (minimal)**

Owns the boundary with the outside world: configs, fixtures, manifests.

**Contract:** `load_config` accepts JSON and a flat-YAML subset. `load_connectome`
/ `save_connectome` handle small JSON fixtures only.

**Planned:** manifest resolution against external object storage, shard readers,
and a run-record writer that captures config + seed + dataset manifests + code
version.

### `viz/` — **implemented (optional)**

Read-only plotting over run outputs. Matplotlib is an optional extra; importing
`mem20_brainz` must never require it.

## Data flow of one timestep

```
spike_history[k-1]  ──►  gather edges whose pre ∈ spikes
                              │
                              ▼
                    sum weights per post  ──►  synaptic current
                              │
                              ▼
        for each neuron: step(current[i], dt, t)  ──►  spike?
                              │
                              ▼
                    spike_history[k], t += dt
```

This is the shape the fast backends must preserve. The optimisation is to
replace the per-neuron loop with an array kernel and the edge gather with a
precomputed CSR slice — not to change the semantics. The array-kernel half of
that is done (`VectorizedEngine`); the CSR half is not.

## Architecture Decision Records

Decisions are recorded as short ADRs in [`docs/adr/`](docs/adr/README.md), each
with context, decision, consequences and status. The index is
[`docs/adr/README.md`](docs/adr/README.md).

| ADR | Decision | Status |
|---|---|---|
| [0001](docs/adr/0001-sparse-connectome-storage.md) | Sparse connectome storage; never a dense adjacency matrix | accepted |
| [0002](docs/adr/0002-neuron-model-tiers.md) | Neuron model tiers, declared per neuron | accepted |
| [0003](docs/adr/0003-simulation-backend.md) | Simulation backend stack and the `Backend` boundary | accepted |
| [0004](docs/adr/0004-provenance-and-versioning.md) | Record-level provenance and immutable dataset versioning | accepted |
| [0005](docs/adr/0005-partitioning-and-chunking.md) | Chunked, partitioned connectome layout | accepted |

An earlier draft of this document listed five provisional decisions under
different titles (implementation language, sparse storage,
reference-before-optimisation, the `measured` flag, and fidelity tiers). Those
were placeholders written before any ADR existed; they are now folded into the
five records above, and `docs/adr/README.md` maps each provisional entry to the
ADR that superseded it.

## Open questions

- **Partitioning strategy.** Anatomy-aware partitioning is the obvious answer,
  but the cut-edge cost of long-range tracts may force a hybrid: region-local
  partitions plus a replicated long-range router.
- **Plasticity at scale.** Online plasticity requires weight updates to be
  local to a partition. Whether STDP can be made partition-local without
  violating causality is unresolved.
- **Fidelity budget.** How to allocate a fixed compute budget across regions is
  a scientific question, not an engineering one, and it needs a stated
  objective (e.g. match a measured functional connectivity matrix).
- **Validation.** What does "correct" mean for a whole-brain model? The project
  needs a validation suite before Phase 3, not after.
- **Precision.** `float64` state is required for bit-identity with the reference,
  but doubles memory versus `float32` (602 GB → 1.2 TB at 86 × 10⁹ neurons for
  the minimal state). A tested precision mode is needed before Phase 4.
