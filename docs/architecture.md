# Architecture overview

A condensed technical companion to [`ARCHITECTURE.md`](../ARCHITECTURE.md). This
file is the one to read if you want the module map and the contracts without the
design rationale.

## Module map

| Package | Responsibility | Status |
|---|---|---|
| `mem20_brainz.neurons` | Per-neuron state and dynamics | placeholder (`LIFNeuron`) |
| `mem20_brainz.synapses` | Single-connection properties | minimal (`Synapse`) |
| `mem20_brainz.regions` | Neuron grouping and composition | container only |
| `mem20_brainz.connectome` | Sparse connectivity storage | reference (COO) |
| `mem20_brainz.simulation` | Time stepping and spike delivery | reference (single process) |
| `mem20_brainz.io` | Config, fixtures, manifests | minimal (JSON) |
| `mem20_brainz.viz` | Plotting over run outputs | optional |

## Dependency direction

```
viz ──► (anything)
io  ──► connectome
simulation ──► neurons, synapses, connectome
neurons, synapses, regions, connectome ──► (nothing internal)
```

Nothing below `simulation` may import `simulation`. The data model must be
usable without the engine.

## Core contracts

**`Neuron.step(i_inj, dt, t) -> bool`**
Advances one timestep; returns True on spike. Must not read global state — this
is what allows a model class to become a struct-of-arrays kernel later.

**`SparseConnectome`**
Four equal-length arrays: `pre` (int32), `post` (int32), `weight` (float32),
`delay_ms` (float32). Bulk construction via `from_arrays`; `add()` is O(n) and
for tests only.

**`SimulationEngine`**
Fixed `dt`. Spikes at step *k* are delivered at step *k + ceil(delay/dt)*.
`spike_history[k]` lists the neuron ids that spiked at step *k*. Raises if the
connectome's `n_neurons` disagrees with the population size.

**`Region`**
Cell-type fractions must sum to 1.0. A region is a declaration, not a data
source.

## One timestep

1. Take `spike_history[k-1]`.
2. Gather edges whose `pre` is in that set.
3. Sum weights per `post` → synaptic current vector.
4. Add external current.
5. Step every neuron; collect spikes.
6. Append to `spike_history`; advance `t`.

Every future backend must preserve these semantics exactly. Optimisation
replaces the per-neuron loop with an array kernel and the edge gather with a
precomputed CSR slice — it does not change the order of operations.

## Scaling plan

| Wall | Mitigation | Phase |
|---|---|---|
| Python object overhead | struct-of-arrays + Numba/JAX | 1 |
| Edge scan cost | CSR + event-driven delivery | 1 |
| Single-machine RAM | sharded on-disk connectome | 4 |
| Inter-node communication | anatomy-aware partitioning | 4 |
| Fidelity cost | per-region fidelity tiers | 3 |

## Storage budget

At ~16 bytes per synapse (two int32 indices, two float32 values), 10¹⁴ synapses
is ~1.6 TB of edge data. That is the number that forces sharding: it does not
fit in RAM on one machine, and it must be streamed rather than loaded.

## What is deliberately absent

- No dense adjacency matrices.
- No implicit default neuron model.
- No silent fallback when a dataset is missing.
- No claim of brain-scale capability anywhere in the code.
