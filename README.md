# mem20-brainz

**A public, open, biologically grounded attempt to model the human brain at the
scale of individual neurons — one modeled unit per neuron, ~86 billion of them,
connected by ~100 trillion synapses.**

> **Status: pre-alpha scaffold (v0.0.1).** This repository currently contains a
> project skeleton, data containers, a reference simulation loop, and a
> vectorised NumPy backend that is bit-identical to that reference. It does
> **not** contain a brain model, real connectome data, or any brain-scale
> capability. Every claim below is labelled as *implemented* or *aspirational*.
> Please read [Current status](#current-status) before anything else.

---

## The scientific goal

The human brain is the only known system that produces a mind, and it does so
with roughly **86 billion neurons** and on the order of **100 trillion
synapses**. Every attempt to understand it computationally has had to make a
choice: model a small circuit faithfully, or model the whole system coarsely.

This project takes the long way around. The target is a model in which **each
neuron in the human brain is represented by its own modeled unit** — with its
own state, its own dynamics, and its own set of incoming and outgoing synapses —
and in which the connectivity is grounded in measured connectome data wherever
such data exists, and explicitly labelled as synthetic where it does not.

That is not a claim that such a model would be conscious, or that it would
reproduce human behaviour. It is a claim about *resolution*: we want to stop
asking "what does the average neuron in this region do" and start asking "what
does this neuron do, given who it is connected to".

## The scale problem

| Quantity | Human brain | What it means for the software |
|---|---|---|
| Neurons | ~86 × 10⁹ | Even 1 byte of state per neuron is 86 GB. Realistic state is 10–100× that. |
| Synapses | ~1 × 10¹⁴ | At ~16 bytes per edge (indices + weight + delay) that is ~1.6 TB, before any working set. |
| Time resolution | ~0.1 ms | 1 second of biological time = 10,000 timesteps. |
| Wall-clock budget | — | A 1-second simulation at 1 ms/s of compute would take ~115 days. |

Three consequences drive the whole architecture:

1. **Memory is the first wall, not compute.** The connectome does not fit in RAM
   on any single machine. Storage must be sharded and streamed, and the
   simulation must be able to touch only the edges it needs this timestep.
2. **Communication dominates.** Once the model is distributed, the cost is
   moving spikes between partitions, not multiplying weights. Partitioning must
   minimise cut edges, which means partitioning must follow anatomy.
3. **Fidelity is a dial, not a constant.** A morphologically detailed neuron
   costs ~10⁴× a point neuron. Any brain-scale run must be able to mix model
   classes across regions and report exactly which regions got which fidelity.

## Modeling approach

### Neuron models

A hierarchy, from cheap to expensive, with the rule that **every neuron record
declares which model class it uses**:

| Class | Cost | Status |
|---|---|---|
| Leaky integrate-and-fire (LIF) | ~10 flops/step | **implemented** (placeholder, uncalibrated) |
| Exponential / adaptive integrate-and-fire (EIF, AdEx) | ~10² flops/step | aspirational |
| Multi-compartment (morphological) | ~10⁴ flops/step | aspirational |
| Detailed channel kinetics (Hodgkin–Huxley family) | ~10⁵ flops/step | aspirational |

### Synapse models

| Feature | Status |
|---|---|
| Static scalar weight + conduction delay + sign | **implemented** |
| Receptor kinetics (AMPA/NMDA/GABA) | aspirational |
| Short-term plasticity (depression/facilitation) | aspirational |
| Long-term plasticity (STDP, homeostatic scaling) | aspirational |

### Region and connectome data sources

No measured data is bundled with this repository. The intended sources, all of
which must be recorded in a manifest with a mandatory `measured: true/false`
flag (see [`data/README.md`](data/README.md)):

- **Mesoscale connectivity** — tracer-based regional connectivity matrices
  (e.g. the Allen Mouse Brain Connectivity Atlas) for region-to-region
  projection strengths.
- **Microscale connectome** — electron-microscopy reconstructions (e.g.
  FlyWire for *Drosophila*, MICrONS for mouse visual cortex) for local
  cell-type-specific wiring rules.
- **Human macro-scale** — diffusion-MRI tractography and post-mortem
  parcellations for long-range human white-matter tracts. These are *statistical
  and indirect*; they constrain tract existence and rough strength, not synapse
  counts.
- **Cell-type atlases** — transcriptomic and morphological atlases for the
  density and composition of cell types per region.

**Honest caveat:** there is no complete measured human connectome, and there is
not going to be one soon. Any whole-human-brain model must therefore *generate*
the majority of its synapses from statistical rules fitted to the data above.
This project's position is that those generated synapses must be labelled as
synthetic at every level — in the manifest, in the config, and in any figure
produced from a run.

### Simulation engine

**Implemented:** a single-process, single-threaded reference loop
(`mem20_brainz.simulation.SimulationEngine`) with fixed timestep, one-step
synaptic delay handling, and spike history recording. It is a correctness
reference for pinning down interfaces. It runs thousands of neurons.

**Also implemented:** `VectorizedEngine` + `NumPyBackend` — the same semantics
with the per-neuron Python loop replaced by an elementwise array kernel. It is
**bit-identical** to the reference, and CI asserts that on a 10⁴-neuron
fixed-seed case. It is still single-process and O(synapses) per step; it is not
a brain-scale engine.

**Aspirational:** the production engine. The intended stack, in order of
adoption:

1. **NumPy vectorised state updates** — batch all neurons of a model class into
   arrays; remove per-neuron Python objects. *(done)*
2. **Numba or JAX kernels** — JIT the inner loop; JAX additionally gives
   autodiff for parameter fitting and `vmap`/`pmap` for batching and multi-device
   execution.
3. **Sparse event-driven delivery** — spikes are sparse; deliver them as index
   lists into a precomputed CSR structure rather than scanning all edges.
4. **MPI or a task-graph runtime** — partition neurons across nodes, exchange
   only boundary spikes each timestep.
5. **GPU offload** — for the dense per-neuron state updates, with the sparse
   delivery staying on CPU or using a sparse-aware GPU path.

### Scaling strategy

- **Sparse, columnar connectome storage.** Coordinate lists now; CSR/CSC and
  sharded on-disk formats later. Never a dense adjacency matrix.
- **Anatomy-aware partitioning.** Partition by region and tract, not by
  round-robin, so most synapses stay inside a partition.
- **Fidelity tiers.** Cheap models for the bulk of the brain, expensive models
  for the regions under study, with the tier map recorded in the run config.
- **Reproducibility first.** Every run records its config, seed, dataset
  manifests and code version, so a result can be attributed to a specific
  model — including which parts were synthetic.

## Current status

What is **actually implemented** in this repository today:

- [x] Python package `mem20_brainz` (v0.0.1) that installs via `pyproject.toml`
- [x] `neurons` — `Neuron` interface and an uncalibrated `LIFNeuron`
- [x] `synapses` — immutable `Synapse` with weight, delay, sign
- [x] `regions` — `Region` container with validated cell-type fractions
- [x] `connectome` — `SparseConnectome` coordinate-list storage, in/out degree,
      JSON round-trip
- [x] `simulation` — reference stepping loop with delay handling and spike history
- [x] `simulation` — `Backend` protocol, `LIFState` struct-of-arrays, and a
      `NumPyBackend` that is **bit-identical** to the reference loop
- [x] `simulation` — `VectorizedEngine`: the reference semantics with the
      per-neuron loop replaced by an array kernel
- [x] `io` — JSON config/fixture loading, flat-YAML fallback parser
- [x] `viz` — optional matplotlib spike raster
- [x] `tests` — smoke tests plus bit-identity equivalence tests (including a
      10⁴-neuron fixed-seed case)
- [x] `scripts` — fixture validator, build helper, micro-benchmark
- [x] JSON Schemas for connectome fixtures, neuron records, dataset manifests
- [x] Five ADRs in [`docs/adr/`](docs/adr/README.md) with an index
- [x] CI: `pytest` + `ruff` on push and pull request, Python 3.11 and 3.12

What is **not** implemented, and is not claimed:

- [ ] Any measured connectome data, or any reader for it
- [ ] Any biologically validated neuron or synapse model
- [ ] Any JIT, GPU, or distributed execution (the NumPy backend is vectorised but
      still single-process and O(synapses) per step)
- [ ] Any CSR / event-driven delivery
- [ ] Any region atlas or anatomical parcellation
- [ ] Any plasticity
- [ ] Any brain-scale run of any kind

If you are looking for a working brain simulation, this is not it yet. If you
are looking for a place to build one in the open, with the honesty constraints
baked in from commit one, this is that place.

## Roadmap phases

| Phase | Goal | Exit criterion |
|---|---|---|
| **0 — Scaffold** *(complete)* | Package, interfaces, tests, docs, ADRs, CI, honesty rules | `pip install -e .` and `pytest` pass; interfaces frozen |
| **1 — Vectorised core** *(in progress)* | Array state, `Backend` protocol, NumPy/Numba/JAX kernels | 10⁶ LIF neurons, 1 s biological time, on one GPU |
| **2 — Real data** | Ingest one measured dataset end to end (mouse mesoscale) | A run whose connectivity traces to a cited, measured source |
| **3 — Region atlas** | Parcellation + cell-type composition per region | Region-level model with per-region fidelity tiers |
| **4 — Distributed** | MPI/task-graph partitioning, boundary spike exchange | 10⁹ neurons across multiple nodes |
| **5 — Human scale** | 86 × 10⁹ neurons, 10¹⁴ synapses, sharded storage | A full-brain run, with a published fidelity map |

See [`ROADMAP.md`](ROADMAP.md) for the detailed version and
[`docs/roadmap.md`](docs/roadmap.md) for the technical breakdown.

## Install and run

Requires Python ≥ 3.10.

```bash
# clone
git clone https://github.com/bythewayz66-glitch/mem20-brainz.git
cd mem20-brainz

# install (editable, with dev tools)
pip install -e ".[dev]"

# run the test suite
pytest

# micro-benchmark the reference engine
python scripts/benchmark.py --neurons 1000 --p 0.01 --steps 100

# validate a small connectome fixture
python scripts/ingest_connectome.py data/fixtures/tiny_connectome.json
```

Minimal usage:

```python
import numpy as np
from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import SimulationEngine

neurons = [LIFNeuron(id=i) for i in range(4)]
conn = SparseConnectome.from_arrays(n_neurons=4, pre=[0, 1], post=[2, 3])
engine = SimulationEngine(neurons=neurons, connectome=conn, dt=0.1)

spikes = engine.run(steps=100, external_current=np.full(4, 50.0))
print(f"{spikes} spikes in {engine.t:.1f} ms of simulated time")
```

The vectorised path — same semantics, bit-identical output:

```python
import numpy as np
from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.simulation import LIFState, VectorizedEngine

conn = SparseConnectome.from_arrays(n_neurons=4, pre=[0, 1], post=[2, 3])
engine = VectorizedEngine(state=LIFState.uniform(4), connectome=conn, dt=0.1)

spikes = engine.run(steps=100, external_current=np.full(4, 50.0))
print(f"{spikes} spikes in {engine.t:.1f} ms of simulated time")
```

## Repository layout

```
mem20-brainz/
├── README.md              # this file — the mission document
├── ARCHITECTURE.md        # system design and module boundaries
├── ROADMAP.md             # phased plan with exit criteria
├── CONTRIBUTING.md        # how to contribute, and the honesty rules
├── LICENSE                # MIT
├── pyproject.toml         # installable package config
├── configs/               # example simulation and region configs
├── data/                  # schemas + tiny fixtures ONLY (see data/README.md)
│   ├── schemas/
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   ├── data-model.md
│   ├── glossary.md
│   └── adr/               # architecture decision records 0001-0005
├── .github/workflows/     # CI: pytest + ruff on 3.11 and 3.12
├── scripts/               # ingestion, build, benchmark helpers
├── src/mem20_brainz/
│   ├── neurons/           # neuron models
│   ├── synapses/          # synapse models
│   ├── regions/           # region containers
│   ├── connectome/        # sparse connectivity storage
│   ├── simulation/        # reference engine + Backend protocol + NumPy backend
│   ├── io/                # config and fixture I/O
│   └── viz/               # plotting (optional)
└── tests/                 # smoke and bit-identity equivalence tests
```

## How to contribute

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first — it contains the two rules that
matter more than any style guide:

1. **Never claim a capability the code does not have.** If a module is a
   placeholder, its docstring says so.
2. **Never mix measured and synthetic data silently.** Every dataset carries a
   `measured` flag; every figure derived from a run states which parts were
   synthetic.

Then: open an issue describing the interface you want to change before changing
it, keep commits small and well-messaged, and make sure `pytest` passes.

## License

MIT — see [`LICENSE`](LICENSE).
