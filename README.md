# mem20-brainz

**A public, open, biologically grounded attempt to model the human brain at the
scale of individual neurons — one modeled unit per neuron, ~86 billion of them,
connected by ~100 trillion synapses.**

> **Status: pre-alpha scaffold (v0.0.1).** This repository currently contains a
> project skeleton, data containers, a reference simulation loop, three engines
> (NumPy, Numba, and a CSR event-driven path) checked against that reference, and
> a live observability layer (streaming server + web frontend) for watching a
> small synthetic network run. It does **not** contain a brain model, real
> connectome data, or any brain-scale capability. Every claim below is labelled
> as *implemented* or *aspirational*. Please read
> [Current status](#current-status) before anything else.

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

**Also implemented:** `NumbaBackend` — the same kernel JIT-compiled, also
**bit-identical** (compiled with `fastmath=False`; FMA contraction would break
it, so the equivalence test is what actually proves it on each platform). Numba
is an optional extra (`pip install 'mem20-brainz[numba]'`), imported lazily.

**Also implemented:** `EventDrivenEngine` over a `CSRConnectome` — delivery
visits only the out-edges of neurons that fired, cutting work by ~97% in the
benchmark configuration. It is **not** bit-identical (different float addition
order; measured drift 5.4 × 10⁻⁹, no spike flips observed) and is gated on a
tolerance test — see [ADR-0006](docs/adr/0006-event-driven-delivery.md).

**The optimisation now wins on wall time too.** Delivery originally used
`np.add.at`, an unbuffered scatter that dominated the per-step cost, so the work
reduction did not translate into time. Replacing it with a segmented reduction
(`np.bincount`) changed that: crossover speedup went from 1.00×/1.08×/0.79×/
1.32×/1.04× to **2.11×/4.01×/2.93×/3.50×/2.10×** across 2k→40k neurons, and
`event_driven` is now the fastest path on the main benchmark (9.76M synapses/s,
up from 6.21M). Numbers: [`docs/benchmarks/`](docs/benchmarks/README.md).

**Also implemented:** `JAXBackend` — the same kernel compiled by XLA. It is
**not** bit-identical and does not claim to be: XLA fuses and reassociates, and
the membrane potential diverges at step 0 (max abs 1.4 × 10⁻¹⁴), staying bounded
at 5.9 × 10⁻¹⁶ relative over 2000 steps. The gate is a tolerance test. It exists
for one reason the others cannot offer — `jax.grad` through a simulation — and
`scripts/fit_jax_example.py` demonstrates it by fitting a drive to a 40 Hz
target (2.0 → 3.4746 nA, loss 2.56e+04 → 0, rate 0.00 → 40.00 Hz). The gradient
is a **surrogate** (straight-through estimator): the forward pass is the true
spike count, only the derivative is substituted, because a spike is
discontinuous at threshold. See [ADR-0008](docs/adr/0008-jax-backend.md).

**Precision:** state is `float64` by default, and that is the only
reference-equivalent precision. `float32` is an explicit opt-in that halves
state memory (2.8 TB vs 5.5 TB at 86 × 10⁹ neurons) but drifts spike count by
~0.14% and agrees on only ~14% of timesteps for a heterogeneous population — so
the saving is **not** banked. See [ADR-0007](docs/adr/0007-numeric-precision.md).

**Aspirational:** the production engine. The intended stack, in order of
adoption:

1. **NumPy vectorised state updates** — batch all neurons of a model class into
   arrays; remove per-neuron Python objects. *(done)*
2. **Numba or JAX kernels** — JIT the inner loop; JAX additionally gives
   autodiff for parameter fitting and `vmap`/`pmap` for batching and multi-device
   execution. *(both done; JAX tolerance-gated)*
3. **Sparse event-driven delivery** — spikes are sparse; deliver them as index
   lists into a precomputed CSR structure rather than scanning all edges.
   *(done, tolerance-gated)*
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
- [x] `simulation` — `NumbaBackend`: JIT kernel, **bit-identical** to the
      reference, optional dependency
- [x] `simulation` — `JAXBackend`: XLA-compiled, tolerance-gated (ADR-0008),
      optional dependency, with `jax_enable_x64` enforced
- [x] `scripts/fit_jax_example.py` — gradient-based parameter fitting via a
      straight-through estimator (surrogate gradient, ADR-0008)
- [x] `simulation` — `precision` mode: `float64` default, `float32` opt-in,
      with the measured divergence pinned by tests (ADR-0007)
- [x] `connectome` — `CSRConnectome`: row-compressed view indexed by
      presynaptic neuron
- [x] `simulation` — `EventDrivenEngine`: CSR delivery, tolerance-gated
      (ADR-0006)
- [x] `simulation` — segmented reduction (`np.bincount`) replacing the
      `np.add.at` scatter: 2.1×–4.0× crossover speedup (ADR-0006)
- [x] `scripts/benchmark.py` — harness with a committed JSON baseline and a
      `--check` mode that fails when the baseline goes stale
- [x] `io` — JSON config/fixture loading, flat-YAML fallback parser
- [x] `viz` — optional matplotlib spike raster
- [x] `scenarios` — `build_demo_network`: a synthetic 5-region network (580
      neurons, ~31k synapses) with generator, seed and parameters recorded in
      provenance
- [x] `readout` — five named states (arousal, valence, surprise, confusion,
      fear-like) computed from measured simulation quantities, each tagged
      `provenance="derived_proxy"` and carrying its raw inputs and formula
- [x] `server` — FastAPI app: REST control (start/pause/reset/step/config/
      stimulus), a versioned frame schema, and a WebSocket stream
- [x] `server` — headless mode (`python -m mem20_brainz.server.headless`) that
      prints a text summary, so the layer is testable without a browser
- [x] `web/` — React + Vite frontend: canvas population view, spike/readout
      timeline, readout gauges with expandable arithmetic, controls, status bar,
      and an explicit disconnected state
- [x] `tests` — smoke tests plus bit-identity equivalence tests (including a
      10⁴-neuron fixed-seed case), readout unit tests, and server/WebSocket tests
- [x] `scripts` — fixture validator, build helper, micro-benchmark, `dev.sh`
- [x] JSON Schemas for connectome fixtures, neuron records, dataset manifests
- [x] Ten ADRs in [`docs/adr/`](docs/adr/README.md) with an index
- [x] CI: `pytest` + `ruff` on push and pull request, Python 3.11 and 3.12,
      plus numba, jax, server, frontend-build and benchmark-baseline jobs

What is **not** implemented, and is not claimed:

- [ ] Any measured connectome data, or any reader for it
- [ ] Any biologically validated neuron or synapse model
- [ ] Any GPU or distributed execution (all engines are single-process; the
      scan-based paths are still O(synapses) per step)
- [ ] Any exact gradient through spiking dynamics (the JAX fit uses a surrogate)
- [ ] Any recurrent delivery inside the traced JAX loop (it is feedforward only)
- [ ] Any region atlas or anatomical parcellation
- [ ] Any plasticity
- [ ] Any brain-scale run of any kind
- [ ] **Any feeling, emotion, or subjective experience.** The affective readouts
      are engineering proxies over simulation state. See
      [`docs/readout.md`](docs/readout.md).

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

# micro-benchmark every engine path (writes docs/benchmarks/baseline.json)
python scripts/benchmark.py --markdown

# fail if the committed baseline is stale relative to the code
python scripts/benchmark.py --check

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

## Watching it run

There is a live observability layer: a streaming server over the existing engine,
and a web frontend that renders the population, the spike timeline, and the
derived readouts in real time.

```bash
# install the server extra
pip install -e ".[dev,server]"

# start backend + frontend together (Ctrl-C stops both)
./scripts/dev.sh
# → backend  http://127.0.0.1:8000
# → frontend http://127.0.0.1:5173   ← open this
```

Or run the halves separately:

```bash
python -m mem20_brainz.server            # backend only, port 8000
cd web && npm install && npm run dev     # frontend only, port 5173
```

No browser? The headless mode prints the same numbers the frontend receives:

```bash
python -m mem20_brainz.server.headless --steps 400 --stimulus threat --every 50
```

```
mem20-brainz — headless run
====================================================================
backend      : numpy
neurons      : 580
synapses     : 30,729
data kind    : synthetic  (synthetic — not measured)
steps        : 400
total spikes : 1,842
throughput   : 412.3 steps/s
stimulus     : threat @ 6.0 for 200.0 ms

  step      t_ms  spikes  readouts
--------------------------------------------------------------------
  1000     100.0      12  arousal=0.216  valence=-0.058  surprise=0.024  confusion=0.187  fear_like=0.000
  1050     105.0      31  arousal=0.402  valence=0.113  surprise=0.198  confusion=0.244  fear_like=0.041
  ...

NOTE: the readouts above are derived proxies over simulation state.
      The model has no feelings. See docs/readout.md.
```

### What you are looking at

- **Population** — one dot per neuron, coloured by region, brightness driven by
  membrane potential, white rings on spikes. The coordinates are *display*
  coordinates from the scenario, not soma positions in any anatomical space.
- **Timeline** — spikes per frame on top, readout traces below.
- **Affective readout** — five gauges. Expand any one to see the measured inputs
  and the exact formula behind the number.
- **Controls** — play/pause, reset, stimulus injection (pick a region, strength,
  duration), and sliders for steps-per-frame, baseline rate and tonic drive.
- **Status bar** — backend, engine, neuron/synapse counts, steps/s, and a
  provenance chip that says `data: synthetic — not measured`.

### The readouts are proxies, not feelings

**The model does not have feelings.** It is a leaky integrate-and-fire network
with placeholder parameters and synthetic connectivity. The named states
(arousal, valence, surprise, confusion, fear-like) are a fixed, documented mapping
from *measurable* simulation quantities — firing rates, excitatory/inhibitory
drive, cross-region dispersion, population synchrony — onto legible labels.

Every readout returns its raw inputs and its formula, is tagged
`provenance="derived_proxy"`, and is labelled as such in the UI. The mapping is a
design choice, not a finding. Read [`docs/readout.md`](docs/readout.md) before
using any of these numbers in a claim.

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
│   ├── readout.md         # what the affective proxies are, and are not
│   ├── adr/               # architecture decision records 0001-0008
│   └── benchmarks/        # committed baseline.json + how to run it
├── .github/workflows/     # CI: pytest + ruff, numba, server, frontend, benchmark
├── scripts/               # ingestion, build, benchmark, dev.sh
├── web/                   # React + Vite frontend for the live stream
│   └── src/
│       ├── components/    # canvas, timeline, gauges, controls, status bar
│       └── lib/           # wire types, WebSocket hook, REST client
├── src/mem20_brainz/
│   ├── neurons/           # neuron models
│   ├── synapses/          # synapse models
│   ├── regions/           # region containers
│   ├── connectome/        # sparse storage + CSR view
│   ├── simulation/        # reference engine, Backend protocol, NumPy/Numba, event-driven
│   ├── scenarios/         # synthetic demo network (provenance-tagged)
│   ├── readout/           # affective proxies (derived_proxy, not feelings)
│   ├── server/            # FastAPI streaming server + headless mode
│   ├── io/                # config and fixture I/O
│   └── viz/               # plotting (optional)
└── tests/                 # smoke, equivalence, readout and server tests
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
