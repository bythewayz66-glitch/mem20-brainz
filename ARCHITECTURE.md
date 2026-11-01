# Architecture

This document describes the system as it is **today**, and marks clearly what is
implemented versus planned. If a section describes something that does not exist
in the code, it says so.

For the *why* behind each major decision, see [`docs/adr/`](docs/adr/README.md).

---

## The shape of the problem

Three numbers determine the entire architecture:

| Quantity | Value | Consequence |
|---|---|---|
| Neurons | ~86 × 10⁹ | State alone is terabytes; must be partitioned |
| Synapses | ~1 × 10¹⁴ | Edges dominate memory; must be sparse and streamed |
| Timestep | ~0.1 ms | 10,000 steps per second of biological time |

Everything below follows from those three. The design rule that falls out of
them: **never materialise anything proportional to N² or to the full edge list
in memory at once.**

---

## Module boundaries

```
src/mem20_brainz/
├── neurons/       neuron models and their state
├── synapses/      synapse models and plasticity
├── regions/       region containers and cell-type composition
├── connectome/    sparse storage, CSR view, partitioning
├── simulation/    engines, Backend protocol, kernels
├── scenarios/     synthetic demo networks (provenance-tagged)
├── readout/       affective proxies over measured state
├── server/        FastAPI streaming server + headless mode
├── io/            config and fixture loading
└── viz/           plotting (optional)
```

The dependency rule: **`simulation` may import from `connectome`, `neurons`,
`synapses`; `server` may import from `simulation` and `readout`; nothing imports
from `server`.** The server is a leaf — it orchestrates and serialises, and
contains no simulation logic. That is what keeps it replaceable.

---

## `neurons/` — **implemented (placeholder)**

`Neuron` is an abstract interface; `LIFNeuron` is a leaky integrate-and-fire
implementation with **uncalibrated** parameters. It is a correctness reference,
not a biological model. Its docstring says so.

Model tiers are declared per neuron (ADR-0002):

| Tier | Name | Status |
|---|---|---|
| `tier0_point_lif` | point LIF | **implemented** |
| `tier1_conductance` | conductance-based | planned |
| `tier2_multicompartment` | multi-compartment | planned |

---

## `synapses/` — **implemented (minimal)**

`Synapse` is an immutable record: `pre_id`, `post_id`, `weight`, `delay_ms`,
`type`, `plasticity_state`, `provenance`. Static only — no plasticity is
implemented.

---

## `connectome/` — **implemented (sparse only)**

`SparseConnectome` stores edges as coordinate lists (`pre`, `post`, `weight`,
`delay_ms`). Never a dense matrix (ADR-0001).

`CSRConnectome` is a row-compressed view indexed by presynaptic neuron:
`indptr` (int64, length N+1), `indices` (int32, postsynaptic ids), `weights`
(float32), `delays` (float32). Built with a **stable** sort on `pre`, so
within-row edge order is preserved from the input edge list.

**Memory layout:**

```
indptr:  [0, 3, 5, 5, 9, ...]        N+1 int64  = 8(N+1) bytes
indices: [4, 7, 2, 0, 9, ...]        E   int32  = 4E bytes
weights: [0.5, 1.2, ...]             E   float32 = 4E bytes
delays:  [1.0, 1.0, ...]             E   float32 = 4E bytes
```

At 10¹⁴ synapses that is 1.2 PB for the edge arrays alone — which is why
partitioning (ADR-0005) is not optional.

---

## `simulation/` — **implemented (single-process)**

### The reference engine

`SimulationEngine` is a per-neuron Python loop. It is slow, correct, and
**frozen**: it is the ground truth every other path is checked against
(ADR-0003). It is never optimised, because an optimised reference is no longer a
reference.

### The `Backend` protocol

```python
class Backend(Protocol):
    def step(self, state: LIFState, i_inj: np.ndarray, dt: float, t: float) -> np.ndarray: ...
```

`LIFState` is a struct-of-arrays: one `float64` array per field, so a step is a
handful of vectorised operations over the whole population. Backends implement
`step` and nothing else; the engine owns delivery, delays, spike history and the
clock.

| Backend | Bit-identical? | Gate | Notes |
|---|---|---|---|
| `NumPyBackend` | **yes** | bit-identity test | default |
| `NumbaBackend` | **yes** | bit-identity test | optional extra, `fastmath=False` |
| `JAXBackend` | **no** | tolerance (1e-9 rel / 1e-12 abs, 1% spikes) | optional extra, needs `jax_enable_x64` |

### `VectorizedEngine`

Same semantics as the reference, with the per-neuron loop replaced by a backend
call. Delivery is still a full edge scan, so it is O(synapses) per step.

### `EventDrivenEngine` and `CSRConnectome`

Delivery visits only the out-edges of neurons that fired. In the benchmark
configuration that is a ~3% edge hit rate, so ~97% of the work is skipped.

**Why it is not bit-identical.** The reference sums contributions in edge-list
order; this path sums them in CSR order (grouped by presynaptic neuron). Float
addition is not associative, so the order is a different order of float
additions; measured drift is up to
5.4 × 10⁻⁹ relative on a high-fan-in case. No spike flips were observed in any
tested regime, but that is an observation about this LIF model, not a guarantee,
so the gate is a tolerance test (`SPIKE_COUNT_TOLERANCE` 1%,
`PER_STEP_AGREEMENT` 95%) rather than a bit-identity claim. One case *is* exact:
when the edge list is already grouped by `pre`, CSR order equals edge-list order.

**Known limitation:** `indices` is `int32`, capping the population at
2,147,483,647 neurons — 86 × 10⁹ exceeds that by 40×. This view is correct for
the 10⁴–10⁸ range Phases 1–4 run at; widening to `int64` costs 400 TB at 10¹⁴
synapses and is deferred.

**Segmented reduction** (ADR-0006, resolution). Delivery originally used
`np.add.at`, an unbuffered scatter that dominated the per-step cost. It is now
`np.bincount(posts, weights=weights, minlength=n_neurons)`, which indexes
directly by post id and needs no sort — unlike `np.add.reduceat`, which would
cost an `argsort` per timestep. Measured effect: crossover speedup went from
1.00×/1.08×/0.79×/1.32×/1.04× to **2.05×/1.30×/1.42×/2.69×/2.30×** across
2k→40k neurons, and `event_driven` became the fastest path on the main
benchmark. `bincount` uses pairwise summation, so the two are still not
bit-identical (measured 5.4 × 10⁻⁷ absolute on sums of order 70, ~7 × 10⁻⁹
relative) — the tolerance gate remains.

**JAX and autodiff** (ADR-0008). `JAXBackend` exists for one reason: `jax.grad`
through a simulation, which no other backend offers. It is **not** bit-identical
— XLA fuses and reassociates, and the membrane potential diverges at step 0
(max abs 1.4 × 10⁻¹⁴), staying bounded at 5.9 × 10⁻¹⁶ relative over 2000 steps.
The gate is a tolerance test (1e-9 relative / 1e-12 absolute on `v`, 1% on spike
count). It requires `jax_enable_x64`, because JAX silently truncates float64 to
float32 otherwise — which per ADR-0007 changes spike output.

Gradients through a spike are **not** the derivative of the model: the spike is
discontinuous at threshold. `scripts/fit_jax_example.py` therefore uses a
straight-through estimator — forward pass is the true spike count, only the
derivative is substituted with a sigmoid surrogate. It is a heuristic search,
not exact gradient descent, and is documented as such.

**Planned:** the partitioner, and recurrent delivery inside the traced JAX loop.

### `io/` — **implemented (minimal)**
