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
   on any single machine: storage must be sharded and streamed, and the
   simulation must touch only the edges it needs this timestep.
2. **Communication dominates.** Once distributed, the cost is moving spikes
   between partitions, not multiplying weights — so partitioning must follow
   anatomy to minimise cut edges.
3. **Fidelity is a dial, not a constant.** A detailed neuron costs ~10⁴× a point
   neuron, so a brain-scale run must mix model classes and report which region
   got which fidelity.

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

- **Mesoscale** — tracer-based regional matrices (e.g. Allen Mouse Brain
  Connectivity Atlas) for region-to-region projection strengths.
- **Microscale** — EM reconstructions (FlyWire, MICrONS) for local
  cell-type-specific wiring rules.
- **Human macro-scale** — diffusion-MRI tractography and post-mortem
  parcellations. These are *statistical and indirect*: they constrain tract
  existence and rough strength, not synapse counts.
- **Cell-type atlases** — transcriptomic and morphological atlases for per-region
  cell-type density.

**Honest caveat:** there is no complete measured human connectome, and there is
not going to be one soon. Any whole-human-brain model must therefore *generate*
most of its synapses from statistical rules fitted to the data above. Those
generated synapses must be labelled synthetic at every level — manifest, config,
and any figure produced from a run.

### Simulation engine

**Implemented:** `SimulationEngine`, a single-process reference loop with fixed
timestep, one-step delay handling and spike history. Slow, correct, frozen as
ground truth. It runs thousands of neurons.

**Also implemented:** `VectorizedEngine` + `NumPyBackend` — same semantics, the
per-neuron loop replaced by an array kernel. **Bit-identical** to the reference,
asserted in CI on a 10⁴-neuron fixed-seed case. Still single-process and
O(synapses) per step.

**Also implemented:** `NumbaBackend` — the same kernel JIT-compiled, also
**bit-identical** (`fastmath=False`; FMA contraction would break it, so the
equivalence test is what proves it per platform). Optional extra, lazy import.

**Also implemented:** `EventDrivenEngine` over a `CSRConnectome` — delivery
visits only the out-edges of neurons that fired, cutting work by ~97% in the
benchmark configuration. It is **not** bit-identical (different float addition
order; measured drift 5.4 × 10⁻⁹, no spike flips observed) and is gated on a
tolerance test — see [ADR-0006](docs/adr/0006-e
