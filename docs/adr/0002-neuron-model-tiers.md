# 0002 — Neuron model tiers, declared per neuron

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz (lead architect)
- **Supersedes:** the provisional "fidelity tiers per region" entry in `ARCHITECTURE.md`

## Context

A morphologically detailed, multi-compartment neuron with active channel
kinetics costs on the order of 10⁴–10⁵× a point neuron per timestep. At
86 × 10⁹ neurons, a whole-brain run at detailed fidelity is not a matter of
buying more hardware; it is off by many orders of magnitude. Any brain-scale
model must therefore be **mixed-fidelity**: cheap models for the bulk, expensive
models where the science requires them.

The question is not whether to mix fidelity, but *where the declaration lives*.

Three candidate scopes:

1. **Global** — one model class for the whole run. Simple, but forces the entire
   brain to the cheapest common denominator, or the most expensive.
2. **Per region** — each anatomical region declares a model class. Natural for
   configuration, and it is how a neuroscientist would describe the intent.
3. **Per neuron** — every neuron record carries its own `model_tier`.

Per-region is tempting because it matches how people think. It fails on a
concrete case: a cortical column contains pyramidal cells and PV/SST
interneurons whose dynamics differ qualitatively, and a region-level declaration
cannot express "detailed pyramidal cells, point interneurons" without inventing a
pseudo-region per cell type. It also fails on reproducibility: if the tier is a
property of a region *config*, then changing the config silently changes the
model, and two runs that claim the same fidelity map can differ.

Per-neuron declaration costs one small integer per neuron — 1 byte at 86 × 10⁹
neurons is 86 GB, or 0.1 byte/neuron if packed into the existing flag byte — and
it makes the fidelity map a *property of the data*, not of the config. That is
the property we need: a run's fidelity map can then be read off the dataset and
compared, rather than trusted from a config file.

## Decision

**Every neuron record declares its own `model_tier`.** There is no implicit
default. A neuron without a tier is invalid data, not a neuron that gets the
cheap model.

The tier vocabulary is closed and versioned:

| Tier | Name | What it is | Cost/step | Status |
|---|---|---|---|---|
| `tier0_point_lif` | Point LIF | Single membrane equation, hard threshold, fixed reset, absolute refractory | ~10 flops | **implemented** (uncalibrated placeholder) |
| `tier1_conductance` | Conductance-based point | EIF / AdEx / Izhikevich: adds adaptation and/or exponential spike onset; still one compartment | ~10² flops | aspirational |
| `tier2_multicompartment` | Multi-compartment | Morphology from `morphology_ref`, cable equation per compartment, passive or active | ~10⁴ flops | aspirational |
| `tier3_channel_kinetics` | Detailed biophysical | Hodgkin–Huxley family with explicit channel state per compartment | ~10⁵ flops | aspirational |

Rules that follow from this:

1. **Tier is per neuron, and the tier string is the contract.** A backend may
   implement only some tiers; it must raise on a tier it does not implement
   rather than substituting a cheaper one. This is the "no silent fallbacks"
   principle applied to fidelity.
2. **A run's fidelity map is derived from the data, not declared separately.**
   The run record reports the tier histogram it actually simulated.
3. **Mixing tiers within a population is a first-class case**, not an edge case.
   The vectorised backend groups neurons by tier and steps each group as its own
   array kernel (see ADR-0003).
4. **Tier changes are data changes.** Re-tiering a neuron produces a new dataset
   version (ADR-0004); it is not an in-place edit.
5. **`biophysical_params` must be sufficient for the declared tier and no more.**
   A `tier0` neuron does not carry channel densities; a `tier2` neuron must carry
   per-compartment parameters or a reference to them.

## Consequences

**Easy:** mixed-fidelity runs are expressible; the fidelity map is auditable from
the data; a region can contain heterogeneous cell types without pseudo-regions;
adding a new tier does not change the schema, only the vocabulary.

**Hard:** the simulation engine must dispatch per tier, so the hot loop is a
small number of grouped kernels rather than one uniform kernel. Population
construction must validate that every neuron has a tier and that its parameters
match. Analysis code must be tier-aware when it aggregates.

**Owed to the reader:** every published result states the tier histogram of the
run. A result from a `tier0`-only run must not be described as if it came from a
biophysically detailed model.

## Alternatives considered

- **Global single tier.** Rejected: forces the whole brain to one cost, and makes
  the interesting regions unaffordable or the bulk unaffordable.
- **Per-region tier only.** Rejected: cannot express heterogeneous cell types
  within a region without inventing pseudo-regions, and makes the fidelity map a
  config property rather than a data property, which breaks reproducibility.
- **Continuous fidelity parameter instead of discrete tiers.** Rejected for now:
  a continuous dial has no stable vocabulary, so two runs cannot be compared by
  their fidelity maps. Discrete tiers are a naming contract; a tier may
  internally expose continuous parameters.
- **Implicit default tier for unlabelled neurons.** Rejected outright: it is
  exactly the silent fallback that makes a fidelity claim unfalsifiable.
