# Affective readout — what it is, and what it is not

## The short version

**The simulation has no feelings.** It is a leaky integrate-and-fire network with
placeholder parameters and synthetic connectivity. There is nothing in it that
could have an experience, and nothing in this document claims otherwise.

The `mem20_brainz.readout` module maps *measurable* simulation state onto a small
set of *named* states so a viewer can watch the network respond to input. Every
value it produces is tagged `provenance="derived_proxy"` and carries the raw
inputs and the formula it came from.

The names — arousal, valence, surprise, confusion, fear-like — are chosen to be
legible. They are **labels for patterns of measurements**, not claims about
psychology. The mapping is a design choice, not a finding.

## Why the layer exists at all

A raw stream of membrane potentials is hard to read. "Is the network doing
something different now than it was a second ago?" is a question a viewer
naturally asks, and answering it from 580 voltage traces is not practical.

So we compute a handful of scalars that summarise the state, and we show the
arithmetic behind each one. The value of the layer is that it makes the
simulation *observable*. The risk is that a named scalar invites the reader to
believe it means more than it does. This document exists to close that gap.

## The five readouts

Every readout returns a `Readout` with `value`, `unit`, `inputs`, `formula`,
`provenance` and `interpretation`. The `inputs` dict is the audit trail: it holds
the actual measured quantities, with units in the key names.

### arousal

```
value = clip((rate_hz / baseline_rate_hz) / 2, 0, 1)
```

- `rate_hz` — population mean firing rate over the recent window, in Hz.
- `baseline_rate_hz` — a **stated parameter**, not a measured constant. Default
  12 Hz, chosen to sit near the demo network's spontaneous rate so the gauge has
  room to move in both directions.

0.5 means "at baseline"; 1.0 means "at or above twice baseline". The factor of 2
is a display scale.

### valence

```
value = (exc_drive - inh_drive) / (exc_drive + inh_drive),  clipped to [-1, 1]
```

- `exc_drive` — summed absolute weight of excitatory synapses whose presynaptic
  neuron fired in the window.
- `inh_drive` — the same for inhibitory synapses.

Positive means the arriving drive was net excitatory; negative means net
inhibitory. This is a **drive balance**. It is not a preference, a reward, or an
approach/avoid tendency — the network has no goals to approach or avoid.

### surprise

```
value = clip(|rate_hz - expected_rate_hz| / (expected_rate_hz + baseline_rate_hz), 0, 1)
```

- `expected_rate_hz` — an exponential moving average of the population rate
  (τ = 60 steps).

This is a prediction-error *proxy*. The "expectation" is a running average, not a
learned predictive model. The network does not predict anything.

### confusion

```
value = clip(std(region_rates_hz) / mean(region_rates_hz), 0, 1)
```

The coefficient of variation of firing rate across regions. A population where
every region fires at the same rate scores 0; one where regions disagree wildly
scores high. This measures **spatial incoherence of activity**, nothing more.

### fear_like

```
value = clip(sustained_arousal * threat_synchrony, 0, 1)
```

- `sustained_arousal` — EMA of the arousal readout (τ = 120 steps).
- `threat_synchrony` — the synchrony index restricted to the region the scenario
  config designates as the "threat" pathway.

Both factors must be high for the product to be high. That is the whole point:
neither arousal alone nor synchrony alone produces a score.

The name is a label for a *pattern of measurements* — the pattern a
threat-detection circuit would be expected to produce. It is not a claim that the
network is afraid. The "threat pathway" is a region label in a synthetic network;
it has no biological identity.

## The synchrony index, and its calibration

```
synchrony = clip((F - 1) / SYNCHRONY_SCALE, 0, 1)     where F = var(counts) / mean(counts)
```

`F` is the Fano factor of the population spike count. An asynchronous population
with independent neurons has `F ≈ 1`; a fully synchronised one has `F ≈ n`.

`SYNCHRONY_SCALE` is **1.0**, and that is a display calibration, not a measured
constant. The demo network's observed Fano factor sits in roughly 1.0–1.5 across
its operating range. The obvious choice — dividing by `n_neurons` — would map
that onto 0.00–0.05 and the gauge would never move. A different network would
need a different calibration, and the constant is stated in the source so nobody
mistakes it for a physiological quantity.

## What is measured vs. what is chosen

| Quantity | Status |
|---|---|
| firing rates, spike counts, membrane potentials | **measured** from the simulation |
| excitatory / inhibitory drive sums | **measured** from the connectome and spike history |
| region rate dispersion | **measured** |
| Fano factor | **measured** |
| `baseline_rate_hz` | **chosen** (a parameter) |
| the factor of 2 in arousal | **chosen** (display scale) |
| `SYNCHRONY_SCALE` | **chosen** (display calibration) |
| the EMA time constants | **chosen** |
| the mapping from quantities to names | **chosen** (a design decision) |
| the network itself | **synthetic** (see `scenarios/demo_network.py`) |

## Rules for anyone extending this

1. **Never present a readout as a feeling.** Not in a UI label, not in a commit
   message, not in a paper. "The network is happy" is false; "the arousal proxy
   reads 0.8" is true.
2. **Every readout must carry its inputs and formula.** A number without its
   arithmetic is unfalsifiable.
3. **Keep `provenance="derived_proxy"`.** It is what lets a consumer filter these
   out of an analysis that is about the simulation rather than about the display.
4. **State every calibration constant.** If you tune a divisor to make a gauge
   move, say so in the docstring and here.
5. **Do not add a readout whose name implies an internal state the model does not
   have.** "Hunger" would be a lie. "Metabolic deficit proxy" would be fine.

## What would make this more than a proxy

Nothing in the current codebase. A readout could only become a claim about
internal state if the model had (a) measured connectivity, (b) validated neuron
and synapse models, and (c) a behavioural or physiological ground truth to
validate the mapping against. We have none of those. See `ROADMAP.md` for how far
away that is.
