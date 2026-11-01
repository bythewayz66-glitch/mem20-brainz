# ADR-0008 — Live observability layer

- **Status:** Accepted
- **Date:** 2026-11-01
- **Deciders:** Mem20-brainz (lead architect)
- **Supersedes:** none
- **Related:** ADR-0001 (sparse storage), ADR-0003 (simulation backend), ADR-0004 (provenance), ADR-0006 (event-driven delivery)

## Context

The engine can run, but the only way to see what it is doing is to read arrays in
a REPL or parse a benchmark log. That is adequate for development and useless for
understanding. Two questions are hard to answer today:

1. *Is the network doing something different now than it was a second ago?*
2. *What happens when I stimulate this region?*

Both are questions about **observability**, not about simulation capability. The
engine already produces everything needed to answer them; there is no way to look
at it.

There is also a temptation worth naming. A live view of a spiking network invites
the viewer to read intention into it — "it looks agitated", "it's calming down".
Any observability layer we build will be used that way unless we design against
it. That is a design constraint, not an afterthought.

## Decision

Add a **live observability layer** consisting of three parts:

1. **A streaming server** (`src/mem20_brainz/server/`) — FastAPI, REST for
   control, WebSocket for frames.
2. **A separate frontend** (`web/`) — React + Vite, canvas rendering.
3. **An affective readout module** (`src/mem20_brainz/readout/`) — named states
   derived from measured quantities, explicitly tagged as proxies.

### 1. The server orchestrates; it does not simulate

The server contains **no simulation arithmetic**. Every voltage update happens
inside the `Backend` behind `VectorizedEngine`. The server's job is:

- run lifecycle (start / pause / reset / step),
- stimulus injection (adding an external current to a region),
- aggregation (per-region rates, synchrony, drive sums),
- serialisation.

This is enforced by review, not by the type system, so it is stated here as a
rule: **if you find yourself writing a membrane equation in `server/`, it belongs
in the backend instead.**

The server reuses the existing `Backend` protocol rather than reaching into the
engine's internals. It works with the NumPy backend today; a Numba or CSR backend
would drop in without a server change, because the server only calls
`engine.step(current)`.

### 2. The frame schema

One frame per streamed step. The schema is versioned (`FRAME_SCHEMA_VERSION = 1`)
and the frontend refuses to render a version it does not know.

```jsonc
{
  "schema_version": 1,
  "step": 1004,
  "t_ms": 100.4,
  "dt_ms": 0.1,
  "v_q": [67, 66, ...],        // membrane potential, 8-bit quantised
  "v_range": [-80.0, -40.0],   // the mV range v_q maps onto
  "v_bin_size": 1,             // 1 = one value per neuron; >1 = averaged
  "spikes": [12, 47, 103],     // neuron indices that fired this step
  "regions": [                 // per-region aggregates
    { "name": "sensory", "n_neurons": 120, "rate_hz": 5.8,
      "active_fraction": 0.31, "mean_v_mv": -63.2 }
  ],
  "readouts": [                // derived proxies, with inputs and formula
    { "name": "arousal", "value": 0.216, "unit": "dimensionless",
      "inputs": { "rate_hz": 5.2, "baseline_rate_hz": 12.0, "ratio": 0.43 },
      "formula": "clip((rate_hz / baseline_rate_hz) / 2, 0, 1)",
      "provenance": "derived_proxy", "interpretation": "..." }
  ],
  "status": { "backend": "numpy", "n_neurons": 580, "n_synapses": 30729,
              "steps_per_sec": 412.0, "data_kind": "synthetic", ... },
  "provenance": { "kind": "synthetic", "generator": "...", "seed": 20261101, ... }
}
```

**Why quantise the voltage.** A float64 per neuron per frame is 4.6 KB for 580
neurons and 688 GB for 86 × 10⁹. Even at demo scale, sending floats wastes
bandwidth for a value that is displayed as a pixel brightness. 8 bits over a
declared `[-80, -40]` mV range is 580 bytes, and the range travels with the frame
so the mapping is exact and reversible. Values outside the range are **clipped,
not wrapped** — a clipped value is visibly saturated, whereas a wrapped one would
be a lie about the voltage.

**Why bin.** Above `DEFAULT_MAX_V_VALUES` (2000) neurons, voltages are averaged
into contiguous bins so the payload stays bounded regardless of population size.
`v_bin_size` states the reduction, so a binned value is never mistaken for a
single neuron's voltage.

**Why the frame carries provenance.** A viewer must be able to tell that this
network is synthetic without reading prose. `provenance.kind` and
`status.data_kind` are in every frame, and the UI renders them as a chip.

### 3. A separate frontend, not server-rendered HTML

The frontend is a separate Vite app rather than templates served by FastAPI.

- The rendering is canvas-based and stateful (spike rings fade over ~180 ms); it
  does not fit a request/response model.
- The dev loop is better: Vite HMR against a running backend, with `/api` and
  `/ws` proxied so the browser sees one origin.
- The frontend can be replaced without touching the server, and vice versa. The
  contract is the frame schema, which is versioned.

The cost is a second toolchain (Node) in CI. Accepted: the alternative is a
worse UI and a worse dev loop.

### 4. The honesty constraint on affective readouts

This is the part of the decision that matters most.

**The model does not have feelings.** It is a leaky integrate-and-fire network
with placeholder parameters and synthetic connectivity. Nothing in it can have an
experience.

The readout layer therefore:

- computes named states **only** from measured simulation quantities,
- returns the raw inputs and the formula with every value,
- tags every value `provenance="derived_proxy"`,
- is documented in `docs/readout.md` as a design choice, not a finding,
- is labelled in the UI as "derived proxy — not a feeling", persistently and
  non-dismissibly, on every gauge.

The names (arousal, valence, surprise, confusion, fear-like) are chosen for
legibility. They label *patterns of measurements*. `fear_like` in particular
requires both sustained high arousal **and** high synchrony in a designated
region — the pattern a threat circuit would be expected to produce — and the
underscore in the name is deliberate: it is fear-*like*, a resemblance, not a
state.

**Rejected alternative:** omit the readouts entirely and show only raw activity.
This was rejected because it makes the simulation much harder to read, and
because the interpretive pressure does not disappear when you omit the labels —
it just moves into the viewer's head, unexamined and undocumented. A labelled,
auditable proxy is more honest than an unlabelled invitation to project.

### 5. Degrade honestly on disconnect

When the WebSocket drops, the frontend **discards the last frame** and shows a
disconnected state. It does not freeze on stale data.

A frozen frame presented as live is a lie about the state of the run. The
`useLiveStream` hook clears `frame` on close, and the timeline history is cleared
too, so a gap in the record is visible rather than papered over.

### 6. One run at a time

The server holds a single process-wide `RunRunner`. This is an observability
tool, not a multi-tenant service. Multiple concurrent runs would need per-session
state, resource limits and a scheduler — none of which is justified by the use
case. If that changes, it is a new ADR.

## Consequences

**Good**

- The simulation is observable: you can watch it run, stimulate a region, and see
  the response in the gauges and the timeline.
- The server is thin, so a faster backend improves the stream without a server
  change.
- The frame schema is versioned and self-describing about provenance.
- The honesty constraint is structural (a tag, a field, a persistent label), not
  a matter of remembering to be careful.

**Bad**

- A second toolchain (Node/Vite) in CI.
- The frame schema is a compatibility surface that must be maintained.
- Quantisation loses voltage precision. Acceptable for display; **not** acceptable
  for analysis, which should read the engine directly.
- The readout layer adds a layer of interpretation that a reader could still
  over-trust, despite the labels. This is mitigated, not eliminated.

**Neutral**

- The demo network is synthetic and small (580 neurons, ~31k synapses). It exists
  to make the layer observable, not to model a brain region.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Server-rendered HTML + polling | Cannot render a fading spike animation; polling wastes bandwidth; worse dev loop. |
| Send full float64 voltages | 8× the bandwidth for a value displayed as pixel brightness; does not scale. |
| Send only spikes, no voltages | Loses the subthreshold dynamics, which is most of what the network is doing. |
| Put readout logic in the frontend | Would duplicate arithmetic in two languages and make it untestable in Python. |
| Omit readouts, show raw activity only | Moves interpretive pressure into the viewer's head, unexamined. See §4. |
| gRPC / binary protocol | Overkill for a local observability tool; JSON is inspectable with `curl`. |
| Multi-run server | Not justified by the use case. See §6. |

## Open questions

1. **Frame rate under load.** At 30 fps with `steps_per_frame = 4`, the stream
   advances 120 steps/s of simulated time. Whether that is the right decimation
   for a viewer to perceive dynamics is untested with real users.
2. **Binning fidelity.** `DEFAULT_MAX_V_VALUES = 2000` is a guess. At what
   population size does binning start to hide structure a viewer would want?
3. **Readout calibration.** `SYNCHRONY_SCALE` and the arousal factor of 2 are
   calibrated to the demo network. A measured connectome would need recalibration,
   and there is no principled procedure for that yet.
4. **Recording.** There is no way to save a run and replay it. That would make the
   layer useful for regression testing, and is the obvious next step.
