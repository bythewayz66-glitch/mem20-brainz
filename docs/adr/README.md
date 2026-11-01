# Architecture Decision Records

An ADR records one decision, the context that forced it, and what it costs. They
are immutable once accepted: to change a decision, write a new ADR that
supersedes the old one and update the old one's `Status` line to
`Superseded by ADR-NNNN`. Do not edit the reasoning of an accepted ADR.

The point is not bureaucracy. This project makes claims about biology, and a
claim is only checkable if the reasoning behind it is written down and dated.

## Template

```markdown
# NNNN — Title

- **Status:** proposed | accepted | superseded by ADR-NNNN | deprecated
- **Date:** YYYY-MM-DD
- **Deciders:** who agreed
- **Supersedes:** ADR-NNNN (if any)

## Context
What forces are at play. Numbers, not adjectives.

## Decision
What we are doing, stated so that it can be violated.

## Consequences
What becomes easy, what becomes hard, what we now owe the reader.

## Alternatives considered
What we rejected, and the specific reason.
```

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-sparse-connectome-storage.md) | Sparse connectome storage; never a dense adjacency matrix | accepted |
| [0002](0002-neuron-model-tiers.md) | Neuron model tiers, declared per neuron | accepted |
| [0003](0003-simulation-backend.md) | Simulation backend stack and the `Backend` boundary | accepted |
| [0004](0004-provenance-and-versioning.md) | Record-level provenance and immutable dataset versioning | accepted |
| [0005](0005-partitioning-and-chunking.md) | Chunked, partitioned connectome layout | accepted |
| [0006](0006-event-driven-delivery.md) | Event-driven spike delivery over a CSR connectome | accepted |
| [0007](0007-numeric-precision.md) | Numeric precision: float64 default, float32 opt-in | accepted |
| [0008](0008-live-observability-layer.md) | Live observability layer: streaming server, frontend, affective readout proxies | accepted |

## Note on numbering

An earlier draft of [`ARCHITECTURE.md`](../../ARCHITECTURE.md) listed five
provisional decisions under different titles (implementation language, sparse
storage, reference-before-optimisation, the `measured` flag, and fidelity
tiers). Those were placeholders written before any ADR existed. They have been
folded into the five records above and the table in `ARCHITECTURE.md` now points
here:

| Provisional | Now recorded in |
|---|---|
| Python + JIT/GPU escape hatches | ADR-0003 |
| Sparse columnar storage, never dense | ADR-0001 |
| Reference implementation before optimisation | ADR-0003 |
| Mandatory `measured` flag | ADR-0004 |
| Fidelity tiers per region | ADR-0002 |

## Writing a new ADR

1. Copy the template into `docs/adr/NNNN-short-title.md`, using the next free
   number.
2. Put the numbers in `Context`. If the decision depends on an arithmetic
   argument, show the arithmetic.
3. Add a row to the index above.
4. If the ADR changes the data model, update
   [`docs/data-model.md`](../data-model.md) and the JSON Schemas in
   [`data/schemas/`](../../data/schemas/) in the same pull request.
