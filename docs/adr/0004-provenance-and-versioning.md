# 0004 — Record-level provenance and immutable dataset versioning

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz (lead architect)
- **Supersedes:** the provisional "mandatory `measured` flag" entry in
  `ARCHITECTURE.md`

## Context

There is no complete measured human connectome, and there will not be one soon.
The best available data is partial and indirect:

- **Electron microscopy** gives synapse-level truth for tiny volumes — FlyWire
  for the whole *Drosophila* brain (~10⁵ neurons), MICrONS for ~1 mm³ of mouse
  visual cortex. These are measured, but they are not human and not whole-brain.
- **Mesoscale tracers** give region-to-region projection strengths in mouse.
  Measured, but region-level, not synapse-level.
- **Diffusion MRI tractography** gives human white-matter tracts. Indirect and
  statistical: it constrains whether a tract exists and roughly how strong it is,
  not how many synapses it contains.
- **Cell-type atlases** give densities and composition per region.

A whole-human-brain model must therefore **generate the large majority of its
synapses** from statistical rules fitted to the data above. That is not a
compromise to be hidden; it is the central scientific fact about any such model.

The failure mode this ADR exists to prevent is specific and common: a model is
built mostly from generated edges, a figure is published, and the figure reads as
if it describes measured biology. A single boolean `measured: true/false` on the
*dataset* does not prevent this, because real datasets are mixed — a connectome
can contain measured tracer-derived inter-region edges alongside synthetic
intra-region edges, in the same file.

Provenance must therefore be **per record**, not per dataset, and it must be
**filterable**, so that any analysis or figure can be restricted to measured
edges and the difference reported.

## Decision

### 1. Every neuron and every synapse carries a `provenance` object

```json
{
  "kind": "measured" | "synthetic" | "hybrid",
  "source_id": "string",
  "citation": "DOI or URL",
  "method": "string (required when kind != measured)",
  "generator": "string (required when kind == synthetic)",
  "generator_params": { "...": "..." },
  "seed": 12345,
  "fitted_to": "source_id of the statistics used"
}
```

- `kind` is mandatory and has no default. A record without it is invalid.
- `measured` requires `source_id` and `citation`.
- `synthetic` requires `generator`, `generator_params`, `seed`, and `fitted_to`.
  A synthetic record that cannot name the statistics it was fitted to is not
  reproducible and is rejected.
- `hybrid` means the record combines measured and generated components (e.g. a
  measured tract with a synthetic synapse count); it must carry both the source
  and the generator fields.

### 2. Provenance is a property of the record, and it survives every transform

Any operation that produces new records — expansion, resampling, pruning,
re-tiering — must propagate or re-derive provenance. An edge that was measured
and then had its weight rescaled is `hybrid`, not `measured`.

### 3. Datasets are immutable and versioned by id + content hash

- A dataset version is identified by `dataset_id` + `version` + `content_hash`
  (SHA-256 over the canonical serialisation of its shards).
- Published versions are **never mutated in place**. A correction publishes a new
  version; the old one stays resolvable.
- Configs reference datasets by `dataset_id@version#hash`, never by path alone.
- The hash is verified on load. A mismatch is a hard error, not a warning.

### 4. Mixed datasets are first-class, and filtering is mandatory in analysis

- A single connectome may contain measured and synthetic edges.
- Every analysis function that aggregates connectivity must accept a provenance
  filter, and every figure derived from a run must state the measured/synthetic
  breakdown of the edges it used.
- The run record stores the provenance histogram it actually simulated.

### 5. Large datasets live outside the repository

The repository holds schemas, small fixtures, and ingestion scripts. It never
holds a connectome. `.gitignore` blocks the relevant extensions and directories,
and a large file in a pull request is a review rejection, not a discussion.

## Consequences

**Easy:** a figure can be restricted to measured edges and the restriction
reported; a synthetic edge can be traced to its generator, parameters and seed;
 a published result can be reproduced from its manifest; a reviewer can check the
measured/synthetic split without reading the code.

**Hard:** every record carries ~100+ bytes of provenance metadata. At 10¹⁴ edges
that is ~10 PB if stored per edge — which is why provenance is stored as a
**dictionary of provenance blocks with integer references**, not inline per edge:
each edge carries a small `provenance_id` (int16, 65 535 distinct blocks is
ample), and the blocks live in a side table. This is a real design constraint
that follows from ADR-0001's arithmetic and must be honoured by the storage
layer.

**Owed to the reader:** no result may be described as measured unless its
provenance histogram says so. "Synthetic" is not a dirty word here; presenting it
as measured is.

## Alternatives considered

- **Single `measured` boolean per dataset.** Rejected: cannot represent mixed
  datasets, which are the normal case, and cannot be filtered at record level.
- **Provenance in prose only (README, methods section).** Rejected: not
  machine-checkable, and it is exactly how synthetic structure gets presented as
  measured biology.
- **Inline provenance per edge.** Rejected on arithmetic: ~10 PB at human scale.
  Integer reference into a side table is the only affordable form.
- **Mutable datasets with a changelog.** Rejected: a result that cites a mutable
  dataset is not reproducible. Immutability plus new versions is the only
  defensible option.
