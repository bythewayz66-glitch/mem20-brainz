# 0005 — Chunked, partitioned connectome layout

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz (lead architect)
- **Supersedes:** nothing (new decision; follows from ADR-0001)

## Context

ADR-0001 establishes that the connectome is 800 TB–1.6 PB and can never be
resident in memory. That leaves one question: how is it laid out on disk so that
a simulation can load only what it needs, and so that ingestion can be parallel
and resumable?

The access patterns are not the same, and they conflict:

| Access pattern | Wants | Phase |
|---|---|---|
| Ingestion (append edges from a source) | Sequential append, resumable, parallel by source | 2 |
| Simulation (deliver spikes from fired neurons) | All out-edges of a *specific set* of neurons, fast | 1–4 |
| Analysis (per-region statistics, provenance histograms) | Columnar scan of a subset of columns over a subset of rows | 2–3 |
| Validation (compare to a measured matrix) | Region-level aggregation | 3 |

A single layout cannot be optimal for all four. The decision is therefore not
"which format" but "which format for which pattern, and what metadata makes them
interoperable".

**Sizing the chunks.** A partition must be small enough to load into a working
set and large enough that per-partition overhead is negligible. With a target
working set of ~8–16 GB per partition and ~8–16 bytes per edge, a partition holds
roughly 10⁹ edges. At 10¹⁴ total edges that is ~10⁵ partitions — a number that
must be recorded, not discovered.

**Why partitioning must follow anatomy.** Mean out-degree is ~1163 (ADR-0001),
and cortical connectivity is strongly local: most synapses of a neuron terminate
within a few hundred micrometres. A round-robin partition by neuron id would cut
essentially every edge, so every spike would cross a partition boundary. An
anatomy-aware partition keeps the majority of edges internal. The cut-edge
fraction is the single number that determines distributed cost, and it must be
measured and reported, not assumed.

## Decision

### 1. Three formats, three jobs

| Format | Role | Why |
|---|---|---|
| **Parquet** | Canonical on-disk edge store, ingestion target, analysis | Columnar: a provenance histogram reads one column, not the whole file. Predicate pushdown and row-group statistics. Splits cleanly across workers. |
| **Zarr / HDF5** | Chunked array access for simulation | Chunk-aligned reads of `indptr`/`indices`/`weight`/`delay`; supports partial reads of a large array without loading it. Zarr for object storage, HDF5 for local POSIX. |
| **COO in memory** | Interchange and tests | Already implemented as `SparseConnectome`. |

CSR is the *traversal* form (ADR-0001) and is materialised per partition at load
time from the Zarr/HDF5 chunks. It is not a separate on-disk format, because
storing both COO and CSR doubles a petabyte-scale dataset for no gain.

### 2. Partitioning is by spatial block, with region as the primary key

Partition key, in order:

1. **Region** (from the parcellation) — the coarsest and most semantically
   meaningful cut, and the one that matches how the data is measured.
2. **Spatial block** within a region — a fixed-size voxel grid (default 100 µm
   edge length) so that locality is preserved and blocks are comparable across
   regions.
3. **Neuron-id range** within a block — the final tiebreak, so every partition
   has a deterministic, contiguous id range.

A partition is therefore identified by `(region, block_index, id_lo, id_hi)`.
Long-range projections are stored in a separate **tract partition set**, keyed by
`(source_region, target_region)`, because they are the cut edges by construction
and should be scheduled as a unit.

### 3. Every dataset carries a metadata index map

A `manifest.json` alongside the shards, containing:

- `dataset_id`, `version`, `content_hash` (ADR-0004)
- `n_neurons`, `n_synapses`, `n_partitions`
- the partition table: for each partition, its key, its shard paths, its edge
  count, its id range, and its byte size
- the **id map**: `neuron_id → (partition, local_index)`, stored as a sorted
  array plus offsets so it can be binary-searched without loading
- the **chunk boundaries** for each Zarr/HDF5 array, so a reader can compute
  which chunks a given id range touches
- the provenance block table (ADR-0004)
- the cut-edge count and fraction, once measured

The id map is the piece that makes partitions interoperable: it is what lets a
simulation translate a global neuron id into a partition-local index without
scanning.

### 4. Ingestion is parallel and resumable

- Each source shard is ingested independently and writes its own partition
  shards; there is no global lock.
- Progress is recorded per source shard in a checkpoint file, so an interrupted
  ingest resumes rather than restarting.
- The final `manifest.json` is written only after all shards complete, and the
  `content_hash` is computed over the completed set. A dataset without a
  manifest is not a dataset.

### 5. Chunk sizes are explicit and justified

Default chunk target: **~64 MB per chunk** for Zarr/HDF5 arrays, aligned to
partition boundaries. Rationale: large enough to amortise per-read overhead,
small enough that a partial read of one partition does not pull in a neighbour.
Any deviation must be justified in the config that sets it.

## Consequences

**Easy:** a simulation loads one partition; ingestion parallelises by source;
analysis reads only the columns it needs; the id map makes global↔local
translation O(log n); resumable ingestion means a 1.6 PB build is not
all-or-nothing.

**Hard:** three formats means three code paths and a conversion step. The
partition table and id map are themselves large at 10⁵ partitions (the id map is
~86 × 10⁹ entries × ~12 B ≈ 1 TB — it must be stored as a sorted array and
binary-searched, never loaded as a dict). Cut edges must be measured, and the
measurement is a real experiment, not a constant.

**Owed to the reader:** any distributed performance claim states the cut-edge
fraction and the partition count it was measured at.

## Alternatives considered

- **Single format for everything (Parquet only).** Rejected: Parquet is poor at
  random partial reads of a large array, which is exactly the simulation access
  pattern.
- **Single format (Zarr only).** Rejected: Zarr is poor at column-subset scans
  and predicate pushdown, which is the analysis pattern.
- **Round-robin partitioning by neuron id.** Rejected: cuts essentially every
  edge, maximising inter-partition traffic. It is the worst possible choice for
  a locally-connected graph.
- **One partition per region, no spatial blocks.** Rejected: a large region
  (cortex) would be a single multi-terabyte partition, defeating the purpose.
- **Store both COO and CSR on disk.** Rejected: doubles a petabyte-scale dataset.
- **Load the id map as a Python dict.** Rejected on arithmetic: ~1 TB and
  ~10¹⁰ dict entries. Sorted array + binary search only.
