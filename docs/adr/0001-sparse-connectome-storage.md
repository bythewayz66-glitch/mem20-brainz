# 0001 — Sparse connectome storage; never a dense adjacency matrix

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** mem20-brainz (lead architect)
- **Supersedes:** the provisional "sparse columnar storage" entry in `ARCHITECTURE.md`

## Context

The target is ~86 × 10⁹ neurons and ~10¹⁴ synapses. Before choosing a storage
format, the arithmetic has to be done, because the arithmetic eliminates most of
the option space on its own.

Let $$N = 8.6 \times 10^{10}$$ and $$S = 10^{14}$$.

**Dense adjacency is not merely expensive, it is physically impossible.**

| Representation | Bytes | Size |
|---|---|---|
| Dense, 1 byte per entry | $$N^2 \times 1$$ | **7.4 ZB** |
| Dense, 4 bytes per entry (float32) | $$N^2 \times 4$$ | **29.6 ZB** |
| Dense, 1 bit per entry (packed) | $$N^2 / 8$$ | **924.5 EB** |

For scale: total data stored by humanity is estimated in the low zettabytes.
A single dense float32 adjacency matrix for the human brain is roughly an order
of magnitude more than that. Even the maximally compressed 1-bit form is ~925 EB,
which is ~10⁶× the largest single-machine memory and far beyond any plausible
storage budget for a side project. Dense storage is not a slow option; it is a
non-option.

The reason is sparsity. Mean out-degree is

$$S / N = 10^{14} / 8.6 \times 10^{10} \approx 1163$$

so the matrix is ~99.9999986 % zeros. Storing the zeros is the entire problem.

**Sparse representations, costed:**

| Representation | Per-edge | Total for $$S$$ | Notes |
|---|---|---|---|
| COO, int32 pre/post + float32 weight/delay | 16 B | **1.6 PB** | Ingestion and streaming form |
| CSR, int32 indices + float32 weight | 8 B | **800 TB** | + `indptr` of $$(N{+}1) \times 8$$ = **688 GB** |
| CSR, int32 indices + float32 weight + float16 delay | 10 B | 1.0 PB | Delay kept explicit |
| Quantised CSR, int32 index + float16 weight + float16 delay | 8 B | 800 TB | Lossy; needs a stated error budget |

So even the leanest lossless-ish sparse form is **hundreds of terabytes to
~1.6 PB**. That number is the single most important fact in this project: it
means the connectome is *never* fully resident in memory, on any machine, at any
point. It must be partitioned on disk and streamed.

For contrast, neuron state is comparatively cheap:

| State | Per neuron | Total for $$N$$ |
|---|---|---|
| Minimal (float32 `v` + float16 refractory + uint8 flags) | 7 B | **602 GB** |
| Comfortable (float32 `v`, `w`, `i_syn`, `t_last`, 4 flags, padding) | 32 B | **2.8 TB** |

Neuron state at 602 GB–2.8 TB is large but *fits* on a well-provisioned node or
a small cluster. The connectome does not. This asymmetry — state is tractable,
edges are not — is what makes event-driven, partition-local delivery the central
design constraint rather than an optimisation.

## Decision

1. **No dense N×N structure may be allocated anywhere in this project**, at any
   layer, including tests and analysis code. A dense matrix over a *region*
   (tens to hundreds of regions) is permitted; a dense matrix over *neurons* is
   not.
2. **COO (four parallel arrays) is the canonical in-memory and interchange
   form.** It is what `SparseConnectome` implements today, what ingestion
   produces, and what JSON fixtures and Parquet shards carry.
3. **CSR is the canonical traversal form** for simulation: `indptr` (int64,
   length $$N{+}1$$), `indices` (int32, length $$S$$), `weight` (float32),
   `delay_ms` (float16 or float32). CSC is derived on demand for the rarer
   column-oriented analyses; it is not stored twice.
4. **On-disk persistence is HDF5, Zarr or Parquet, chosen per access pattern**
   (see ADR-0005): Parquet for columnar analytics and ingestion, Zarr/HDF5 for
   chunked array access during simulation.
5. **Indices are `int32` while $$N < 2^{31}$$**, which holds for the human brain
   with ~25× headroom. `indptr` is `int64` because cumulative edge counts exceed
   `int32` at $$S > 2.1 \times 10^9$$, which is reached at ~2 % of human scale.
   This mixed-width choice is deliberate and must not be "simplified".
6. **Any in-memory working set must be justified against an explicit budget** in
   the code that allocates it. "It fit on my machine" is not a justification.

## Consequences

**Easy:** the data model is representable at all; ingestion can stream; a
simulation can load one partition; the same arrays feed NumPy, Numba and JAX
without conversion.

**Hard:** every algorithm must be written for streaming or partitioned access.
Anything that wants "the whole matrix" — spectral analysis, global normalisation,
all-pairs operations — needs a different formulation or an approximation. This is
a real cost and it will bite during Phase 3 validation.

**Owed to the reader:** any figure or claim derived from connectivity must state
how many edges were actually loaded, and whether that was the whole connectome or
a partition.

## Alternatives considered

- **Dense adjacency (any dtype).** Rejected: 7.4–29.6 ZB. Physically impossible.
- **Dense bit-packed adjacency.** Rejected: 924.5 EB, still ~10⁶× too large, and
  bit-unpacking costs more than the sparse gather it replaces.
- **Per-synapse Python objects.** Rejected: a Python object is ~56 B minimum
  before its payload, so ~5.6 PB plus interpreter overhead, and it destroys
  vectorisation. `Synapse` remains as a *description* type for single edges and
  tests, never as the storage form.
- **Graph database (Neo4j et al.).** Rejected for the simulation path: per-edge
  storage overhead is far above 16 B and traversal is not array-shaped. May be
  revisited for interactive exploration of small subgraphs.
- **Store only region-level strengths and generate synapses at load time.**
  Rejected as the *only* form: it forecloses measured edge-level data. It is
  retained as a legitimate synthetic generation strategy (ADR-0004), not as a
  storage format.
